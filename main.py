"""
cloud_run/main.py

Two entry points:
  POST /review   — CI guardrail. Diffs two commits, reviews only the
                   .sql files that changed, posts a GitHub comment.
  POST /analyze  — On-demand audit. Reviews every .sql file in the repo
                   at a given ref. No diff, no GitHub comment.

Both share the same analysis core (dry run -> Gemini rewrite -> dry run
-> deterministic type-risk detection) and emit the same ui_report shape,
so the dashboard renders either one unchanged.

RAG: instead of a hand-maintained dict mapping each SQL file to "the"
Beam consumer, every Beam *function* (including helpers) is embedded
once and indexed. At review time the SQL text itself is embedded and
matched by cosine similarity against that index, so the consumer is
discovered, not configured. QUERY_RUNTIME_CONFIG's beam_consumer field
becomes the deterministic fallback when retrieval confidence is low —
retrieval never gets to fail silently.

Hackathon-scoped: no retries, minimal error handling,
--allow-unauthenticated on the Cloud Run service.
"""
import os
import ast
import json
import subprocess
import tempfile
import shutil
import datetime
import logging
import re
import hashlib
import time
from collections import Counter

import flask
from flask import render_template
from google.cloud import bigquery
from google.cloud import storage
from google import genai
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sql-review-bot")

app = flask.Flask(__name__)

PROJECT_ID = os.environ["PROJECT_ID"]
LOCATION = os.environ.get("LOCATION", "us-central1")


BUCKET_NAME = "sql-review-ui-report"
DEFAULT_REPO_URL = os.environ.get("DEFAULT_REPO_URL")
BYTES_PER_GB = 1024 ** 3
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
if GITHUB_TOKEN:
    logger.info(f"GITHUB_TOKEN loaded (length {len(GITHUB_TOKEN)}, starts with {GITHUB_TOKEN[:4]}...)")
else:
    logger.warning("GITHUB_TOKEN is NOT set in environment")

SQL_DIR = "resources/sql"
BEAM_DIR = "src/beam"

genai_client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
bq_client = bigquery.Client(project=PROJECT_ID)
storage_client = storage.Client(project=PROJECT_ID)

# -----------------------------------------------------------------
# runs_per_day / dag_task_id are NOT derivable from anything the bot
# can see (no Airflow/Composer access). Hand-maintained demo
# assumptions, not detected values. Be explicit about that if asked.
#
# beam_consumer is now only the FALLBACK mapping — used when RAG
# retrieval confidence is too low to trust. The primary path finds
# the consumer via semantic search (see the RAG section below).
# -----------------------------------------------------------------
QUERY_RUNTIME_CONFIG = {
    "resources/sql/retail_lending_portfolio.sql": {
        "dag_task_id": "rm_report_daily", "runs_per_day": 4,
        "beam_consumer": "rm_report.py"},
    "resources/sql/customer_risk_dashboard.sql": {
        "dag_task_id": "risk_dashboard_hourly", "runs_per_day": 7,
        "beam_consumer": "dashboard.py"},
    "resources/sql/delinquency_alerts.sql": {
        "dag_task_id": "delinquency_scan", "runs_per_day": 12,
        "beam_consumer": "delinquency_alerts.py"},
    "resources/sql/employer_concentration.sql": {
        "dag_task_id": "employer_concentration_weekly", "runs_per_day": 1,
        "beam_consumer": "employer_concentration.py"},
    "resources/sql/high_risk_watchlist.sql": {
        "dag_task_id": "risk_watchlist", "runs_per_day": 3,
        "beam_consumer": "watchlist.py"},
    "resources/sql/portfolio_stress_test.sql": {
        "dag_task_id": "portfolio_stress_quarterly", "runs_per_day": 8,
        "beam_consumer": "stress_test.py"},
}
DEFAULT_RUNTIME_CONFIG = {"dag_task_id": "unscheduled", "runs_per_day": 1}


def _authed_url(url: str) -> str:
    """Inject the token for private repos. Never logged, never returned."""
    if GITHUB_TOKEN and url.startswith("https://github.com/"):
        return url.replace("https://", f"https://x-access-token:{GITHUB_TOKEN}@", 1)
    return url


# =================================================================
# Git helpers
# =================================================================
def run_git(*args, cwd):
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr}")
    return result.stdout


def get_commit_message(repo_dir: str, sha: str) -> str:
    try:
        return run_git("log", "-1", "--format=%s", sha, cwd=repo_dir).strip()
    except Exception as e:
        logger.warning(f"Could not fetch commit message for {sha}: {e}")
        return sha


def _file_exists_at(repo_dir, sha, path):
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{sha}:{path}"], cwd=repo_dir, capture_output=True
    )
    return result.returncode == 0


def _read_beam_dir(tmp_dir: str):
    """Concatenate every .py under src/beam into one context blob."""
    beam_context, beam_filenames = "", []
    beam_full_path = os.path.join(tmp_dir, BEAM_DIR)
    if os.path.isdir(beam_full_path):
        for fname in sorted(os.listdir(beam_full_path)):
            if fname.endswith(".py"):
                beam_filenames.append(fname)
                with open(os.path.join(beam_full_path, fname)) as f:
                    beam_context += f"\n--- {fname} ---\n{f.read()}"
    return beam_context, beam_filenames


def clone_and_diff(repo_clone_url: str, before_sha: str, after_sha: str):
    """Clone the repo, return changed .sql files with old/new content,
    the Beam context, the Beam filenames, and the after_sha message."""
    tmp_dir = tempfile.mkdtemp()
    run_git("clone", _authed_url(repo_clone_url), tmp_dir, cwd="/tmp")

    commit_message = get_commit_message(tmp_dir, after_sha)

    changed_files = run_git(
        "diff", "--name-only", before_sha, after_sha, cwd=tmp_dir
    ).splitlines()
    changed_sql = [f for f in changed_files if f.startswith(SQL_DIR) and f.endswith(".sql")]

    results = []
    for path in changed_sql:
        old_exists = _file_exists_at(tmp_dir, before_sha, path)
        new_exists = _file_exists_at(tmp_dir, after_sha, path)

        old_content = run_git("show", f"{before_sha}:{path}", cwd=tmp_dir) if old_exists else None
        new_content = run_git("show", f"{after_sha}:{path}", cwd=tmp_dir) if new_exists else None

        if new_content is None:
            logger.info(f"Skipping {path}: deleted between {before_sha} and {after_sha}")
            continue

        results.append({"path": path, "old": old_content, "new": new_content})

    beam_context, beam_filenames = _read_beam_dir(tmp_dir)
    committed_schemas = read_committed_avro_schemas(tmp_dir)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return results, beam_context, beam_filenames, commit_message, committed_schemas


def clone_all_sql(repo_clone_url: str, ref: str = "main"):
    """Every .sql under resources/sql at `ref`. No before/after diff."""
    tmp_dir = tempfile.mkdtemp()
    run_git("clone", _authed_url(repo_clone_url), tmp_dir, cwd="/tmp")

    try:
        run_git("checkout", ref, cwd=tmp_dir)
    except RuntimeError as e:
        logger.warning(f"checkout {ref} failed ({e}); staying on default branch")

    head_sha = run_git("rev-parse", "--short", "HEAD", cwd=tmp_dir).strip()

    results = []
    sql_full_path = os.path.join(tmp_dir, SQL_DIR)
    if os.path.isdir(sql_full_path):
        for fname in sorted(os.listdir(sql_full_path)):
            if fname.endswith(".sql"):
                with open(os.path.join(sql_full_path, fname)) as f:
                    results.append({
                        "path": f"{SQL_DIR}/{fname}",
                        "old": None,
                        "new": f.read(),
                    })

    beam_context, beam_filenames = _read_beam_dir(tmp_dir)
    committed_schemas = read_committed_avro_schemas(tmp_dir)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return results, beam_context, beam_filenames, head_sha, committed_schemas


# =================================================================
# BigQuery dry run + costing
# =================================================================
BYTES_PER_TB = 1024 ** 4
COST_PER_TB_USD = 6.25


def dry_run_bytes(sql_text: str) -> tuple[int, bool]:
    """Returns (bytes_scanned, success)."""
    if not sql_text:
        return 0, True

    sql_text = extract_sql(sql_text)
    if not sql_text.strip():
        return 0, True

    try:
        job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        job = bq_client.query(sql_text, job_config=job_config)
        return job.total_bytes_processed, True
    except Exception:
        logger.exception(f"Dry run failed:\n{sql_text}")
        return 0, False


def bytes_to_cost(num_bytes: int) -> float:
    if not num_bytes:
        return 0.0
    return (num_bytes / BYTES_PER_TB) * COST_PER_TB_USD


def _extract_tables(sql_text: str):
    """Fully-qualified table names enclosed in backticks."""
    return list(set(re.findall(r'`([^`]+)`', sql_text)))


def _count_tables_by_dataset(changed_results):
    seen = set()
    counts = Counter()
    for r in changed_results:
        for t in _extract_tables(r["new_sql"]):
            if t in seen:
                continue
            seen.add(t)
            parts = t.split(".")
            dataset = parts[-2] if len(parts) >= 2 else "unknown"
            counts[dataset] += 1
    return counts


_schema_manifest_cache: dict[str, str] = {}  # table ref → manifest chunk, lives for process lifetime

def build_schema_manifest(sql_text: str) -> str:
    tables = _extract_tables(sql_text)
    logger.info(f"Extracted tables: {tables}")
    manifest = []

    for table in tables:
        try:
            if table in _schema_manifest_cache:
                logger.info(f"Schema cache HIT for {table}")
                manifest.append(_schema_manifest_cache[table])
                continue

            parts = table.split(".")
            if len(parts) == 3:
                project, dataset, table_name = parts
            elif len(parts) == 2:
                project = PROJECT_ID
                dataset, table_name = parts
            else:
                logger.warning(f"Skipping unsupported table reference: {table}")
                continue

            logger.info(f"Fetching INFORMATION_SCHEMA for {project}.{dataset}.{table_name}")

            # INFORMATION_SCHEMA.TABLE_STORAGE (unlike .COLUMNS) doesn't
            # reliably auto-resolve its query location from the dataset
            # reference alone — left unpinned it can default to the
            # multi-region "US" and 404 on a dataset that actually lives
            # in a specific region (e.g. us-central1), even though the
            # exact same dataset works fine for .COLUMNS. Look up the
            # dataset's real location once and pin both queries to it;
            # if even the lookup fails, fall back to unpinned (location=
            # None behaves the same as omitting it) rather than losing
            # the whole table's schema entry over a metadata query.
            try:
                dataset_location = bq_client.get_dataset(f"{project}.{dataset}").location
            except Exception as e:
                logger.warning(f"Could not determine location of {project}.{dataset}, "
                               f"querying without a pinned location: {e}")
                dataset_location = None

            column_query = f"""
            SELECT column_name, data_type, is_nullable,
                   is_partitioning_column, clustering_ordinal_position
            FROM `{project}.{dataset}.INFORMATION_SCHEMA.COLUMNS`
            WHERE table_name='{table_name}'
            ORDER BY ordinal_position
            """
            columns = list(bq_client.query(column_query, location=dataset_location).result())

            table_query = f"""
            SELECT table_name, row_count, size_bytes
            FROM `{project}.{dataset}.INFORMATION_SCHEMA.TABLE_STORAGE`
            WHERE table_name='{table_name}'
            """
            try:
                table_info = list(bq_client.query(table_query, location=dataset_location).result())
            except Exception as e:
                logger.warning(f"Could not fetch TABLE_STORAGE for {table}: {e}")
                table_info = []

            chunk = [f"\n===== TABLE : {table} ====="]
            if table_info:
                t = table_info[0]
                chunk.append(f"Rows : {t.row_count}")
                chunk.append(f"Storage : {t.size_bytes} bytes")

            partition_cols, clustering_cols = [], []
            for c in columns:
                nullable_marker = " [NULLABLE]" if c.is_nullable == "YES" else ""
                chunk.append(f"- {c.column_name} ({c.data_type}){nullable_marker}")
                if c.is_partitioning_column == "YES":
                    partition_cols.append(c.column_name)
                if c.clustering_ordinal_position:
                    clustering_cols.append(c.column_name)

            chunk.append(f"Partition Columns : {partition_cols if partition_cols else 'None'}")
            chunk.append(f"Cluster Columns : {clustering_cols if clustering_cols else 'None'}")

            chunk_str = "\n".join(chunk)
            _schema_manifest_cache[table] = chunk_str
            manifest.append(chunk_str)

        except Exception:
            logger.exception(f"Failed to discover schema for {table}")

    return "\n".join(manifest)


# =================================================================
# Vertex AI context cache
#
# Only the schema manifest is cached now. It's the part that's
# genuinely stable and shared across queries (most queries here hit
# the same two tables). The Beam consumer content is no longer cached
# here — RAG retrieves a small, query-specific slice of it fresh on
# every call (see below), so caching the whole Beam directory would
# just be redundant with what retrieval already sends inline.
# =================================================================
CACHE_STORAGE_USD_PER_MILLION_TOKENS_PER_HOUR = 1.00
CACHE_WRITE_USD_PER_MILLION_TOKENS = 0.30   # billed once, at creation
CACHE_READ_USD_PER_MILLION_TOKENS = 0.03    # billed per reusing call


def cache_storage_cost_per_hour(token_count: int) -> float:
    return (token_count / 1_000_000) * CACHE_STORAGE_USD_PER_MILLION_TOKENS_PER_HOUR


def cache_write_cost(token_count: int) -> float:
    return (token_count / 1_000_000) * CACHE_WRITE_USD_PER_MILLION_TOKENS


def cache_read_cost(token_count: int) -> float:
    return (token_count / 1_000_000) * CACHE_READ_USD_PER_MILLION_TOKENS


def find_existing_cache(display_name):
    try:
        for cache in genai_client.caches.list():
            if cache.display_name == display_name:
                logger.info(f"Found existing cache: {cache.name} (expires {cache.expire_time})")
                return cache.name
    except Exception as e:
        logger.warning(f"Cache lookup failed ({e}), will attempt to create a new one")
    return None


def get_cache_token_count(cache_name: str) -> int:
    """Exact token count from the API, not our len()//4 estimate."""
    try:
        cache = genai_client.caches.get(name=cache_name)
        return cache.usage_metadata.total_token_count
    except Exception as e:
        logger.warning(f"Could not fetch cache token count: {e}")
        return 0


def get_or_create_cache(schema_manifest: str):
    """Cache key is the schema manifest ALONE (see module note above)."""
    PROMPT_VERSION = "v2-rag-schema-only"
    combined = PROMPT_VERSION + schema_manifest
    cache_hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()[:12]
    display_name = f"grid-schema-{cache_hash}"

    existing = find_existing_cache(display_name)
    if existing:
        token_count = get_cache_token_count(existing)
        logger.info(
            f"Reusing existing cache: {existing} | {token_count:,} tokens | "
            f"storage: ${cache_storage_cost_per_hour(token_count):.6f}/hr | "
            f"this read: ${cache_read_cost(token_count):.6f}"
        )
        return {"name": existing, "status": "HIT", "token_count": token_count}

    approx_tokens = len(combined) // 4
    logger.info(f"No existing cache. Creating one, ~{approx_tokens} tokens ({len(combined)} chars)")
    try:
        cache = genai_client.caches.create(
            model="gemini-2.5-flash",
            config={"contents": [combined], "ttl": "86400s", "display_name": display_name},
        )
        actual_tokens = cache.usage_metadata.total_token_count
        logger.info(
            f"Cache created: {cache.name} | {actual_tokens:,} tokens | "
            f"write: ${cache_write_cost(actual_tokens):.6f} | "
            f"storage: ${cache_storage_cost_per_hour(actual_tokens):.6f}/hr"
        )
        return {"name": cache.name, "status": "MISS", "token_count": actual_tokens}
    except Exception as e:
        logger.warning(f"Cache creation failed ({e}), falling back to inline context")
        return {"name": None, "status": "DISABLED", "token_count": 0}


# =================================================================
# RAG: Beam consumer retrieval
#
# Replaces "look up the consumer file in a hand-maintained dict" with
# "embed every Beam function once, embed the SQL, retrieve the closest
# matches." This is what makes consumer discovery scale past a small,
# manually-curated repo — QUERY_RUNTIME_CONFIG's beam_consumer field
# becomes the safety net for when confidence is too low to trust.
#
# Two things this deliberately does NOT touch:
#   - detect_type_risks() still reads the FULL text of the resolved
#     consumer file, not the retrieved snippet. Retrieval picks WHICH
#     file to trust; once picked, the deterministic AST scan needs the
#     whole file (helper functions included) to be correct.
#   - If the embedding call fails for any reason (model unavailable,
#     SDK signature mismatch, quota), indexing/retrieval fails open:
#     the index comes back empty, retrieval reports "no match", and
#     the pipeline behaves exactly as it did before RAG existed.
# =================================================================
RAG_EMBED_MODEL = "text-embedding-004"
RAG_TOP_K = 3
RAG_MIN_SIMILARITY = 0.55  # below this, don't trust retrieval — use config fallback
RAG_MIN_MARGIN = 0.03      # top-1 must clearly beat runner-up, or defer to config
#   Two structurally similar queries (e.g. both aggregating max_dti /
#   max_revol_util / total_outstanding from near-identical CTEs) can be
#   semantically closer to EACH OTHER than either is to its own real
#   consumer. An absolute similarity threshold alone doesn't catch this
#   — both candidates can clear it. A margin check does: if the top-1
#   and top-2 scores are nearly tied, that's the model telling us it's
#   genuinely unsure, and a known-good config mapping is safer than a
#   coin-flip guess.

# Retrieval score = containment (dominant) + cosine similarity (tie-break),
# computed per FILE, not per function.
#
# Two rounds of measurement against this repo's real data:
#
# 1. Column-signature cosine similarity ALONE got 4 of 6 queries wrong —
#    dense embeddings treat "customer_name city state employer occupation"
#    as semantically similar regardless of which EXACT columns overlap,
#    because that's generic lending-domain vocabulary shared by nearly
#    every function here.
#
# 2. Switching to per-FUNCTION column containment fixed that, but broke
#    2 of 6 differently: a tiny single-column predicate (e.g. a filter
#    needing only max_dti and max_revol_util) trivially hits 100%
#    containment against almost any risk-related query, out-ranking the
#    real formatter function that needs 9 columns and can mathematically
#    never reach 1.0. Scoring at the FILE level (union of every row-
#    function's columns in that file) fixes this: a file's helper
#    predicates and its real formatter share the same denominator, so a
#    narrow predicate can't win on the file's behalf if its siblings need
#    columns the query doesn't have.
#
# Verified by hand + a standalone validation script against all 6 real
# (query, file) pairs in this repo (see chat/PR discussion): file-level
# containment gets all 6 right — 3 via confident retrieval with a clear
# margin, 3 via the margin/threshold safety net correctly recognizing a
# genuine near-tie and deferring to the known-good config default rather
# than guessing. That's the fallback working as designed, not masking a
# broken retrieval.
#
# Cosine similarity remains a minority tie-breaker (RAG_COSINE_WEIGHT) —
# it's what keeps this a genuine embeddings-based retrieval system rather
# than pure keyword matching, and it's still useful for ranking WHICH
# function(s) within the winning file to show Gemini as context.
RAG_CONTAINMENT_WEIGHT = 0.85
RAG_COSINE_WEIGHT = 0.15


def _containment(function_columns: list[str], query_columns: list[str]) -> float:
    """What fraction of `function_columns` are present in `query_columns`.
    1.0 means the query supplies every column this function needs,
    regardless of how many extra columns the query also has — full
    containment, not symmetric overlap (Jaccard would unfairly penalize
    functions matched against wide `SELECT *` queries)."""
    if not function_columns:
        return 0.0
    fset, qset = set(function_columns), set(query_columns)
    return len(fset & qset) / len(fset)

_BEAM_INDEX_CACHE: dict[str, list[dict]] = {}  # beam_context hash -> indexed functions
#   NOTE: this is a plain in-process dict, unlike the Vertex cache above.
#   It resets on cold start and isn't shared across instances — fine here,
#   since indexing ~20-30 Beam functions costs one small embedding call
#   and takes well under a second.


def _split_beam_context(beam_context: str) -> dict:
    """Reverse _read_beam_dir's '--- fname ---\\ncontent' concatenation
    back into {filename: source}."""
    parts = re.split(r"\n--- (.+?) ---\n", beam_context)
    sources = {}
    for i in range(1, len(parts), 2):
        fname = parts[i]
        content = parts[i + 1] if i + 1 < len(parts) else ""
        sources[fname] = content
    return sources


def _consumes_row(node: ast.FunctionDef) -> bool:
    """Only functions that actually accept a `row` argument are real
    per-record consumers. Orchestration functions like pipeline.run()
    wire branches together and mention every SQL file/table by name —
    which makes their embedding generically close to any query, often
    closer than the real narrow formatter function is. Left unfiltered,
    that lets the orchestrator win retrieval it has no business winning,
    and since it never touches row[...] it's also useless to
    detect_type_risks. Excluding non-row functions from the index fixes
    both at once."""
    return "row" in {a.arg for a in node.args.args}


def _row_columns(source: str) -> list[str]:
    """Every row["col"] subscript in this source, deduplicated and
    sorted. Same extraction detect_type_risks uses, generalized to ALL
    subscripts (not just ones inside arithmetic/comparison) — here
    we're using it as a signature of what the function reads, not
    scanning for bugs."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    cols = {
        n.slice.value for n in ast.walk(tree)
        if isinstance(n, ast.Subscript)
        and isinstance(n.slice, ast.Constant)
        and isinstance(n.slice.value, str)
    }
    return sorted(cols)


def build_function_index(beam_sources: dict) -> list[dict]:
    """One entry per row-consuming function def (including nested/
    helper functions) across all Beam files. Each entry carries its
    real source (shown to Gemini once retrieved) AND its column
    signature (what actually gets embedded — see module note above on
    why raw source is too noisy a retrieval signal here)."""
    index = []
    for fname, src in beam_sources.items():
        try:
            tree = ast.parse(src)
        except SyntaxError:
            logger.warning(f"RAG index: could not parse {fname}, skipping")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and _consumes_row(node):
                snippet = ast.get_source_segment(src, node)
                if snippet and snippet.strip():
                    index.append({
                        "file": fname,
                        "function": node.name,
                        "source": snippet,
                        "columns": _row_columns(snippet),
                    })
    return index


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Vertex AI embeddings, one vector per input text.

    NOTE: unverified against a live project in this environment — the
    google-genai SDK's embed_content signature has shifted across
    versions. Before relying on this in the demo, run a one-off sanity
    check (see the message accompanying this file) and adjust the
    call shape if it errors. Every caller of this function already
    wraps it in try/except and fails open to the static config, so a
    signature mismatch degrades gracefully rather than breaking /analyze.
    """
    if not texts:
        return []
    result = genai_client.models.embed_content(model=RAG_EMBED_MODEL, contents=texts)
    return [e.values for e in result.embeddings]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


# =================================================================
# BigQuery-backed vector store for the RAG function index
#
# The in-process dict above (_BEAM_INDEX_CACHE) is an L1 cache — fast,
# but it resets on every Cloud Run cold start / new instance, so a
# fresh instance re-calls the embedding API even when the Beam code
# hasn't changed. This table is the L2 cache: one row per indexed
# function, keyed by content_hash (same hash used for the in-process
# cache). A cold start checks BigQuery before re-embedding — if this
# exact Beam code was already indexed by ANY instance, the vectors are
# reused, not recomputed. It also makes the vectors inspectable with a
# plain SQL query instead of being invisible internal state:
#   bq query "SELECT file, function, columns FROM \`PROJECT.rag_index.beam_function_embeddings\`"
# =================================================================
VECTOR_DATASET = os.environ.get("VECTOR_DATASET", "rag_index")
VECTOR_TABLE = "beam_function_embeddings"
VECTOR_TABLE_ID = f"{PROJECT_ID}.{VECTOR_DATASET}.{VECTOR_TABLE}"

_vector_table_ready = False


def _ensure_vector_table():
    """Creates the dataset/table on first use. Idempotent — safe to
    call on every cold start; a no-op after the first successful call
    in this instance's lifetime."""
    global _vector_table_ready
    if _vector_table_ready:
        return

    dataset_ref = bigquery.DatasetReference(PROJECT_ID, VECTOR_DATASET)
    try:
        bq_client.get_dataset(dataset_ref)
    except Exception:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = LOCATION
        bq_client.create_dataset(dataset, exists_ok=True)
        logger.info(f"Created BigQuery dataset {VECTOR_DATASET} in {LOCATION}")

    bq_client.query(
        f"""
        CREATE TABLE IF NOT EXISTS `{VECTOR_TABLE_ID}` (
            content_hash STRING,
            file STRING,
            function STRING,
            columns ARRAY<STRING>,
            source STRING,
            embedding ARRAY<FLOAT64>,
            indexed_at TIMESTAMP
        )
        """,
        location=LOCATION,
    ).result()
    _vector_table_ready = True


def _load_index_from_bigquery(content_hash: str) -> list[dict]:
    """Returns [] if this content_hash has never been indexed before —
    that's the normal, expected case for the first request after a
    Beam code change, not an error."""
    _ensure_vector_table()
    job = bq_client.query(
        f"""
        SELECT file, function, columns, source, embedding
        FROM `{VECTOR_TABLE_ID}`
        WHERE content_hash = @content_hash
        """,
        job_config=bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("content_hash", "STRING", content_hash),
        ]),
        location=LOCATION,
    )
    return [
        {
            "file": r.file, "function": r.function,
            "columns": list(r.columns), "source": r.source,
            "embedding": list(r.embedding),
        }
        for r in job.result()
    ]


def _save_index_to_bigquery(content_hash: str, index: list[dict]):
    """Streaming insert — fire-and-forget from the caller's perspective
    (wrapped in try/except there). Non-fatal if it fails: the index
    still works for this request via the in-process cache, it just
    won't survive to the next cold start."""
    _ensure_vector_table()
    now = datetime.datetime.utcnow().isoformat()
    rows = [
        {
            "content_hash": content_hash,
            "file": e["file"],
            "function": e["function"],
            "columns": e["columns"],
            "source": e["source"],
            "embedding": e["embedding"],
            "indexed_at": now,
        }
        for e in index
    ]
    errors = bq_client.insert_rows_json(VECTOR_TABLE_ID, rows)
    if errors:
        logger.warning(f"BigQuery vector insert had errors (non-fatal): {errors}")
    else:
        logger.info(f"Persisted {len(rows)} function embeddings to {VECTOR_TABLE_ID}")


def get_function_index(beam_context: str, beam_sources: dict) -> list[dict]:
    """L1 (in-process dict) -> L2 (BigQuery) -> rebuild-from-scratch,
    in that order. Each layer is checked only if the previous one
    missed, so a warm instance never touches BigQuery and a cold
    instance only touches the embedding API when BigQuery also misses
    (i.e. this exact Beam code has genuinely never been indexed)."""
    key = hashlib.sha256(beam_context.encode("utf-8")).hexdigest()[:16]
    if key in _BEAM_INDEX_CACHE:
        return _BEAM_INDEX_CACHE[key]

    try:
        index = _load_index_from_bigquery(key)
        if index:
            logger.info(f"RAG index loaded from BigQuery: {len(index)} functions (content_hash={key})")
            _BEAM_INDEX_CACHE[key] = index
            return index
    except Exception:
        logger.exception("BigQuery vector lookup failed — falling back to rebuilding index")

    index = []
    try:
        index = build_function_index(beam_sources)
        if index:
            # Embed the column signature, not the raw source — see the
            # module note above. Fall back to raw source only for the
            # rare function that takes `row` but touches no row["col"]
            # (shouldn't normally happen given _consumes_row already
            # filtered to row-taking functions, but stay safe).
            embed_inputs = [" ".join(e["columns"]) or e["source"] for e in index]
            vectors = embed_texts(embed_inputs)
            for entry, vec in zip(index, vectors):
                entry["embedding"] = vec
        logger.info(f"RAG index built: {len(index)} functions across {len(beam_sources)} files")
        if index:
            try:
                _save_index_to_bigquery(key, index)
            except Exception:
                logger.exception("Failed to persist RAG index to BigQuery (non-fatal)")
    except Exception:
        logger.exception("RAG indexing failed — retrieval disabled for this request, using config fallback")
        index = []

    _BEAM_INDEX_CACHE[key] = index
    return index


def _extract_query_columns(sql_text: str, schema_manifest: str) -> list[str]:
    """Approximate the columns this query is about: every column name
    known from INFORMATION_SCHEMA (already fetched into schema_manifest
    for this query — no extra BigQuery call) that appears as a whole
    word OUTSIDE any CAST(...) expression in the SQL text.

    Why strip CAST(...) contents: this repo's queries route almost every
    raw numeric column through a CAST(...), frequently renamed in the
    same breath — e.g. `SUM(CAST(loan_amnt AS FLOAT64)) AS total_amount`.
    The raw pre-cast name there is a computation INPUT, not necessarily a
    genuine output column; counting "loan_amnt" as present because it's
    the cast argument is often wrong once the value is renamed to
    total_amount. Stripping CAST(...) contents keeps the signal that
    actually matters: alias names the query itself computes (the
    strongest, most query-specific evidence there is) and explicit,
    non-renamed column references (JOIN keys, WHERE-clause columns, plain
    SELECT-list columns).

    Known limitation: for a bare `SELECT *` / `table.*`, this can't see
    columns that ONLY exist via wildcard pass-through and are never
    otherwise named in the query text (e.g. a column nobody CASTs,
    filters on, or aliases). Verified this doesn't silently produce a
    wrong answer in practice — for the one real demo query this affects,
    the margin/config-fallback safety net below catches it and defers to
    the known-good default instead of guessing.
    """
    known_cols = set(re.findall(r"^- (\w+) \(", schema_manifest, re.MULTILINE))
    stripped = re.sub(r"CAST\([^()]*\)", " ", sql_text, flags=re.IGNORECASE)
    stripped = re.sub(r"CAST\([^()]*\)", " ", stripped, flags=re.IGNORECASE)  # one nesting level
    present = [c for c in known_cols if re.search(rf"\b{re.escape(c)}\b", stripped)]
    return sorted(present)


def retrieve_consumer_context(sql_text: str, index: list[dict], fallback_file: str,
                              beam_sources: dict, schema_manifest: str):
    """Returns (context_to_send_to_gemini, consumer_filename, matched, confidence).

    Retrieval decides WHICH FILE is the consumer, scored by that file's
    row-functions taken TOGETHER (union of columns across every row-
    function in the file) — not by ranking individual functions. See the
    module note above on why: an individual tiny helper predicate can
    trivially hit 100% containment and out-rank the real, larger
    formatter function that needs many more columns. Aggregating to the
    file level fixes that, since the file is what QUERY_RUNTIME_CONFIG's
    beam_consumer actually names and what the rest of the pipeline
    (detect_type_risks, the UI report) treats as the unit of "consumer."

    `matched` is False whenever retrieval didn't run, errored, or came
    back below RAG_MIN_SIMILARITY / RAG_MIN_MARGIN — in every one of
    those cases this falls back to the exact same behavior as the
    pre-RAG config lookup: the configured file's full source, nothing
    inferred.
    """
    if not index:
        return beam_sources.get(fallback_file, ""), fallback_file, False, 0.0

    query_cols = _extract_query_columns(sql_text, schema_manifest)
    embed_input = " ".join(query_cols) if query_cols else sql_text
    try:
        query_vec = embed_texts([embed_input])[0]
    except Exception:
        logger.exception("RAG query embedding failed — falling back to configured consumer")
        return beam_sources.get(fallback_file, ""), fallback_file, False, 0.0

    by_file: dict[str, list[dict]] = {}
    for e in index:
        by_file.setdefault(e["file"], []).append(e)

    # Combined score per FILE: containment of the file's UNION of needed
    # columns does almost all the work (see module note above); cosine
    # similarity (max across the file's functions) only breaks near-ties
    # between files with equal/near-equal containment.
    scored = []
    for fname, entries in by_file.items():
        file_cols: set[str] = set()
        for e in entries:
            file_cols |= set(e["columns"])
        cont = _containment(list(file_cols), query_cols)
        cos = max((_cosine(query_vec, e["embedding"]) for e in entries), default=0.0)
        combined = RAG_CONTAINMENT_WEIGHT * cont + RAG_COSINE_WEIGHT * cos
        scored.append((combined, fname, entries))
    scored.sort(key=lambda t: t[0], reverse=True)

    if not scored:
        return beam_sources.get(fallback_file, ""), fallback_file, False, 0.0

    best_score = scored[0][0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    margin = best_score - second_score

    if best_score < RAG_MIN_SIMILARITY:
        logger.warning(
            f"RAG confidence low ({best_score:.2f} < {RAG_MIN_SIMILARITY}) "
            f"— falling back to configured consumer '{fallback_file}'"
        )
        return beam_sources.get(fallback_file, ""), fallback_file, False, best_score

    if margin < RAG_MIN_MARGIN and fallback_file:
        logger.warning(
            f"RAG top match ambiguous — top='{scored[0][1]}' ({best_score:.3f}) "
            f"vs runner-up='{scored[1][1]}' ({second_score:.3f}), "
            f"margin={margin:.3f} < {RAG_MIN_MARGIN} "
            f"— deferring to configured consumer '{fallback_file}'"
        )
        return beam_sources.get(fallback_file, ""), fallback_file, False, best_score

    _, primary_file, entries = scored[0]
    entries_ranked = sorted(
        entries, key=lambda e: _cosine(query_vec, e["embedding"]), reverse=True
    )
    top = entries_ranked[:RAG_TOP_K]
    context = "\n\n".join(
        f"--- {e['file']} :: {e['function']}() ---\n{e['source']}"
        for e in top
    )
    return context, primary_file, True, best_score


# =================================================================
# Avro schema risk detection
#
# Real-world shape this mirrors: the Beam pipeline is the last stop
# before a record gets written to GCS as Avro, so the Beam formatter's
# output dict effectively IS the Avro schema in production. This check
# never executes SQL or Beam (same fail-open, static-analysis-only
# policy as the rest of the app) — it compares two things it can both
# derive without running anything:
#
#   1. "Expected" Avro schema  — every column BigQuery says this
#      query's source tables have, mapped to its Avro type. What
#      COULD be delivered.
#   2. "Dataset resulting from the Beam pipeline" — the resolved
#      consumer's primary formatter function's `return {...}` dict,
#      statically resolved field-by-field back to the row["col"]
#      expression(s) that build each value. What the Beam code
#      actually COMMITS to delivering, read from its source rather
#      than executed.
#
# A field in (2) whose source column isn't in (1) means the SQL no
# longer produces something the Avro record still promises — that
# will KeyError in production before Avro serialization ever runs.
# Flagged as a HIGH risk, in the exact same shape as the existing
# STRING-in-arithmetic risks, so it shows up in the same "Type Risks"
# section rather than needing its own UI plumbing.
# =================================================================
def _select_primary_function(file_source: str) -> dict | None:
    """Within one Beam file, picks the function most likely to be the
    actual output-record builder (the one whose return dict becomes
    the Avro record) rather than a filter/predicate helper.

    Heuristic: the row-consuming function with the LARGEST column
    signature, name prefix as a tie-break. Same insight the file-level
    RAG fix relies on — a predicate needs a handful of columns to
    decide keep/drop, a formatter needs most/all of them to build the
    output record, so column-count is a reliable proxy for "this is
    the formatter" without needing a naming convention to hold.
    """
    try:
        tree = ast.parse(file_source)
    except SyntaxError:
        return None

    candidates = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and _consumes_row(node):
            snippet = ast.get_source_segment(file_source, node)
            if snippet and snippet.strip():
                candidates.append({
                    "name": node.name,
                    "source": snippet,
                    "columns": _row_columns(snippet),
                })
    if not candidates:
        return None

    def sort_key(c):
        name_bonus = 1 if re.match(r"^(format_|build_|to_|make_)", c["name"]) else 0
        return (len(c["columns"]), name_bonus)

    candidates.sort(key=sort_key, reverse=True)
    return candidates[0]


def _bq_type_to_avro(bq_type: str) -> str:
    """Best-effort BigQuery -> Avro primitive type mapping, covering
    the types this repo's tables actually use. Anything unrecognized
    falls back to "string" (Avro's most permissive type) so a mapping
    gap produces an overly-lenient comparison rather than a crash."""
    mapping = {
        "STRING": "string", "BYTES": "bytes",
        "INT64": "long", "INTEGER": "long",
        "FLOAT64": "double", "FLOAT": "double",
        "NUMERIC": "double", "BIGNUMERIC": "double",
        "BOOL": "boolean", "BOOLEAN": "boolean",
        "TIMESTAMP": "long", "DATE": "string",
        "DATETIME": "string", "TIME": "string",
    }
    return mapping.get(bq_type.upper(), "string")


def build_expected_avro_schema(sql_text: str, schema_manifest: str) -> dict:
    """Step 1: {column_name: avro_type}, one entry per column the QUERY
    actually outputs, typed via its declared BigQuery type. Reuses the
    same whole-word, CAST-aware extraction RAG retrieval already
    validated (_extract_query_columns) — deliberately NOT "every
    column the underlying table has": schema_manifest comes from
    INFORMATION_SCHEMA on the source TABLE, which still lists a column
    even after a rewrite prunes it from the SELECT list. The whole
    point of this check is to catch exactly that drift (a SQL rewrite
    quietly dropping a column Beam still expects), so "expected" has
    to mean "what this query currently selects," not "what the table
    has.\""""
    query_cols = set(_extract_query_columns(sql_text, schema_manifest))
    types = dict(re.findall(r"^- (\w+) \((\w+)\)", schema_manifest, re.MULTILINE))
    return {c: _bq_type_to_avro(types[c]) for c in query_cols if c in types}


def build_beam_output_fields(primary_function: dict) -> dict | None:
    """Step 2: "the dataset resulting from the beam pipeline" — parses
    the primary formatter's `return {...}` dict literal and maps each
    output key to the row["col"] column(s) feeding its value. Not
    executed data; the field-by-field shape the function's source
    statically commits to producing, consistent with this app never
    running Beam code, only reading it.

    Returns None if the function has no dict-literal return (e.g. it
    returns a pre-built object) — the caller treats that as "not
    applicable," not a mismatch.
    """
    try:
        tree = ast.parse(primary_function["source"])
    except SyntaxError:
        return None

    func_node = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)), None)
    if not func_node:
        return None

    return_dict = None
    for node in ast.walk(func_node):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            return_dict = node.value
            break
    if return_dict is None:
        return None

    # Resolve one level of local-variable indirection. A common real
    # pattern (e.g. dashboard.py's format_dashboard_row): compute a
    # derived value once, name it, then return the name —
    # `effective_rate = row["emp_length"] * 0.05; return
    # {"effective_rate": effective_rate, ...}`. The dict VALUE there is
    # a bare Name, not a Subscript, so without this the source column
    # would be invisible to the comparison below — silently treating a
    # real dependency as "computed, nothing to check." One level of
    # lookup catches this without needing full data-flow analysis.
    local_assigns = {}
    for node in ast.walk(func_node):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            local_assigns[node.targets[0].id] = node.value

    def resolve_cols(node):
        if isinstance(node, ast.Name) and node.id in local_assigns:
            node = local_assigns[node.id]
        return [n.slice.value for n in ast.walk(node)
                if isinstance(n, ast.Subscript)
                and isinstance(n.slice, ast.Constant)
                and isinstance(n.slice.value, str)]

    fields = {}
    for key_node, value_node in zip(return_dict.keys, return_dict.values):
        if not (isinstance(key_node, ast.Constant) and isinstance(key_node.value, str)):
            continue
        fields[key_node.value] = resolve_cols(value_node)
    return fields


def build_avro_schema_document(expected_schema: dict, record_name: str) -> dict:
    """Turns {column: avro_type} into an actual, valid Avro schema
    document — the literal .avsc content a real Avro-to-GCS delivery
    would carry, not just an internal comparison structure. Every
    field is nullable (["null", type]) since BigQuery columns default
    to NULLABLE and schema_manifest doesn't currently track REQUIRED
    per-column — a documented simplification, not a hidden one."""
    return {
        "type": "record",
        "name": record_name,
        "namespace": "bq_devops_cost_guardrail",
        "fields": [
            {"name": col, "type": ["null", avro_type], "default": None}
            for col, avro_type in sorted(expected_schema.items())
        ],
    }


def upload_avro_schema(schema_doc: dict, query_path: str) -> str | None:
    """Writes the generated .avsc to GCS — a real file you can open,
    not just numbers in a report — mirroring how this repo's real
    Avro-to-GCS delivery would publish its schema alongside the data.
    Fails open: this is an inspection artifact, it should never block
    analysis if the upload itself fails."""
    try:
        blob_name = f"avro_schemas/{query_path.replace('/', '_')}.avsc"
        storage_client.bucket(BUCKET_NAME).blob(blob_name).upload_from_string(
            json.dumps(schema_doc, indent=2), content_type="application/json"
        )
        gcs_path = f"gs://{BUCKET_NAME}/{blob_name}"
        logger.info(f"Uploaded generated Avro schema to {gcs_path}")
        return gcs_path
    except Exception:
        logger.exception(f"Failed to upload Avro schema for {query_path} (non-fatal)")
        return None


def read_committed_avro_schemas(tmp_dir: str) -> dict:
    """Read every .avsc from resources/avro/ in the cloned repo.
    Returns {query_name: avsc_doc} where query_name is the SQL filename
    without extension (e.g. 'customer_risk_dashboard').
    Returns {} if the directory doesn't exist — fails open, never blocks analysis."""
    avro_dir = os.path.join(tmp_dir, "resources", "avro")
    schemas = {}
    if not os.path.isdir(avro_dir):
        logger.info("No resources/avro/ directory found in repo — committed schema check disabled")
        return schemas
    for fname in sorted(os.listdir(avro_dir)):
        if fname.endswith(".avsc"):
            path = os.path.join(avro_dir, fname)
            try:
                with open(path) as f:
                    schemas[fname[:-5]] = json.load(f)
                logger.info(f"Loaded committed Avro schema: {fname}")
            except Exception as e:
                logger.warning(f"Could not load committed schema {fname}: {e}")
    return schemas


def check_committed_avro_contract(primary_function: dict | None, consumer_file: str,
                                   query_path: str, committed_schemas: dict) -> list:
    """Compare the committed .avsc contract (resources/avro/) against what
    the current Beam formatter actually produces.

    The committed .avsc is the authoritative schema contract for what the
    Beam pipeline must deliver. This check catches two classes of drift:

      HIGH — Fields the contract declares but the formatter no longer
              returns. Downstream Avro readers will break on these.
      LOW  — Fields the formatter produces that aren't in the contract.
              Schema is stale and needs updating.

    This is complementary to check_avro_schema_risk(), which verifies the
    SQL→Beam direction (source columns still exist in the query output).
    This function guards the Beam→contract direction."""
    query_name = os.path.basename(query_path).replace(".sql", "")
    committed = committed_schemas.get(query_name)
    if not committed:
        logger.info(f"No committed .avsc for '{query_name}' — skipping contract check")
        return []

    committed_fields = {f["name"] for f in committed.get("fields", [])}

    if primary_function is None:
        logger.info(f"No primary function for {query_path} — skipping contract check")
        return []

    beam_fields = build_beam_output_fields(primary_function)
    if beam_fields is None:
        return []
    beam_output_keys = set(beam_fields.keys())

    avsc_path = f"resources/avro/{query_name}.avsc"
    risks = []

    for field in sorted(committed_fields - beam_output_keys):
        risks.append({
            "severity": "HIGH",
            "beam_file": consumer_file,
            "column": field,
            "issue": f"Contracted Avro field '{field}' is no longer produced by {consumer_file}",
            "detail": (f"{avsc_path} declares '{field}' as a required output field, "
                       f"but the current {consumer_file} formatter does not return it. "
                       f"Downstream Avro readers expecting this field will fail."),
            "fix": (f"Restore '{field}' to the formatter's return dict in {consumer_file}, "
                    f"or remove it from {avsc_path} if the field is intentionally retired."),
        })

    for field in sorted(beam_output_keys - committed_fields):
        risks.append({
            "severity": "LOW",
            "beam_file": consumer_file,
            "column": field,
            "issue": f"Beam output field '{field}' is not declared in {avsc_path}",
            "detail": (f"{consumer_file} produces '{field}' but {avsc_path} does not declare it. "
                       f"Avro readers relying on the committed schema won't know this field exists."),
            "fix": f"Add '{field}' to {avsc_path} to keep the contract current.",
        })

    high = sum(1 for r in risks if r["severity"] == "HIGH")
    low = sum(1 for r in risks if r["severity"] == "LOW")
    if risks:
        logger.info(f"{query_path}: contract check — {high} HIGH, {low} LOW violations vs {avsc_path}")
    else:
        logger.info(f"{query_path}: committed Avro contract fully satisfied by {consumer_file}")
    return risks


def check_avro_schema_risk(sql_text: str, schema_manifest: str, consumer_file: str,
                           beam_sources: dict, query_path: str,
                           committed_schemas: dict | None = None) -> dict:
    """Step 1+2+3 together: generates the expected .avsc, resolves the
    Beam-pipeline output fields, compares them, and flags mismatches
    as risks on this query. `sql_text` should be the OPTIMIZED SQL —
    the query as it will actually run — since the entire point is
    catching a rewrite that drops a column Beam still depends on.

    Returns {"risks": [...], "status": "...", "schema_doc": {...} or
    None, "schema_gcs_path": "gs://..." or None}. schema_doc/
    schema_gcs_path are only None when the check couldn't run at all
    (no consumer resolved, no dict-literal return) — see `status` for
    why. Fails open at every step rather than raising, so a pipeline
    this check can't cleanly analyze just gets skipped instead of
    breaking analysis for that query."""
    empty = {"risks": [], "status": "", "schema_doc": None, "schema_gcs_path": None}

    file_source = beam_sources.get(consumer_file, "")
    if not file_source:
        return {**empty, "status": "N/A — no Beam consumer resolved for this query"}

    primary = _select_primary_function(file_source)
    if not primary:
        return {**empty, "status": "N/A — no row-building function found in this consumer"}

    beam_fields = build_beam_output_fields(primary)
    if beam_fields is None:
        return {**empty, "status": f"N/A — {primary['name']}() doesn't return a plain dict literal"}

    # Step 1: generate the expected schema FIRST, as its own real
    # artifact — independent of whether Beam turns out to match it.
    expected_schema = build_expected_avro_schema(sql_text, schema_manifest)
    schema_doc = build_avro_schema_document(expected_schema, primary["name"])
    gcs_path = upload_avro_schema(schema_doc, query_path)

    # Step 2 (beam_fields, above) vs Step 1 (expected_schema): compare.
    risks = []
    matched = 0
    for field, source_cols in beam_fields.items():
        if not source_cols:
            # Computed/constant/helper-derived field (e.g. a boolean
            # flag from a predicate call, or a literal) — nothing in
            # the query schema to check it against, not a risk.
            matched += 1
            continue
        missing = [c for c in source_cols if c not in expected_schema]
        if missing:
            risks.append({
                "severity": "HIGH",
                "beam_file": consumer_file,
                "column": ", ".join(missing),
                "issue": f"Avro field '{field}' references column(s) "
                         f"{missing} not present in this query's output",
                "detail": f"{primary['name']}() builds \"{field}\" from "
                          f"row[{missing[0]!r}], but the current SELECT "
                          f"list doesn't produce {missing[0]!r} — this "
                          f"will KeyError before the record is ever "
                          f"serialized to Avro, not just mis-serialize it.",
                "fix": f"Add {missing} back to the SQL SELECT list, or "
                       f"remove '{field}' from {primary['name']}() if the "
                       f"field is no longer needed downstream.",
            })
        else:
            matched += 1

    # Check committed .avsc contract (Beam→contract direction)
    if committed_schemas:
        contract_risks = check_committed_avro_contract(primary, consumer_file, query_path, committed_schemas)
        risks = risks + contract_risks

    total = len(beam_fields)
    summary = (f"{matched}/{total} Avro fields matched ({primary['name']}() "
               f"in {consumer_file})") if total else "No output fields detected"
    return {"risks": risks, "status": summary, "schema_doc": schema_doc, "schema_gcs_path": gcs_path}


# =================================================================
# Gemini
# =================================================================
def ask_gemini_for_rewrite(old_sql, new_sql, cache_name, schema_manifest,
                           consumer_context, original_bytes, consumer_file="") -> str:
    # The schema is cached (see get_or_create_cache) — only inline it
    # when there's no cache to point at. consumer_context is always
    # inlined: it's the small, query-specific slice RAG just retrieved,
    # never the whole Beam directory, so there's nothing to gain by
    # caching it separately.
    schema_block = "" if cache_name else f"""
**Schema:**
{schema_manifest}
"""

    prompt = f"""You are an expert Google BigQuery SQL optimizer.

**Strict Rules (never break these):**
- Preserve EXACT business logic, output rows, column names/types, ordering, joins, filters, window functions, and NULL/timestamp semantics.
- The downstream Beam pipeline (shown below) is the ground truth for required columns. Do NOT remove any column it uses.
- Keep the exact same JOIN structure (same tables, same join type, same ON conditions).
- Prefer explicit column lists over SELECT * when safe.
- Only remove truly unused columns/expressions.
- Separately, check the reverse direction: if any downstream Beam/Spark
  code references a field that does NOT appear in the query's SELECT
  list at all, that's a correctness bug (would cause a KeyError at
  runtime) — report it under "recommendations", do not silently add it.
{schema_block}
**Downstream Beam Consumer** (retrieved via semantic search — the
function(s) most relevant to this query, from `{consumer_file}`):
{consumer_context}

**Current SQL to optimize:**
{new_sql}

**Task:**
Rewrite the SQL to be more efficient (mainly by pruning unused columns/projections) while being **100% semantically identical**.

Return **ONLY** valid JSON (no markdown, no extra text):

{{
  "business_logic": {{
    "status": "PASS",
    "reason": "Brief explanation"
  }},
  "optimized_sql": "the complete rewritten SQL query here",
  "summary": "One sentence summary of changes",
  "changes": [
    {{"change": "Removed unused column X", "reason": "Not used by Beam pipeline"}}
  ],
  "risks": [
    {{
      "severity": "HIGH",
      "beam_file": "filename.py",
      "column": "column_name",
      "issue": "One-line description of the type mismatch",
      "detail": "Exact expression that fails and why",
      "fix": "Suggested fix in SQL or Beam"
    }}
  ],
  "recommendations": ["List of further suggestions for human review"]
}}

Do not estimate bytes/cost. The app will dry-run it.
"""

    if cache_name:
        logger.info(f"Calling Gemini WITH cached schema: {cache_name}")
        response = genai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={"cached_content": cache_name, "temperature": 0},
        )
    else:
        logger.info("Calling Gemini WITHOUT cache (inline schema fallback)")
        response = genai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={"temperature": 0},
        )

    return response.text.strip()


def extract_sql(response_text: str) -> str:
    """Pull SQL out of a Gemini markdown response."""
    if not response_text:
        return ""

    match = re.search(r"```sql\s*(.*?)```", response_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    match = re.search(r"```\s*(.*?)```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return response_text.strip()


def parse_gemini_json(rewrite: str, fallback_sql: str) -> dict:
    """Strip code fences and parse. Falls back to the original SQL."""
    rewrite = rewrite.strip()
    if rewrite.startswith("```json"):
        rewrite = rewrite[7:]
    if rewrite.startswith("```"):
        rewrite = rewrite[3:]
    if rewrite.endswith("```"):
        rewrite = rewrite[:-3]

    try:
        return json.loads(rewrite.strip())
    except json.JSONDecodeError:
        logger.error(f"JSON parse failed:\n{rewrite}")
        return {
            "optimized_sql": fallback_sql,
            "business_logic": {"status": "FAIL", "reason": "Gemini JSON parse failed"},
            "summary": "Fallback to original",
            "changes": [],
            "risks": [],
            "recommendations": [],
        }


# =================================================================
# Type-risk detection (deterministic AST pass + Gemini validation)
#
# Unchanged by RAG on purpose. This still reads the FULL text of
# whichever file was resolved as the consumer (by RAG or by fallback)
# — never the retrieved snippet — so it keeps seeing helper functions
# and can't miss a bug because retrieval only grabbed part of a file.
# =================================================================
def detect_type_risks(beam_dir_context: str, consumer_file: str, schema_manifest: str) -> list:
    """Static risk detector — no LLM. Scans ALL Beam files (not just the
    consumer) so helper functions in sibling files are also covered.

    Detects:
      HIGH   — STRING/BYTES column in arithmetic or numeric comparison
      HIGH   — STRING column passed to int()/float()/numeric conversion
      MEDIUM — NULLABLE column in arithmetic/comparison without None guard
      MEDIUM — NULLABLE column as divisor (None → TypeError in division)
      LOW    — STRING column in == / != comparison (logic bug, not a crash;
               Python silently returns False instead of raising)
    """
    types = dict(re.findall(r"^- (\w+) \((\w+)\)", schema_manifest, re.MULTILINE))
    nullable_cols = set(re.findall(r"^- (\w+) \([^)]+\) \[NULLABLE\]", schema_manifest, re.MULTILINE))

    # Scan ALL Beam files, not just the consumer — helper functions in
    # sibling files that also touch row["col"] are covered this way.
    # We tag each risk with the file it was found in.
    file_blocks = re.findall(
        r"--- (.+?) ---\n(.*?)(?=\n--- |\Z)", beam_dir_context, re.DOTALL
    )
    if not file_blocks:
        # Fallback: try single-file pattern (consumer file only)
        m = re.search(rf"--- {re.escape(consumer_file)} ---\n(.*?)(?=\n--- |\Z)",
                      beam_dir_context, re.DOTALL)
        file_blocks = [(consumer_file, m.group(1))] if m else []

    risks, seen = [], set()
    NUMERIC_CONVERSIONS = {"int", "float", "round", "abs", "divmod", "pow"}

    for fname, source in file_blocks:
        class Visitor(ast.NodeVisitor):
            def _cols(self, node):
                """All row["col"] subscripts anywhere inside this node."""
                return [n.slice.value for n in ast.walk(node)
                        if isinstance(n, ast.Subscript)
                        and isinstance(n.slice, ast.Constant)
                        and isinstance(n.slice.value, str)]

            def _flag(self, node, kind, severity="HIGH"):
                for col in self._cols(node):
                    key = (col, kind)
                    if key in seen:
                        continue
                    t = types.get(col)
                    if t in ("STRING", "BYTES"):
                        seen.add(key)
                        risks.append({
                            "severity": severity,
                            "beam_file": fname,
                            "column": col,
                            "issue": f"{col} is {t} in BigQuery but used in {kind} in Python",
                            "detail": f"`{ast.unparse(node)}` raises TypeError — "
                                      f"Python does not implicitly convert {t} to a number.",
                            "fix": f"CAST({col} AS INT64/FLOAT64) in the SQL SELECT list, "
                                   f"or convert in Beam before use.",
                        })
                    elif col in nullable_cols:
                        seen.add(key)
                        risks.append({
                            "severity": "MEDIUM",
                            "beam_file": fname,
                            "column": col,
                            "issue": f"{col} is NULLABLE in BigQuery but used in {kind} without a None check",
                            "detail": f"`{ast.unparse(node)}` will raise TypeError when {col} is NULL — "
                                      f"BigQuery returns None for NULLABLE columns.",
                            "fix": f"Guard with `if row['{col}'] is not None` before use, "
                                   f"or use `COALESCE({col}, 0)` in the SQL.",
                        })

            def visit_BinOp(self, node):
                if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
                    self._flag(node, "arithmetic")
                elif isinstance(node.op, ast.Div):
                    self._flag(node, "arithmetic")
                    # Additionally flag NULLABLE columns used as the DIVISOR —
                    # dividing by None raises TypeError even when the numerator
                    # is fine. Check right-hand side of the division only.
                    for col in self._cols(node.right):
                        key = (col, "division-by-nullable")
                        if key not in seen and col in nullable_cols:
                            seen.add(key)
                            risks.append({
                                "severity": "MEDIUM",
                                "beam_file": fname,
                                "column": col,
                                "issue": f"{col} is NULLABLE and used as a divisor — will TypeError when NULL",
                                "detail": f"`{ast.unparse(node)}` — Python raises TypeError when "
                                          f"dividing by None. {col} is NULLABLE in BigQuery.",
                                "fix": f"Use `COALESCE({col}, 1)` in SQL or guard with "
                                       f"`if row['{col}'] else <default>` in Beam.",
                            })
                self.generic_visit(node)

            def visit_Compare(self, node):
                # Numeric comparisons (>, <, >=, <=) — crash with TypeError on STRING
                if any(isinstance(o, (ast.Gt, ast.Lt, ast.GtE, ast.LtE)) for o in node.ops):
                    self._flag(node, "numeric comparison")
                # Equality (==, !=) on STRING — silent logic bug, not a crash
                if any(isinstance(o, (ast.Eq, ast.NotEq)) for o in node.ops):
                    for col in self._cols(node):
                        key = (col, "equality comparison")
                        if key not in seen and types.get(col) in ("STRING", "BYTES"):
                            seen.add(key)
                            risks.append({
                                "severity": "LOW",
                                "beam_file": fname,
                                "column": col,
                                "issue": f"{col} is STRING in BigQuery but compared with == / != to a number",
                                "detail": f"`{ast.unparse(node)}` — Python silently returns False "
                                          f"when comparing STRING to int/float (no crash, but always wrong). "
                                          f"e.g. `'36' == 36` is False in Python.",
                                "fix": f"CAST({col} AS INT64/FLOAT64) in the SQL SELECT list "
                                       f"before comparing, or compare to a string literal instead.",
                            })
                self.generic_visit(node)

            def visit_Call(self, node):
                # int(row["col"]), float(row["col"]) etc. on STRING columns
                # can raise ValueError if the string isn't cleanly numeric
                # (e.g. "94107-1234", "35 months", "n/a")
                func_name = (
                    node.func.id if isinstance(node.func, ast.Name)
                    else node.func.attr if isinstance(node.func, ast.Attribute)
                    else None
                )
                if func_name in NUMERIC_CONVERSIONS:
                    for arg in node.args:
                        for col in self._cols(arg):
                            key = (col, f"{func_name}() conversion")
                            if key not in seen and types.get(col) in ("STRING", "BYTES"):
                                seen.add(key)
                                risks.append({
                                    "severity": "HIGH",
                                    "beam_file": fname,
                                    "column": col,
                                    "issue": f"{col} is STRING — {func_name}(row['{col}']) raises ValueError if not cleanly numeric",
                                    "detail": f"`{ast.unparse(node)}` — {func_name}() raises ValueError "
                                              f"for strings like '35 months', '94107-1234', 'n/a', or empty string. "
                                              f"{col} is stored as STRING in BigQuery with no format guarantee.",
                                    "fix": f"CAST({col} AS INT64/FLOAT64) in the SQL SELECT list to let "
                                           f"BigQuery handle the conversion (it raises a query error on bad values "
                                           f"rather than crashing individual Beam records), or wrap in try/except.",
                                })
                self.generic_visit(node)

        try:
            Visitor().visit(ast.parse(source))
        except SyntaxError:
            logger.warning(f"Could not parse {fname}")

    return risks


def validate_risks_with_gemini(candidates: list, optimized_sql: str,
                               consumer_file: str, cache_name) -> list:
    """AST found candidates; Gemini confirms each is a real failure."""
    if not candidates:
        return []

    prompt = f"""You are validating suspected type-mismatch bugs in a Beam pipeline.

A static analyzer flagged these columns as STRING in BigQuery but used in
numeric operations in `{consumer_file}`. Confirm or reject each one.

**Candidates:**
{json.dumps(candidates, indent=2)}

**SQL that produces these rows:**
{optimized_sql}

**For each candidate decide:**
- CONFIRMED if the column reaches Python as STRING and the expression raises
  TypeError. Note: CAST in a WHERE / ORDER BY / window function does NOT change
  the output type — only an explicit CAST in the SELECT list does.
- REJECTED only if the column IS explicitly CAST in the SELECT list, or the
  expression cannot actually execute.

Python never implicitly converts types: `"5" > 2` raises TypeError regardless
of the string's contents.

Return ONLY valid JSON — the confirmed risks, with `detail` and `fix` rewritten
to be specific to this query. Return [] if all are rejected.

[
  {{
    "severity": "HIGH",
    "beam_file": "...",
    "column": "...",
    "issue": "...",
    "detail": "...",
    "fix": "..."
  }}
]
"""
    try:
        resp = genai_client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt, config={"temperature": 0}
        )
        text = resp.text.strip()
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
        validated = json.loads(text)
        logger.info(f"Risk validation: {len(candidates)} candidates -> {len(validated)} confirmed")
        return validated
    except Exception:
        logger.exception("Risk validation failed, falling back to raw AST candidates")
        return candidates   # fail open — never lose a real bug


# =================================================================
# Shared analysis core — used by BOTH /review and /analyze
# =================================================================
def analyze_one(change: dict, beam_context: str, beam_sources: dict, beam_index: list,
                committed_schemas: dict | None = None):
    """Dry run -> RAG consumer lookup -> Gemini rewrite -> dry run ->
    risk detection for one SQL file. Returns (result_dict, cache_info)."""
    new_bytes, _ = dry_run_bytes(change["new"])

    schema_manifest = build_schema_manifest(change["new"])
    cache_info = get_or_create_cache(schema_manifest)
    cache_name = cache_info["name"]

    fallback_consumer = QUERY_RUNTIME_CONFIG.get(change["path"], {}).get("beam_consumer", "")
    consumer_context, consumer, matched, confidence = retrieve_consumer_context(
        change["new"], beam_index, fallback_consumer, beam_sources, schema_manifest
    )

    # Cross-validate RAG's pick against the committed .avsc contract.
    # A wide query (SELECT * with 150+ columns) can give a tiny formatter's
    # small column set a near-perfect containment score even when it's the
    # wrong consumer — e.g. delinquency_alerts.py (6 columns) against a
    # stress-test query that happens to include all 6. The committed .avsc
    # is the authoritative contract: if the picked consumer's formatter
    # output keys barely overlap with those contracted fields, RAG guessed
    # wrong and the config fallback is safer.
    if matched and committed_schemas:
        query_name = os.path.basename(change["path"]).replace(".sql", "")
        avsc_doc = committed_schemas.get(query_name)
        if avsc_doc:
            contracted = {f["name"] for f in avsc_doc.get("fields", [])}
            primary_check = _select_primary_function(beam_sources.get(consumer, ""))
            beam_keys_check = set(build_beam_output_fields(primary_check) or {}) if primary_check else set()
            overlap = len(beam_keys_check & contracted) / max(len(contracted), 1)
            if overlap < 0.3:
                logger.warning(
                    f"RAG picked '{consumer}' for {change['path']} but only "
                    f"{overlap:.0%} field overlap with committed .avsc ({len(contracted)} contracted fields) "
                    f"— reverting to config fallback '{fallback_consumer}'"
                )
                consumer = fallback_consumer
                consumer_context = beam_sources.get(fallback_consumer, "")
                matched = False
                confidence = 0.0

    logger.info(
        f"{change['path']}: consumer={consumer!r} "
        f"via={'RAG' if matched else 'config fallback'} confidence={confidence:.2f}"
    )

    rewrite = ask_gemini_for_rewrite(
        change.get("old"), change["new"], cache_name,
        schema_manifest, consumer_context, new_bytes, consumer_file=consumer,
    )
    rewrite_json = parse_gemini_json(rewrite, change["new"])

    optimized_sql = rewrite_json.get("optimized_sql") or change["new"]
    rewrite_bytes, rewrite_ok = dry_run_bytes(optimized_sql)

    # Deterministic risk scan ALWAYS sees the resolved consumer's full
    # file text, not the RAG snippet — see the module note above.
    candidates = detect_type_risks(beam_context, consumer, schema_manifest)
    validated = validate_risks_with_gemini(candidates, optimized_sql, consumer, cache_name)
    risks = validated if validated else candidates
    logger.info(f"{change['path']}: {len(candidates)} candidates, {len(validated)} confirmed")

    # Avro schema check — independent of the type-risk scan above, but
    # merged into the same `risks` list so it surfaces in the same
    # report section without needing separate UI plumbing. Checked
    # against optimized_sql (not the original) since the point is to
    # catch the rewrite itself breaking the downstream Avro delivery.
    # Also writes the generated .avsc to GCS — a real, openable file,
    # not just a number in the report.
    avro_result = check_avro_schema_risk(
        optimized_sql, schema_manifest, consumer, beam_sources, change["path"],
        committed_schemas=committed_schemas,
    )
    risks = risks + avro_result["risks"]
    logger.info(f"{change['path']}: avro check — {avro_result['status']} "
                f"(schema: {avro_result['schema_gcs_path']})")

    return {
        "path": change["path"],
        "old_bytes": None,
        "new_bytes": new_bytes,
        "rewrite_bytes": rewrite_bytes,
        "rewrite_ok": rewrite_ok,
        "savings_pct": (1 - rewrite_bytes / new_bytes) * 100 if new_bytes else 0,
        "new_sql": change["new"],
        "optimized_sql": optimized_sql,
        "summary": rewrite_json.get("summary", ""),
        "changes": rewrite_json.get("changes", []),
        "recommendations": rewrite_json.get("recommendations", []),
        "business_logic": rewrite_json.get("business_logic", {}),
        "risks": risks,
        "rag_consumer": consumer,
        "rag_matched": matched,
        "rag_confidence": round(confidence, 3),
        "avro_status": avro_result["status"],
        "avro_schema": avro_result["schema_doc"],
        "avro_schema_gcs_path": avro_result["schema_gcs_path"],
        "schema_manifest": schema_manifest,  # reused by test harness, avoids re-fetch
    }, cache_info


# =================================================================
# GitHub
# =================================================================
def post_github_comment(repo_owner: str, repo_name: str, commit_sha: str, body: str):
    if not GITHUB_TOKEN:
        logger.warning("No GITHUB_TOKEN set, skipping comment post.")
        logger.info(f"Comment body was:\n{body}")
        return
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/commits/{commit_sha}/comments"
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}
    logger.info(f"Posting comment to {url}")
    try:
        resp = requests.post(url, json={"body": body}, headers=headers)
        resp.raise_for_status()
        logger.info(f"Comment posted successfully, id={resp.json().get('id')}")
    except requests.exceptions.HTTPError as e:
        logger.error(f"GitHub comment post failed: {e} — response body: {resp.text}")
        raise


def build_comment_section(r: dict, cache_info: dict) -> str:
    """One GitHub comment section for a single analysed SQL file."""
    old_bytes = r.get("old_bytes")
    new_bytes = r["new_bytes"]
    rewrite_bytes = r["rewrite_bytes"]
    savings_usd = bytes_to_cost(new_bytes) - bytes_to_cost(rewrite_bytes)
    tokens = cache_info["token_count"]

    section = f"## `{r['path']}`\n\n"

    if old_bytes is not None:
        section += f"**Previous:** {old_bytes:,} bytes scanned (${bytes_to_cost(old_bytes):.6f})\n\n"
    section += f"**Current:** {new_bytes:,} bytes scanned (${bytes_to_cost(new_bytes):.6f})\n\n"
    if not r["rewrite_ok"]:
        section += ("**Gemini Rewrite:** dry-run failed — this SQL may be invalid, "
                    "do not merge without manual review\n\n")
    else:
        section += (f"**Gemini Rewrite:** {rewrite_bytes:,} bytes scanned "
                    f"(${bytes_to_cost(rewrite_bytes):.6f})\n\n")
    section += f"**Savings:** ${savings_usd:.6f} ({r['savings_pct']:.1f}% reduction)\n\n"

    if "rag_consumer" in r:
        method = "semantic retrieval" if r["rag_matched"] else "static config fallback"
        section += (
            f"### Consumer Discovery\n"
            f"- Resolved: `{r['rag_consumer']}`\n"
            f"- Method: {method} (confidence {r['rag_confidence']:.2f})\n\n"
        )

    if r.get("avro_status"):
        section += f"### Avro Schema Check\n- {r['avro_status']}\n"
        if r.get("avro_schema_gcs_path"):
            section += f"- Generated schema: `{r['avro_schema_gcs_path']}`\n"
        section += "\n"

    section += (
        f"### Schema Cache\n"
        f"- Size: {tokens:,} tokens\n"
        f"- Storage cost: ${cache_storage_cost_per_hour(tokens):.6f}/hr "
        f"(${cache_storage_cost_per_hour(tokens) * 24:.4f}/day if held for 24h)\n"
        f"- Cost of this cached call: ${cache_read_cost(tokens):.6f} "
        f"(vs. ${(tokens / 1_000_000) * 0.30:.6f} if sent inline, uncached)\n\n"
    )

    bl = r["business_logic"]
    section += "### Business Logic\n"
    section += f"- Status: **{bl.get('status', 'UNKNOWN')}**\n- Reason: {bl.get('reason', '')}\n\n"

    section += "### Summary\n" + r["summary"] + "\n\n"

    if r["changes"]:
        section += "### Changes Applied\n"
        for item in r["changes"]:
            section += f"- **{item['change']}**\n  - {item['reason']}\n"
        section += "\n"

    if r["risks"]:
        section += f"### Type Risks ({len(r['risks'])})\n"
        for risk in r["risks"]:
            section += (
                f"- **[{risk['severity']}] `{risk['column']}`** in `{risk['beam_file']}`\n"
                f"  - {risk['issue']}\n"
                f"  - {risk['detail']}\n"
                f"  - _Fix:_ {risk['fix']}\n"
            )
        section += "\n"

    if r["recommendations"]:
        section += "### Recommendations\n"
        for rec in r["recommendations"]:
            section += f"- {rec}\n"
        section += "\n"

    section += (
        "<details>\n<summary><b>Optimized SQL</b></summary>\n\n"
        "```sql\n" + r["optimized_sql"] + "\n```\n</details>\n"
    )
    return section


# =================================================================
# UI report
# =================================================================
def compute_guardrail_status(all_business_logic: list) -> str:
    if all(bl.get("status") == "PASS" for bl in all_business_logic):
        return "PASSED GUARDRAIL"
    return "FAILED GUARDRAIL"


def build_ui_report(branch: str, commit_message: str, changed_results: list,
                    beam_filenames: list, cache_info) -> dict:
    queries = []
    all_business_logic = []
    total_original_cost = 0.0
    total_optimized_cost = 0.0

    cache_info = cache_info or {"status": "N/A", "token_count": 0}

    for r in changed_results:
        config = QUERY_RUNTIME_CONFIG.get(r["path"], DEFAULT_RUNTIME_CONFIG)
        runs_per_day = config["runs_per_day"]

        original_cost_per_run = bytes_to_cost(r["new_bytes"])
        optimized_cost_per_run = bytes_to_cost(r["rewrite_bytes"])

        total_original_cost += original_cost_per_run * runs_per_day * 365
        total_optimized_cost += optimized_cost_per_run * runs_per_day * 365

        queries.append({
            "query_id": f"q{len(queries) + 1}",
            "file_name": r["path"].split("/")[-1],
            "dag_task_id": config["dag_task_id"],
            "runs_per_day": runs_per_day,
            "original_bytes_scanned": r["new_bytes"],
            "original_cost_per_run": round(original_cost_per_run, 6),
            "optimized_bytes_scanned": r["rewrite_bytes"],
            "optimized_cost_per_run": round(optimized_cost_per_run, 6),
            "individual_savings_percent": round(r["savings_pct"], 1),
            "original_sql": r["new_sql"],
            "optimized_sql": r["optimized_sql"],
            "ai_explanation": r["summary"],
            "risks": r.get("risks", []),
            "avro_status": r.get("avro_status", "N/A"),
            "avro_schema": r.get("avro_schema"),
            "avro_schema_gcs_path": r.get("avro_schema_gcs_path"),
        })
        all_business_logic.append(r["business_logic"])

    net_savings = total_original_cost - total_optimized_cost
    savings_pct = (net_savings / total_original_cost * 100) if total_original_cost else 0.0

    dataset_counts = _count_tables_by_dataset(changed_results)
    bq_schemas_checked = ", ".join(
        f"{dataset} ({count} Table{'s' if count != 1 else ''})"
        for dataset, count in dataset_counts.items()
    ) or "None"

    # Aggregate the per-query Avro check (see check_avro_schema_risk)
    # into one summary line. A query's avro_status starts with "N/A"
    # when the check couldn't run for it (no consumer resolved, no
    # dict-literal return, etc.) — those don't count toward "checked."
    avro_checked = [r for r in changed_results
                    if r.get("avro_status") and not r["avro_status"].startswith("N/A")]
    avro_mismatch_count = sum(
        1 for r in changed_results for risk in r.get("risks", [])
        if risk.get("issue", "").startswith("Avro field")
    )
    if avro_checked:
        avro_summary = (
            f"{len(avro_checked)}/{len(changed_results)} "
            f"quer{'y' if len(changed_results) == 1 else 'ies'} checked — "
            + (f"{avro_mismatch_count} mismatch"
               f"{'es' if avro_mismatch_count != 1 else ''} found"
               if avro_mismatch_count else "all fields matched")
        )
    else:
        avro_summary = "N/A — no Avro-producing consumer resolved for these queries"

    return {
        "release_id": commit_message,
        "branch": branch,
        "guardrail_status": compute_guardrail_status(all_business_logic),
        "summary_metrics": {
            "total_queries_evaluated": len(queries),
            "annual_original_cost_usd": round(total_original_cost, 2),
            "annual_optimized_cost_usd": round(total_optimized_cost, 2),
            "net_annual_savings_usd": round(net_savings, 2),
            "savings_percentage": round(savings_pct, 1),
        },
        "queries": queries,
        "context_verification": {
            "bq_schemas_checked": bq_schemas_checked,
            "avro_mappings_matched": avro_summary,
            "dataflow_logic_validated": ", ".join(sorted(set(beam_filenames))) or "None",
            "vertex_cache": {
                "status": cache_info["status"],
                "token_count": cache_info["token_count"],
                "storage_cost_per_hour":
                    round(cache_storage_cost_per_hour(cache_info["token_count"]), 6),
                "read_cost":
                    round(cache_read_cost(cache_info["token_count"]), 6),
            },
        },
    }


def upload_report(blob_name: str, ui_report: dict):
    try:
        storage_client.bucket(BUCKET_NAME).blob(blob_name).upload_from_string(
            json.dumps(ui_report), content_type="application/json"
        )
        logger.info(f"Uploaded report to gs://{BUCKET_NAME}/{blob_name}")
    except Exception:
        logger.exception(f"Failed to upload {blob_name} to GCS")


# =================================================================
# Routes
# =================================================================
@app.route("/review", methods=["POST"])
def review():
    """CI guardrail — only the SQL files changed between two commits."""
    payload = flask.request.get_json()
    repo_clone_url = payload["repo_clone_url"]
    repo_owner = payload["repo_owner"]
    repo_name = payload["repo_name"]
    before_sha = payload["before_sha"]
    after_sha = payload["after_sha"]
    branch = payload.get("branch", "unknown")

    changed, beam_context, beam_filenames, commit_message, committed_schemas = clone_and_diff(
        repo_clone_url, before_sha, after_sha
    )
    if not changed:
        return flask.jsonify({"status": "no_sql_changes"})

    # Build the RAG index ONCE per request — every file in this push
    # shares the same Beam corpus, so there's no reason to re-embed it
    # per file.
    beam_sources = _split_beam_context(beam_context)
    beam_index = get_function_index(beam_context, beam_sources)

    comment_sections = []
    changed_results = []
    cache_info = None

    for change in changed:
        old_bytes, _ = dry_run_bytes(change["old"]) if change["old"] else (None, True)

        r, cache_info = analyze_one(change, beam_context, beam_sources, beam_index,
                                    committed_schemas=committed_schemas)
        r["old_bytes"] = old_bytes
        changed_results.append(r)

        comment_sections.append(build_comment_section(r, cache_info))

    comment_body = "## SQL Cost Review\n\n" + "\n\n---\n\n".join(comment_sections)
    post_github_comment(repo_owner, repo_name, after_sha, comment_body)

    ui_report = build_ui_report(branch, commit_message, changed_results,
                                beam_filenames, cache_info)
    upload_report("latest_report.json", ui_report)

    return flask.jsonify({
        "status": "ok",
        "changed_files": [c["path"] for c in changed],
        "ui_report": ui_report,
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    """On-demand audit — every SQL file in the repo at `ref`."""
    payload = flask.request.get_json(silent=True) or {}
    repo_clone_url = payload.get("repo_clone_url") or DEFAULT_REPO_URL
    if not repo_clone_url:
        return flask.jsonify({
            "status": "error",
            "message": "repo_clone_url required (or set DEFAULT_REPO_URL)",
        }), 400
    ref = payload.get("ref", "main")

    all_sql, beam_context, beam_filenames, head_sha, committed_schemas = clone_all_sql(repo_clone_url, ref)
    if not all_sql:
        return flask.jsonify({"status": "no_sql_found"}), 404

    logger.info(f"Full scan: {len(all_sql)} SQL files at {ref}@{head_sha}")

    beam_sources = _split_beam_context(beam_context)
    beam_index = get_function_index(beam_context, beam_sources)

    results, cache_info = [], None
    for i, change in enumerate(all_sql):
        if i > 0:
            time.sleep(10)  # avoid Gemini 429 rate limit between files
        try:
            r, cache_info = analyze_one(change, beam_context, beam_sources, beam_index,
                                        committed_schemas=committed_schemas)
            results.append(r)
        except Exception:
            logger.exception(f"Analysis failed for {change['path']}, skipping")

    if not results:
        return flask.jsonify({"status": "error", "message": "all files failed"}), 500

    ui_report = build_ui_report(
        ref, "Full repository scan",
        results, beam_filenames, cache_info,
    )
    upload_report("latest_scan.json", ui_report)

    return flask.jsonify({
        "status": "ok",
        "files_analyzed": len(results),
        "files_found": len(all_sql),
        "ui_report": ui_report,
    })


# =================================================================
# Test Data Harness
# (originally query_test_harness.py by team lead — integrated here
#  so it shares PROJECT_ID, bq_client, genai_client, and
#  build_schema_manifest with the rest of the guardrail)
# =================================================================
PROD_DATASET = "loan_data"
TEST_DATASET = "loan_data_test"

_BQ_TYPE_MAP = {
    "STRING": "STRING", "BYTES": "BYTES",
    "INT64": "INT64", "INTEGER": "INTEGER", "INT": "INT64",
    "FLOAT64": "FLOAT64", "FLOAT": "FLOAT64", "NUMERIC": "NUMERIC",
    "BOOL": "BOOL", "BOOLEAN": "BOOL",
    "DATE": "DATE", "DATETIME": "DATETIME",
    "TIMESTAMP": "TIMESTAMP", "TIME": "TIME",
    "ARRAY": "STRING", "STRUCT": "STRING",
}

_TEST_GEMINI_PROMPT = """
You are a SQL test engineer. You have the EXACT BigQuery schema for every table
this query references (fetched live from INFORMATION_SCHEMA). Use it to generate
comprehensive test cases covering every logical branch.

REAL BIGQUERY SCHEMA (column names and types are exact — use them as-is):
{schema_block}

SQL QUERY TO TEST:
{sql}

Generate test cases for:
  POSITIVE — rows that SHOULD appear in the final result
  NEGATIVE — rows that SHOULD be excluded
  EDGE     — boundary values, NULLs, CAST on strings, window ties

Rules:
  - Use EXACT column names from the schema above
  - Match BQ types: STRING columns get string values (even numeric-looking ones),
    INT64 gets integers, FLOAT64 gets floats
  - For columns the query CASTs, keep them as STRING in the test row
  - Make customer_id unique per test case (prefix with test case id)
  - For multi-table queries, provide rows for ALL tables with matching join keys

Return ONLY valid JSON — an array of objects:
{{
  "id": "TC_P01_description",
  "category": "positive" | "negative" | "edge",
  "description": "one sentence",
  "logic_tested": "CTE / filter / expression being exercised",
  "expected": "in_result" | "not_in_result" | "exactly_N_rows",
  "expected_count": 1,
  "rows_per_table": {{
    "project.dataset.table1": [ {{ col: val, ... }}, ... ],
    "project.dataset.table2": [ {{ col: val, ... }}, ... ]
  }}
}}
""".strip()


def _parse_manifest_to_schema(manifest: str) -> dict[str, dict[str, str]]:
    """Converts build_schema_manifest() output → {table: {col: bq_type}}."""
    schemas: dict[str, dict[str, str]] = {}
    current_table = None
    for line in manifest.splitlines():
        line = line.strip()
        m = re.match(r"=+ TABLE\s*:\s*(.+?)\s*=+", line)
        if m:
            current_table = m.group(1).strip()
            schemas[current_table] = {}
            continue
        if current_table and line.startswith("- "):
            m2 = re.match(r"- (\w+)\s+\(([^)]+)\)", line)
            if m2:
                schemas[current_table][m2.group(1)] = m2.group(2)
    return schemas


def _schema_to_block(schemas: dict[str, dict[str, str]]) -> str:
    lines = []
    for table, cols in schemas.items():
        lines.append(f"TABLE: {table}")
        for col, bq_type in cols.items():
            lines.append(f"  - {col}  ({bq_type})")
        lines.append("")
    return "\n".join(lines)


def _generate_test_cases(sql_text: str, schema_block: str) -> list[dict]:
    prompt = _TEST_GEMINI_PROMPT.format(schema_block=schema_block, sql=sql_text)
    resp = genai_client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt, config={"temperature": 0}
    )
    raw = re.sub(r"^```(?:json)?|```$", "", resp.text.strip(), flags=re.MULTILINE).strip()
    return json.loads(raw)


def _create_test_table(table_ref: str, schema: dict[str, str]) -> str:
    _, _, table_name = table_ref.split(".")
    test_table_id = f"{PROJECT_ID}.{TEST_DATASET}.{table_name}"
    fields = [
        bigquery.SchemaField(col, _BQ_TYPE_MAP.get(bq_type, "STRING"))
        for col, bq_type in schema.items()
    ]
    bq_client.delete_table(test_table_id, not_found_ok=True)
    bq_client.create_table(bigquery.Table(test_table_id, schema=fields))
    return test_table_id


def _setup_test_tables(schemas: dict[str, dict[str, str]]) -> dict[str, str]:
    ds = bigquery.Dataset(f"{PROJECT_ID}.{TEST_DATASET}")
    ds.location = LOCATION
    bq_client.create_dataset(ds, exists_ok=True)
    return {ref: _create_test_table(ref, schema) for ref, schema in schemas.items()}


def _load_test_data(test_cases: list[dict], table_mapping: dict[str, str]):
    rows_by_table: dict[str, list[dict]] = {tid: [] for tid in table_mapping.values()}
    for tc in test_cases:
        for prod_ref, rows in tc.get("rows_per_table", {}).items():
            test_id = table_mapping.get(prod_ref) or next(
                (v for k, v in table_mapping.items()
                 if k.endswith(prod_ref.split(".")[-1])), None
            )
            if test_id and rows:
                cleaned = [{k: v for k, v in r.items() if v is not None} for r in rows]
                rows_by_table.setdefault(test_id, []).extend(cleaned)
    for test_id, rows in rows_by_table.items():
        if rows:
            errs = bq_client.insert_rows_json(test_id, rows)
            if errs:
                logger.warning(f"Insert errors in {test_id}: {errs}")
            else:
                logger.info(f"Loaded {len(rows)} rows → {test_id}")


def _rewrite_to_test_dataset(sql_text: str) -> str:
    return sql_text.replace(
        f"`{PROJECT_ID}.{PROD_DATASET}.",
        f"`{PROJECT_ID}.{TEST_DATASET}."
    )


def _validate_test_cases(test_cases: list[dict], actual_rows: list[dict]) -> list[dict]:
    actual_ids = set(str(r.get("customer_id", "")) for r in actual_rows)
    results = []
    for tc in test_cases:
        tc_ids = {
            str(row.get("customer_id"))
            for rows in tc.get("rows_per_table", {}).values()
            for row in rows
            if row.get("customer_id")
        }
        expected = tc.get("expected", "in_result")
        found = sum(1 for cid in actual_ids if cid in tc_ids)
        if expected == "in_result":
            passed = found > 0
            note = f"Found {found} row(s)" + (" ✓" if passed else " — expected ≥1")
        elif expected == "not_in_result":
            passed = found == 0
            note = f"Found {found} row(s)" + (" ✓" if passed else " — expected 0")
        elif expected == "exactly_N_rows":
            n = tc.get("expected_count", 1)
            passed = found == n
            note = f"Found {found} row(s) — expected {n}" + (" ✓" if passed else " ✗")
        else:
            passed, note = False, f"Unknown expected: {expected}"
        results.append({**tc, "passed": passed, "note": note})
    return results


def _upload_test_report_html(results: list[dict], sql_path: str,
                              test_sql: str, manifest: str) -> str | None:
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed
    pct = int(passed / total * 100) if total else 0

    by_cat: dict[str, list] = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r)

    def rows_html(cases):
        html = ""
        for r in cases:
            bg = "#d4edda" if r["passed"] else "#f8d7da"
            icon = "✅" if r["passed"] else "❌"
            html += (f"<tr style='background:{bg}'>"
                     f"<td><code>{r['id']}</code></td>"
                     f"<td style='text-align:center'>{icon}</td>"
                     f"<td>{r['description']}</td>"
                     f"<td><em>{r['logic_tested']}</em></td>"
                     f"<td><code>{r['expected']}</code></td>"
                     f"<td>{r['note']}</td></tr>")
        return html

    cat_meta = {
        "positive": ("Positive Cases", "#d4edda", "#155724"),
        "negative": ("Negative Cases", "#f8d7da", "#721c24"),
        "edge":     ("Edge Cases",     "#fff3cd", "#856404"),
    }
    sections = ""
    for cat, cases in by_cat.items():
        title, bg, fg = cat_meta.get(cat, (cat.title(), "#eee", "#000"))
        sections += (f"<h2>{title}</h2><table>"
                     f"<tr><th>Test Case</th><th>Result</th><th>Description</th>"
                     f"<th>Logic Tested</th><th>Expected</th><th>Outcome</th></tr>"
                     f"{rows_html(cases)}</table>")

    overall = (
        "<p style='color:#28a745;font-weight:700'>✅ All test cases passed.</p>"
        if failed == 0 else
        f"<p style='color:#dc3545;font-weight:700'>❌ {failed} test case(s) failed.</p>"
    )

    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>Test Report — {sql_path}</title>
<style>
body{{font-family:-apple-system,sans-serif;margin:40px;color:#212529;background:#f8f9fa}}
h1{{color:#343a40}}h2{{color:#495057;border-bottom:2px solid #dee2e6;padding-bottom:6px;margin-top:40px}}
.summary{{display:flex;gap:20px;margin:24px 0}}
.card{{background:#fff;border-radius:8px;padding:20px 28px;box-shadow:0 1px 4px rgba(0,0,0,.1);text-align:center}}
.num{{font-size:2.4rem;font-weight:700}}.lbl{{font-size:.85rem;color:#6c757d;margin-top:4px}}
.pass{{color:#28a745}}.fail{{color:#dc3545}}.tot{{color:#343a40}}
table{{width:100%;border-collapse:collapse;background:#fff;border-radius:8px;
       overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.1);margin-bottom:24px}}
th{{background:#343a40;color:#fff;padding:10px 14px;text-align:left;font-size:.85rem}}
td{{padding:10px 14px;border-bottom:1px solid #dee2e6;font-size:.88rem;vertical-align:top}}
pre{{background:#f1f3f5;padding:16px;border-radius:6px;font-size:.8rem;overflow-x:auto;
     max-height:350px;white-space:pre-wrap}}
</style></head><body>
<h1>🧪 Query Test Report</h1>
<p><strong>SQL:</strong> <code>{sql_path}</code><br>
<strong>Test dataset:</strong> <code>{PROJECT_ID}.{TEST_DATASET}</code><br>
<strong>Generated:</strong> {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
<div class="summary">
  <div class="card"><div class="num tot">{total}</div><div class="lbl">Total</div></div>
  <div class="card"><div class="num pass">{passed}</div><div class="lbl">Passed</div></div>
  <div class="card"><div class="num fail">{failed}</div><div class="lbl">Failed</div></div>
  <div class="card"><div class="num {'pass' if pct==100 else 'fail'}">{pct}%</div>
    <div class="lbl">Pass Rate</div></div>
</div>
{overall}{sections}
<h2>Schema (from INFORMATION_SCHEMA)</h2><pre>{manifest}</pre>
<h2>SQL Run Against Test Data</h2><pre>{test_sql}</pre>
</body></html>"""

    try:
        sql_stem = sql_path.replace("/", "_").replace(".sql", "")
        blob_name = f"test_reports/{sql_stem}_report.html"
        storage_client.bucket(BUCKET_NAME).blob(blob_name).upload_from_string(
            html, content_type="text/html"
        )
        gcs_path = f"gs://{BUCKET_NAME}/{blob_name}"
        logger.info(f"Test report uploaded: {gcs_path}")
        return gcs_path
    except Exception:
        logger.exception("Failed to upload test report to GCS (non-fatal)")
        return None


@app.route("/test", methods=["POST"])
def run_tests():
    """Generate synthetic test data from the real BQ schema, run the
    SQL against it, validate results, upload an HTML report to GCS.

    Body (JSON):
      sql_path      — relative path, e.g. "resources/sql/portfolio_stress_test.sql"
      sql_text      — the SQL to test (optimized version recommended)
      dry_run       — if true, return generated test cases without touching BQ
    """
    payload = flask.request.get_json(silent=True) or {}
    sql_path = payload.get("sql_path", "unknown.sql")
    sql_text = payload.get("sql_text", "")
    dry_run = payload.get("dry_run", False)

    if not sql_text:
        return flask.jsonify({"status": "error", "message": "sql_text is required"}), 400

    try:
        # Step 1: fetch real schema via the same function the guardrail uses
        manifest = build_schema_manifest(sql_text)
        schemas = _parse_manifest_to_schema(manifest)
        if not schemas:
            return flask.jsonify({"status": "error",
                                  "message": "No tables found in manifest"}), 400
        schema_block = _schema_to_block(schemas)

        # Step 2: ask Gemini to generate test cases from the real schema
        test_cases = _generate_test_cases(sql_text, schema_block)
        logger.info(f"Generated {len(test_cases)} test cases for {sql_path}")

        if dry_run:
            return flask.jsonify({
                "status": "dry_run",
                "sql_path": sql_path,
                "test_cases": test_cases,
                "summary": {cat: sum(1 for t in test_cases if t["category"] == cat)
                            for cat in ("positive", "negative", "edge")},
            })

        # Step 3: create test tables mirroring real schema
        table_mapping = _setup_test_tables(schemas)

        # Step 4: load synthetic rows
        _load_test_data(test_cases, table_mapping)

        # Step 5: rewrite SQL to point at test dataset and run it
        test_sql = _rewrite_to_test_dataset(sql_text)
        time.sleep(5)   # allow streaming inserts to settle
        actual_rows = [dict(r) for r in bq_client.query(test_sql).result()]
        logger.info(f"Test query returned {len(actual_rows)} row(s)")

        # Step 6: validate
        results = _validate_test_cases(test_cases, actual_rows)
        passed = sum(1 for r in results if r["passed"])

        # Step 7: upload HTML report to GCS
        report_gcs_path = _upload_test_report_html(results, sql_path, test_sql, manifest)

        return flask.jsonify({
            "status": "ok",
            "sql_path": sql_path,
            "test_dataset": f"{PROJECT_ID}.{TEST_DATASET}",
            "total": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "pass_rate_pct": round(passed / len(results) * 100) if results else 0,
            "results": results,
            "report_gcs_path": report_gcs_path,
        })

    except Exception:
        logger.exception(f"Test harness failed for {sql_path}")
        return flask.jsonify({"status": "error",
                              "message": "Test harness failed — check logs"}), 500


@app.route("/api/latest-report", methods=["GET"])
def latest_report():
    """?source=scan reads the full-repo audit; default is the PR review."""
    blob_name = ("latest_scan.json"
                 if flask.request.args.get("source") == "scan"
                 else "latest_report.json")
    try:
        blob = storage_client.bucket(BUCKET_NAME).blob(blob_name)
        if not blob.exists():
            return flask.jsonify({"status": "error", "message": "No report available."}), 404
        return flask.jsonify(json.loads(blob.download_as_bytes().decode("utf-8")))
    except Exception as e:
        logger.exception("Failed to fetch latest report")
        return flask.jsonify({"status": "error", "message": str(e)}), 500


@app.route("/")
def ui():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
