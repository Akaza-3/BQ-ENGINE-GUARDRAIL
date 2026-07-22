"""
cloud_run/main.py

Triggered by the GitHub Actions workflow on every push to main.
Given a before/after commit SHA, it:
  1. Clones the repo and diffs the two commits.
  2. Pulls out changed .sql files (resources/sql/) and their paired
     Beam/Spark consumer code (src/beam/).
  3. Dry-runs the SQL at both commits (before vs after) — free,
     deterministic cost signal, no Gemini involved.
  4. Sends the changed SQL + consumer code + schema to Gemini
     (via a Vertex AI context cache) asking for a rewrite that
     preserves business logic.
  5. Dry-runs Gemini's suggested rewrite too.
  6. Posts a comment on the PR/commit with all three numbers.
  7. Returns a structured "ui_report" JSON block for the UI, alongside
     the existing GitHub comment.

This is a hackathon-scoped skeleton: no retries, minimal error
handling, --allow-unauthenticated on the Cloud Run service for
simplicity. Tighten both before using it on anything real.
"""
import os
import json
import subprocess
import tempfile
import shutil
import datetime
import logging
import re
import hashlib
from collections import Counter

import flask
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
BYTES_PER_GB = 1024 ** 3
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")  # PAT with repo scope, set as Cloud Run secret
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
# UI report config — runs_per_day / dag_task_id are NOT derivable
# from anything the bot can see (no Airflow/Composer access). These
# are hand-maintained assumptions for the demo, not detected values.
# Be explicit about that if asked.
# -----------------------------------------------------------------
QUERY_RUNTIME_CONFIG = {
    "resources/sql/retail_lending_portfolio.sql": {"dag_task_id": "rm_report_daily", "runs_per_day": 4},
    "resources/sql/customer_risk_dashboard.sql": {"dag_task_id": "risk_dashboard_hourly", "runs_per_day": 24},
    "resources/sql/delinquency_alerts.sql": {"dag_task_id": "delinquency_scan", "runs_per_day": 12},
    "resources/sql/employer_concentration.sql": {"dag_task_id": "employer_concentration_weekly", "runs_per_day": 1},
}
DEFAULT_RUNTIME_CONFIG = {"dag_task_id": "unscheduled", "runs_per_day": 1}


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


def clone_and_diff(repo_clone_url: str, before_sha: str, after_sha: str):
    """Clone the repo, return list of changed .sql files with old/new
    content, the concatenated Beam context, the list of Beam filenames
    seen, and the after_sha commit message."""
    tmp_dir = tempfile.mkdtemp()
    run_git("clone", repo_clone_url, tmp_dir, cwd="/tmp")

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
            # File was deleted in this diff — nothing to review or
            # optimize, skip it rather than crashing.
            logger.info(f"Skipping {path}: deleted between {before_sha} and {after_sha}")
            continue

        results.append({"path": path, "old": old_content, "new": new_content})

    beam_context = ""
    beam_filenames = []
    beam_full_path = os.path.join(tmp_dir, BEAM_DIR)
    if os.path.isdir(beam_full_path):
        for fname in os.listdir(beam_full_path):
            if fname.endswith(".py"):
                beam_filenames.append(fname)
                with open(os.path.join(beam_full_path, fname)) as f:
                    beam_context += f"\n--- {fname} ---\n{f.read()}"

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return results, beam_context, beam_filenames, commit_message


def _file_exists_at(repo_dir, sha, path):
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{sha}:{path}"], cwd=repo_dir, capture_output=True
    )
    return result.returncode == 0


def dry_run_bytes(sql_text: str) -> tuple[int, bool]:
    """Returns (bytes_scanned, success)."""
    if not sql_text:
        return 0, True

    sql_text = extract_sql(sql_text)

    if not sql_text.strip():
        return 0, True

    try:
        job_config = bigquery.QueryJobConfig(
            dry_run=True,
            use_query_cache=False
        )
        job = bq_client.query(sql_text, job_config=job_config)
        return job.total_bytes_processed, True
    except Exception as e:
        logger.exception(f"Dry run failed:\n{sql_text}")
        return 0, False


BYTES_PER_TB = 1024 ** 4
COST_PER_TB_USD = 6.25


def bytes_to_cost(num_bytes: int) -> float:
    if not num_bytes:
        return 0.0
    return (num_bytes / BYTES_PER_TB) * COST_PER_TB_USD


def _extract_tables(sql_text: str):
    """
    Extract fully-qualified table names from SQL enclosed in backticks.
    Example:
    `project.dataset.table`
    """
    return list(set(re.findall(r'`([^`]+)`', sql_text)))


def _count_tables_by_dataset(changed_results):
    counts = Counter()
    for r in changed_results:
        for t in _extract_tables(r["new_sql"]):
            parts = t.split(".")
            dataset = parts[-2] if len(parts) >= 2 else "unknown"
            counts[dataset] += 1
    return counts


def build_schema_manifest(sql_text: str) -> str:
    tables = _extract_tables(sql_text)
    logger.info(f"Extracted tables: {tables}")

    manifest = []

    for table in tables:
        try:
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

            column_query = f"""
            SELECT
                column_name,
                data_type,
                is_partitioning_column,
                clustering_ordinal_position
            FROM `{project}.{dataset}.INFORMATION_SCHEMA.COLUMNS`
            WHERE table_name='{table_name}'
            ORDER BY ordinal_position
            """

            columns = list(bq_client.query(column_query).result())

            table_query = f"""
            SELECT
                table_name,
                row_count,
                size_bytes
            FROM `{project}.{dataset}.INFORMATION_SCHEMA.TABLE_STORAGE`
            WHERE table_name='{table_name}'
            """

            try:
                table_info = list(bq_client.query(table_query).result())
            except Exception as e:
                logger.warning(f"Could not fetch TABLE_STORAGE for {table}: {e}")
                table_info = []

            manifest.append(f"\n===== TABLE : {table} =====")

            if table_info:
                t = table_info[0]
                manifest.append(f"Rows : {t.row_count}")
                manifest.append(f"Storage : {t.size_bytes} bytes")

            partition_cols = []
            clustering_cols = []

            for c in columns:
                manifest.append(f"- {c.column_name} ({c.data_type})")

                if c.is_partitioning_column == "YES":
                    partition_cols.append(c.column_name)

                if c.clustering_ordinal_position:
                    clustering_cols.append(c.column_name)

            manifest.append(
                f"Partition Columns : {partition_cols if partition_cols else 'None'}"
            )

            manifest.append(
                f"Cluster Columns : {clustering_cols if clustering_cols else 'None'}"
            )

        except Exception:
            logger.exception(f"Failed to discover schema for {table}")

    return "\n".join(manifest)


def find_existing_cache(display_name):
    """Look for a live, non-expired cache with our display name."""
    try:
        for cache in genai_client.caches.list():
            if cache.display_name == display_name:
                logger.info(f"Found existing cache: {cache.name} (expires {cache.expire_time})")
                return cache.name
    except Exception as e:
        logger.warning(f"Cache lookup failed ({e}), will attempt to create a new one")
    return None


CACHE_STORAGE_USD_PER_MILLION_TOKENS_PER_HOUR = 1.00
CACHE_WRITE_USD_PER_MILLION_TOKENS = 0.30   # billed once, at creation
CACHE_READ_USD_PER_MILLION_TOKENS = 0.03    # billed per call that reuses it


def cache_storage_cost_per_hour(token_count: int) -> float:
    return (token_count / 1_000_000) * CACHE_STORAGE_USD_PER_MILLION_TOKENS_PER_HOUR


def cache_write_cost(token_count: int) -> float:
    return (token_count / 1_000_000) * CACHE_WRITE_USD_PER_MILLION_TOKENS


def cache_read_cost(token_count: int) -> float:
    return (token_count / 1_000_000) * CACHE_READ_USD_PER_MILLION_TOKENS


def get_cache_token_count(cache_name: str) -> int:
    """Fetch the real token count for a cache from the API, not our
    rough len()//4 estimate — usageMetadata.totalTokenCount is exact."""
    try:
        cache = genai_client.caches.get(name=cache_name)
        return cache.usage_metadata.total_token_count
    except Exception as e:
        logger.warning(f"Could not fetch cache token count: {e}")
        return 0


def get_or_create_cache(schema_manifest: str, beam_context: str):
    PROMPT_VERSION = "v1"
    combined = PROMPT_VERSION + schema_manifest + "\n\n[DOWNSTREAM]\n" + beam_context
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
        return {
            "name": existing,
            "status": "HIT",
            "token_count": token_count
        }

    approx_tokens = len(combined) // 4
    logger.info(f"No existing cache found. Creating new one, manifest ~{approx_tokens} tokens (~{len(combined)} chars)")
    try:
        cache = genai_client.caches.create(
            model="gemini-2.5-flash",
            config={"contents": [combined], "ttl": "86400s", "display_name": display_name},
        )
        actual_tokens = cache.usage_metadata.total_token_count
        logger.info(
            f"Context cache created successfully: {cache.name} | actual size: {actual_tokens:,} tokens | "
            f"write cost: ${cache_write_cost(actual_tokens):.6f} | "
            f"storage: ${cache_storage_cost_per_hour(actual_tokens):.6f}/hr (${cache_storage_cost_per_hour(actual_tokens) * 24:.4f}/day)"
        )
        return {
            "name": cache.name,
            "status": "MISS",
            "token_count": actual_tokens
        }
    except Exception as e:
        logger.warning(f"Cache creation failed ({e}), falling back to inline context")
        return {
            "name": None,
            "status": "DISABLED",
            "token_count": 0
        }


def ask_gemini_for_rewrite(old_sql: str, new_sql: str, cache_name, schema_manifest, beam_context, original_bytes: int) -> str:
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
- For "risks": scan the Beam consumer code for any column used in arithmetic,
    comparison, or type-sensitive operation where the SQL schema shows that column
    as STRING. Each such mismatch is a risk entry. If none found, return an empty list.
    IMPORTANT: a column's output type is its INFORMATION_SCHEMA type UNLESS it is
    explicitly CAST in the SELECT list itself. CAST in WHERE clauses, ORDER BY, or
    window function arguments does NOT change the output type. Example:
    `SELECT delinq_2yrs FROM t WHERE CAST(delinq_2yrs AS INT64) > 0` — delinq_2yrs
    is still STRING when it reaches the Beam consumer. Flag any STRING column used
    in arithmetic (+,-,*,/) or numeric comparison (>,<,>=,<=) as severity HIGH.

**Input:**
{schema_manifest}

**Downstream Beam Consumer:**
{beam_context}

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
        logger.info(f"Calling Gemini WITH cached context: {cache_name}")
        response = genai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={
                "cached_content": cache_name,
                "temperature": 0,
            },
        )
    else:
        logger.info("Calling Gemini WITHOUT cache (inline context fallback)")
        full_prompt = (
            f"[SCHEMA]\n{schema_manifest}\n\n"
            f"[DOWNSTREAM CODE]\n{beam_context}\n\n"
            f"{prompt}"
        )
        response = genai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt,
            config={
                "temperature": 0,
            },
        )

    return response.text.strip()


def extract_sql(response_text: str) -> str:
    """
    Extract SQL from a Gemini markdown response.
    """
    if not response_text:
        return ""

    match = re.search(r"```sql\s*(.*?)```", response_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    match = re.search(r"```\s*(.*?)```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return response_text.strip()


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


def compute_guardrail_status(all_business_logic: list) -> str:
    if all(bl.get("status") == "PASS" for bl in all_business_logic):
        return "PASSED GUARDRAIL"
    return "FAILED GUARDRAIL"


def build_ui_report(branch: str, commit_message: str, changed_results: list, beam_filenames: list, cache_info) -> dict:
    queries = []
    all_business_logic = []
    total_original_cost = 0.0
    total_optimized_cost = 0.0

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
        })
        all_business_logic.append(r["business_logic"])

    net_savings = total_original_cost - total_optimized_cost
    savings_pct = (net_savings / total_original_cost * 100) if total_original_cost else 0.0

    dataset_counts = _count_tables_by_dataset(changed_results)
    bq_schemas_checked = ", ".join(
        f"{dataset} ({count} Table{'s' if count != 1 else ''})"
        for dataset, count in dataset_counts.items()
    ) or "None"

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
            "avro_mappings_matched": "N/A — no Avro schema in this pipeline",
            "dataflow_logic_validated": ", ".join(sorted(set(beam_filenames))) or "None",
            "vertex_cache": {
                "status": cache_info["status"],
                "token_count": cache_info["token_count"],
                "storage_cost_per_hour":
                    round(cache_storage_cost_per_hour(cache_info["token_count"]), 6),
                "read_cost":
                    round(cache_read_cost(cache_info["token_count"]), 6)
            }
        }
    }


@app.route("/review", methods=["POST"])
def review():
    payload = flask.request.get_json()
    repo_clone_url = payload["repo_clone_url"]
    repo_owner = payload["repo_owner"]
    repo_name = payload["repo_name"]
    before_sha = payload["before_sha"]
    after_sha = payload["after_sha"]
    branch = payload.get("branch", "unknown")

    changed, beam_context, beam_filenames, commit_message = clone_and_diff(
        repo_clone_url, before_sha, after_sha
    )
    if not changed:
        return flask.jsonify({"status": "no_sql_changes"})

    comment_sections = []
    changed_results = []  # accumulates data for the UI report

    for change in changed:
        old_bytes, old_ok = dry_run_bytes(change["old"]) if change["old"] else (None, True)
        new_bytes, new_ok = dry_run_bytes(change["new"])

        schema_manifest = build_schema_manifest(change["new"])
        cache_info = get_or_create_cache(schema_manifest, beam_context)
        cache_name = cache_info["name"]

        rewrite = ask_gemini_for_rewrite(
            change["old"],
            change["new"],
            cache_name,
            schema_manifest,
            beam_context,
            new_bytes,
        )

        # -----------------------------
        # Parse Gemini JSON
        # -----------------------------
        rewrite = rewrite.strip()
        if rewrite.startswith("```json"):
            rewrite = rewrite[7:]
        if rewrite.startswith("```"):
            rewrite = rewrite[3:]
        if rewrite.endswith("```"):
            rewrite = rewrite[:-3]
        rewrite = rewrite.strip()

        try:
            rewrite_json = json.loads(rewrite)
        except json.JSONDecodeError:
            logger.error(f"JSON parse failed:\n{rewrite}")
            rewrite_json = {
                "optimized_sql": change["new"],
                "business_logic": {"status": "FAIL", "reason": "Gemini JSON parse failed"},
                "summary": "Fallback to original",
                "changes": [],
                "risks": [],
                "recommendations": []
            }

        optimized_sql = rewrite_json["optimized_sql"]
        summary = rewrite_json.get("summary", "")
        changes = rewrite_json.get("changes", [])
        recommendations = rewrite_json.get("recommendations", [])
        business_logic = rewrite_json.get("business_logic", {})

        # -----------------------------
        # Dry run ONLY the optimized SQL
        # -----------------------------
        rewrite_bytes, rewrite_ok = dry_run_bytes(optimized_sql)

        savings_usd = bytes_to_cost(new_bytes) - bytes_to_cost(rewrite_bytes)
        savings_pct = (1 - rewrite_bytes / new_bytes) * 100 if new_bytes else 0
        risks = rewrite_json.get("risks", [])

        changed_results.append({
            "path": change["path"],
            "old_bytes": old_bytes,
            "new_bytes": new_bytes,
            "rewrite_bytes": rewrite_bytes,
            "savings_pct": savings_pct,
            "new_sql": change["new"],
            "optimized_sql": optimized_sql,
            "summary": summary,
            "business_logic": business_logic,
            "risks": risks, 
        })

        # -----------------------------
        # Build GitHub comment
        # -----------------------------
        section = f"## `{change['path']}`\n\n"

        if old_bytes is not None:
            section += f"**Previous:** {old_bytes:,} bytes scanned (${bytes_to_cost(old_bytes):.6f})\n\n"
        section += f"**Current:** {new_bytes:,} bytes scanned (${bytes_to_cost(new_bytes):.6f})\n\n"
        if not rewrite_ok:
            section += "**Gemini Rewrite:** ⚠️ dry-run failed — this SQL may be invalid, do not merge without manual review\n\n"
        else:
            section += f"**Gemini Rewrite:** {rewrite_bytes:,} bytes scanned (${bytes_to_cost(rewrite_bytes):.6f})\n\n"
        section += f"**Savings:** ${savings_usd:.6f} ({savings_pct:.1f}% reduction)\n\n"

        section += (
            f"### Context Cache\n"
            f"- Size: {cache_info['token_count']:,} tokens\n"
            f"- Storage cost: ${cache_storage_cost_per_hour(cache_info['token_count']):.6f}/hr "
            f"(${cache_storage_cost_per_hour(cache_info['token_count']) * 24:.4f}/day if held for 24h)\n"
            f"- Cost of this cached call: ${cache_read_cost(cache_info['token_count']):.6f} "
            f"(vs. ${(cache_info['token_count'] / 1_000_000) * 0.30:.6f} if sent inline, uncached)\n\n"
        )

        section += "### Business Logic\n"
        section += (
            f"- Status: **{business_logic.get('status', 'UNKNOWN')}**\n"
            f"- Reason: {business_logic.get('reason', '')}\n\n"
        )

        section += "### Summary\n"
        section += summary + "\n\n"

        if changes:
            section += "### Changes Applied\n"
            for item in changes:
                section += (
                    f"- **{item['change']}**\n"
                    f"  - {item['reason']}\n"
                )
            section += "\n"

        if recommendations:
            section += "### Recommendations\n"
            for rec in recommendations:
                section += f"- {rec}\n"
            section += "\n"

        section += (
            "<details>\n"
            "<summary><b>Optimized SQL</b></summary>\n\n"
            "```sql\n"
            f"{optimized_sql}\n"
            "```\n"
            "</details>\n"
        )

        comment_sections.append(section)

    comment_body = "## SQL Cost Review\n\n" + "\n\n---\n\n".join(comment_sections)
    post_github_comment(repo_owner, repo_name, after_sha, comment_body)

    ui_report = build_ui_report(branch, commit_message, changed_results, beam_filenames, cache_info)

    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob("latest_report.json")
        blob.upload_from_string(
            json.dumps(ui_report),
            content_type="application/json",
        )
    except Exception:
        logger.exception("Failed to upload report to GCS")

    return flask.jsonify({
        "status": "ok",
        "changed_files": [c["path"] for c in changed],
        "ui_report": ui_report,
    })

@app.route("/api/latest-report", methods=["GET"])
def latest_report():
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob("latest_report.json")

        if not blob.exists():
            return flask.jsonify({
                "status": "error",
                "message": "No report available."
            }), 404

        report = json.loads(blob.download_as_bytes().decode("utf-8"))

        return flask.jsonify(report)

    except Exception as e:
        logger.exception("Failed to fetch latest report")
        return flask.jsonify({
            "status": "error",
            "message": str(e)
        }), 500


from flask import render_template

@app.route("/")
def ui():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))