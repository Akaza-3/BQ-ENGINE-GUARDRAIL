"""
rag_visualize.py

Standalone, one-off script — NOT part of the deployed service.
Run this locally (same machine you tested test_embed.py on, so your
gcloud ADC credentials are already set up) to generate slide-ready
visuals of the RAG retrieval this project actually performs:

  1. rag_similarity_heatmap.png
     Every SQL query x every Beam function, cosine similarity.
     Rows are queries, columns are functions. The brightest cell in
     each row is the function RAG picks as that query's consumer.

  2. rag_embedding_space.png
     A 2D projection (PCA, no extra dependencies) of every embedded
     SQL query and Beam function. Points that are semantically related
     land near each other — this is literally what "retrieval by
     meaning, not keywords" looks like in space.

  3. rag_retrieval_table.csv
     query -> best-matched function -> file -> similarity score, as
     a plain table for a slide or the printed console output.

Lives in BQ-ENGINE-GUARDRAIL; reads SQL/Beam source from the sibling
grid_frequency_hackathon repo (override with the SOURCE_REPO env var
if your layout differs). Output images/CSV are written next to this
script, not into the source repo.

Usage:
    cd BQ-ENGINE-GUARDRAIL
    pip3 install matplotlib numpy --break-system-packages   # if needed
    python3 rag_visualize.py
"""
import os
import ast
import csv
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from google import genai

PROJECT_ID = os.environ.get("PROJECT_ID", "project-ff7c2ef5-8d88-401a-b86")
LOCATION = os.environ.get("LOCATION", "us-central1")
EMBED_MODEL = "text-embedding-004"

# This script lives in BQ-ENGINE-GUARDRAIL, but the SQL/Beam files it
# reads live in the sibling grid_frequency_hackathon repo. Override with
# SOURCE_REPO if your layout differs; outputs are still written next to
# this script (OUTPUT_DIR), not into the source repo.
OUTPUT_DIR = Path(__file__).parent
SOURCE_REPO = Path(os.environ.get(
    "SOURCE_REPO",
    str(OUTPUT_DIR.parent / "grid_frequency_hackathon"),
))
SQL_DIR = SOURCE_REPO / "resources" / "sql"
BEAM_DIR = SOURCE_REPO / "src" / "beam"

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)


def embed(texts: list[str]) -> np.ndarray:
    result = client.models.embed_content(model=EMBED_MODEL, contents=texts)
    return np.array([e.values for e in result.embeddings])


def load_sql_files() -> dict[str, str]:
    return {p.name: p.read_text() for p in sorted(SQL_DIR.glob("*.sql"))}


def _consumes_row(node: ast.FunctionDef) -> bool:
    """Only functions that take `row` as an argument are real per-record
    consumers. Orchestration functions like pipeline.run() mention every
    SQL file/table by name and would otherwise win retrieval generically
    — same filter as main.py's build_function_index."""
    return "row" in {a.arg for a in node.args.args}


def _row_columns(source: str) -> list[str]:
    """Every row["col"] subscript in this source. Same extraction as
    main.py's _row_columns — this is what actually gets embedded for
    retrieval, not the raw source (see module note on why raw text is
    too noisy a signal for this corpus)."""
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


def load_beam_functions() -> list[dict]:
    """One entry per row-consuming function def (top-level and nested/
    helper), same extraction logic as main.py's build_function_index."""
    entries = []
    for p in sorted(BEAM_DIR.glob("*.py")):
        src = p.read_text()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and _consumes_row(node):
                snippet = ast.get_source_segment(src, node)
                if snippet and snippet.strip():
                    entries.append({
                        "file": p.name,
                        "function": node.name,
                        "source": snippet,
                        "columns": _row_columns(snippet),
                    })
    return entries


def _extract_query_columns(sql_text: str, known_cols: set) -> list[str]:
    """Which of the known columns (union of every column any Beam
    function references — this offline script has no BigQuery access,
    so it approximates INFORMATION_SCHEMA with that union) appear as a
    whole word OUTSIDE any CAST(...) expression in this SQL. Same idea
    as main.py's _extract_query_columns (which uses the real schema
    manifest): stripping CAST(...) contents keeps alias names the query
    itself computes and explicit, non-renamed column references, while
    discarding computation-input mentions that often get renamed via
    `AS <alias>` in the same breath — e.g.
    `SUM(CAST(loan_amnt AS FLOAT64)) AS total_amount` should count as
    "this query produces total_amount," not "this query produces
    loan_amnt.\""""
    stripped = re.sub(r"CAST\([^()]*\)", " ", sql_text, flags=re.IGNORECASE)
    stripped = re.sub(r"CAST\([^()]*\)", " ", stripped, flags=re.IGNORECASE)  # one nesting level
    return sorted(c for c in known_cols if re.search(rf"\b{re.escape(c)}\b", stripped))


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
# 2. Per-FUNCTION column containment fixed that, but broke 2 of 6
#    differently: a tiny single-column predicate (e.g. a filter needing
#    only max_dti and max_revol_util) trivially hits 100% containment
#    against almost any risk-related query, out-ranking the real
#    formatter function that needs 9 columns and can mathematically never
#    reach 1.0. Scoring at the FILE level (union of every row-function's
#    columns in that file) fixes this — a narrow predicate can't win on
#    the file's behalf if its siblings need columns the query doesn't have.
#
# File-level containment + the CAST-stripping extraction above gets all
# 6 real demo queries right — 3 via confident retrieval with a clear
# margin, 3 via the margin/threshold safety net correctly recognizing a
# genuine near-tie and deferring to the known-good config default.
RAG_CONTAINMENT_WEIGHT = 0.85
RAG_COSINE_WEIGHT = 0.15


def _containment(function_columns: list[str], query_columns: list[str]) -> float:
    """What fraction of `function_columns` are present in `query_columns`.
    1.0 = the query supplies every column this function needs, regardless
    of how many extra columns the query also carries — full containment,
    not symmetric overlap (Jaccard unfairly penalizes functions matched
    against wide `SELECT *` queries)."""
    if not function_columns:
        return 0.0
    fset, qset = set(function_columns), set(query_columns)
    return len(fset & qset) / len(fset)


def main():
    print(f"Reading SQL/Beam source from: {SOURCE_REPO}")
    print("Loading SQL files...")
    sql_files = load_sql_files()
    print(f"  {len(sql_files)} SQL files: {list(sql_files)}")

    print("Extracting Beam functions...")
    functions = load_beam_functions()
    print(f"  {len(functions)} functions across "
          f"{len(set(f['file'] for f in functions))} files")

    if not sql_files or not functions:
        raise SystemExit(
            f"\nNo SQL files or Beam functions found under {SOURCE_REPO}.\n"
            f"Expected: {SQL_DIR} and {BEAM_DIR}\n"
            f"Set SOURCE_REPO to the correct path if grid_frequency_hackathon "
            f"isn't a sibling of this script, e.g.:\n"
            f"  SOURCE_REPO=/path/to/grid_frequency_hackathon python3 rag_visualize.py"
        )

    known_cols = set()
    for f in functions:
        known_cols |= set(f["columns"])
    print(f"  {len(known_cols)} distinct columns referenced across all functions")

    print("Embedding SQL queries (as column signatures, not raw SQL text)...")
    sql_names = list(sql_files.keys())
    sql_col_sigs = []
    sql_cols_list = []
    for name, text in sql_files.items():
        cols = _extract_query_columns(text, known_cols)
        sig = " ".join(cols) if cols else text
        sql_col_sigs.append(sig)
        sql_cols_list.append(cols)
        print(f"    {name}: {cols}")
    sql_vecs = embed(sql_col_sigs)

    print("Embedding Beam functions (as column signatures, not raw source)...")
    fn_labels = [f"{f['file']}::{f['function']}" for f in functions]
    fn_col_sigs = [" ".join(f["columns"]) or f["source"] for f in functions]
    fn_vecs = embed(fn_col_sigs)

    # --- Cosine similarity matrix: queries (rows) x functions (cols) ---
    def cosine_matrix(a, b):
        a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
        return a_norm @ b_norm.T

    sim = cosine_matrix(sql_vecs, fn_vecs)

    # --- File grouping: retrieval decides WHICH FILE is the consumer,
    # scored by that file's row-functions taken TOGETHER (union of
    # columns), not by ranking individual functions. See module note
    # above on why. ---
    file_names = sorted(set(f["file"] for f in functions))
    file_indices = {fn: [i for i, f in enumerate(functions) if f["file"] == fn] for fn in file_names}
    file_union_cols = {
        fn: sorted(set().union(*(set(functions[i]["columns"]) for i in idxs)))
        for fn, idxs in file_indices.items()
    }

    # --- Containment matrix: queries (rows) x FILES (cols) ---
    # containment[i, k] = fraction of file k's UNION of needed columns
    # that query i's column set actually supplies.
    file_containment = np.zeros((len(sql_names), len(file_names)))
    for i, qcols in enumerate(sql_cols_list):
        qset = set(qcols)
        for k, fn in enumerate(file_names):
            fcols = file_union_cols[fn]
            file_containment[i, k] = len(set(fcols) & qset) / len(fcols) if fcols else 0.0

    # --- Cosine matrix: queries (rows) x FILES (cols), max over that
    # file's individual function similarities ---
    file_cosine = np.zeros((len(sql_names), len(file_names)))
    for i in range(len(sql_names)):
        for k, fn in enumerate(file_names):
            file_cosine[i, k] = max(sim[i, j] for j in file_indices[fn])

    # --- Combined retrieval score per file: this decides the match ---
    file_combined = RAG_CONTAINMENT_WEIGHT * file_containment + RAG_COSINE_WEIGHT * file_cosine

    # --- Visual 1: heatmap ---
    # Plots the COMBINED score (containment + cosine) per FILE, since
    # that's what actually decides the match now — retrieval picks a
    # consumer FILE, not an individual function (see module note above).
    fig, ax = plt.subplots(figsize=(max(8, len(file_names) * 1.1), max(4, len(sql_names) * 0.6)))
    im = ax.imshow(file_combined, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(file_names)))
    ax.set_xticklabels(file_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(sql_names)))
    ax.set_yticklabels(sql_names, fontsize=9)
    ax.set_title(
        f"RAG retrieval: SQL query × Beam file combined score\n"
        f"({RAG_CONTAINMENT_WEIGHT:.0%} column containment + {RAG_COSINE_WEIGHT:.0%} cosine similarity; "
        f"brightest cell per row = the file RAG picks as consumer)",
        fontsize=10)
    for i in range(len(sql_names)):
        best_k = int(np.argmax(file_combined[i]))
        ax.add_patch(plt.Rectangle((best_k - 0.5, i - 0.5), 1, 1,
                                    fill=False, edgecolor="red", linewidth=2))
    fig.colorbar(im, ax=ax, label="combined retrieval score")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "rag_similarity_heatmap.png", dpi=180)
    print("Saved rag_similarity_heatmap.png")

    # --- Visual 2: 2D projection of the embedding space (manual PCA) ---
    all_vecs = np.vstack([sql_vecs, fn_vecs])
    all_labels = [f"SQL: {n}" for n in sql_names] + [f"Beam: {l}" for l in fn_labels]
    all_groups = (["query"] * len(sql_names)) + [f["file"] for f in functions]

    centered = all_vecs - all_vecs.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ vt[:2].T  # top-2 principal components

    fig2, ax2 = plt.subplots(figsize=(9, 7))
    unique_groups = sorted(set(all_groups))
    cmap = plt.get_cmap("tab10")
    for gi, group in enumerate(unique_groups):
        idx = [i for i, g in enumerate(all_groups) if g == group]
        marker = "*" if group == "query" else "o"
        size = 220 if group == "query" else 70
        ax2.scatter(proj[idx, 0], proj[idx, 1], label=group,
                    marker=marker, s=size, color=cmap(gi % 10),
                    edgecolors="black", linewidths=0.5)
    for i, label in enumerate(all_labels):
        short = label.split("::")[-1].split(": ")[-1]
        ax2.annotate(short, (proj[i, 0], proj[i, 1]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax2.set_title("Beam function + SQL query embeddings, projected to 2D (PCA)\n"
                  "Stars = SQL queries, circles = Beam functions, colored by source file",
                  fontsize=10)
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.legend(fontsize=8, loc="best")
    fig2.tight_layout()
    fig2.savefig(OUTPUT_DIR / "rag_embedding_space.png", dpi=180)
    print("Saved rag_embedding_space.png")

    # --- Table: best match per query, with runner-up + margin ---
    # Ranked by the per-FILE COMBINED score (containment + cosine), same
    # metric and same granularity main.py's retrieve_consumer_context()
    # now uses. A close top-1/top-2 gap is exactly how a genuinely
    # ambiguous query confuses retrieval — this is the diagnostic that
    # motivated main.py's RAG_MIN_MARGIN check, applied here to the
    # combined per-file score. MIN_SIM mirrors RAG_MIN_SIMILARITY.
    MIN_MARGIN = 0.03
    MIN_SIM = 0.55
    rows = []
    for i, qname in enumerate(sql_names):
        order = np.argsort(file_combined[i])[::-1]
        best_k, second_k = int(order[0]), int(order[1]) if len(order) > 1 else int(order[0])
        best_score = float(file_combined[i, best_k])
        second_score = float(file_combined[i, second_k])
        margin = best_score - second_score
        low_conf = best_score < MIN_SIM
        ambiguous = margin < MIN_MARGIN
        rows.append({
            "sql_query": qname,
            "matched_file": file_names[best_k],
            "combined_score": round(best_score, 3),
            "containment": round(float(file_containment[i, best_k]), 3),
            "cosine_sim": round(float(file_cosine[i, best_k]), 3),
            "runner_up_file": file_names[second_k],
            "runner_up_combined_score": round(second_score, 3),
            "margin": round(margin, 3),
            "low_confidence": low_conf,
            "ambiguous": ambiguous,
            "would_use_config_fallback": low_conf or ambiguous,
        })

    fieldnames = ["sql_query", "matched_file", "combined_score", "containment", "cosine_sim",
                  "runner_up_file", "runner_up_combined_score", "margin",
                  "low_confidence", "ambiguous", "would_use_config_fallback"]
    with open(OUTPUT_DIR / "rag_retrieval_table.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("Saved rag_retrieval_table.csv\n")

    print(f"{'SQL query':<32} {'matched file':<26} {'score':>6}  {'contain':>7}  {'cosine':>6}   "
          f"{'runner-up file':<26} {'score':>6}   {'margin':>6}  flag")
    print("-" * 155)
    for r in rows:
        flag = "-> config fallback used in main.py" if r["would_use_config_fallback"] else "-> RAG match used directly"
        print(f"{r['sql_query']:<32} {r['matched_file']:<26} {r['combined_score']:>6}  "
              f"{r['containment']:>7}  {r['cosine_sim']:>6}   "
              f"{r['runner_up_file']:<26} {r['runner_up_combined_score']:>6}   "
              f"{r['margin']:>6}  {flag}")


if __name__ == "__main__":
    main()
