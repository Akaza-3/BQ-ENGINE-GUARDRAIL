"""
query_test_harness.py

Generates comprehensive test data for a SQL query using the SAME
schema the RAG review bot already fetched from BigQuery INFORMATION_SCHEMA.

Flow:
  1. Call build_schema_manifest() from the review bot — same function,
     same BQ schema, same Vertex AI cache the bot already built
  2. Parse manifest → {table: {column: bq_type}} — the real schema
  3. Send SQL + real schema to Gemini → structured test cases
     (positive, negative, edge) with concrete row values
  4. Create test tables in loan_data_test mirroring the real schema
  5. Load test data, rewrite SQL to point at test dataset, run it
  6. Validate actual results vs expected, generate HTML report

Usage:
    export PROJECT_ID=project-ff7c2ef5-8d88-401a-b86
    cd grid_frequency_hackathon-main

    # Uses Gemini to auto-generate test cases from the real BQ schema
    python3 query_test_harness.py --sql resources/sql/portfolio_stress_test.sql

    # Preview test cases without loading to BQ
    python3 query_test_harness.py --sql resources/sql/portfolio_stress_test.sql --dry-run
"""

import os, re, sys, json, time, argparse, datetime, hashlib
from pathlib import Path

# ── Pull build_schema_manifest directly from the review bot ──────────────────
# This is the same function the bot uses — same INFORMATION_SCHEMA queries,
# same Vertex AI cache. We get the real schema for free.
GUARDRAIL_DIR = Path(__file__).parent.parent / "BQ-ENGINE-GUARDRAIL-main"
sys.path.insert(0, str(GUARDRAIL_DIR))
from main import build_schema_manifest, _extract_tables   # noqa: E402

from google.cloud import bigquery
from google import genai

PROJECT_ID   = os.environ.get("PROJECT_ID", "project-ff7c2ef5-8d88-401a-b86")
PROD_DATASET = "loan_data"
TEST_DATASET = "loan_data_test"

bq = bigquery.Client(project=PROJECT_ID)
ai = genai.Client(vertexai=True, project=PROJECT_ID, location="us-central1")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Parse the RAG schema manifest into a usable dict
# ─────────────────────────────────────────────────────────────────────────────

def parse_manifest(manifest: str) -> dict[str, dict[str, str]]:
    """
    Converts build_schema_manifest() output into:
      { "project.dataset.table": { "col_name": "BQ_TYPE", ... }, ... }

    Manifest format (one block per table):
      ===== TABLE : project.dataset.table =====
      Rows : N
      Storage : N bytes
      - column_name (DATA_TYPE)
      ...
      Partition Columns : [...]
      Cluster Columns   : [...]
    """
    schemas: dict[str, dict[str, str]] = {}
    current_table = None

    for line in manifest.splitlines():
        line = line.strip()

        # New table block
        m = re.match(r"=+ TABLE\s*:\s*(.+?)\s*=+", line)
        if m:
            current_table = m.group(1).strip()
            schemas[current_table] = {}
            continue

        # Column line: "- column_name (TYPE)"
        if current_table and line.startswith("- "):
            m2 = re.match(r"- (\w+)\s+\(([^)]+)\)", line)
            if m2:
                col, bq_type = m2.group(1), m2.group(2)
                schemas[current_table][col] = bq_type

    return schemas


def manifest_to_schema_block(schemas: dict[str, dict[str, str]]) -> str:
    """Renders schema as a clean block for the Gemini prompt."""
    lines = []
    for table, cols in schemas.items():
        lines.append(f"TABLE: {table}")
        for col, bq_type in cols.items():
            lines.append(f"  - {col}  ({bq_type})")
        lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Ask Gemini to generate test cases using the real schema
# ─────────────────────────────────────────────────────────────────────────────

GEMINI_PROMPT = """
You are a SQL test engineer. You have the EXACT BigQuery schema for every table
this query references (fetched live from INFORMATION_SCHEMA). Use it to generate
comprehensive test cases covering every logical branch.

REAL BIGQUERY SCHEMA (column names and types are exact — use them as-is):
{schema_block}

SQL QUERY TO TEST:
{sql}

Generate test cases for:
  POSITIVE — rows that SHOULD appear in the final result (one case per valid
             filter combination, JOIN path, and window-function outcome)
  NEGATIVE — rows that SHOULD be excluded (one case per filter that can reject rows,
             per JOIN that can drop rows, per window rank that deduplicates)
  EDGE     — boundary values (exactly at > / >= thresholds), NULLs in aggregated
             columns, CAST on strings that look numeric, window functions with
             ties, NTILE distribution, multi-row deduplication

Rules for generating row values:
  - Use the EXACT column names from the schema above — no invented columns
  - Match BQ types exactly: STRING columns get string values (even numeric ones
    like "35.0"), INT64/INTEGER get integers, FLOAT64 gets floats, BOOL gets bool
  - For columns the query CASTs (e.g. CAST(dti AS FLOAT64)), keep them as STRING
    in the test row — that is the real storage type
  - Make customer_id and loan_id unique per test case (prefix with the test case id)
  - For multi-table queries, provide rows for ALL tables with matching join keys

Return ONLY valid JSON — an array of objects, each with:
{{
  "id":             "TC_P01_description",
  "category":       "positive" | "negative" | "edge",
  "description":    "one sentence",
  "logic_tested":   "CTE / filter / expression being exercised",
  "expected":       "in_result" | "not_in_result" | "exactly_N_rows",
  "expected_count": 1,
  "rows_per_table": {{
    "project.dataset.table1": [ {{ col: val, ... }}, ... ],
    "project.dataset.table2": [ {{ col: val, ... }}, ... ]
  }}
}}
""".strip()


def generate_test_cases(sql_text: str, schema_block: str) -> list[dict]:
    print("  Asking Gemini to generate test cases from the real BQ schema...")
    prompt = GEMINI_PROMPT.format(schema_block=schema_block, sql=sql_text)
    resp = ai.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={"temperature": 0}
    )
    raw = resp.text.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    cases = json.loads(raw)
    print(f"  Gemini generated {len(cases)} test cases.")
    return cases


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Create BQ test tables mirroring the real schema
# ─────────────────────────────────────────────────────────────────────────────

# Map BQ type strings → bigquery.SchemaField types
BQ_TYPE_MAP = {
    "STRING": "STRING", "BYTES": "BYTES",
    "INT64": "INT64", "INTEGER": "INTEGER", "INT": "INT64",
    "FLOAT64": "FLOAT64", "FLOAT": "FLOAT64", "NUMERIC": "NUMERIC",
    "BOOL": "BOOL", "BOOLEAN": "BOOL",
    "DATE": "DATE", "DATETIME": "DATETIME",
    "TIMESTAMP": "TIMESTAMP", "TIME": "TIME",
    "ARRAY": "STRING",   # simplify nested types for test data
    "STRUCT": "STRING",
}


def create_test_table(table_ref: str, schema: dict[str, str]) -> str:
    """
    Mirror the prod table in TEST_DATASET with the same column names/types.
    Returns the test table id.
    """
    _, _, table_name = table_ref.split(".")
    test_table_id = f"{PROJECT_ID}.{TEST_DATASET}.{table_name}"

    fields = [
        bigquery.SchemaField(col, BQ_TYPE_MAP.get(bq_type, "STRING"))
        for col, bq_type in schema.items()
    ]
    bq.delete_table(test_table_id, not_found_ok=True)
    bq.create_table(bigquery.Table(test_table_id, schema=fields))
    print(f"    Created {test_table_id}  ({len(fields)} columns from real schema)")
    return test_table_id


def setup_test_tables(schemas: dict[str, dict[str, str]]) -> dict[str, str]:
    """Create all test tables. Returns {prod_table_ref: test_table_id}."""
    ds = bigquery.Dataset(f"{PROJECT_ID}.{TEST_DATASET}")
    ds.location = "us-central1"
    bq.create_dataset(ds, exists_ok=True)

    mapping = {}
    for table_ref, schema in schemas.items():
        test_id = create_test_table(table_ref, schema)
        mapping[table_ref] = test_id
    return mapping


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Load test rows into BQ
# ─────────────────────────────────────────────────────────────────────────────

def load_test_data(test_cases: list[dict], table_mapping: dict[str, str]):
    """
    Collect all rows from test cases and insert into the correct test tables.
    table_mapping: {prod_table_ref: test_table_id}
    """
    # Aggregate rows per test table
    rows_by_table: dict[str, list[dict]] = {tid: [] for tid in table_mapping.values()}

    for tc in test_cases:
        for prod_ref, rows in tc.get("rows_per_table", {}).items():
            test_id = table_mapping.get(prod_ref)
            if not test_id:
                # Try fuzzy match by table name suffix
                suffix = prod_ref.split(".")[-1]
                test_id = next(
                    (v for k, v in table_mapping.items() if k.endswith(suffix)),
                    None
                )
            if test_id and rows:
                # Strip None values so BQ treats them as NULL
                cleaned = [{k: v for k, v in r.items() if v is not None} for r in rows]
                rows_by_table.setdefault(test_id, []).extend(cleaned)

    for test_id, rows in rows_by_table.items():
        if rows:
            errs = bq.insert_rows_json(test_id, rows)
            if errs:
                print(f"    [!] Insert errors in {test_id}: {errs}")
            else:
                print(f"    Loaded {len(rows)} rows → {test_id}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Rewrite SQL to point at test dataset and run it
# ─────────────────────────────────────────────────────────────────────────────

def rewrite_sql(sql_text: str) -> str:
    return sql_text.replace(
        f"`{PROJECT_ID}.{PROD_DATASET}.",
        f"`{PROJECT_ID}.{TEST_DATASET}."
    )


def run_query(sql_text: str) -> list[dict]:
    time.sleep(3)   # allow streaming inserts to settle
    return [dict(r) for r in bq.query(sql_text).result()]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Validate actual vs expected
# ─────────────────────────────────────────────────────────────────────────────

def validate(test_cases: list[dict], actual_rows: list[dict]) -> list[dict]:
    """
    For each test case, find its rows in the actual result by customer_id
    (or the first unique key column), then check the expected outcome.
    """
    # Collect all customer_ids in actual output
    actual_ids = set(str(r.get("customer_id", "")) for r in actual_rows)

    results = []
    for tc in test_cases:
        # Collect all unique customer_ids this test case inserted
        tc_ids = set()
        for rows in tc.get("rows_per_table", {}).values():
            for row in rows:
                cid = row.get("customer_id")
                if cid:
                    tc_ids.add(str(cid))

        expected = tc.get("expected", "in_result")
        found_count = sum(1 for cid in actual_ids if cid in tc_ids)

        if expected == "in_result":
            passed = found_count > 0
            note = f"Found {found_count} row(s)" + (" ✓" if passed else " — expected ≥1")

        elif expected == "not_in_result":
            passed = found_count == 0
            note = f"Found {found_count} row(s)" + (" ✓" if passed else " — expected 0")

        elif expected == "exactly_N_rows":
            n = tc.get("expected_count", 1)
            passed = found_count == n
            note = f"Found {found_count} row(s) — expected {n}" + (" ✓" if passed else " ✗")

        else:
            passed, note = False, f"Unknown expected: {expected}"

        results.append({**tc, "passed": passed, "note": note})

    return results


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — HTML report
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(
    results: list[dict],
    sql_file: str,
    test_sql: str,
    manifest: str,
    out_path: str,
):
    total  = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed
    pct    = int(passed / total * 100) if total else 0

    by_cat: dict[str, list] = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r)

    def rows_html(cases):
        html = ""
        for r in cases:
            bg   = "#d4edda" if r["passed"] else "#f8d7da"
            icon = "✅" if r["passed"] else "❌"
            html += f"""
            <tr style="background:{bg}">
              <td><code>{r['id']}</code></td>
              <td style="text-align:center">{icon}</td>
              <td>{r['description']}</td>
              <td><em>{r['logic_tested']}</em></td>
              <td><code>{r['expected']}</code></td>
              <td>{r['note']}</td>
            </tr>"""
        return html

    overall_msg = (
        "<p style='color:#28a745;font-weight:700;font-size:1.1rem'>"
        "✅ All test cases passed — query is safe to finalize.</p>"
        if failed == 0 else
        f"<p style='color:#dc3545;font-weight:700;font-size:1.1rem'>"
        f"❌ {failed} test case(s) failed — review before finalizing.</p>"
    )

    sections = ""
    cat_meta = {
        "positive": ("Positive Cases", "should appear in result",  "#d4edda", "#155724"),
        "negative": ("Negative Cases", "should NOT appear",        "#f8d7da", "#721c24"),
        "edge":     ("Edge Cases",     "boundaries / NULLs / window functions", "#fff3cd", "#856404"),
    }
    for cat, cases in by_cat.items():
        title, subtitle, bg, fg = cat_meta.get(cat, (cat.title(), "", "#eee", "#000"))
        sections += f"""
        <h2>{title}
          <span style="font-size:.75rem;font-weight:600;padding:3px 10px;
                border-radius:12px;background:{bg};color:{fg};margin-left:8px">
            {subtitle}
          </span>
        </h2>
        <table>
          <tr><th>Test Case</th><th>Result</th><th>Description</th>
              <th>Logic Tested</th><th>Expected</th><th>Outcome</th></tr>
          {rows_html(cases)}
        </table>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Query Test Report — {Path(sql_file).name}</title>
<style>
  body  {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           margin: 40px; color: #212529; background: #f8f9fa; }}
  h1    {{ color: #343a40; }}
  h2    {{ color: #495057; border-bottom: 2px solid #dee2e6;
           padding-bottom: 6px; margin-top: 40px; }}
  .summary {{ display:flex; gap:20px; margin:24px 0; flex-wrap:wrap; }}
  .card {{ background:#fff; border-radius:8px; padding:20px 28px;
           box-shadow:0 1px 4px rgba(0,0,0,.1); text-align:center; }}
  .num  {{ font-size:2.4rem; font-weight:700; }}
  .lbl  {{ font-size:.85rem; color:#6c757d; margin-top:4px; }}
  .pass {{ color:#28a745; }} .fail {{ color:#dc3545; }} .tot {{ color:#343a40; }}
  table {{ width:100%; border-collapse:collapse; background:#fff; border-radius:8px;
           overflow:hidden; box-shadow:0 1px 4px rgba(0,0,0,.1); margin-bottom:24px; }}
  th    {{ background:#343a40; color:#fff; padding:10px 14px;
           text-align:left; font-size:.85rem; }}
  td    {{ padding:10px 14px; border-bottom:1px solid #dee2e6;
           font-size:.88rem; vertical-align:top; }}
  tr:last-child td {{ border-bottom:none; }}
  pre   {{ background:#f1f3f5; padding:16px; border-radius:6px; font-size:.8rem;
           overflow-x:auto; max-height:350px; white-space:pre-wrap; }}
  .badge {{ background:#e9ecef; color:#495057; padding:2px 8px;
            border-radius:4px; font-size:.8rem; font-family:monospace; }}
</style>
</head>
<body>

<h1>🧪 Query Test Report</h1>
<p>
  <strong>SQL file:</strong> <code>{sql_file}</code><br>
  <strong>Schema source:</strong> BigQuery INFORMATION_SCHEMA (via RAG bot's
  <code>build_schema_manifest()</code>)<br>
  <strong>Test dataset:</strong>
  <span class="badge">{PROJECT_ID}.{TEST_DATASET}</span>
  &nbsp;(prod is never touched)<br>
  <strong>Generated:</strong>
  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
</p>

<div class="summary">
  <div class="card"><div class="num tot">{total}</div><div class="lbl">Total Cases</div></div>
  <div class="card"><div class="num pass">{passed}</div><div class="lbl">Passed</div></div>
  <div class="card"><div class="num fail">{failed}</div><div class="lbl">Failed</div></div>
  <div class="card">
    <div class="num {'pass' if pct == 100 else 'fail'}">{pct}%</div>
    <div class="lbl">Pass Rate</div>
  </div>
</div>

{overall_msg}

{sections}

<h2>Schema Used (from RAG bot — INFORMATION_SCHEMA)</h2>
<pre>{manifest}</pre>

<h2>SQL Executed Against Test Data</h2>
<pre>{test_sql}</pre>

</body>
</html>"""

    Path(out_path).write_text(html)
    print(f"\n  Report saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sql",     required=True, help="SQL file to test")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show test cases without loading to BigQuery")
    args = parser.parse_args()

    sql_path = args.sql
    sql_text = Path(sql_path).read_text()
    test_sql = rewrite_sql(sql_text)
    report_path = Path(sql_path).stem + "_report.html"

    print(f"\n{'='*65}")
    print(f"  Query Test Harness  —  schema from RAG / INFORMATION_SCHEMA")
    print(f"  SQL    : {sql_path}")
    print(f"  Report : {report_path}")
    print(f"{'='*65}\n")

    # ── Step 1: Fetch the real BQ schema via the RAG bot's manifest builder
    print("STEP 1 — Fetching schema via build_schema_manifest() (same as RAG bot)")
    manifest = build_schema_manifest(sql_text)
    schemas  = parse_manifest(manifest)

    if not schemas:
        print("  [!] No tables found in manifest. Check GCP credentials and PROJECT_ID.")
        sys.exit(1)

    for table, cols in schemas.items():
        print(f"  {table}: {len(cols)} columns")
    schema_block = manifest_to_schema_block(schemas)

    # ── Step 2: Ask Gemini to generate test cases from the real schema
    print("\nSTEP 2 — Generating test cases (Gemini + real schema)")
    test_cases = generate_test_cases(sql_text, schema_block)

    if args.dry_run:
        print("\n[DRY RUN] Generated test cases:")
        for tc in test_cases:
            row_counts = {t.split(".")[-1]: len(r)
                          for t, r in tc.get("rows_per_table", {}).items()}
            print(f"  [{tc['category'].upper():8s}] {tc['id']}")
            print(f"             {tc['description']}")
            print(f"             rows: {row_counts}  expected: {tc['expected']}")
        cat_counts = {}
        for tc in test_cases:
            cat_counts[tc["category"]] = cat_counts.get(tc["category"], 0) + 1
        print(f"\n  Total: {len(test_cases)}  |  " +
              "  |  ".join(f"{k}: {v}" for k, v in cat_counts.items()))
        return

    # ── Step 3: Create test tables using the real schema
    print("\nSTEP 3 — Creating test tables in", TEST_DATASET)
    table_mapping = setup_test_tables(schemas)

    # ── Step 4: Load test data
    print("\nSTEP 4 — Loading test data")
    load_test_data(test_cases, table_mapping)

    # ── Step 5: Run the query
    print("\nSTEP 5 — Running query against test data")
    print(f"  (SQL rewritten: {PROD_DATASET} → {TEST_DATASET})")
    actual_rows = run_query(test_sql)
    print(f"  Query returned {len(actual_rows)} row(s)")

    # ── Step 6: Validate
    print("\nSTEP 6 — Validating results")
    results = validate(test_cases, actual_rows)
    passed  = sum(1 for r in results if r["passed"])
    for r in results:
        icon = "✅" if r["passed"] else "❌"
        print(f"  {icon} [{r['category'].upper():8s}] {r['id']}: {r['note']}")

    # ── Step 7: Report
    print("\nSTEP 7 — Generating HTML report")
    generate_report(results, sql_path, test_sql, manifest, report_path)

    print(f"\n{'='*65}")
    if passed == len(results):
        print(f"  ✅ ALL {len(results)} TESTS PASSED — safe to finalize.")
    else:
        print(f"  ❌ {len(results) - passed}/{len(results)} FAILED — fix before finalizing.")
    print(f"  Report: {report_path}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
