"""
TruLens Replay Evaluator

Usage:
    python tru_eval_replay.py --csv path/to/log.csv \
        --app-name MyApp --app-version 1.0 --run-name run1 \
        --snowflake-account <account> --snowflake-user <user> --snowflake-password <pw> 

What it does:
- Loads a CSV of existing LLM runs/logs (rows should contain input/output and optionally ground_truth/context).
- Auto-detects common column names and maps them to TruLens span attributes.
- Creates a TruApp and RunConfig, registers the app with Snowflake, creates a run, starts invocation using the CSV as a DATAFRAME source, and computes appropriate metrics (only metrics possible given available columns).

Notes & assumptions:
- This code follows examples from TruLens docs. Some SDK function signatures may vary depending on the installed package version. If you hit an error, check the TruLens package docs for exact class names and parameter signatures and modify the small adapter functions below.
- Snowflake connection here is simplified; we assume a SnowflakeConnector wrapper class exists and accepts connection parameters shown. If your code uses a different connector factory, replace SnowflakeConnector(...) with your project's connector creation code.

"""
import os
import argparse
import pandas as pd
from typing import Dict, List

# --- TruLens imports ---
try:
    from trulens.core.otel.instrument import instrument
    from trulens.otel.semconv.trace import SpanAttributes
    from trulens.sdk import TruApp, RunConfig  # <-- adjust if the package path differs
    from trulens.connectors.snowflake import SnowflakeConnector  # <-- adjust to your environment
except Exception as e:
    print("Warning: Could not import TruLens SDK classes. Make sure `trulens` is installed and import paths match your SDK version.")
    print(e)
    # We'll still generate the script. If runtime imports fail, user must fix environment.

COMMON_INPUT_NAMES = ["input", "prompt", "query", "user_query", "question", "prompt_text"]
COMMON_OUTPUT_NAMES = ["output", "response", "answer", "generated", "model_output"]
COMMON_GT_NAMES = ["ground_truth", "golden_answer", "golden", "expected", "target"]
COMMON_CONTEXT_NAMES = ["context", "retrieved_contexts", "contexts", "kb", "passages"]

METRICS_IF_AVAILABLE = {
    "correctness": ["RECORD_ROOT.INPUT", "RECORD_ROOT.OUTPUT", "RECORD_ROOT.GROUND_TRUTH_OUTPUT"],
    "answer_relevance": ["RECORD_ROOT.INPUT", "RECORD_ROOT.OUTPUT"],
    "context_relevance": ["RETRIEVAL.QUERY_TEXT", "RETRIEVAL.RETRIEVED_CONTEXTS"],
    "groundedness": ["RETRIEVAL.RETRIEVED_CONTEXTS", "RECORD_ROOT.OUTPUT"],
    "coherence": ["RECORD_ROOT.OUTPUT"]
}


def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Detect column names in the user's CSV and return a mapping from span-attributes to df column names."""
    cols = {c.lower(): c for c in df.columns}

    def find_one(candidates: List[str]):
        for cand in candidates:
            if cand in cols:
                return cols[cand]
        return None

    mapping = {}
    in_col = find_one(COMMON_INPUT_NAMES)
    out_col = find_one(COMMON_OUTPUT_NAMES)
    gt_col = find_one(COMMON_GT_NAMES)
    ctx_col = find_one(COMMON_CONTEXT_NAMES)

    if in_col:
        mapping["RECORD_ROOT.INPUT"] = in_col
        mapping["RETRIEVAL.QUERY_TEXT"] = in_col  # many logs use same col for query
    if out_col:
        mapping["RECORD_ROOT.OUTPUT"] = out_col
    if gt_col:
        mapping["RECORD_ROOT.GROUND_TRUTH_OUTPUT"] = gt_col
    if ctx_col:
        mapping["RETRIEVAL.RETRIEVED_CONTEXTS"] = ctx_col

    return mapping


def pick_metrics_from_mapping(mapping: Dict[str, str]) -> List[str]:
    """Return a list of metrics that can be computed given the available span-attribute -> column mapping."""
    available = set(mapping.keys())
    chosen = []
    for metric, reqs in METRICS_IF_AVAILABLE.items():
        if all(r in available for r in reqs):
            chosen.append(metric)
    # Coherence only needs output; include it if output present
    if "RECORD_ROOT.OUTPUT" in available and "coherence" not in chosen:
        chosen.append("coherence")
    return chosen


class ReplayApp:
    """A tiny app used to "replay" recorded outputs from the CSV.

    Assumption: When TruApp.run starts with source_type=DATAFRAME, the SDK will call
    main_method with each row's *input* and optionally other columns. Because SDK
    implementations differ, we implement the main_method to accept a dictionary-like
    `record` and return the recorded output. If your TruLens SDK calls the method
    signature differently, adapt the wrapper below (adapter function `make_main_method`).
    """

    def __init__(self, output_col: str):
        self.output_col = output_col

    # Keep this method simple; decorated at registration time if needed.
    def replay(self, record: Dict) -> str:
        # record might be a pandas Series or dict - handle both
        if isinstance(record, pd.Series):
            return record.get(self.output_col, "")
        elif isinstance(record, dict):
            return record.get(self.output_col, "")
        else:
            # fallback
            try:
                return getattr(record, self.output_col)
            except Exception:
                return ""


def make_snowflake_connector_from_env(args) -> object:
    # Replace this with your org's Snowflake connector wrapper
    # This is a placeholder to show required parameters; the real SnowflakeConnector
    # constructor may differ.
    try:
        return SnowflakeConnector(
            account=args.snowflake_account,
            user=args.snowflake_user,
            password=args.snowflake_password,
            warehouse=args.snowflake_warehouse,
            database=args.snowflake_database,
            schema=args.snowflake_schema,
        )
    except Exception as e:
        print("Ensure SnowflakeConnector class and parameters match your SDK. Error:", e)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CSV file with logs")
    parser.add_argument("--app-name", required=True)
    parser.add_argument("--app-version", default="1.0")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--llm-judge", default=None, help="Optional LLM judge name (e.g. mistral-large2)")

    # Snowflake args (can be set via env vars in production)
    parser.add_argument("--snowflake-account", default=os.environ.get("SNOWFLAKE_ACCOUNT"))
    parser.add_argument("--snowflake-user", default=os.environ.get("SNOWFLAKE_USER"))
    parser.add_argument("--snowflake-password", default=os.environ.get("SNOWFLAKE_PASSWORD"))
    parser.add_argument("--snowflake-warehouse", default=os.environ.get("SNOWFLAKE_WAREHOUSE"))
    parser.add_argument("--snowflake-database", default=os.environ.get("SNOWFLAKE_DATABASE"))
    parser.add_argument("--snowflake-schema", default=os.environ.get("SNOWFLAKE_SCHEMA"))

    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")

    mapping = detect_columns(df)
    print("Detected mapping:")
    for k, v in mapping.items():
        print(f"  {k} -> {v}")

    metrics = pick_metrics_from_mapping(mapping)
    print("Metrics that will be computed based on available columns:", metrics)

    # Create Snowflake connector
    connector = make_snowflake_connector_from_env(args)

    # Create Replay app instance
    output_col = mapping.get("RECORD_ROOT.OUTPUT")
    if output_col is None:
        raise ValueError("CSV does not contain an output column. One of these column names is required: " + ",".join(COMMON_OUTPUT_NAMES))

    replay_app = ReplayApp(output_col=output_col)

    # Register app in TruLens (adjust constructor names as necessary for your SDK version)
    try:
        tru_app = TruApp(
            test_app=replay_app,
            app_name=args.app_name,
            app_version=args.app_version,
            connector=connector,
            main_method=replay_app.replay
        )
    except Exception as e:
        print("Error creating TruApp. Make sure the TruLens SDK import paths and constructor signature match your environment.")
        raise

    # Create RunConfig using the detected mapping
    run_config = RunConfig(
        run_name=args.run_name,
        description=f"Replay run from CSV {os.path.basename(args.csv)}",
        label="replay_import",
        source_type="DATAFRAME",
        dataset_name=os.path.basename(args.csv),
        dataset_spec=mapping,
        llm_judge_name=args.llm_judge or None,
    )

    run = tru_app.add_run(run_config=run_config)
    print("Run created. Starting invocation (this call blocks until invocation completes or times out)...")

    # Start run using DataFrame source
    run.start(input_df=df)

    # Wait and poll status until invocation completes or partially completes (simple polling)
    import time
    while True:
        status = run.get_status()
        print("Run status:", status)
        if status in ("INVOCATION_COMPLETED", "INVOCATION_PARTIALLY_COMPLETED", "INVOCATION_FAILED"):
            break
        time.sleep(5)

    print("Invocation finished. Starting metric computation...")
    # compute metrics; if none found, compute at least coherence if possible
    if not metrics:
        metrics = ["coherence"] if "RECORD_ROOT.OUTPUT" in mapping else []

    if metrics:
        compute_job = run.compute_metrics(metrics=metrics)
        print("Triggered metric computation job. compute_metrics is non-blocking.")
        print("You can check job status from SnowSight AI & ML -> Evaluations or via SDK methods.")
    else:
        print("No metrics available to compute with the provided CSV columns.")


if __name__ == "__main__":
    main()
