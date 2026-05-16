"""
Nomura AI Assistant Log Classification Pipeline
================================================
Steps:
    0 - Setup: Load taxonomy, build embedding vectors
    1 - Ingest & Preprocess: Load Excel, session grouping, metadata enrichment
    2 - Hybrid Semantic Scoring: Prototype (50%) + Keyword (30%) + Definition (20%)
    3 - Threshold Decision: High confidence → label, Low → llm_classify, Very low → unclassified
    4 - LLM Fallback: SKIPPED (routed to llm_classify)
    5 - Governance Flagging: Flag high/medium risk domain+intent combos + keyword watchlist
    6 - Output & Storage: Write labeled records, governance alerts, llm queue

Usage:
    python nomura_pipeline.py \
        --input_path /path/to/logs.xlsx \
        --taxonomy_path /path/to/nomura_taxonomy.json \
        --embedding_model_path /path/to/embedding/model \
        --output_dir /path/to/output/
"""

import os
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

# Scoring weights
WEIGHT_PROTOTYPE   = 0.50
WEIGHT_KEYWORD     = 0.30
WEIGHT_DEFINITION  = 0.20

# Confidence thresholds (tune from 200-log pilot)
HIGH_CONF_THRESHOLD = 0.75
LOW_CONF_THRESHOLD  = 0.50

# Metadata boost when department matches predicted domain keyword
METADATA_BOOST = 1.20

# Minimum prompt token length — shorter prompts merged with next turn
MIN_PROMPT_TOKENS = 5

# Max characters of prompt sent to LLM (when implemented)
LLM_PROMPT_TRUNCATE = 300

# Department → domain code mapping (extend as needed)
DEPARTMENT_DOMAIN_MAP = {
    "trading":       "D02",
    "execution":     "D02",
    "risk":          "D03",
    "market risk":   "D03",
    "credit risk":   "D03",
    "compliance":    "D04",
    "regulatory":    "D04",
    "legal":         "D04",
    "m&a":           "D05",
    "advisory":      "D05",
    "dcm":           "D05",
    "ecm":           "D05",
    "finance":       "D06",
    "accounting":    "D06",
    "technology":    "D07",
    "engineering":   "D07",
    "it":            "D07",
    "research":      "D08",
    "analytics":     "D08",
    "hr":            "D09",
    "people":        "D09",
    "human resources": "D09",
}

# Suspicious keyword watchlist for anomaly layer (maintain with compliance team)
SUSPICIOUS_WATCHLIST = [
    "launder", "laundering", "money laundering",
    "insider", "MNPI", "non-public",
    "manipulate", "manipulation", "front run", "front-run",
    "bypass", "circumvent", "avoid detection",
    "off the record", "delete this", "don't tell",
    "shell company", "nominee", "fictitious",
    "bribe", "kickback", "corrupt",
]


# ─────────────────────────────────────────────
# STEP 0 — Setup
# ─────────────────────────────────────────────

def load_taxonomy(taxonomy_path: str) -> dict:
    """Load taxonomy JSON from disk."""
    log.info(f"Loading taxonomy from {taxonomy_path}")
    with open(taxonomy_path, "r") as f:
        taxonomy = json.load(f)
    log.info(
        f"Taxonomy loaded: {len(taxonomy['domains'])} domains, "
        f"{len(taxonomy['intents'])} intents"
    )
    return taxonomy


def build_taxonomy_vectors(taxonomy: dict, model: SentenceTransformer) -> dict:
    """
    Pre-compute and cache all embedding vectors for domains and intents.

    Returns:
        {
            "domains": {
                "D01": {
                    "name": ...,
                    "definition_vector": np.array,
                    "keyword_vectors": [np.array, ...],
                    "phrase_vectors": [np.array, ...]
                },
                ...
            },
            "intents": { same structure }
        }
    """
    vectors = {"domains": {}, "intents": {}}

    for category_type in ["domains", "intents"]:
        log.info(f"Building vectors for {category_type}...")
        for category in taxonomy[category_type]:
            code = category["code"]
            log.debug(f"  Embedding {code} - {category['name']}")

            definition_vector = model.encode(
                category["definition"], convert_to_numpy=True
            )
            keyword_vectors = model.encode(
                category["keywords"], convert_to_numpy=True
            )
            phrase_vectors = model.encode(
                category["phrases"], convert_to_numpy=True
            )

            vectors[category_type][code] = {
                "name":               category["name"],
                "definition_vector":  definition_vector,
                "keyword_vectors":    keyword_vectors,
                "phrase_vectors":     phrase_vectors,
            }

    log.info("Taxonomy vectors built successfully")
    return vectors


def load_governance_matrix(taxonomy: dict) -> dict:
    """
    Load governance flag matrix.
    Returns dict keyed by (domain_code, intent_code) → {level, reason}
    """
    matrix = {}
    gfm = taxonomy.get("governance_flag_matrix", {})

    for entry in gfm.get("high_risk", []):
        key = (entry["domain"], entry["intent"])
        matrix[key] = {"level": "HIGH", "reason": entry["reason"]}

    for entry in gfm.get("medium_risk", []):
        key = (entry["domain"], entry["intent"])
        matrix[key] = {"level": "MEDIUM", "reason": entry["reason"]}

    log.info(f"Governance matrix loaded: {len(matrix)} flagged combinations")
    return matrix


# ─────────────────────────────────────────────
# STEP 1 — Data Ingestion & Preprocessing
# ─────────────────────────────────────────────

def load_logs(input_path: str) -> pd.DataFrame:
    """Load logs from Excel file."""
    log.info(f"Loading logs from {input_path}")
    df = pd.read_excel(input_path)

    required_columns = [
        "session_id", "query_timestamp", "username",
        "prompt", "co", "user_type", "division"
    ]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    log.info(f"Loaded {len(df)} raw log rows")
    return df


def preprocess_logs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare raw logs:
        - Normalise text fields
        - Sort by session + timestamp
        - Flag short prompts for merging
    """
    log.info("Preprocessing logs...")

    df = df.copy()
    df["prompt"]   = df["prompt"].fillna("").astype(str).str.strip()
    df["division"] = df["division"].fillna("").astype(str).str.strip().str.lower()
    df["user_type"]= df["user_type"].fillna("").astype(str).str.strip().str.lower()
    df["co"]       = df["co"].fillna("").astype(str).str.strip().str.lower()

    df["query_timestamp"] = pd.to_datetime(df["query_timestamp"], errors="coerce")
    df = df.sort_values(["session_id", "query_timestamp"]).reset_index(drop=True)

    # Token length approximation (split on whitespace)
    df["prompt_token_count"] = df["prompt"].apply(lambda x: len(x.split()))
    df["is_short_prompt"]    = df["prompt_token_count"] < MIN_PROMPT_TOKENS

    log.info(
        f"Preprocessing done. Short prompts: "
        f"{df['is_short_prompt'].sum()} / {len(df)}"
    )
    return df


def group_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group rows by session_id, merge short prompts with adjacent turns,
    and produce one combined_prompt per session with enriched metadata.

    Returns one row per session.
    """
    log.info("Grouping sessions...")
    sessions = []

    for session_id, group in df.groupby("session_id"):
        group = group.reset_index(drop=True)
        prompts = group["prompt"].tolist()
        is_short = group["is_short_prompt"].tolist()

        # Merge short prompts with next turn
        merged_prompts = []
        i = 0
        while i < len(prompts):
            if is_short[i] and i + 1 < len(prompts):
                merged_prompts.append(prompts[i] + " " + prompts[i + 1])
                i += 2
            else:
                merged_prompts.append(prompts[i])
                i += 1

        combined_prompt = " | ".join(merged_prompts)

        # Metadata: take most common non-null value per session
        division  = group["division"].mode()[0]  if not group["division"].mode().empty  else ""
        user_type = group["user_type"].mode()[0] if not group["user_type"].mode().empty else ""
        co        = group["co"].mode()[0]        if not group["co"].mode().empty        else ""
        username  = group["username"].iloc[0]

        # Domain prior from department
        domain_prior = None
        for dept_keyword, domain_code in DEPARTMENT_DOMAIN_MAP.items():
            if dept_keyword in division:
                domain_prior = domain_code
                break

        sessions.append({
            "session_id":      session_id,
            "username":        username,
            "co":              co,
            "user_type":       user_type,
            "division":        division,
            "combined_prompt": combined_prompt,
            "turn_count":      len(group),
            "domain_prior":    domain_prior,
            "first_timestamp": group["query_timestamp"].min(),
            "last_timestamp":  group["query_timestamp"].max(),
        })

    sessions_df = pd.DataFrame(sessions)
    log.info(f"Session grouping done: {len(sessions_df)} sessions from {len(df)} rows")
    return sessions_df


# ─────────────────────────────────────────────
# STEP 2 — Hybrid Semantic Scoring
# ─────────────────────────────────────────────

def score_against_category(
    input_vector: np.ndarray,
    category_vectors: dict,
    domain_prior: str | None,
    category_code: str
) -> float:
    """
    Compute weighted hybrid score for one input vector against one category.

        score = 0.5 * max_prototype_similarity
              + 0.3 * max_keyword_similarity
              + 0.2 * definition_similarity

    Applies metadata boost if domain_prior matches this category.
    """
    input_2d = input_vector.reshape(1, -1)

    # Prototype score — max similarity to any phrase
    phrase_sims  = cosine_similarity(input_2d, category_vectors["phrase_vectors"])[0]
    proto_score  = float(np.max(phrase_sims))

    # Keyword score — max similarity to any keyword
    keyword_sims = cosine_similarity(input_2d, category_vectors["keyword_vectors"])[0]
    kw_score     = float(np.max(keyword_sims))

    # Definition score
    def_sim      = cosine_similarity(
        input_2d, category_vectors["definition_vector"].reshape(1, -1)
    )[0][0]
    def_score    = float(def_sim)

    score = (
        WEIGHT_PROTOTYPE  * proto_score +
        WEIGHT_KEYWORD    * kw_score    +
        WEIGHT_DEFINITION * def_score
    )

    # Metadata boost
    if domain_prior and domain_prior == category_code:
        score = min(score * METADATA_BOOST, 1.0)

    return score


def classify_session(
    combined_prompt: str,
    domain_prior: str | None,
    model: SentenceTransformer,
    taxonomy_vectors: dict
) -> dict:
    """
    Run hybrid scoring for a single session.
    Returns best domain, best intent, and their confidence scores.
    """
    input_vector = model.encode(combined_prompt, convert_to_numpy=True)

    # Score all domains
    domain_scores = {}
    for code, vectors in taxonomy_vectors["domains"].items():
        domain_scores[code] = score_against_category(
            input_vector, vectors, domain_prior, code
        )

    # Score all intents (no prior for intents)
    intent_scores = {}
    for code, vectors in taxonomy_vectors["intents"].items():
        intent_scores[code] = score_against_category(
            input_vector, vectors, None, code
        )

    best_domain      = max(domain_scores, key=domain_scores.get)
    domain_confidence = domain_scores[best_domain]

    best_intent      = max(intent_scores, key=intent_scores.get)
    intent_confidence = intent_scores[best_intent]

    combined_confidence = (domain_confidence + intent_confidence) / 2

    return {
        "predicted_domain":     best_domain,
        "domain_confidence":    round(domain_confidence, 4),
        "predicted_intent":     best_intent,
        "intent_confidence":    round(intent_confidence, 4),
        "combined_confidence":  round(combined_confidence, 4),
        "all_domain_scores":    {k: round(v, 4) for k, v in domain_scores.items()},
        "all_intent_scores":    {k: round(v, 4) for k, v in intent_scores.items()},
    }


def run_semantic_scoring(
    sessions_df: pd.DataFrame,
    model: SentenceTransformer,
    taxonomy_vectors: dict
) -> pd.DataFrame:
    """Run hybrid scoring across all sessions."""
    log.info(f"Running semantic scoring on {len(sessions_df)} sessions...")

    results = []
    for i, row in sessions_df.iterrows():
        if i % 500 == 0:
            log.info(f"  Scoring session {i+1}/{len(sessions_df)}...")

        scores = classify_session(
            combined_prompt=row["combined_prompt"],
            domain_prior=row["domain_prior"],
            model=model,
            taxonomy_vectors=taxonomy_vectors
        )
        results.append(scores)

    scores_df = pd.DataFrame(results)
    sessions_df = pd.concat(
        [sessions_df.reset_index(drop=True), scores_df.reset_index(drop=True)],
        axis=1
    )

    log.info("Semantic scoring complete")
    return sessions_df


# ─────────────────────────────────────────────
# STEP 3 — Threshold Decision
# ─────────────────────────────────────────────

def apply_threshold(sessions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign label_source based on confidence thresholds:
        >= HIGH_CONF_THRESHOLD  → "semantic"
        >= LOW_CONF_THRESHOLD   → "llm_classify"  (LLM fallback, to be implemented)
        <  LOW_CONF_THRESHOLD   → "unclassified"  (manual review)
    """
    log.info("Applying confidence thresholds...")

    def get_label_source(conf):
        if conf >= HIGH_CONF_THRESHOLD:
            return "semantic"
        elif conf >= LOW_CONF_THRESHOLD:
            return "llm_classify"
        else:
            return "unclassified"

    sessions_df["label_source"] = sessions_df["combined_confidence"].apply(
        get_label_source
    )

    counts = sessions_df["label_source"].value_counts().to_dict()
    log.info(f"Label source distribution: {counts}")
    return sessions_df


# ─────────────────────────────────────────────
# STEP 4 — LLM Fallback (SKIPPED)
# ─────────────────────────────────────────────

def llm_fallback(sessions_df: pd.DataFrame) -> pd.DataFrame:
    """
    LLM fallback classification for low-confidence sessions.
    SKIPPED for now — sessions remain labeled as 'llm_classify'.

    When implemented:
        - Truncate combined_prompt to LLM_PROMPT_TRUNCATE characters
        - Call SLM/LLM with taxonomy definitions
        - Parse JSON response: domain, intent, confidence, reason
        - Apply BUDGET_CAP guard
        - Update label_source to "llm"
    """
    log.info("Step 4 (LLM Fallback) skipped — llm_classify sessions saved for later")
    return sessions_df


# ─────────────────────────────────────────────
# STEP 5 — Governance Flagging
# ─────────────────────────────────────────────

def check_watchlist(prompt: str) -> tuple[bool, str]:
    """Check prompt against suspicious keyword watchlist."""
    prompt_lower = prompt.lower()
    for keyword in SUSPICIOUS_WATCHLIST:
        if keyword.lower() in prompt_lower:
            return True, f"suspicious keyword detected: '{keyword}'"
    return False, ""


def apply_governance_flags(
    sessions_df: pd.DataFrame,
    governance_matrix: dict
) -> pd.DataFrame:
    """
    Apply governance flags based on:
        1. Domain + Intent combination matrix
        2. Suspicious keyword watchlist (anomaly layer)

    Only flags sessions that have a definitive label (semantic or llm).
    llm_classify and unclassified sessions get flagged by watchlist only.
    """
    log.info("Applying governance flags...")

    governance_flags   = []
    governance_reasons = []

    for _, row in sessions_df.iterrows():
        flag   = "NONE"
        reason = ""

        # Matrix-based flagging (only for labeled sessions)
        if row["label_source"] == "semantic":
            pair = (row["predicted_domain"], row["predicted_intent"])
            if pair in governance_matrix:
                flag   = governance_matrix[pair]["level"]
                reason = governance_matrix[pair]["reason"]

        # Watchlist override — escalates to HIGH regardless of label source
        watchlist_hit, watchlist_reason = check_watchlist(row["combined_prompt"])
        if watchlist_hit:
            flag   = "HIGH"
            reason = (reason + " | " + watchlist_reason).strip(" | ")

        governance_flags.append(flag)
        governance_reasons.append(reason)

    sessions_df["governance_flag"]   = governance_flags
    sessions_df["governance_reason"] = governance_reasons

    flag_counts = sessions_df["governance_flag"].value_counts().to_dict()
    log.info(f"Governance flag distribution: {flag_counts}")
    return sessions_df


# ─────────────────────────────────────────────
# STEP 6 — Output & Storage
# ─────────────────────────────────────────────

def save_outputs(sessions_df: pd.DataFrame, output_dir: str) -> None:
    """
    Save pipeline outputs to separate files:
        - classification_logs.csv     → all sessions with labels
        - governance_alerts.csv       → HIGH flag sessions
        - governance_monitoring.csv   → MEDIUM flag sessions
        - llm_classify_queue.csv      → sessions needing LLM classification
        - unclassified.csv            → very low confidence sessions
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Columns for final output record
    output_columns = [
        "session_id", "username", "co", "user_type", "division",
        "combined_prompt", "turn_count",
        "predicted_domain", "domain_confidence",
        "predicted_intent", "intent_confidence",
        "combined_confidence", "label_source",
        "governance_flag", "governance_reason",
        "domain_prior", "first_timestamp", "last_timestamp"
    ]

    # 1 — Full classification log
    all_logs_path = output_path / f"classification_logs_{run_timestamp}.csv"
    sessions_df[output_columns].to_csv(all_logs_path, index=False)
    log.info(f"Saved {len(sessions_df)} records → {all_logs_path}")

    # 2 — Governance alerts (HIGH)
    high_risk = sessions_df[sessions_df["governance_flag"] == "HIGH"]
    if len(high_risk) > 0:
        alerts_path = output_path / f"governance_alerts_{run_timestamp}.csv"
        high_risk[output_columns].to_csv(alerts_path, index=False)
        log.info(f"Saved {len(high_risk)} HIGH governance alerts → {alerts_path}")

    # 3 — Governance monitoring (MEDIUM)
    medium_risk = sessions_df[sessions_df["governance_flag"] == "MEDIUM"]
    if len(medium_risk) > 0:
        monitoring_path = output_path / f"governance_monitoring_{run_timestamp}.csv"
        medium_risk[output_columns].to_csv(monitoring_path, index=False)
        log.info(f"Saved {len(medium_risk)} MEDIUM governance flags → {monitoring_path}")

    # 4 — LLM classify queue
    llm_queue = sessions_df[sessions_df["label_source"] == "llm_classify"]
    if len(llm_queue) > 0:
        llm_path = output_path / f"llm_classify_queue_{run_timestamp}.csv"
        llm_queue[output_columns].to_csv(llm_path, index=False)
        log.info(f"Saved {len(llm_queue)} sessions to LLM queue → {llm_path}")

    # 5 — Unclassified (very low confidence)
    unclassified = sessions_df[sessions_df["label_source"] == "unclassified"]
    if len(unclassified) > 0:
        unclassified_path = output_path / f"unclassified_{run_timestamp}.csv"
        unclassified[output_columns].to_csv(unclassified_path, index=False)
        log.info(f"Saved {len(unclassified)} unclassified sessions → {unclassified_path}")

    # Summary
    log.info("─" * 50)
    log.info("PIPELINE SUMMARY")
    log.info(f"  Total sessions:       {len(sessions_df)}")
    log.info(f"  Labeled (semantic):   {(sessions_df['label_source'] == 'semantic').sum()}")
    log.info(f"  LLM queue:            {(sessions_df['label_source'] == 'llm_classify').sum()}")
    log.info(f"  Unclassified:         {(sessions_df['label_source'] == 'unclassified').sum()}")
    log.info(f"  HIGH governance:      {(sessions_df['governance_flag'] == 'HIGH').sum()}")
    log.info(f"  MEDIUM governance:    {(sessions_df['governance_flag'] == 'MEDIUM').sum()}")
    log.info("─" * 50)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def run_pipeline(
    input_path: str,
    taxonomy_path: str,
    embedding_model_path: str,
    output_dir: str
) -> pd.DataFrame:
    """Run the full classification pipeline end to end."""

    log.info("=" * 60)
    log.info("NOMURA AI LOG CLASSIFICATION PIPELINE")
    log.info("=" * 60)

    # Step 0 — Setup
    log.info("[STEP 0] Loading taxonomy and embedding model...")
    taxonomy         = load_taxonomy(taxonomy_path)
    governance_matrix = load_governance_matrix(taxonomy)

    log.info(f"Loading embedding model from {embedding_model_path}")
    model            = SentenceTransformer(embedding_model_path)
    taxonomy_vectors = build_taxonomy_vectors(taxonomy, model)

    # Step 1 — Ingest & Preprocess
    log.info("[STEP 1] Ingesting and preprocessing logs...")
    raw_df      = load_logs(input_path)
    clean_df    = preprocess_logs(raw_df)
    sessions_df = group_sessions(clean_df)

    # Step 2 — Hybrid Semantic Scoring
    log.info("[STEP 2] Running hybrid semantic scoring...")
    sessions_df = run_semantic_scoring(sessions_df, model, taxonomy_vectors)

    # Step 3 — Threshold Decision
    log.info("[STEP 3] Applying confidence thresholds...")
    sessions_df = apply_threshold(sessions_df)

    # Step 4 — LLM Fallback (skipped)
    log.info("[STEP 4] LLM fallback (skipped)...")
    sessions_df = llm_fallback(sessions_df)

    # Step 5 — Governance Flagging
    log.info("[STEP 5] Applying governance flags...")
    sessions_df = apply_governance_flags(sessions_df, governance_matrix)

    # Step 6 — Save Outputs
    log.info("[STEP 6] Saving outputs...")
    save_outputs(sessions_df, output_dir)

    log.info("Pipeline complete.")
    return sessions_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Nomura AI Log Classification Pipeline"
    )
    parser.add_argument(
        "--input_path",
        required=True,
        help="Path to input Excel file with log data"
    )
    parser.add_argument(
        "--taxonomy_path",
        required=True,
        help="Path to nomura_taxonomy.json"
    )
    parser.add_argument(
        "--embedding_model_path",
        required=True,
        help="Local path to sentence-transformers embedding model"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write output CSV files"
    )
    parser.add_argument(
        "--high_conf_threshold",
        type=float,
        default=HIGH_CONF_THRESHOLD,
        help=f"High confidence threshold (default: {HIGH_CONF_THRESHOLD})"
    )
    parser.add_argument(
        "--low_conf_threshold",
        type=float,
        default=LOW_CONF_THRESHOLD,
        help=f"Low confidence threshold (default: {LOW_CONF_THRESHOLD})"
    )

    args = parser.parse_args()

    # Override thresholds if passed
    HIGH_CONF_THRESHOLD = args.high_conf_threshold
    LOW_CONF_THRESHOLD  = args.low_conf_threshold

    run_pipeline(
        input_path=args.input_path,
        taxonomy_path=args.taxonomy_path,
        embedding_model_path=args.embedding_model_path,
        output_dir=args.output_dir
    )
