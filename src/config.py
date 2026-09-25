"""
Central configuration for the Business Entity Resolution pipeline.

All paths, constants, and hyperparameters live here so every other module
imports from one place.
"""


from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
# Resolve relative to this file so the config works regardless of cwd.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Directory layout ──────────────────────────────────────────────────────────
TRAIN_DIR   = PROJECT_ROOT / "dataset" / "train"
TEST_DIR    = PROJECT_ROOT / "dataset" / "test"
OUTPUT_DIR  = PROJECT_ROOT / "output"
CACHE_DIR   = PROJECT_ROOT / "cache"          # Parquet caches go here

# Create dirs that may not exist yet
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Source file paths ─────────────────────────────────────────────────────────
SOURCES = {
    "train": {
        "s1": TRAIN_DIR / "train_source1.tsv",
        "s2": TRAIN_DIR / "train_source2.tsv",
        "s3": TRAIN_DIR / "train_source3.tsv",
        "gt": TRAIN_DIR / "train_ground_truth.tsv",
    },
    "test": {
        "s1": TEST_DIR / "test_source1.tsv",
        "s2": TEST_DIR / "test_source2.tsv",
        "s3": TEST_DIR / "test_source3.tsv",
    },
}

# ── Column names ──────────────────────────────────────────────────────────────
SOURCE_COLUMNS    = ["entity_id", "business_name", "business_address", "country"]
GT_COLUMNS        = ["source1_entity_id", "matched_entity_ids"]
MATCHING_COLUMNS  = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_COLUMNS = ["source1_entity_id", "candidate_entity_ids"]

# ── Data loading ──────────────────────────────────────────────────────────────
CHUNK_SIZE = 500_000       # rows per chunk when reading TSVs
RANDOM_SEED = 42

# ── Blocking hyperparameters ──────────────────────────────────────────────────
TFIDF_TOP_K        = 50            # top-K nearest neighbours per S1 entity
TFIDF_NGRAM_RANGE  = (3, 4)        # character n-gram range
TFIDF_ANALYZER     = "char_wb"     # character n-grams respecting word boundaries
MIN_TOKEN_OVERLAP  = 2             # inverted-index blocking threshold
MAX_CANDIDATES     = 100           # safety cap per S1 entity after union

# ── Model hyperparameters ─────────────────────────────────────────────────────
VAL_FRACTION = 0.20                # fraction of S1 entities held out for validation

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "n_jobs": -1,
    "seed": RANDOM_SEED,
}

# ── Threshold sweep ──────────────────────────────────────────────────────────
THRESHOLD_MIN  = 0.05
THRESHOLD_MAX  = 0.96
THRESHOLD_STEP = 0.01

# ── Output files ──────────────────────────────────────────────────────────────
MATCHING_OUTPUT  = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_OUTPUT = OUTPUT_DIR / "candidate_pairs.tsv"
