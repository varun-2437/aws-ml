"""
Memory-efficient data loading for the Entity Resolution pipeline.

Handles:
  - TSV reading with dtype=str to prevent ID coercion
  - Chunked loading for large files
  - Parquet caching for fast reloads
  - Ground-truth parsing into a dict mapping
"""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

from src.config import (
    CACHE_DIR,
    CHUNK_SIZE,
    GT_COLUMNS,
    SOURCE_COLUMNS,
    SOURCES,
)

logger = logging.getLogger(__name__)


# ─── TSV → DataFrame ─────────────────────────────────────────────────────────

def load_source_tsv(
    path: Path,
    *,
    use_cache: bool = True,
    chunk_size: int = CHUNK_SIZE,
) -> pd.DataFrame:
    """Load a source TSV (S1/S2/S3) into a DataFrame.

    Uses Parquet caching so the first load is slow but subsequent loads
    are ~10x faster.  Every column is read as ``str`` to prevent pandas
    from coercing entity IDs or addresses.
    """
    cache_path = CACHE_DIR / f"{path.stem}.parquet"

    if use_cache and cache_path.exists():
        logger.info("Loading cached %s", cache_path.name)
        df = pd.read_parquet(cache_path)
        logger.info("  → %s rows loaded from cache", f"{len(df):,}")
        return df

    logger.info("Reading %s (chunked, chunk_size=%s) …", path.name, f"{chunk_size:,}")
    chunks: list[pd.DataFrame] = []
    for i, chunk in enumerate(
        pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=False,   # treat empty strings as "", not NaN
            chunksize=chunk_size,
            encoding="utf-8",
        )
    ):
        chunks.append(chunk)
        if (i + 1) % 5 == 0:
            logger.info("  … read %s rows so far", f"{sum(len(c) for c in chunks):,}")

    df = pd.concat(chunks, ignore_index=True)
    logger.info("  → %s rows loaded from TSV", f"{len(df):,}")

    # Validate columns
    expected = SOURCE_COLUMNS
    if list(df.columns) != expected:
        logger.warning(
            "Column mismatch in %s: got %s, expected %s",
            path.name, list(df.columns), expected,
        )

    # Optimise memory: country as category
    if "country" in df.columns:
        df["country"] = df["country"].astype("category")

    # Cache to Parquet
    if use_cache:
        df.to_parquet(cache_path, index=False)
        logger.info("  → cached to %s", cache_path.name)

    return df


def load_source(split: str, source: str, **kwargs) -> pd.DataFrame:
    """Convenience wrapper: ``load_source("train", "s1")``."""
    path = SOURCES[split][source]
    return load_source_tsv(path, **kwargs)


# ─── Ground-truth loading ────────────────────────────────────────────────────

def load_ground_truth(
    path: Optional[Path] = None,
) -> Dict[str, List[str]]:
    """Parse ``train_ground_truth.tsv`` into a dict.

    Returns
    -------
    dict
        ``{source1_entity_id: [matched_id_1, matched_id_2, ...]}``
        Singletons map to an empty list.
    """
    if path is None:
        path = SOURCES["train"]["gt"]

    logger.info("Loading ground truth from %s …", path.name)
    gt: Dict[str, List[str]] = {}

    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        assert header == GT_COLUMNS, f"Unexpected header: {header}"

        for row in reader:
            s1_id = row[0]
            matched_raw = row[1].strip() if len(row) > 1 else ""
            if matched_raw:
                gt[s1_id] = matched_raw.split(",")
            else:
                gt[s1_id] = []

    n_singleton = sum(1 for v in gt.values() if len(v) == 0)
    n_with_match = sum(1 for v in gt.values() if len(v) > 0)
    logger.info(
        "  → %s S1 entities (%s singletons, %s with ≥1 match)",
        f"{len(gt):,}", f"{n_singleton:,}", f"{n_with_match:,}",
    )
    return gt


def get_ground_truth_sets(
    gt: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Set[str]]:
    """Return ground truth with values as sets (for fast lookup)."""
    if gt is None:
        gt = load_ground_truth()
    return {k: set(v) for k, v in gt.items()}


# ─── Quick stats ──────────────────────────────────────────────────────────────

def print_source_stats(df: pd.DataFrame, label: str = "") -> None:
    """Print basic statistics for a source DataFrame."""
    print(f"\n{'═' * 60}")
    print(f"  Source: {label}")
    print(f"{'═' * 60}")
    print(f"  Rows            : {len(df):,}")
    print(f"  Columns         : {list(df.columns)}")

    for col in ["business_name", "business_address", "country"]:
        if col in df.columns:
            missing = (df[col] == "").sum()
            print(f"  {col:18s}: {missing:,} missing ({missing / len(df) * 100:.2f}%)")

    if "country" in df.columns:
        print(f"  Country dist    :")
        for country, count in df["country"].value_counts().items():
            print(f"    {country:12s}: {count:>10,} ({count / len(df) * 100:.1f}%)")

    # Devanagari detection (sample first 100k for speed)
    if "business_name" in df.columns:
        sample = df["business_name"].head(100_000)
        devanagari_count = sample.str.contains(r"[\u0900-\u097F]", regex=True).sum()
        pct = devanagari_count / len(sample) * 100
        print(f"  Devanagari names: ~{pct:.1f}% (sampled first 100k)")

    print(f"{'═' * 60}\n")


# ─── Entrypoint for quick testing ─────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load all training sources and print stats
    for source_key in ["s1", "s2", "s3"]:
        df = load_source("train", source_key)
        print_source_stats(df, label=f"Train {source_key.upper()}")
        del df  # free memory

    # Load and verify ground truth
    gt = load_ground_truth()
    gt_sets = get_ground_truth_sets(gt)

    # Quick distribution
    from collections import Counter
    match_counts = Counter(len(v) for v in gt.values())
    print("\nGround-truth match-count distribution:")
    for k in sorted(match_counts):
        print(f"  {k} matches: {match_counts[k]:,}")
