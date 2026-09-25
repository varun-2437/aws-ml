"""
Candidate generation orchestrator.

Ties together data loading, normalization, and blocking into a single
workflow that processes data by country partition to manage memory.
"""

import gc
import logging
from typing import Dict, Set

import pandas as pd

from src.blocker import (
    generate_candidates_for_country,
    measure_blocking_recall,
)
from src.config import CACHE_DIR
from src.data_loader import (
    get_ground_truth_sets,
    load_ground_truth,
    load_source,
)
from src.normalizer import normalize_dataframe

logger = logging.getLogger(__name__)


def _load_and_normalize(split: str, source: str) -> pd.DataFrame:
    """Load a source TSV and apply normalization."""
    logger.info("Loading and normalizing %s/%s …", split, source)
    df = load_source(split, source)
    df = normalize_dataframe(df)
    return df


def generate_all_candidates(
    split: str = "train",
    use_address_blocking: bool = False,
    save_cache: bool = True,
) -> Dict[str, Set[str]]:
    """Run candidate generation for all countries in a split.

    Processes one country at a time to limit memory usage.

    Parameters
    ----------
    split : str
        "train" or "test".
    use_address_blocking : bool
        Whether to run address-number blocking (slow on large data).
    save_cache : bool
        Whether to cache normalized DataFrames and candidates.

    Returns
    -------
    dict
        ``{s1_id: set_of_candidate_ids}`` for ALL S1 entities across all countries.
    """
    # Load S1
    s1 = _load_and_normalize(split, "s1")

    # Load S2 + S3 and concatenate
    s2 = _load_and_normalize(split, "s2")
    s3 = _load_and_normalize(split, "s3")
    s23 = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    gc.collect()

    # Cache normalized data
    if save_cache:
        s1.to_parquet(CACHE_DIR / f"{split}_s1_normalized.parquet", index=False)
        s23.to_parquet(CACHE_DIR / f"{split}_s23_normalized.parquet", index=False)
        logger.info("Cached normalized DataFrames to %s", CACHE_DIR)

    # Get unique countries
    countries = sorted(s1["country"].unique())
    logger.info("Countries in %s: %s", split, countries)

    all_candidates: Dict[str, Set[str]] = {}

    for country in countries:
        s1_country = s1[s1["country"] == country].reset_index(drop=True)
        s23_country = s23[s23["country"] == country].reset_index(drop=True)

        country_cands = generate_candidates_for_country(
            s1_df=s1_country,
            s23_df=s23_country,
            country=country,
            use_address_blocking=use_address_blocking,
        )

        all_candidates.update(country_cands)

        # Free country-specific data
        del s1_country, s23_country
        gc.collect()

    # Ensure every S1 entity has an entry
    for eid in s1["entity_id"]:
        all_candidates.setdefault(eid, set())

    logger.info(
        "Total candidates: %s S1 entities, avg %.1f candidates/entity",
        f"{len(all_candidates):,}",
        sum(len(v) for v in all_candidates.values()) / len(all_candidates),
    )

    return all_candidates


def run_train_candidate_generation() -> dict:
    """Full training candidate generation with blocking recall measurement.

    Returns
    -------
    dict with keys:
        candidates: {s1_id: set_of_candidate_ids}
        blocking_recall: dict of recall metrics
    """
    candidates = generate_all_candidates(split="train", use_address_blocking=False)

    # Measure blocking recall
    gt = get_ground_truth_sets()
    recall_metrics = measure_blocking_recall(candidates, gt)

    return {
        "candidates": candidates,
        "blocking_recall": recall_metrics,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )

    result = run_train_candidate_generation()

    print("\n" + "═" * 60)
    print("  BLOCKING RECALL RESULTS")
    print("═" * 60)
    for k, v in result["blocking_recall"].items():
        if isinstance(v, float):
            print(f"  {k:20s}: {v:.4f}")
        else:
            print(f"  {k:20s}: {v:,}" if isinstance(v, int) else f"  {k:20s}: {v}")
    print("═" * 60)
