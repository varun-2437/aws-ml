"""
End-to-end prediction pipeline.

Runs the full pipeline on test data:
  1. Load + normalize
  2. Generate candidates (blocking)
  3. Compute pairwise features
  4. Run trained model
  5. Apply threshold
  6. Write output files
"""

import gc
import logging
from pathlib import Path
from typing import Dict, Set

import numpy as np
import pandas as pd

from src.blocker import generate_candidates_for_country
from src.candidate_generator import _load_and_normalize
from src.config import (
    CACHE_DIR,
    CANDIDATE_OUTPUT,
    MATCHING_OUTPUT,
)
from src.data_loader import load_source
from src.features import FEATURE_NAMES
from src.model import load_model, predictions_to_entity_matches
from src.normalizer import normalize_dataframe
from src.pair_generator import build_record_lookup, generate_pairs_for_prediction

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Output writers
# ═════════════════════════════════════════════════════════════════════════════

def write_matching_results(
    all_s1_ids: list,
    matches: Dict[str, Set[str]],
    path: Path = MATCHING_OUTPUT,
):
    """Write matching_results.tsv with one row per S1 entity."""
    logger.info("Writing %s …", path)
    with open(path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in all_s1_ids:
            matched = matches.get(s1_id, set())
            matched_str = ",".join(sorted(matched)) if matched else ""
            f.write(f"{s1_id}\t{matched_str}\n")
    logger.info("  → %s rows written", f"{len(all_s1_ids):,}")


def write_candidate_pairs(
    all_s1_ids: list,
    candidates: Dict[str, Set[str]],
    path: Path = CANDIDATE_OUTPUT,
):
    """Write candidate_pairs.tsv with one row per S1 entity."""
    logger.info("Writing %s …", path)
    with open(path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in all_s1_ids:
            cands = candidates.get(s1_id, set())
            cands_str = ",".join(sorted(cands)) if cands else ""
            f.write(f"{s1_id}\t{cands_str}\n")
    logger.info("  → %s rows written", f"{len(all_s1_ids):,}")


# ═════════════════════════════════════════════════════════════════════════════
# Full prediction pipeline
# ═════════════════════════════════════════════════════════════════════════════

def run_test_prediction(model_path: Path = None):
    """Run the complete prediction pipeline on test data.

    Steps:
      1. Load and normalize all test sources
      2. Generate blocking candidates by country
      3. Compute pairwise features
      4. Predict with trained model
      5. Apply optimized threshold
      6. Write both output files
    """
    # Load model and threshold
    model, threshold = load_model(model_path)
    logger.info("Using threshold: %.3f", threshold)

    # Load and normalize test data
    s1 = _load_and_normalize("test", "s1")
    s2 = _load_and_normalize("test", "s2")
    s3 = _load_and_normalize("test", "s3")
    s23 = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    gc.collect()

    all_s1_ids = list(s1["entity_id"])
    countries = sorted(s1["country"].unique())
    logger.info("Test countries: %s", countries)
    logger.info("Test S1 entities: %s", f"{len(all_s1_ids):,}")

    # Build lookups
    s1_lookup = build_record_lookup(s1)
    s23_lookup = build_record_lookup(s23)

    all_candidates: Dict[str, Set[str]] = {}
    all_matches: Dict[str, Set[str]] = {}

    for country in countries:
        logger.info("\n═══ Processing %s ═══", country)
        s1_country = s1[s1["country"] == country].reset_index(drop=True)
        s23_country = s23[s23["country"] == country].reset_index(drop=True)

        # Generate candidates
        country_cands = generate_candidates_for_country(
            s1_df=s1_country,
            s23_df=s23_country,
            country=country,
            use_address_blocking=False,
        )
        all_candidates.update(country_cands)

        # Compute features
        X, pair_ids = generate_pairs_for_prediction(
            candidates=country_cands,
            s1_lookup=s1_lookup,
            s23_lookup=s23_lookup,
        )

        if len(pair_ids) > 0:
            # Predict
            probabilities = model.predict(X)

            # Apply threshold
            country_matches = predictions_to_entity_matches(
                pair_ids, probabilities, threshold
            )
            all_matches.update(country_matches)

        del s1_country, s23_country, X
        gc.collect()

    # Ensure every S1 has entries
    for s1_id in all_s1_ids:
        all_candidates.setdefault(s1_id, set())
        all_matches.setdefault(s1_id, set())

    # Ensure matches are subset of candidates
    for s1_id in all_s1_ids:
        all_candidates[s1_id] |= all_matches.get(s1_id, set())

    # Write output files
    write_matching_results(all_s1_ids, all_matches)
    write_candidate_pairs(all_s1_ids, all_candidates)

    # Summary
    n_matched = sum(1 for v in all_matches.values() if v)
    n_singleton = sum(1 for v in all_matches.values() if not v)
    total_matches = sum(len(v) for v in all_matches.values())
    logger.info(
        "\nPrediction summary: %s matched, %s singletons, %s total match links",
        f"{n_matched:,}", f"{n_singleton:,}", f"{total_matches:,}",
    )


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    run_test_prediction()
