"""
Pair generator: creates labeled (S1, candidate) pairs for training.

Builds a lookup dict from DataFrames and generates positive/negative
pairs based on blocking candidates and ground truth.
"""

import logging
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from src.features import FEATURE_NAMES, compute_features_batch

logger = logging.getLogger(__name__)


def build_record_lookup(df: pd.DataFrame) -> Dict[str, dict]:
    """Convert a DataFrame to a dict of {entity_id: {field: value}}.

    This is the fastest lookup structure for feature computation.
    Uses positional indexing for robustness against special column names.
    """
    records = {}
    columns = list(df.columns)
    eid_idx = columns.index("entity_id")
    for row in df.itertuples(index=False):
        eid = row[eid_idx]
        records[eid] = {col: row[i] for i, col in enumerate(columns)}
    return records


def generate_pairs_and_labels(
    candidates: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
    s1_lookup: Dict[str, dict],
    s23_lookup: Dict[str, dict],
    max_neg_per_entity: int = 50,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]:
    """Generate feature matrix and labels from candidates + ground truth.

    For each S1 entity:
      - All true matches found in candidates → positive pairs
      - Up to max_neg_per_entity non-matches from candidates → negative pairs
      - Singletons with candidates → all negative pairs

    Parameters
    ----------
    candidates : dict
        ``{s1_id: set_of_candidate_ids}``
    ground_truth : dict
        ``{s1_id: set_of_true_match_ids}``
    s1_lookup : dict
        Record lookup for S1 entities.
    s23_lookup : dict
        Record lookup for S2/S3 entities.
    max_neg_per_entity : int
        Maximum negative samples per S1 entity (to control class imbalance).
    seed : int
        Random seed for negative sampling.

    Returns
    -------
    X : np.ndarray of shape (n_pairs, n_features)
    y : np.ndarray of shape (n_pairs,) with 0/1 labels
    pair_ids : list of (s1_id, cand_id) tuples
    """
    rng = np.random.RandomState(seed)

    all_pairs = []
    all_labels = []

    for s1_id, cand_ids in candidates.items():
        if not cand_ids:
            continue

        true_ids = ground_truth.get(s1_id, set())

        # Filter candidates to only those we have records for
        valid_cands = [c for c in cand_ids if c in s23_lookup]

        # Positives
        positives = [c for c in valid_cands if c in true_ids]
        for cand_id in positives:
            all_pairs.append((s1_id, cand_id))
            all_labels.append(1)

        # Negatives
        negatives = [c for c in valid_cands if c not in true_ids]
        if len(negatives) > max_neg_per_entity:
            negatives = list(rng.choice(negatives, max_neg_per_entity, replace=False))
        for cand_id in negatives:
            all_pairs.append((s1_id, cand_id))
            all_labels.append(0)

    logger.info(
        "Generated %s pairs: %s positive, %s negative (%.2f%% positive)",
        f"{len(all_pairs):,}",
        f"{sum(all_labels):,}",
        f"{len(all_labels) - sum(all_labels):,}",
        sum(all_labels) / len(all_labels) * 100 if all_labels else 0,
    )

    # Compute features in batch
    logger.info("Computing features for %s pairs …", f"{len(all_pairs):,}")
    X = compute_features_batch(all_pairs, s1_lookup, s23_lookup)
    y = np.array(all_labels, dtype=np.int32)

    return X, y, all_pairs


def generate_pairs_for_prediction(
    candidates: Dict[str, Set[str]],
    s1_lookup: Dict[str, dict],
    s23_lookup: Dict[str, dict],
) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    """Generate feature matrix for prediction (no labels).

    Returns
    -------
    X : np.ndarray of shape (n_pairs, n_features)
    pair_ids : list of (s1_id, cand_id) tuples
    """
    all_pairs = []

    for s1_id, cand_ids in candidates.items():
        valid_cands = [c for c in cand_ids if c in s23_lookup]
        for cand_id in valid_cands:
            all_pairs.append((s1_id, cand_id))

    logger.info("Generating features for %s prediction pairs …", f"{len(all_pairs):,}")

    if not all_pairs:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32), all_pairs

    X = compute_features_batch(all_pairs, s1_lookup, s23_lookup)
    return X, all_pairs
