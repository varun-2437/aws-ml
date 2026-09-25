"""
Pairwise similarity features for the Entity Resolution pipeline.

For each (S1, candidate) pair, computes a feature vector describing
how similar the two records are across name, address, and meta signals.
"""

import logging
from typing import List, Optional, Set

import numpy as np
import Levenshtein
import jellyfish

from src.normalizer import (
    char_ngrams,
    extract_numbers,
    extract_pin_zip,
    is_devanagari,
    tokenize,
)

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Primitive similarity functions
# ═════════════════════════════════════════════════════════════════════════════

def _safe_ratio(a: str, b: str) -> float:
    """Normalized Levenshtein similarity (0–1, higher is more similar)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return Levenshtein.ratio(a, b)


def _jaro_winkler(a: str, b: str) -> float:
    """Jaro-Winkler similarity (0–1)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return jellyfish.jaro_winkler_similarity(a, b)


def _jaccard(set_a: Set[str], set_b: Set[str]) -> float:
    """Jaccard similarity on two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _containment(short: str, long: str) -> float:
    """Whether the shorter string is contained in the longer one.
    Returns 1.0 if yes, 0.0 if no."""
    if not short or not long:
        return 0.0
    s, l_ = (short, long) if len(short) <= len(long) else (long, short)
    return 1.0 if s in l_ else 0.0


def _len_ratio(a: str, b: str) -> float:
    """Ratio of lengths: min/max. Returns 1.0 for equal length, 0 for empty."""
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    if la == 0 or lb == 0:
        return 0.0
    return min(la, lb) / max(la, lb)


def _common_prefix_len(a: str, b: str) -> int:
    """Length of the longest common prefix."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def _sorted_token_sim(a: str, b: str) -> float:
    """Levenshtein similarity on alphabetically sorted tokens.
    Handles word-order differences."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    sorted_a = " ".join(sorted(a.split()))
    sorted_b = " ".join(sorted(b.split()))
    return Levenshtein.ratio(sorted_a, sorted_b)


# ═════════════════════════════════════════════════════════════════════════════
# Feature vector computation for a single pair
# ═════════════════════════════════════════════════════════════════════════════

# Feature names (in order) — used by the model
FEATURE_NAMES = [
    # Name features
    "name_levenshtein",
    "name_jaro_winkler",
    "name_jaccard_token",
    "name_jaccard_char3",
    "name_containment",
    "name_sorted_token_sim",
    "name_len_ratio",
    "name_common_prefix_ratio",
    "name_first_word_match",
    "name_legal_suffix_match",
    "name_token_count_diff",
    # Address features
    "addr_levenshtein",
    "addr_jaccard_token",
    "addr_number_match",
    "addr_pin_match",
    "addr_len_ratio",
    "addr_missing",
    # Meta features
    "source_is_s3",
    "is_devanagari_mismatch",
    "combined_score",
    "name_addr_agreement",
]


def compute_pair_features(
    s1_norm_name: str,
    s1_norm_address: str,
    s1_legal_suffix: str,
    s1_business_name: str,
    s1_business_address: str,
    cand_norm_name: str,
    cand_norm_address: str,
    cand_legal_suffix: str,
    cand_business_name: str,
    cand_business_address: str,
    cand_entity_id: str,
) -> np.ndarray:
    """Compute the feature vector for a single (S1, candidate) pair.

    Returns a 1D numpy array of shape (len(FEATURE_NAMES),).
    """
    features = np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    # ── Name features ─────────────────────────────────────────────────────
    features[0] = _safe_ratio(s1_norm_name, cand_norm_name)           # name_levenshtein
    features[1] = _jaro_winkler(s1_norm_name, cand_norm_name)         # name_jaro_winkler

    s1_name_tokens = tokenize(s1_norm_name)
    cand_name_tokens = tokenize(cand_norm_name)
    features[2] = _jaccard(s1_name_tokens, cand_name_tokens)          # name_jaccard_token

    s1_char3 = char_ngrams(s1_norm_name, 3)
    cand_char3 = char_ngrams(cand_norm_name, 3)
    features[3] = _jaccard(s1_char3, cand_char3)                      # name_jaccard_char3

    features[4] = _containment(s1_norm_name, cand_norm_name)          # name_containment
    features[5] = _sorted_token_sim(s1_norm_name, cand_norm_name)     # name_sorted_token_sim
    features[6] = _len_ratio(s1_norm_name, cand_norm_name)            # name_len_ratio

    # Common prefix ratio — normalized to [0, 1]
    max_len = max(len(s1_norm_name), len(cand_norm_name), 1)
    features[7] = _common_prefix_len(s1_norm_name, cand_norm_name) / max_len  # name_common_prefix_ratio

    # First word match
    s1_first = s1_norm_name.split()[0] if s1_norm_name else ""
    cand_first = cand_norm_name.split()[0] if cand_norm_name else ""
    features[8] = 1.0 if s1_first and s1_first == cand_first else 0.0  # name_first_word_match

    # Legal suffix match
    features[9] = 1.0 if s1_legal_suffix == cand_legal_suffix and s1_legal_suffix else 0.0

    # Token count difference (normalized)
    n1 = len(s1_name_tokens)
    n2 = len(cand_name_tokens)
    features[10] = abs(n1 - n2) / max(n1, n2, 1)                     # name_token_count_diff

    # ── Address features ──────────────────────────────────────────────────
    features[11] = _safe_ratio(s1_norm_address, cand_norm_address)    # addr_levenshtein

    s1_addr_tokens = tokenize(s1_norm_address)
    cand_addr_tokens = tokenize(cand_norm_address)
    features[12] = _jaccard(s1_addr_tokens, cand_addr_tokens)         # addr_jaccard_token

    # Address number match
    s1_nums = set(extract_numbers(s1_business_address))
    cand_nums = set(extract_numbers(cand_business_address))
    if s1_nums and cand_nums:
        features[13] = len(s1_nums & cand_nums) / max(len(s1_nums), len(cand_nums))
    else:
        features[13] = 0.0                                           # addr_number_match

    # PIN/ZIP match
    s1_pin = extract_pin_zip(s1_business_address)
    cand_pin = extract_pin_zip(cand_business_address)
    features[14] = 1.0 if s1_pin and cand_pin and s1_pin == cand_pin else 0.0  # addr_pin_match

    features[15] = _len_ratio(s1_norm_address, cand_norm_address)     # addr_len_ratio

    # Either address missing
    features[16] = 1.0 if not s1_norm_address or not cand_norm_address else 0.0  # addr_missing

    # ── Meta features ─────────────────────────────────────────────────────
    features[17] = 1.0 if cand_entity_id.startswith("S3-") else 0.0  # source_is_s3

    # Devanagari mismatch
    s1_dev = is_devanagari(s1_business_name)
    cand_dev = is_devanagari(cand_business_name)
    features[18] = 1.0 if s1_dev != cand_dev else 0.0                # is_devanagari_mismatch

    # Combined score — use Jaro-Winkler (more discriminative) for name
    name_sim = features[1]  # Jaro-Winkler
    addr_sim = features[11]  # address Levenshtein
    features[19] = 0.7 * name_sim + 0.3 * addr_sim                   # combined_score

    # Name-address agreement: both > 0.5
    features[20] = 1.0 if name_sim > 0.5 and addr_sim > 0.5 else 0.0  # name_addr_agreement

    return features


# ═════════════════════════════════════════════════════════════════════════════
# Batch feature computation
# ═════════════════════════════════════════════════════════════════════════════

def compute_features_batch(
    pairs: list,
    s1_lookup: dict,
    s23_lookup: dict,
) -> np.ndarray:
    """Compute features for a batch of pairs.

    Parameters
    ----------
    pairs : list of (s1_id, cand_id)
    s1_lookup : dict
        ``{entity_id: {field: value}}`` for S1 records.
    s23_lookup : dict
        ``{entity_id: {field: value}}`` for S2/S3 records.

    Returns
    -------
    np.ndarray of shape (len(pairs), n_features)
    """
    n = len(pairs)
    n_features = len(FEATURE_NAMES)
    X = np.zeros((n, n_features), dtype=np.float32)

    for i, (s1_id, cand_id) in enumerate(pairs):
        s1 = s1_lookup[s1_id]
        cand = s23_lookup[cand_id]

        X[i] = compute_pair_features(
            s1_norm_name=s1.get("norm_name", ""),
            s1_norm_address=s1.get("norm_address", ""),
            s1_legal_suffix=s1.get("legal_suffix_type", ""),
            s1_business_name=s1.get("business_name", ""),
            s1_business_address=s1.get("business_address", ""),
            cand_norm_name=cand.get("norm_name", ""),
            cand_norm_address=cand.get("norm_address", ""),
            cand_legal_suffix=cand.get("legal_suffix_type", ""),
            cand_business_name=cand.get("business_name", ""),
            cand_business_address=cand.get("business_address", ""),
            cand_entity_id=cand_id,
        )

        if (i + 1) % 1_000_000 == 0:
            logger.info("  … computed features for %s pairs", f"{i + 1:,}")

    return X
