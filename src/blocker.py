"""
Blocking / Candidate Generation for the Entity Resolution pipeline.

Implements multiple blocking strategies and unions their results to
achieve high recall while keeping the candidate set manageable.

Strategies:
  1. Country blocking (hard filter — always applied first)
  2. TF-IDF character n-gram cosine similarity (top-K per S1)
  3. Token-based inverted index (≥ N shared name tokens)
  4. Address number + city token blocking

All strategies are run within a single country partition to manage memory.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Set

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from src.config import (
    MAX_CANDIDATES,
    MIN_TOKEN_OVERLAP,
    TFIDF_ANALYZER,
    TFIDF_NGRAM_RANGE,
    TFIDF_TOP_K,
)
from src.normalizer import tokenize

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Strategy 1: TF-IDF character n-gram blocking
# ═════════════════════════════════════════════════════════════════════════════

def tfidf_blocking(
    s1_names: pd.Series,
    s1_ids: pd.Series,
    s23_names: pd.Series,
    s23_ids: pd.Series,
    top_k: int = TFIDF_TOP_K,
    ngram_range: tuple = TFIDF_NGRAM_RANGE,
    analyzer: str = TFIDF_ANALYZER,
) -> Dict[str, Set[str]]:
    """Find top-K most similar S2/S3 candidates per S1 entity using TF-IDF.

    Parameters
    ----------
    s1_names : pd.Series
        Normalized business names for S1 entities.
    s1_ids : pd.Series
        Entity IDs corresponding to s1_names.
    s23_names : pd.Series
        Normalized business names for S2+S3 entities.
    s23_ids : pd.Series
        Entity IDs corresponding to s23_names.
    top_k : int
        Number of nearest neighbours per S1 entity.

    Returns
    -------
    dict
        ``{s1_id: set_of_candidate_s23_ids}``
    """
    logger.info(
        "TF-IDF blocking: %s S1 × %s S2/S3, top_k=%d, ngrams=%s",
        f"{len(s1_names):,}", f"{len(s23_names):,}", top_k, ngram_range,
    )

    # Build TF-IDF on S2/S3 names
    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        max_features=500_000,      # cap vocabulary for memory
        sublinear_tf=True,
        dtype=np.float32,
    )

    # Fit on S2/S3, transform both
    tfidf_s23 = vectorizer.fit_transform(s23_names)
    tfidf_s1 = vectorizer.transform(s1_names)

    logger.info(
        "  TF-IDF matrix: S1 %s, S2/S3 %s, vocab %s",
        tfidf_s1.shape, tfidf_s23.shape, len(vectorizer.vocabulary_),
    )

    # Sparse top-K dot product: for each S1 row, find the top-K S2/S3 columns
    # sp_matmul_topn computes tfidf_s1 @ tfidf_s23.T and keeps only top-K per row
    similarity = sp_matmul_topn(
        tfidf_s1,
        tfidf_s23.T,
        top_n=top_k,
        threshold=0.0,   # keep everything in top-K regardless of score
        n_threads=-1,
    )

    # Convert sparse result to candidate dict
    candidates: Dict[str, Set[str]] = {}
    coo = similarity.tocoo()

    # Group by S1 row
    row_to_cols = defaultdict(set)
    for row, col in zip(coo.row, coo.col):
        row_to_cols[row].add(col)

    s1_id_arr = s1_ids.values
    s23_id_arr = s23_ids.values

    for row_idx, col_indices in row_to_cols.items():
        s1_id = s1_id_arr[row_idx]
        candidates[s1_id] = {s23_id_arr[c] for c in col_indices}

    # Ensure every S1 has an entry (possibly empty)
    for s1_id in s1_id_arr:
        candidates.setdefault(s1_id, set())

    logger.info(
        "  TF-IDF blocking: avg %.1f candidates/entity",
        np.mean([len(v) for v in candidates.values()]),
    )

    return candidates


# ═════════════════════════════════════════════════════════════════════════════
# Strategy 2: Token-based inverted index blocking
# ═════════════════════════════════════════════════════════════════════════════

def token_blocking(
    s1_tokens_series: pd.Series,
    s1_ids: pd.Series,
    s23_tokens_series: pd.Series,
    s23_ids: pd.Series,
    min_overlap: int = MIN_TOKEN_OVERLAP,
) -> Dict[str, Set[str]]:
    """Find candidates sharing ≥ min_overlap name tokens with each S1 entity.

    Parameters
    ----------
    s1_tokens_series : pd.Series
        Space-joined sorted tokens for S1 entities (from normalizer).
    s1_ids : pd.Series
        Entity IDs corresponding to s1_tokens_series.
    s23_tokens_series : pd.Series
        Space-joined sorted tokens for S2+S3 entities.
    s23_ids : pd.Series
        Entity IDs corresponding to s23_tokens_series.
    min_overlap : int
        Minimum shared tokens to qualify as a candidate.

    Returns
    -------
    dict
        ``{s1_id: set_of_candidate_s23_ids}``
    """
    logger.info(
        "Token blocking: %s S1 × %s S2/S3, min_overlap=%d",
        f"{len(s1_ids):,}", f"{len(s23_ids):,}", min_overlap,
    )

    # Build inverted index: token → set of S2/S3 indices
    inverted: Dict[str, List[int]] = defaultdict(list)
    for idx, token_str in enumerate(s23_tokens_series):
        if not token_str:
            continue
        for token in token_str.split():
            # Skip very common tokens (would create huge candidate sets)
            inverted[token].append(idx)

    # Filter out tokens that appear in > 5% of records (too common to be useful)
    max_freq = len(s23_ids) * 0.05
    inverted = {t: idxs for t, idxs in inverted.items() if len(idxs) <= max_freq}

    logger.info("  Inverted index: %s unique tokens (after filtering)", f"{len(inverted):,}")

    s23_id_arr = s23_ids.values
    candidates: Dict[str, Set[str]] = {}

    for s1_idx, (s1_id, token_str) in enumerate(zip(s1_ids, s1_tokens_series)):
        if not token_str:
            candidates[s1_id] = set()
            continue

        s1_tokens = set(token_str.split())

        # Count how many tokens each S2/S3 record shares with this S1
        overlap_count: Dict[int, int] = defaultdict(int)
        for token in s1_tokens:
            if token in inverted:
                for s23_idx in inverted[token]:
                    overlap_count[s23_idx] += 1

        # Keep those with >= min_overlap
        cands = {s23_id_arr[idx] for idx, count in overlap_count.items()
                 if count >= min_overlap}
        candidates[s1_id] = cands

        if (s1_idx + 1) % 500_000 == 0:
            logger.info("  … processed %s S1 entities", f"{s1_idx + 1:,}")

    logger.info(
        "  Token blocking: avg %.1f candidates/entity",
        np.mean([len(v) for v in candidates.values()]),
    )

    return candidates


# ═════════════════════════════════════════════════════════════════════════════
# Strategy 3: Address number + city blocking
# ═════════════════════════════════════════════════════════════════════════════

def address_number_blocking(
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
) -> Dict[str, Set[str]]:
    """Block by matching street numbers + overlapping city/locality tokens.

    Only produces candidates for S1 records that have at least one
    address number. This is a precision-oriented blocker.
    """
    logger.info(
        "Address-number blocking: %s S1 × %s S2/S3",
        f"{len(s1_df):,}", f"{len(s23_df):,}",
    )

    # Build index: first_address_number → list of (s23_id, address_tokens)
    num_index: Dict[str, List[tuple]] = defaultdict(list)
    for _, row in s23_df.iterrows():
        nums = row.get("address_numbers", "")
        if not nums:
            continue
        first_num = nums.split(",")[0]
        addr_tokens = set(row.get("norm_address_tokens", "").split()) if row.get("norm_address_tokens") else set()
        num_index[first_num].append((row["entity_id"], addr_tokens))

    candidates: Dict[str, Set[str]] = {}

    for _, row in s1_df.iterrows():
        s1_id = row["entity_id"]
        nums = row.get("address_numbers", "")
        if not nums:
            candidates[s1_id] = set()
            continue

        first_num = nums.split(",")[0]
        s1_addr_tokens = set(row.get("norm_address_tokens", "").split()) if row.get("norm_address_tokens") else set()

        cands = set()
        if first_num in num_index:
            for s23_id, s23_addr_tokens in num_index[first_num]:
                # Require at least 1 shared address token beyond the number
                shared = s1_addr_tokens & s23_addr_tokens
                if len(shared) >= 2:  # number + at least one location token
                    cands.add(s23_id)
        candidates[s1_id] = cands

    logger.info(
        "  Address blocking: avg %.1f candidates/entity",
        np.mean([len(v) for v in candidates.values()]),
    )

    return candidates


# ═════════════════════════════════════════════════════════════════════════════
# Union of all strategies
# ═════════════════════════════════════════════════════════════════════════════

def generate_candidates_for_country(
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    country: str,
    use_address_blocking: bool = True,
    address_tfidf_top_k: int = 20,
) -> Dict[str, Set[str]]:
    """Run all blocking strategies for a single country partition and union results.

    Parameters
    ----------
    s1_df : DataFrame
        Normalized S1 records for this country.
    s23_df : DataFrame
        Normalized S2+S3 records for this country.
    country : str
        Country name (for logging).
    use_address_blocking : bool
        Whether to run the address-number blocker (slower, small gain).
    address_tfidf_top_k : int
        Top-K for address-based TF-IDF blocking (smaller than name blocking).

    Returns
    -------
    dict
        ``{s1_id: set_of_candidate_s23_ids}``  capped at MAX_CANDIDATES.
    """
    logger.info(
        "═══ Generating candidates for %s: %s S1, %s S2/S3 ═══",
        country, f"{len(s1_df):,}", f"{len(s23_df):,}",
    )

    # Strategy 1: TF-IDF on names
    tfidf_cands = tfidf_blocking(
        s1_names=s1_df["norm_name"],
        s1_ids=s1_df["entity_id"],
        s23_names=s23_df["norm_name"],
        s23_ids=s23_df["entity_id"],
    )

    # Strategy 2: Token overlap on names
    token_cands = token_blocking(
        s1_tokens_series=s1_df["norm_name_tokens"],
        s1_ids=s1_df["entity_id"],
        s23_tokens_series=s23_df["norm_name_tokens"],
        s23_ids=s23_df["entity_id"],
    )

    # Strategy 3: TF-IDF on addresses (catches Devanagari name mismatches)
    # Only run if there are records with non-empty addresses
    addr_tfidf_cands = {}
    s1_has_addr = s1_df["norm_address"].str.len() > 0
    s23_has_addr = s23_df["norm_address"].str.len() > 0
    if s1_has_addr.any() and s23_has_addr.any():
        logger.info("  Running address TF-IDF blocking …")
        addr_tfidf_cands = tfidf_blocking(
            s1_names=s1_df.loc[s1_has_addr, "norm_address"],
            s1_ids=s1_df.loc[s1_has_addr, "entity_id"],
            s23_names=s23_df.loc[s23_has_addr, "norm_address"],
            s23_ids=s23_df.loc[s23_has_addr, "entity_id"],
            top_k=address_tfidf_top_k,
        )

    # Strategy 4: Address number matching (optional, slow on large data)
    if use_address_blocking:
        addr_num_cands = address_number_blocking(s1_df, s23_df)
    else:
        addr_num_cands = {}

    # Union all strategies
    all_s1_ids = set(s1_df["entity_id"])
    candidates: Dict[str, Set[str]] = {}

    for s1_id in all_s1_ids:
        union = set()
        union |= tfidf_cands.get(s1_id, set())
        union |= token_cands.get(s1_id, set())
        union |= addr_tfidf_cands.get(s1_id, set())
        union |= addr_num_cands.get(s1_id, set())

        # Cap at MAX_CANDIDATES (keep the ones from TF-IDF first, since
        # they're score-ranked; then add token/addr extras)
        if len(union) > MAX_CANDIDATES:
            # Prioritise name TF-IDF candidates (they have similarity scores)
            tfidf_set = tfidf_cands.get(s1_id, set())
            extras = union - tfidf_set
            union = tfidf_set | set(list(extras)[: MAX_CANDIDATES - len(tfidf_set)])

        candidates[s1_id] = union

    avg_cands = np.mean([len(v) for v in candidates.values()])
    non_empty = sum(1 for v in candidates.values() if v)
    logger.info(
        "  %s TOTAL: %s entities, avg %.1f candidates, %s non-empty (%.1f%%)",
        country, f"{len(candidates):,}", avg_cands,
        f"{non_empty:,}", non_empty / len(candidates) * 100,
    )

    return candidates


# ═════════════════════════════════════════════════════════════════════════════
# Blocking recall measurement
# ═════════════════════════════════════════════════════════════════════════════

def measure_blocking_recall(
    candidates: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
) -> dict:
    """Measure how many true matches are captured by the candidate sets.

    Returns
    -------
    dict
        - recall: macro-averaged blocking recall
        - n_entities: number of entities evaluated
        - n_perfect: entities where all true matches are in candidates
        - n_missed_some: entities where some true matches are missing
        - total_true: total ground-truth matches across all entities
        - total_found: total true matches found in candidates
    """
    total_true = 0
    total_found = 0
    n_perfect = 0
    n_missed_some = 0
    recalls = []

    for s1_id, true_ids in ground_truth.items():
        if not true_ids:
            continue  # singletons — no recall to measure

        cand_ids = candidates.get(s1_id, set())
        found = true_ids & cand_ids
        total_true += len(true_ids)
        total_found += len(found)

        r = len(found) / len(true_ids)
        recalls.append(r)

        if r == 1.0:
            n_perfect += 1
        else:
            n_missed_some += 1

    result = {
        "recall": np.mean(recalls) if recalls else 0.0,
        "n_entities": len(recalls),
        "n_perfect": n_perfect,
        "n_missed_some": n_missed_some,
        "total_true": total_true,
        "total_found": total_found,
        "pct_found": total_found / total_true * 100 if total_true else 0,
    }

    logger.info(
        "Blocking recall: %.4f (%.1f%% of true matches found, %s/%s entities perfect)",
        result["recall"], result["pct_found"],
        f"{n_perfect:,}", f"{len(recalls):,}",
    )

    return result
