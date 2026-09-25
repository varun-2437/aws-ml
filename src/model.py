"""
Model training, threshold optimization, and prediction.

Wraps LightGBM with the competition-specific F₀.₅ optimization loop.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import joblib
import lightgbm as lgb
import numpy as np

from src.config import (
    CACHE_DIR,
    LGBM_PARAMS,
    RANDOM_SEED,
    THRESHOLD_MAX,
    THRESHOLD_MIN,
    THRESHOLD_STEP,
    VAL_FRACTION,
)
from src.evaluation import evaluate_detailed, f05_per_entity, macro_f05
from src.features import FEATURE_NAMES

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Train / val split by S1 entity
# ═════════════════════════════════════════════════════════════════════════════

def split_by_entity(
    pair_ids: List[Tuple[str, str]],
    X: np.ndarray,
    y: np.ndarray,
    val_fraction: float = VAL_FRACTION,
    seed: int = RANDOM_SEED,
) -> dict:
    """Split pairs into train/val such that each S1 entity is entirely in one split.

    Returns
    -------
    dict with keys:
        X_train, y_train, pairs_train,
        X_val, y_val, pairs_val,
        val_s1_ids (set)
    """
    rng = np.random.RandomState(seed)

    # Get unique S1 IDs
    s1_ids = list({s1 for s1, _ in pair_ids})
    rng.shuffle(s1_ids)

    n_val = int(len(s1_ids) * val_fraction)
    val_s1_set = set(s1_ids[:n_val])
    train_s1_set = set(s1_ids[n_val:])

    train_mask = np.array([s1 in train_s1_set for s1, _ in pair_ids])
    val_mask = ~train_mask

    logger.info(
        "Split: %s train S1 entities (%s pairs), %s val S1 entities (%s pairs)",
        f"{len(train_s1_set):,}", f"{train_mask.sum():,}",
        f"{len(val_s1_set):,}", f"{val_mask.sum():,}",
    )

    return {
        "X_train": X[train_mask],
        "y_train": y[train_mask],
        "pairs_train": [pair_ids[i] for i in range(len(pair_ids)) if train_mask[i]],
        "X_val": X[val_mask],
        "y_val": y[val_mask],
        "pairs_val": [pair_ids[i] for i in range(len(pair_ids)) if val_mask[i]],
        "val_s1_ids": val_s1_set,
    }


# ═════════════════════════════════════════════════════════════════════════════
# LightGBM training
# ═════════════════════════════════════════════════════════════════════════════

def train_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict = None,
    num_boost_round: int = 1000,
    early_stopping_rounds: int = 50,
) -> lgb.Booster:
    """Train a LightGBM model with early stopping on validation loss.

    Returns the trained Booster.
    """
    if params is None:
        params = LGBM_PARAMS.copy()

    # Set class weight based on imbalance
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    if n_pos > 0:
        params["scale_pos_weight"] = n_neg / n_pos
        logger.info(
            "Class balance: %s pos, %s neg, scale_pos_weight=%.2f",
            f"{n_pos:,}", f"{n_neg:,}", params["scale_pos_weight"],
        )

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_NAMES, reference=dtrain)

    callbacks = [
        lgb.log_evaluation(period=50),
        lgb.early_stopping(stopping_rounds=early_stopping_rounds),
    ]

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    logger.info("Best iteration: %d", model.best_iteration)

    return model


def get_feature_importance(model: lgb.Booster) -> List[Tuple[str, float]]:
    """Return feature importance sorted by gain."""
    importance = model.feature_importance(importance_type="gain")
    names = model.feature_name()
    pairs = sorted(zip(names, importance), key=lambda x: -x[1])
    return pairs


# ═════════════════════════════════════════════════════════════════════════════
# Threshold optimization
# ═════════════════════════════════════════════════════════════════════════════

def predictions_to_entity_matches(
    pair_ids: List[Tuple[str, str]],
    probabilities: np.ndarray,
    threshold: float,
) -> Dict[str, Set[str]]:
    """Convert pair-level probabilities to entity-level match sets.

    For each S1 entity, collect all candidate IDs whose probability
    exceeds the threshold.
    """
    matches: Dict[str, Set[str]] = defaultdict(set)
    for (s1_id, cand_id), prob in zip(pair_ids, probabilities):
        if prob >= threshold:
            matches[s1_id].add(cand_id)
    return dict(matches)


def optimize_threshold(
    pair_ids: List[Tuple[str, str]],
    probabilities: np.ndarray,
    ground_truth: Dict[str, Set[str]],
    val_s1_ids: Set[str] = None,
    threshold_min: float = THRESHOLD_MIN,
    threshold_max: float = THRESHOLD_MAX,
    threshold_step: float = THRESHOLD_STEP,
) -> Tuple[float, float, List[Tuple[float, float]]]:
    """Sweep thresholds and find the one that maximizes macro F₀.₅.

    Parameters
    ----------
    pair_ids : list of (s1_id, cand_id)
    probabilities : array of predicted match probabilities
    ground_truth : {s1_id: set_of_true_ids} — full ground truth
    val_s1_ids : set of S1 IDs in the validation set (if None, use all in GT)

    Returns
    -------
    best_threshold : float
    best_f05 : float
    sweep_results : list of (threshold, f05)
    """
    # Filter ground truth to val entities only
    if val_s1_ids is not None:
        gt_val = {k: v for k, v in ground_truth.items() if k in val_s1_ids}
    else:
        gt_val = ground_truth

    sweep = []
    best_threshold = 0.5
    best_f05 = 0.0

    thresholds = np.arange(threshold_min, threshold_max, threshold_step)

    for t in thresholds:
        preds = predictions_to_entity_matches(pair_ids, probabilities, t)

        # Ensure all val entities have a prediction entry (empty = singleton)
        full_preds = {s1_id: preds.get(s1_id, set()) for s1_id in gt_val}

        score = macro_f05(full_preds, gt_val)
        sweep.append((float(t), float(score)))

        if score > best_f05:
            best_f05 = score
            best_threshold = float(t)

    logger.info(
        "Threshold sweep: best=%.3f → F₀.₅=%.4f (tested %d thresholds)",
        best_threshold, best_f05, len(thresholds),
    )

    return best_threshold, best_f05, sweep


# ═════════════════════════════════════════════════════════════════════════════
# Model persistence
# ═════════════════════════════════════════════════════════════════════════════

def save_model(model: lgb.Booster, threshold: float, path: Path = None):
    """Save model + threshold to disk."""
    if path is None:
        path = CACHE_DIR / "model.pkl"
    joblib.dump({"model": model, "threshold": threshold}, path)
    logger.info("Model saved to %s", path)


def load_model(path: Path = None) -> Tuple[lgb.Booster, float]:
    """Load model + threshold from disk."""
    if path is None:
        path = CACHE_DIR / "model.pkl"
    data = joblib.load(path)
    logger.info("Model loaded from %s", path)
    return data["model"], data["threshold"]
