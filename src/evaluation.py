"""
Evaluation metrics for the Entity Resolution pipeline.

Implements the competition's macro-averaged F₀.₅ metric, which weights
precision 2× more heavily than recall.
"""

from typing import Dict, Set


def f05_per_entity(predicted_ids: Set[str], true_ids: Set[str]) -> float:
    """Compute F₀.₅ for a single Source-1 entity.

    Parameters
    ----------
    predicted_ids : set of str
        The entity IDs predicted as matches.
    true_ids : set of str
        The ground-truth matched entity IDs.

    Returns
    -------
    float
        F₀.₅ score in [0, 1].

    Special cases (per competition rules):
      - Both empty (correct singleton)  → 1.0
      - Predicted empty, true non-empty → 0.0 (missed all matches)
      - Predicted non-empty, true empty → 0.0 (false merge on singleton)
    """
    if len(true_ids) == 0 and len(predicted_ids) == 0:
        return 1.0
    if len(predicted_ids) == 0 or len(true_ids) == 0:
        return 0.0

    tp = len(predicted_ids & true_ids)
    if tp == 0:
        return 0.0

    precision = tp / len(predicted_ids)
    recall = tp / len(true_ids)

    # F_beta with beta = 0.5
    # F₀.₅ = (1 + 0.25) * P * R / (0.25 * P + R)
    beta_sq = 0.25
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def macro_f05(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
) -> float:
    """Compute macro-averaged F₀.₅ across all Source-1 entities.

    Parameters
    ----------
    predictions : dict
        ``{s1_entity_id: set_of_predicted_match_ids}``
    ground_truth : dict
        ``{s1_entity_id: set_of_true_match_ids}``

    Returns
    -------
    float
        Macro-averaged F₀.₅.
    """
    if not ground_truth:
        return 0.0

    total = 0.0
    for s1_id, true_ids in ground_truth.items():
        pred_ids = predictions.get(s1_id, set())
        total += f05_per_entity(pred_ids, true_ids)

    return total / len(ground_truth)


def evaluate_detailed(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
    entity_countries: Dict[str, str] = None,
) -> dict:
    """Compute detailed evaluation metrics.

    Returns a dict with:
      - overall_f05: macro F₀.₅ over all entities
      - per_country_f05: {country: f05} if entity_countries is provided
      - n_entities: total count
      - n_correct_singletons: singletons correctly identified
      - n_false_merges: singletons that got false matches
      - n_missed_all: entities where all true matches were missed
      - avg_precision: macro-averaged precision
      - avg_recall: macro-averaged recall
    """
    scores = []
    precisions = []
    recalls = []
    n_correct_singletons = 0
    n_false_merges = 0
    n_missed_all = 0

    country_scores = {}  # {country: [f05_scores]}

    for s1_id, true_ids in ground_truth.items():
        pred_ids = predictions.get(s1_id, set())
        score = f05_per_entity(pred_ids, true_ids)
        scores.append(score)

        # Track per-country
        if entity_countries and s1_id in entity_countries:
            country = entity_countries[s1_id]
            country_scores.setdefault(country, []).append(score)

        # Detailed stats
        if len(true_ids) == 0:
            if len(pred_ids) == 0:
                n_correct_singletons += 1
            else:
                n_false_merges += 1
        else:
            tp = len(pred_ids & true_ids)
            if len(pred_ids) > 0:
                precisions.append(tp / len(pred_ids))
            else:
                precisions.append(0.0)
            recalls.append(tp / len(true_ids))
            if tp == 0:
                n_missed_all += 1

    result = {
        "overall_f05": sum(scores) / len(scores) if scores else 0.0,
        "n_entities": len(ground_truth),
        "n_correct_singletons": n_correct_singletons,
        "n_false_merges": n_false_merges,
        "n_missed_all": n_missed_all,
        "avg_precision": sum(precisions) / len(precisions) if precisions else 0.0,
        "avg_recall": sum(recalls) / len(recalls) if recalls else 0.0,
    }

    if entity_countries:
        result["per_country_f05"] = {
            country: sum(s) / len(s) if s else 0.0
            for country, s in country_scores.items()
        }

    return result


# ─── Quick self-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test cases from the competition definition
    assert f05_per_entity(set(), set()) == 1.0, "Correct singleton should be 1.0"
    assert f05_per_entity({"S2-1"}, set()) == 0.0, "False merge on singleton should be 0.0"
    assert f05_per_entity(set(), {"S2-1"}) == 0.0, "Missed all should be 0.0"

    # Perfect match
    assert f05_per_entity({"S2-1", "S3-2"}, {"S2-1", "S3-2"}) == 1.0

    # Partial match: predicted {S2-1}, true {S2-1, S3-2}
    # precision = 1.0, recall = 0.5
    # F0.5 = 1.25 * 1.0 * 0.5 / (0.25 * 1.0 + 0.5) = 0.625 / 0.75 = 0.8333
    score = f05_per_entity({"S2-1"}, {"S2-1", "S3-2"})
    assert abs(score - 0.8333) < 0.001, f"Expected ~0.833, got {score}"

    # Partial with false positive: predicted {S2-1, S2-999}, true {S2-1, S3-2}
    # tp=1, precision=0.5, recall=0.5
    # F0.5 = 1.25 * 0.5 * 0.5 / (0.25 * 0.5 + 0.5) = 0.3125 / 0.625 = 0.5
    score = f05_per_entity({"S2-1", "S2-999"}, {"S2-1", "S3-2"})
    assert abs(score - 0.5) < 0.001, f"Expected 0.5, got {score}"

    print("✓ All evaluation tests passed!")

    # Test macro_f05
    gt = {
        "S1-1": {"S2-1", "S3-1"},
        "S1-2": set(),                # singleton
        "S1-3": {"S2-3"},
    }
    preds = {
        "S1-1": {"S2-1", "S3-1"},     # perfect
        "S1-2": set(),                 # correct singleton
        "S1-3": set(),                 # missed
    }
    m = macro_f05(preds, gt)
    expected = (1.0 + 1.0 + 0.0) / 3
    assert abs(m - expected) < 0.001, f"Expected {expected}, got {m}"
    print(f"✓ Macro F₀.₅ test passed: {m:.4f}")
