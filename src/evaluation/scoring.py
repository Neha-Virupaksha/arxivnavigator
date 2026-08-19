import math

WEIGHTS = {
    'faithfulness': 0.4,
    'answer_relevancy': 0.4,
    'nv_context_relevance': 0.2,
}


def _is_valid(v):
    if v is None:
        return False
    try:
        if math.isnan(v):
            return False
    except TypeError:
        return False
    return True


def compute_composite_score(scores_dict: dict) -> float:
    """
    Composite = (Faithfulness x 0.4) + (Answer Relevancy x 0.4) + (Context Relevance x 0.2)

    A nan metric means the judge failed to produce a verdict — it is UNKNOWN, not bad.
    Rather than treating it as 0 (which would unfairly punish an otherwise-good answer
    just because the judge stumbled on formatting), we exclude it and reweight the
    remaining metrics proportionally so the composite stays on a comparable 0-1 scale.
    This matches how healer.py already picks the best attempt internally.
    """
    valid_weight_sum = sum(
        weight for metric, weight in WEIGHTS.items() if _is_valid(scores_dict.get(metric))
    )

    if valid_weight_sum == 0:
        return 0.0  # every metric was nan — genuinely no signal, composite is 0

    weighted_sum = sum(
        scores_dict.get(metric) * weight
        for metric, weight in WEIGHTS.items()
        if _is_valid(scores_dict.get(metric))
    )

    # Reweight so the result stays on a 0-1 scale even if some metrics were excluded
    return round(weighted_sum / valid_weight_sum, 4)
