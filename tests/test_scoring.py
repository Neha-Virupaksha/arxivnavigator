from src.evaluation.scoring import compute_composite_score

def test_all_metrics_present():
    scores = {'faithfulness': 0.8, 'answer_relevancy': 0.9, 'nv_context_relevance': 0.5}
    result = compute_composite_score(scores)
    expected = round(0.8 * 0.4 + 0.9 * 0.4 + 0.5 * 0.2, 4)
    assert result == expected

def test_nan_metric_excluded_and_reweighted():
    scores = {'faithfulness': float('nan'), 'answer_relevancy': 0.9, 'nv_context_relevance': 0.5}
    result = compute_composite_score(scores)
    # only answer_relevancy (0.4) and context_relevance (0.2) are valid -> reweight over 0.6
    expected = round((0.9 * 0.4 + 0.5 * 0.2) / 0.6, 4)
    assert result == expected

def test_all_nan_returns_zero():
    scores = {'faithfulness': float('nan'), 'answer_relevancy': None, 'nv_context_relevance': float('nan')}
    assert compute_composite_score(scores) == 0.0

def test_missing_keys_treated_as_invalid():
    scores = {'faithfulness': 0.6}
    result = compute_composite_score(scores)
    assert result == 0.6  # only faithfulness valid, reweighted to full scale
