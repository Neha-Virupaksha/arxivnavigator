from src.evaluation.healer import RAGHealer

def test_check_need_healing_all_pass():
    healer = RAGHealer()
    scores = {'faithfulness': 0.8, 'answer_relevancy': 0.8, 'nv_context_relevance': 0.8}
    needs_healing, failed, _ = healer.check_need_healing(scores)
    assert needs_healing is False
    assert failed == []

def test_check_need_healing_faithfulness_fails():
    healer = RAGHealer()
    scores = {'faithfulness': 0.1, 'answer_relevancy': 0.8, 'nv_context_relevance': 0.8}
    needs_healing, failed, _ = healer.check_need_healing(scores)
    assert needs_healing is True
    assert 'faithfulness' in failed

def test_nan_is_not_treated_as_failure():
    """A nan (judge failed to parse) should NOT by itself count as a failed metric —
    it's inconclusive, not bad."""
    healer = RAGHealer()
    scores = {'faithfulness': float('nan'), 'answer_relevancy': 0.8, 'nv_context_relevance': 0.8}
    needs_healing, failed, _ = healer.check_need_healing(scores)
    assert 'faithfulness' not in failed

def test_is_fully_resolved_requires_real_numbers():
    """A nan metric should NOT count as 'resolved' even though it's not 'failed' —
    this is the bug we found where healing declared false victory."""
    healer = RAGHealer()
    scores = {'faithfulness': float('nan'), 'answer_relevancy': 0.8, 'nv_context_relevance': 0.8}
    assert healer._is_fully_resolved(scores) is False

def test_is_fully_resolved_true_when_all_pass():
    healer = RAGHealer()
    scores = {'faithfulness': 0.8, 'answer_relevancy': 0.8, 'nv_context_relevance': 0.8}
    assert healer._is_fully_resolved(scores) is True

def test_score_sum_ignores_nan():
    healer = RAGHealer()
    scores = {'faithfulness': float('nan'), 'answer_relevancy': 0.9, 'nv_context_relevance': 0.5}
    assert healer._score_sum(scores) == 1.4

def test_score_sum_all_nan_returns_negative_one():
    healer = RAGHealer()
    scores = {'faithfulness': float('nan'), 'answer_relevancy': float('nan'), 'nv_context_relevance': float('nan')}
    assert healer._score_sum(scores) == -1
