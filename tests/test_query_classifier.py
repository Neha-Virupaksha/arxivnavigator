from src.classification.query_classifier import classify_query

def test_comparison_query():
    assert classify_query("compare BERT and GPT architectures") == "Comparison-based"
    assert classify_query("RLHF vs DPO for LLM alignment") == "Comparison-based"

def test_recent_query():
    assert classify_query("latest research on multimodal LLMs") == "Recent-based"
    assert classify_query("new developments in AI agents 2024") == "Recent-based"

def test_paper_query():
    assert classify_query("key papers on attention mechanism") == "Paper-based"

def test_application_query():
    assert classify_query("how is NLP used in healthcare") == "Application-based"

def test_technique_query_default():
    assert classify_query("what is the best method for fine-tuning LLMs") == "Technique-based"
    assert classify_query("how does chain of thought prompting work") == "Technique-based"
