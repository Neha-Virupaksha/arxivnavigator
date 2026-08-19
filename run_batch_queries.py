import requests
import time

API_URL = "http://127.0.0.1:8000/query"

# A diverse mix pulled from the project brief, spanning all 5 query types
# and multiple categories, kept to a manageable batch size for a first run.
QUERIES = [
    ("best techniques for reducing hallucinations in RAG", "cs_cl"),
    ("how does chain of thought prompting work", "cs_ai"),
    ("what is the best method for fine-tuning LLMs on limited data", "cs_lg"),
    ("compare BERT and GPT architectures", "cs_cl"),
    ("difference between RAG and fine-tuning", "cs_ai"),
    ("RLHF vs DPO for LLM alignment", "cs_ai"),
    ("latest research on multimodal LLMs", "cs_cv"),
    ("recent advances in vision language models", "cs_cv"),
    ("new developments in AI agents 2024", "cs_ai"),
    ("key papers on attention mechanism", "cs_lg"),
    ("important papers on retrieval augmented generation", "cs_ir"),
    ("how is NLP used in healthcare", "cs_cl"),
    ("applications of computer vision in autonomous driving", "cs_cv"),
    ("how is RL used to train LLMs", "cs_ai"),
    ("how does flash attention work", "cs_lg"),
]

print(f"Running {len(QUERIES)} queries against {API_URL}...\n")

for i, (query, category) in enumerate(QUERIES, 1):
    print(f"[{i}/{len(QUERIES)}] {query} ({category})")
    try:
        start = time.time()
        response = requests.post(API_URL, json={"query": query, "category": category}, timeout=600)
        elapsed = time.time() - start
        if response.status_code == 200:
            data = response.json()
            print(f"    -> composite={data['composite_score']} healed={data['healing_triggered']} "
                  f"strategy={data.get('winning_strategy')} ({elapsed:.1f}s)")
        else:
            print(f"    -> FAILED [{response.status_code}]: {response.text}")
    except Exception as e:
        print(f"    -> ERROR: {e}")
    print()

print("Batch complete.")
