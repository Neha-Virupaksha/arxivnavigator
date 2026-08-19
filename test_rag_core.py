from src.retrieval.retriever import PaperRetriever
from src.generation.generator import ResponseGenerator

print("--- Testing RAG Core ---")

# 1. Retrieve
retriever = PaperRetriever()
results = retriever.retrieve("What is the latest in AI research?", namespace="cs_ai", k=2)

context = "\n\n".join([f"ArXiv ID: {r.metadata['arxiv_id']}\n{r.page_content}" for r in results])
print(f"\n--- Retrieved Context ---\n{context}\n-------------------------\n")

# 2. Generate
generator = ResponseGenerator()
answer = generator.generate("What is the latest in AI research?", context)

print(f"\nAI Answer:\n{answer}")
