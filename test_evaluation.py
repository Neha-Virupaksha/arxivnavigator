from src.retrieval.retriever import PaperRetriever
from src.generation.generator import ResponseGenerator
from src.evaluation.ragas_evaluator import RagasEvaluator

# 1. Setup
query = "What is complexity-aware reasoning in AI agents?"
retriever = PaperRetriever()
generator = ResponseGenerator()
evaluator = RagasEvaluator()

# 2. Pipeline
docs = retriever.retrieve(query, namespace="cs_ai", k=3)
context = [d.page_content for d in docs]
answer = generator.generate(query, "\n".join(context))

# 3. Evaluate
scores = evaluator.evaluate_response(query, answer, context)

print(f"\n--- Evaluation Results ---\n{scores}")
