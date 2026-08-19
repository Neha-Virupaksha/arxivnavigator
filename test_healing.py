from src.retrieval.retriever import PaperRetriever
from src.generation.generator import ResponseGenerator
from src.evaluation.ragas_evaluator import RagasEvaluator
from src.evaluation.healer import RAGHealer

print("--- Testing Self-Healing RAG Loop ---")

query = "What is complexity-aware reasoning in AI agents?"
category = "cs_ai"

retriever = PaperRetriever()
generator = ResponseGenerator()
evaluator = RagasEvaluator()
healer = RAGHealer()

# 1. Initial attempt
docs = retriever.retrieve(query, namespace=category, k=3)
context = [d.page_content for d in docs]
answer = generator.generate(query, "\n".join(context))

print(f"\n--- Initial Answer ---\n{answer}")

original_scores = evaluator.evaluate_response(query, answer, context)
needs_healing, failed_metrics, original_dict = healer.check_need_healing(original_scores)

print(f"\n--- Initial Scores ---\n{original_dict}")
print(f"Needs healing: {needs_healing} | Failed metrics: {failed_metrics}")

# 2. Heal if needed
if needs_healing:
    result = healer.heal(query, retriever, generator, evaluator, category, original_scores)
    if result and result['improved']:
        print(f"\n--- HEALED Answer ---\n{result['answer']}")
        print(f"\n--- HEALED Scores ---\n{result['scores']}")
    else:
        print("\n--- Healing did not find a better answer. Keeping original. ---")
else:
    print("\n--- No healing needed, original answer was good enough. ---")
