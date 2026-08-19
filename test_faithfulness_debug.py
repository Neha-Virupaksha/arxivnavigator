import logging
import sys
from unittest.mock import MagicMock
sys.modules['langchain_community.chat_models.vertexai'] = MagicMock()

import nest_asyncio
nest_asyncio.apply()

# Turn on verbose logging so we can see the raw statements/verdicts ragas extracts
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("httpx").setLevel(logging.WARNING)  # quiet the noisy HTTP logs

from src.retrieval.retriever import PaperRetriever
from src.generation.generator import ResponseGenerator
from src.evaluation.ragas_evaluator import RagasEvaluator

query = "What is complexity-aware reasoning in AI agents?"
retriever = PaperRetriever()
generator = ResponseGenerator()
evaluator = RagasEvaluator()

docs = retriever.retrieve(query, namespace="cs_ai", k=3)
context = [d.page_content for d in docs]
answer = generator.generate(query, "\n".join(context))

print(f"\n=== ANSWER ===\n{answer}\n")
print(f"=== CONTEXT ===\n{context}\n")

scores = evaluator.evaluate_response(query, answer, context)
print(f"\n=== FINAL SCORES ===\n{scores}")
