import os
import sys
from unittest.mock import MagicMock

# Trick Ragas into thinking VertexAI exists so it stops crashing
sys.modules['langchain_community.chat_models.vertexai'] = MagicMock()

import nest_asyncio
nest_asyncio.apply()

import numpy as np
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, ContextRelevance
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset


def _extract_scores_dict(scores):
    """Reliably pulls a {metric_name: value} dict out of a ragas EvaluationResult,
    regardless of ragas version quirks around to_dict()/dict(). Uses .scores,
    which is a list of per-row result dicts (one row here since we evaluate one
    sample at a time)."""
    if hasattr(scores, 'scores'):
        data = scores.scores
        if isinstance(data, list) and len(data) > 0:
            return dict(data[0])
        return {}
    elif isinstance(scores, dict):
        return scores
    return {}


class RagasEvaluator:
    """Grades AI responses using Faithfulness, Answer Relevancy, and Context Relevance."""

    def __init__(self, model_name="llama3.2:1b", faithfulness_samples=1):
        self.llm = ChatOllama(model=model_name, base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))

        self.ragas_llm = LangchainLLMWrapper(self.llm)
        self.ragas_embeddings = LangchainEmbeddingsWrapper(self.embeddings)
        self.context_relevance = ContextRelevance(llm=self.ragas_llm)

        # Faithfulness is by far the noisiest metric with a small local judge model —
        # the same query/answer/context can score very differently run to run.
        # Sampling it multiple times and averaging smooths out that variance.
        self.faithfulness_samples = faithfulness_samples

    def _make_dataset(self, query, answer, context):
        data = {
            "question": [query],
            "answer": [answer],
            "contexts": [context],
            "user_input": [query],
            "response": [answer],
            "retrieved_contexts": [context]
        }
        return Dataset.from_dict(data)

    def _run_faithfulness_samples(self, query, answer, context):
        """Runs faithfulness N times and returns the mean of the valid (non-nan) scores."""
        dataset = self._make_dataset(query, answer, context)
        results = []
        for _ in range(self.faithfulness_samples):
            try:
                scores = evaluate(
                    dataset,
                    metrics=[faithfulness],
                    llm=self.ragas_llm,
                    embeddings=self.ragas_embeddings
                )
                sd = _extract_scores_dict(scores)
                value = sd.get('faithfulness')
                if value is not None and not (isinstance(value, float) and np.isnan(value)):
                    results.append(value)
            except Exception:
                continue  # a single failed sample shouldn't kill the whole evaluation

        if not results:
            return float('nan')  # every sample failed — genuinely inconclusive
        return float(np.mean(results))

    def evaluate_response(self, query: str, answer: str, context: list):
        """Scores a single interaction. Faithfulness is averaged over multiple samples
        to reduce judge-model noise; other metrics run once (they're more stable)."""
        dataset = self._make_dataset(query, answer, context)

        # Single-pass metrics
        scores = evaluate(
            dataset,
            metrics=[answer_relevancy, self.context_relevance],
            llm=self.ragas_llm,
            embeddings=self.ragas_embeddings
        )
        scores_dict = _extract_scores_dict(scores)

        # Averaged faithfulness
        scores_dict['faithfulness'] = self._run_faithfulness_samples(query, answer, context)

        return scores_dict
