import re

RECENCY_KEYWORDS = ['latest', 'recent', 'new', 'newest', 'current', 'up to date', '2024', '2025', '2026']

CATEGORY_KEYWORDS = {
    'cs_cv': ['image', 'vision', 'visual', 'clip', 'segmentation', 'object detection', 'diffusion', 'gan'],
    'cs_cl': ['nlp', 'language model', 'text', 'translation', 'tokenizer', 'sentiment', 'parsing'],
    'cs_ir': ['retrieval', 'search', 'ranking', 'recommendation', 'indexing'],
    'cs_lg': ['training', 'optimizer', 'gradient', 'neural network', 'learning rate', 'generalization'],
    'cs_ai': ['agent', 'reasoning', 'planning', 'reinforcement learning', 'multi-agent'],
}


class QueryExpansionStrategy:
    """Strategy 1 — expands technical abbreviations and adds domain synonyms via the LLM,
    so retrieval matches full technical terminology used in ArXiv abstracts."""
    name = "Query Expansion"

    def run(self, original_query, retriever, generator, category, k=5):
        expansion_prompt = (
            f"Rewrite this research question by expanding any abbreviations and adding "
            f"2-3 relevant technical synonyms or related terms, to improve search recall. "
            f"Return ONLY the rewritten query, nothing else.\n\nQuestion: {original_query}"
        )
        expanded_query = generator.llm.invoke(expansion_prompt).strip()
        docs = retriever.retrieve(expanded_query, namespace=category, k=k)
        return docs


class DateAwareStrategy:
    """Strategy 2 — detects recency keywords and filters to papers published in the
    last 12 months, so 'latest research on X' doesn't retrieve old papers."""
    name = "Date-Aware Retrieval"

    def run(self, original_query, retriever, generator, category, k=5):
        from datetime import datetime, timedelta
        # Pinecone's $gte filter requires numeric metadata, but published_date is
        # stored as a string (e.g. "2026-01-15") from ingestion — so we filter
        # client-side instead of relying on a server-side Pinecone filter.
        cutoff = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        candidates = retriever.retrieve(original_query, namespace=category, k=k * 3)

        recent = [
            d for d in candidates
            if str(d.metadata.get('published_date', '')) >= cutoff
        ]

        if recent:
            return recent[:k]
        # No recent papers found — fall back to the general top-k rather than
        # returning nothing.
        return candidates[:k]

    @staticmethod
    def is_applicable(query):
        q = query.lower()
        return any(kw in q for kw in RECENCY_KEYWORDS)


class CategoryFilterStrategy:
    """Strategy 3 — detects the research domain from the query and restricts retrieval
    to the matching Pinecone namespace, avoiding cross-category noise."""
    name = "Category Filtering"

    def run(self, original_query, retriever, generator, category, k=5):
        detected_category = self.detect_category(original_query) or category
        docs = retriever.retrieve(original_query, namespace=detected_category, k=k)
        return docs

    @staticmethod
    def detect_category(query):
        q = query.lower()
        for cat, keywords in CATEGORY_KEYWORDS.items():
            if any(kw in q for kw in keywords):
                return cat
        return None


class RerankStrategy:
    """Strategy 4 — retrieves a larger candidate pool, then reranks by title and
    abstract relevance separately, catching papers with a highly relevant title
    even if the abstract itself is dense."""
    name = "Abstract Reranking"

    def run(self, original_query, retriever, generator, category, k=5):
        return retriever.retrieve_reranked(original_query, namespace=category, fetch_k=k * 2, top_k=k)


class MMRDiversityStrategy:
    """Strategy 5 — expands the candidate pool and applies Maximal Marginal Relevance
    to pick diverse papers, avoiding redundant near-duplicate retrievals."""
    name = "Expand K with MMR Diversity"

    def run(self, original_query, retriever, generator, category, k=5):
        return retriever.retrieve_mmr(original_query, namespace=category, k=k, fetch_k=k * 3)


# Strategies in the order the healer will try them by default
ALL_STRATEGIES = [
    QueryExpansionStrategy(),
    DateAwareStrategy(),
    CategoryFilterStrategy(),
    RerankStrategy(),
    MMRDiversityStrategy(),
]


def select_strategy_order(original_query, failed_metrics):
    """Picks a sensible order to try strategies in, based on the query and which
    metrics failed. Recency-flavored queries always try Date-Aware first."""
    strategies = list(ALL_STRATEGIES)

    if DateAwareStrategy.is_applicable(original_query):
        strategies.sort(key=lambda s: 0 if isinstance(s, DateAwareStrategy) else 1)
    elif 'nv_context_relevance' in failed_metrics:
        # Context relevance failures are usually a retrieval-scope problem —
        # try narrowing to the right category first, then diversify.
        strategies.sort(key=lambda s: (
            0 if isinstance(s, CategoryFilterStrategy) else
            1 if isinstance(s, QueryExpansionStrategy) else
            2 if isinstance(s, MMRDiversityStrategy) else 3
        ))
    elif 'faithfulness' in failed_metrics:
        # Faithfulness failures need genuinely different/better context, not just
        # more of the same — expansion and reranking are most likely to help.
        strategies.sort(key=lambda s: (
            0 if isinstance(s, QueryExpansionStrategy) else
            1 if isinstance(s, RerankStrategy) else
            2 if isinstance(s, MMRDiversityStrategy) else 3
        ))
    else:
        # answer_relevancy failing — the context was probably fine, tighten the query
        strategies.sort(key=lambda s: (
            0 if isinstance(s, QueryExpansionStrategy) else
            1 if isinstance(s, RerankStrategy) else 2
        ))

    return strategies
