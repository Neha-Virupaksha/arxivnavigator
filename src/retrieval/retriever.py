import os
import numpy as np
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

class PaperRetriever:
    """Fetches relevant paper chunks from Pinecone."""

    def __init__(self, index_name="arxivnavigator"):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pc.Index(index_name)
        self.index_name = index_name
        # Cache one vector store per namespace instead of rebuilding on every call
        self._vector_stores = {}

    def _vector_store(self, namespace: str):
        if namespace not in self._vector_stores:
            self._vector_stores[namespace] = PineconeVectorStore(
                index=self.index,
                embedding=self.embeddings,
                namespace=namespace
            )
        return self._vector_stores[namespace]

    def retrieve(self, query: str, namespace: str, k: int = 5, filter: dict = None):
        vector_store = self._vector_store(namespace)
        if filter:
            return vector_store.similarity_search(query, k=k, filter=filter)
        return vector_store.similarity_search(query, k=k)

    def retrieve_mmr(self, query: str, namespace: str, k: int = 5, fetch_k: int = 15, lambda_mult: float = 0.5):
        vector_store = self._vector_store(namespace)
        return vector_store.max_marginal_relevance_search(
            query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
        )

    def retrieve_reranked(self, query: str, namespace: str, fetch_k: int = 10, top_k: int = 5):
        docs = self.retrieve(query, namespace=namespace, k=fetch_k)
        if not docs:
            return []

        query_vec = np.array(self.embeddings.embed_query(query))

        def cos_sim(a, b):
            a, b = np.array(a), np.array(b)
            denom = (np.linalg.norm(a) * np.linalg.norm(b))
            return float(np.dot(a, b) / denom) if denom > 0 else 0.0

        scored = []
        for doc in docs:
            title = doc.metadata.get('title', '')
            title_vec = self.embeddings.embed_query(title) if title else query_vec
            title_score = cos_sim(query_vec, title_vec)
            abstract_score = cos_sim(query_vec, self.embeddings.embed_query(doc.page_content))
            combined = title_score * 0.6 + abstract_score * 0.4
            scored.append((combined, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]
