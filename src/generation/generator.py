import os
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

class ResponseGenerator:
    """Generates answers using Ollama and the retrieved context."""
    
    def __init__(self, model_name="llama3.2:3b"):
        self.llm = OllamaLLM(model=model_name, base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(2),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def generate(self, query: str, context: str):
        prompt = ChatPromptTemplate.from_template(
            "You are an expert AI research assistant. Answer the user's question using ONLY the "
            "information in the context below. Do not use any outside knowledge, and do not invent "
            "or guess any paper titles, ArXiv IDs, authors, or facts that are not explicitly present "
            "in the context. If the context does not contain enough information to answer the question, "
            "say so clearly instead of making something up.\n\n"
            "When citing a paper, use only the exact ArXiv ID given in the context for that chunk.\n\n"
            "Context:\n{context}\n\n"
            "Question: {query}\n\n"
            "Answer:"
        )
        chain = prompt | self.llm
        return chain.invoke({"query": query, "context": context})
