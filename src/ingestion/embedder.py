import os
from typing import List, Dict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv

# Load the secret API key from your .env file
load_dotenv()

class PaperEmbedder:
    """Translates text into numbers (vectors) and stores them in Pinecone."""
    
    def __init__(self, index_name="arxivnavigator"):
        print("Loading HuggingFace embedding model (this may take a minute the first time)...")
        # We use all-MiniLM-L6-v2 which creates 384-dimensional vectors
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        print("Connecting to Pinecone database...")
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = index_name
        
    def embed_and_store(self, chunks: List[Dict]):
        """Uploads the chunks to their specific category sections in Pinecone."""
        if not chunks:
            print("No chunks to embed.")
            return
            
        # Group chunks by their specific AI category (namespaces)
        namespace_chunks = {}
        for chunk in chunks:
            ns = chunk['metadata']['namespace']
            if ns not in namespace_chunks:
                namespace_chunks[ns] = {"texts": [], "metadatas": [], "ids": []}
            
            namespace_chunks[ns]["texts"].append(chunk["text"])
            namespace_chunks[ns]["metadatas"].append(chunk["metadata"])
            namespace_chunks[ns]["ids"].append(chunk["id"])
            
        # Upload each group to its own section in Pinecone
        for ns, data in namespace_chunks.items():
            print(f"Uploading {len(data['texts'])} chunks to category '{ns}'...")
            PineconeVectorStore.from_texts(
                texts=data["texts"],
                embedding=self.embeddings,
                metadatas=data["metadatas"],
                ids=data["ids"],
                index_name=self.index_name,
                namespace=ns
            )
            print(f"Successfully uploaded to {ns}!")
