from src.ingestion.loader import ArxivLoader
from src.ingestion.chunker import PaperChunker
from src.ingestion.embedder import PaperEmbedder

print("--- ArXivNavigator: Phase 1 Data Ingestion ---")

# 1. Download Papers
print("\n1. Fetching recent AI papers from the internet...")
loader = ArxivLoader()
# Fetching just 2 papers per category (10 total) for our first real run
papers = loader.fetch_papers(max_results=2) 

# 2. Chunk Papers
print(f"\n2. Breaking down {len(papers)} papers into readable paragraphs...")
chunker = PaperChunker()
chunks = chunker.chunk_papers(papers)
print(f"Created {len(chunks)} total text chunks.")

# 3. Save to Pinecone
print("\n3. Translating text and saving permanently to Pinecone database...")
embedder = PaperEmbedder()
embedder.embed_and_store(chunks)

print("\n--- Success! Phase 1 is officially complete! ---")
