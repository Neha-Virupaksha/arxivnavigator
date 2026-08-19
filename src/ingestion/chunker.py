from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict

class PaperChunker:
    """Breaks down full research papers into smaller, bite-sized chunks for the AI."""
    
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        # We split the text into chunks of 1000 characters. 
        # The 200 character overlap ensures we don't cut a sentence in half and lose context.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def chunk_papers(self, papers: List[Dict]) -> List[Dict]:
        chunked_data = []
        
        for paper in papers:
            # Combine the title and abstract into one readable block
            full_text = f"Title: {paper['title']}\nAbstract: {paper['abstract']}"
            
            # Split the block into pieces
            chunks = self.splitter.split_text(full_text)
            
            for i, chunk in enumerate(chunks):
                # Save each piece with its original tracking info (metadata)
                chunked_data.append({
                    "id": f"{paper['arxiv_id']}-chunk-{i}",
                    "text": chunk,
                    "metadata": {
                        "title": paper['title'],
                        "authors": paper['authors'],
                        "published_date": paper['published_date'],
                        "arxiv_id": paper['arxiv_id'],
                        "namespace": paper['namespace']
                    }
                })
                
        return chunked_data
