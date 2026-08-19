import urllib.request
import xml.etree.ElementTree as ET
import time
from typing import List, Dict

class ArxivLoader:
    """Fetches daily AI research from ArXiv API."""
    BASE_URL = "http://export.arxiv.org/api/query?"
    
    def __init__(self):
        self.categories = ["cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.IR"]
        
    def fetch_papers(self, max_results: int = 10) -> List[Dict]:
        all_papers = []
        for cat in self.categories:
            # FIX: Changed sortOrder=desc to sortOrder=descending
            url = f"{self.BASE_URL}search_query=cat:{cat}&sortBy=submittedDate&sortOrder=descending&max_results={max_results}"
            try:
                # Adding a User-Agent header is best practice to avoid being blocked
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response:
                    xml_data = response.read()
                    all_papers.extend(self._parse_xml(xml_data, cat))
                time.sleep(3) 
            except Exception as e:
                print(f"Error fetching {cat}: {e}")
        return all_papers

    def _parse_xml(self, xml_data: bytes, category: str) -> List[Dict]:
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        root = ET.fromstring(xml_data)
        papers = []
        for entry in root.findall('atom:entry', ns):
            papers.append({
                "title": entry.find('atom:title', ns).text.strip(),
                "abstract": entry.find('atom:summary', ns).text.strip(),
                "authors": ", ".join([a.find('atom:name', ns).text for a in entry.findall('atom:author', ns)]),
                "published_date": entry.find('atom:published', ns).text,
                "arxiv_id": entry.find('atom:id', ns).text.split('/')[-1],
                "namespace": category.replace(".", "_").lower()
            })
        return papers
