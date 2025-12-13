import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import os
import time
from pathlib import Path
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss #faiss-cpu or faiss-gpu
import numpy as np


def fetch_arxiv_papers(category='cs.CL', max_results=50):
    """
    Fetch arXiv papers from specified category and download PDFs
    
    Args:
        category: arXiv category (default: cs.CL for Computation and Language)
        max_results: Number of papers to fetch (default: 50)
    """
    # Create directory for PDFs
    pdf_dir = Path('arxiv_pdfs')
    pdf_dir.mkdir(exist_ok=True)
    
    # arXiv API base URL
    base_url = 'http://export.arxiv.org/api/query?'
    
    # Build query parameters
    params = {
        'search_query': f'cat:{category}',
        'start': 0,
        'max_results': max_results,
        'sortBy': 'submittedDate',
        'sortOrder': 'descending'
    }
    
    query_url = base_url + urllib.parse.urlencode(params)
    
    print(f"Fetching {max_results} papers from {category}...")
    print(f"Query URL: {query_url}\n")
    
    try:
        # Fetch the feed
        with urllib.request.urlopen(query_url) as response:
            data = response.read()
        
        # Parse XML
        root = ET.fromstring(data)
        
        # Define namespace
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        # Find all entries
        entries = root.findall('atom:entry', ns)
        
        print(f"Found {len(entries)} papers\n")
        
        # Download each PDF
        for idx, entry in enumerate(entries, 1):
            # Get paper ID
            paper_id = entry.find('atom:id', ns).text.split('/abs/')[-1]
            
            # Get title
            title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
            
            # Get PDF link
            pdf_link = None
            for link in entry.findall('atom:link', ns):
                if '/pdf/' in link.get('href'):
                    pdf_link = link.get('href')
                    break
            
            if pdf_link:
                # Create safe filename
                safe_title = "".join(c for c in title[:50] if c.isalnum() or c in (' ', '-', '_')).strip()
                filename = f"{paper_id.replace('/', '_')}_{safe_title}.pdf"
                filepath = pdf_dir / filename
                
                print(f"[{idx}/{len(entries)}] Downloading: {title[:80]}...")
                print(f"    ID: {paper_id}")
                print(f"    Saving to: {filename}")
                
                try:
                    urllib.request.urlretrieve(pdf_link, filepath)
                    print(f"    ✓ Downloaded successfully\n")
                except Exception as e:
                    print(f"    ✗ Error downloading: {e}\n")
                
                # Be nice to the arXiv API - add delay between downloads
                if idx < len(entries):
                    time.sleep(3)  # 3 second delay between requests
            else:
                print(f"[{idx}/{len(entries)}] No PDF link found for: {title[:80]}\n")
        
        print(f"\nDownload complete! PDFs saved to: {pdf_dir.absolute()}")
        print(f"Successfully processed {len(entries)} papers")
        
    except Exception as e:
        print(f"Error fetching papers: {e}")


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Open a PDF and extract all text as a single string.
    """
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        page_text = page.get_text()  # get raw text from page
        # (Optional) clean page_text here (remove headers/footers)
        pages.append(page_text)
    full_text = "\n".join(pages)
    
    return full_text


def chunk_text(text: str, max_tokens: int = 512, overlap: int = 50) -> list[str]:
    tokens = text.split()
    chunks = []
    step = max_tokens - overlap
    for i in range(0, len(tokens), step):
        chunk = tokens[i:i + max_tokens]
        chunks.append(" ".join(chunk))
    return chunks

model = SentenceTransformer('all-MiniLM-L6-v2')
    #all-mpnet-base-v2
    #paraphrase-MiniLM-L6-v2
def embedding(chunks: list[str]):
    embeddings = model.encode(chunks)  # embeds each text chunk into a 384-d vecto
    return embeddings

def faiss_indexing(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)  # using a simple L2 index
    index.add(np.array(embeddings))  # add all chunk vectors
    return index

def query(faiss_index, query_text):
    query_embedding = model.encode(query_text, normalize_embeddings=True)
    k = 3
    distances, indices = faiss_index.search(np.array([query_embedding]), k)
    return indices[0]