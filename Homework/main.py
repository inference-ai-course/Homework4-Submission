import fitz  # PyMuPDF
import requests
import faiss
import numpy as np
import requests
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer
from typing import List
import uvicorn
from fastapi import FastAPI
import numpy as np
import os

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Open a PDF and extract all text as a single string.
    """
    if pdf_path.startswith("http://") or pdf_path.startswith("https://"):
        response = requests.get(pdf_path)
        doc = fitz.open(stream=response.content, filetype="pdf")
    else:
        doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        page_text = page.get_text()  # get raw text from page
        # (Optional) clean page_text here (remove headers/footers)
        pages.append(page_text)
    full_text = "\n".join(pages)
    return full_text

def chunk_text(text: str, max_tokens: int = 500, overlap: int = 50) -> List[str]:
    tokens = text.split()
    chunks = []
    step = max_tokens - overlap
    for i in range(0, len(tokens), step):
        chunk = tokens[i:i + max_tokens]
        chunks.append(" ".join(chunk))
    return chunks

# load pdfs and extract text
def query_arxiv(category, max_results=50):  # Limited to 50 for size; change to 200 if needed
    url = f"http://export.arxiv.org/api/query?search_query=cat:{category}&sortBy=submittedDate&sortOrder=descending&max_results={max_results}"
    response = requests.get(url)
    root = ET.fromstring(response.content)
    entries = []
    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
        pdf_url = entry.find('{http://www.w3.org/2005/Atom}id').text.replace('abs', 'pdf') + '.pdf'
        paper = {
            'pdf_url': pdf_url,
            'title': entry.find('{http://www.w3.org/2005/Atom}title').text.strip(),
        }
        entries.append(paper)
    return entries

# Embed chunks using SentenceTransformer
def embed_chunks(chunks: List[str], model: SentenceTransformer) -> List[List[float]]:
     
     embeddings = model.encode(chunks)
     return embeddings

# Assume embeddings is a 2D numpy array of shape (num_chunks, dim)
def create_faiss_index(embeddings: List[float]) -> faiss.Index: 
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)  # using a simple L2 index
    index.add(np.array(embeddings))  # add all chunk vectors
    return index

# Initialize model, chunks, and index at module level
# process multiple arxiv papers
arxiv_papers = query_arxiv('cs.AI', max_results=10)  # e.g., most 10 papers from AI research
texts = []
for paper in arxiv_papers:
    text = extract_text_from_pdf(paper['pdf_url'])
    texts.append(text)
full_text = "\n".join(texts)

# setting up embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# chunking and embedding
chunks = chunk_text(full_text)
embeddings = embed_chunks(chunks, model)
index = create_faiss_index(np.array(embeddings))

app = FastAPI()

@app.get("/search")
async def search(q: str):
    """
    Receive a query 'q', embed it, retrieve top-3 passages, and return them.
    """
    # TODO: Embed the query 'q' using your embedding model
    query_vector = model.encode([q])[0]   # e.g., model.encode([q])[0]
    # Perform FAISS search
    k = 3  # top-k results
    distances, indices = index.search(np.array([query_vector]), k)
    # Retrieve the corresponding chunks (assuming 'chunks' list and 'indices' shape [1, k])
    results = []
    for idx in indices[0]:
        results.append(chunks[idx])
    return {"query": q, "results": results}

if __name__ == "__main__":

    # Start FastAPI with Uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "True").lower() == "true"
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )