from sentence_transformers import SentenceTransformer

import fitz  # PyMuPDF

import faiss

from fastapi import FastAPI
import numpy as np

from pathlib import Path

import json

app = FastAPI()

@app.get("/search")
async def search(q: str):
    """
    Receive a query 'q', embed it, retrieve top-3 passages, and return them.
    """
    # TODO: Embed the query 'q' using your embedding model
    query_vector = model.encode([q])[0]
    # Perform FAISS search
    k = 3
    distances, indices = index.search(np.array([query_vector]), k)
    # Retrieve the corresponding chunks (assuming 'chunks' list and 'indices' shape [1, k])
    results = []
    for idx in indices[0]:
        results.append(chunks[idx])
    return {"query": q, "results": results}

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

path = "../class2/pdf/"
list_of_chunks = []
for f in Path(path).iterdir():
    pdf_text = extract_text_from_pdf(f)
    chunks = chunk_text(pdf_text)
    list_of_chunks.extend(chunks)
    print(f"Processed {f}, extracted {len(chunks)} chunks.")

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(list_of_chunks)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, "faiss_index.bin")

with open("chunks.json", "w") as f:
    json.dump(chunks, f)
