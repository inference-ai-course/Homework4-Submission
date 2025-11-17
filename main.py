# main.py
import os
import json
import pickle
from typing import List

from fastapi import FastAPI, Query
from pydantic import BaseModel

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# =========================================
# 1. CONFIG
# =========================================
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

CHUNKS_JSON = os.path.join(DATA_DIR, "chunks.json")
FAISS_INDEX = os.path.join(DATA_DIR, "faiss.index")
EMB_DIM = 384  # MiniLM embedding dimension
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# =========================================
# 2. FastAPI APP
# =========================================
app = FastAPI(title="Paper Retrieval System")

model = SentenceTransformer(MODEL_NAME)


# =========================================
# 3. Helper — chunking text
# =========================================
def split_into_chunks(text: str, chunk_size=150, overlap=30) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks


# =========================================
# 4. Build dataset (simulate 50 chunks)
# =========================================
def build_dataset():
    fake_text = """
    Deep learning has transformed natural language processing.
    Attention mechanisms allow models to capture long-range dependencies.
    Large Language Models (LLMs) such as GPT, LLaMA, and Mistral achieve state-of-the-art results.
    Knowledge retrieval is essential for grounding model outputs.
    Vector databases like FAISS accelerate similarity search.
    """

    chunks = split_into_chunks(fake_text * 10)[:50]  # 50 chunks

    with open(CHUNKS_JSON, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"✔ Saved {len(chunks)} chunks to {CHUNKS_JSON}")
    return chunks


# =========================================
# 5. Build FAISS index
# =========================================
def build_faiss_index():
    if not os.path.exists(CHUNKS_JSON):
        chunks = build_dataset()
    else:
        chunks = json.load(open(CHUNKS_JSON, "r", encoding="utf-8"))

    embeddings = model.encode(chunks)
    embeddings = embeddings.astype("float32")

    index = faiss.IndexFlatL2(EMB_DIM)
    index.add(embeddings)

    faiss.write_index(index, FAISS_INDEX)
    print(f"✔ FAISS index saved to {FAISS_INDEX}")
    return index, chunks


# Load index on startup
@app.on_event("startup")
def load_index():
    global index, chunks

    if not os.path.exists(FAISS_INDEX):
        print("Index not found → building...")
        index, chunks = build_faiss_index()
    else:
        print("Loading FAISS index...")
        index = faiss.read_index(FAISS_INDEX)
        chunks = json.load(open(CHUNKS_JSON, "r", encoding="utf-8"))


# =========================================
# 6. Retrieval API
# =========================================
class RetrievalResult(BaseModel):
    query: str
    top_passages: List[str]


@app.get("/search", response_model=RetrievalResult)
def search(query: str = Query(..., description="Search query")):
    q_emb = model.encode([query]).astype("float32")
    D, I = index.search(q_emb, 3)  # top 3

    top_passages = [chunks[i] for i in I[0]]
    return RetrievalResult(query=query, top_passages=top_passages)


# =========================================
# 7. Generate Retrieval Report
# =========================================
@app.get("/report")
def generate_report():
    test_queries = [
        "What is deep learning?",
        "Explain attention mechanism",
        "What are LLMs?",
        "Why do we need retrieval?",
        "What is FAISS used for?"
    ]

    report_lines = []

    for q in test_queries:
        q_emb = model.encode([q]).astype("float32")
        D, I = index.search(q_emb, 3)

        report_lines.append(f"===== QUERY: {q} =====")
        for rank, idx in enumerate(I[0]):
            report_lines.append(f"Top {rank+1}: {chunks[idx]}")
        report_lines.append("\n")

    report_path = os.path.join(DATA_DIR, "retrieval_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    return {
        "message": "Report generated",
        "file": report_path
    }
