from sentence_transformers import SentenceTransformer
import json
import pickle
from fastapi import FastAPI
import numpy as np

app = FastAPI()

model = SentenceTransformer("all-MiniLM-L6-v2")

with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

with open("index.pkl", "rb") as f:
    faiss_index = pickle.load(f)


@app.get("/search")
async def search(q: str):
    """
    Receive a query 'q', embed it, retrieve top-3 passages, and return them.
    """
    # TODO: Embed the query 'q' using your embedding model
    query_vector = model.encode([q])[0]
    # Perform FAISS search
    k = 3
    distances, indices = faiss_index.search(np.array([query_vector]), k)

    return {
        "request": q,
        "response": [chunks[index] for index in indices[0]]
    }