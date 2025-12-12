# Week 4 Homework Submission – READER  
**MLE in GenAI – Week 4: RAG System (PDF → Chunks → Embeddings → FAISS → API)**  
**Author:** Wei Yang  
**Notebook:** `Wei_Yang_week4_submission.ipynb`

---

## 🔍 Overview

This Week 4 homework implements a complete **Retrieval-Augmented Generation (RAG)** pipeline.  
The system processes a PDF, converts it into text chunks, embeds them, stores them in a FAISS vector index, and exposes a retrieval API using FastAPI.

This assignment forms the foundation for Week 5’s hybrid search and RRF re-ranking extensions.

---

## 🧩 Homework Objectives

The notebook fulfills the core Week 4 requirements:

---

### ✔️ 1. PDF Loading and Text Extraction

- Uses **PyMuPDF (fitz)** to load a PDF document.
- Extracts text page by page.
- Stores extracted text in Python structures for later chunking.

---

### ✔️ 2. Chunking Strategy

- Implements text chunking using sliding windows or fixed-size segments.
- Ensures overlapping chunks for better semantic coherence.
- Stores chunks alongside metadata such as:
  - page number
  - chunk index
  - source filename

---

### ✔️ 3. Embedding Generation

- Uses a **SentenceTransformer** model to generate dense embedding vectors.
- Converts each chunk into an embedding.
- Verifies embedding dimension and vector structure.

---

### ✔️ 4. Vector Store: FAISS Index

- Builds a **FAISS IndexFlatL2** or similar index type.
- Adds all embedding vectors to the index for similarity search.
- Saves the index and metadata to disk:
  - `faiss.index`
  - `chunks.pkl`
  - `meta.pkl`

This allows the RAG system to be reused later without reprocessing the entire PDF.

---

### ✔️ 5. Search Pipeline

- Performs similarity search in FAISS.
- Retrieves top-k relevant chunks for a given query.
- Combines FAISS distances with metadata to produce readable results.

---

### ✔️ 6. FastAPI Endpoints

Implements a small API with endpoints such as:

- `/search` – returns top-k FAISS search results
- `/ask` – (optional) query + retrieval + LLM answer synthesis
- (Optional) JSON formatting helpers

This enables programmatic access to the RAG system outside the notebook.

---

## 📂 Notebook Structure

| Section | Description |
|--------|-------------|
| **PDF Loading** | Extract text from PDF using PyMuPDF |
| **Chunking** | Split text into overlapping chunks |
| **Embeddings** | Embed chunks using SentenceTransformer |
| **FAISS Indexing** | Build and store vector database |
| **Search** | Retrieve relevant chunks from FAISS |
| **FastAPI** | Define simple API endpoints |
| **Testing** | Run example queries to validate pipeline |

---

## 🛠️ How to Run This Notebook

1. **Install Dependencies** (inside your course environment)

   ```bash
   pip install sentence-transformers faiss-cpu pymupdf fastapi uvicorn

Activate Environment

conda activate mle_genai


Launch Jupyter

jupyter notebook


Run all cells in Wei_Yang_week4_submission.ipynb.

✅ Results

Running this notebook successfully demonstrates:

End-to-end RAG pipeline working locally

Ability to generate embeddings and perform similarity search

FAISS vector index creation and persistence

Integration via FastAPI for external queries

Preparation for Week 5 hybrid retrieval (FAISS + SQL + RRF)

This completes all Week 4 requirements and lays the groundwork for advanced retrieval pipelines in Week 5.
