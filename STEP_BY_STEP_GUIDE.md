# Week 4 RAG Homework - Step-by-Step Guide

This guide walks you through completing the Week 4 RAG homework assignment step by step.

## Overview

You'll build a Retrieval-Augmented Generation (RAG) system using arXiv cs.CL papers. The system will:
1. Download and process 50 research papers
2. Extract text and chunk it into searchable segments
3. Generate embeddings and build a FAISS index
4. Provide a FastAPI service for semantic search

## Prerequisites

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Packages
- `fastapi` - Web framework for the API
- `uvicorn` - ASGI server
- `sentence-transformers` - For generating embeddings
- `faiss-cpu` - For similarity search
- `pymupdf` - For PDF text extraction
- `numpy` - Numerical operations

---

## Step 1: Data Collection (Download arXiv Papers)

### What to do:
Run the data collection cells in `src/assignment.ipynb` to download 50 arXiv cs.CL papers.

### Code location:
- `src/utils/arxiv_search.py` - Contains functions for searching and downloading papers
- `src/utils/data_collection.py` - Helper functions for organizing data

### In the notebook:

```python
from utils.arxiv_search import build_arxiv_search_url, run_scraper_in_background
from utils.arxiv_search import scrape_arxiv_details_from_json_threaded
from utils.arxiv_search import save_arxiv_scraped_details, get_pdf_arxiv

# Build search URL for cs.CL papers
url = build_arxiv_search_url(query="cat:cs.CL", size=50)

# Set up directories
scrapped_data_dir = "data/scrapped"
os.makedirs(scrapped_data_dir, exist_ok=True)

# Download paper metadata
target_file_path = f"{scrapped_data_dir}/research_paper.json"
run_scraper_in_background(url=url, output_file=target_file_path)

# Scrape detailed information
scraped_data = scrape_arxiv_details_from_json_threaded(target_file_path)

# Save cleaned data
output_file = f"{scrapped_data_dir}/scrapped_clean.json"
save_arxiv_scraped_details(scraped_data, output_file)

# Download PDFs
pdf_dir = "data/downloads/pdf"
get_pdf_arxiv(cleaned_json=output_file, save_dir=pdf_dir)
```

### Expected output:
- PDFs saved in `src/data/downloads/pdf/`
- Metadata saved in `src/data/scrapped/`

---

## Step 2: Text Extraction from PDFs

### What to do:
Extract text from all downloaded PDFs using PyMuPDF.

### Code location:
- `src/utils/text_extraction.py` - Contains `extract_text_from_pdf_threaded()`

### In the notebook:

```python
from pathlib import Path
from utils.text_extraction import extract_text_from_pdf_threaded

# Find all PDF files
pdf_download_dir = Path(pdf_dir)
pdf_files = [str(f) for f in pdf_download_dir.glob("*.pdf")]

# Extract text from each PDF
documents = []
document_metadata = []

for pdf_file in pdf_files:
    text = extract_text_from_pdf_threaded(pdf_file, max_workers=4)
    documents.append(text)
    
    # Store metadata
    paper_id = Path(pdf_file).stem
    document_metadata.append({
        'paper_id': paper_id,
        'file_path': pdf_file
    })
```

### Expected output:
- List of extracted text documents
- Corresponding metadata for each document

---

## Step 3: Text Chunking

### What to do:
Split each document into smaller chunks (≤512 tokens) with overlap for better retrieval.

### Code location:
- `src/utils/text_chunking.py` - Contains `chunk_text()`

### In the notebook:

```python
from utils.text_chunking import chunk_text

# Chunk all documents
all_chunks = []
chunk_metadata = []

for i, (doc, meta) in enumerate(zip(documents, document_metadata)):
    # Chunk with 512 token max and 50 token overlap
    doc_chunks = chunk_text(doc, max_tokens=512, overlap=50)
    
    # Add chunks
    all_chunks.extend(doc_chunks)
    
    # Add metadata for each chunk
    for chunk_idx, chunk in enumerate(doc_chunks):
        chunk_metadata.append({
            'paper_id': meta['paper_id'],
            'file_path': meta['file_path'],
            'chunk_index': chunk_idx,
            'total_chunks': len(doc_chunks)
        })

print(f"Total chunks created: {len(all_chunks)}")
```

### Expected output:
- List of text chunks (typically 1000-3000 chunks from 50 papers)
- Metadata for each chunk

### Why chunking matters:
- Smaller chunks (250-512 tokens) give more precise retrieval
- Overlap ensures context isn't lost at boundaries
- Each chunk becomes a searchable unit

---

## Step 4: Embedding Generation

### What to do:
Generate dense vector embeddings for all chunks using sentence-transformers.

### Code location:
- `src/utils/embedding_generation.py` - Contains `get_embedding()`

### In the notebook:

```python
from utils.embedding_generation import get_embedding
import numpy as np

# Generate embeddings for all chunks
embeddings = get_embedding(all_chunks)

print(f"Generated embeddings with shape: {embeddings.shape}")
print(f"Embedding dimension: {embeddings.shape[1]}")
```

### Expected output:
- NumPy array of shape `(num_chunks, 384)`
- 384 is the dimension for the `all-MiniLM-L6-v2` model

### What's happening:
- Each text chunk is converted to a 384-dimensional vector
- Similar chunks will have similar vectors (measured by distance)
- This enables semantic search

---

## Step 5: FAISS Indexing

### What to do:
Build a FAISS index for efficient similarity search.

### Code location:
- `src/utils/faiss_indexing.py` - Contains `FAISSIndex` class

### In the notebook:

```python
from utils.faiss_indexing import FAISSIndex

# Create FAISS index
dimension = embeddings.shape[1]
faiss_index = FAISSIndex(dimension=dimension)

# Add embeddings to index
faiss_index.add_embeddings(embeddings, all_chunks, chunk_metadata)

# Get statistics
stats = faiss_index.get_stats()
print(f"Index Statistics: {stats}")

# Save the index
index_path = "data/faiss_index.index"
chunks_path = "data/faiss_chunks.pkl"
faiss_index.save(index_path, chunks_path)
```

### Expected output:
- FAISS index file: `src/data/faiss_index.index`
- Chunks pickle file: `src/data/faiss_chunks.pkl`

### What's FAISS?
- Facebook AI Similarity Search
- Efficiently finds nearest neighbors in high-dimensional space
- IndexFlatL2 uses L2 (Euclidean) distance

---

## Step 6: Query and Retrieval Demo

### What to do:
Test the system with sample queries in the notebook.

### In the notebook:

```python
from utils.embedding_generation import get_single_embedding

def search_query(query: str, k: int = 3):
    """Search for relevant passages given a query."""
    print(f"\nQuery: {query}\n")
    
    # Generate query embedding
    query_embedding = get_single_embedding(query)
    
    # Search
    distances, indices, chunks, metadata = faiss_index.search(query_embedding, k=k)
    
    # Display results
    for i, (dist, idx, chunk, meta) in enumerate(zip(distances[0], indices[0], chunks, metadata)):
        print(f"Result {i+1} (Distance: {dist:.4f})")
        print(f"Paper ID: {meta.get('paper_id', 'N/A')}")
        print(f"Text: {chunk[:300]}...\n")
    
    return distances, indices, chunks, metadata

# Test with example queries
search_query("What are transformer architectures?", k=3)
search_query("How does attention mechanism work?", k=3)
search_query("What are large language models?", k=3)
```

### Expected output:
- Top-3 most relevant chunks for each query
- Distance scores (lower = more similar)
- Paper IDs and chunk metadata

---

## Step 7: FastAPI Service

### What to do:
Run the FastAPI service to provide a REST API for searching.

### Code location:
- `src/main.py` - FastAPI application

### Start the service:

```bash
cd src
python main.py
```

Or using uvicorn directly:

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### Test the API:

**Health check:**
```bash
curl http://localhost:8000/health
```

**Get statistics:**
```bash
curl http://localhost:8000/stats
```

**Search query:**
```bash
curl "http://localhost:8000/search?q=transformer%20architecture&k=3"
```

### API Endpoints:

1. **GET /** - Root endpoint with API documentation
2. **GET /health** - Health check
3. **GET /stats** - Index statistics
4. **GET /search?q={query}&k={num_results}** - Search for relevant passages

### Expected response format:

```json
{
  "query": "transformer architecture",
  "results": [
    {
      "rank": 1,
      "chunk_id": 42,
      "distance": 0.8234,
      "text": "Transformers are a type of neural network...",
      "metadata": {
        "paper_id": "2511.04886",
        "chunk_index": 5,
        "total_chunks": 120
      }
    },
    ...
  ],
  "num_results": 3
}
```

---

## Step 8: Create Retrieval Report

### What to do:
Document your system's performance with 5 example queries.

### Create a file: `RETRIEVAL_REPORT.md`

Include:
1. **Query 1-5**: Different types of questions
2. **Top-3 results for each**: Show the retrieved passages
3. **Analysis**: Comment on retrieval quality
4. **Observations**: What works well? What could be improved?

### Example structure:

```markdown
# Retrieval Report

## Query 1: "What are transformer architectures?"

### Result 1 (Distance: 0.7234)
- Paper ID: 2511.04886
- Text: "Transformers are neural network architectures..."

### Result 2 (Distance: 0.8123)
...

## Analysis
The system successfully retrieved relevant passages about transformers...
```

---

## Deliverables Checklist

- [ ] **Code**: Complete notebook with all steps (`src/assignment.ipynb`)
- [ ] **FAISS Index**: Saved index files (`src/data/faiss_index.index`, `src/data/faiss_chunks.pkl`)
- [ ] **FastAPI Service**: Working API (`src/main.py`)
- [ ] **Retrieval Report**: Document with 5 example queries and results
- [ ] **README**: Instructions for running the project

---

## Tips for Success

### Experiment with Parameters:
- Try different chunk sizes (256, 512, 1024 tokens)
- Adjust overlap (25, 50, 100 tokens)
- Test different embedding models (`all-mpnet-base-v2`, `paraphrase-MiniLM-L6-v2`)

### Improve Retrieval:
- Add reranking with a cross-encoder model
- Filter by paper metadata (year, authors)
- Combine with keyword search (hybrid retrieval)

### Debug Common Issues:

**Issue**: "FAISS index not loaded"
- **Solution**: Make sure you've run the notebook to create the index files first

**Issue**: "Embedding dimension mismatch"
- **Solution**: Ensure you're using the same model for queries as you used for indexing

**Issue**: "Out of memory"
- **Solution**: Process documents in batches, reduce chunk size, or use a smaller embedding model

---

## Project Structure

```
Homework4-Submission/
├── src/
│   ├── assignment.ipynb          # Main notebook
│   ├── main.py                   # FastAPI service
│   ├── data/
│   │   ├── downloads/pdf/        # Downloaded PDFs
│   │   ├── scrapped/             # Paper metadata
│   │   ├── faiss_index.index     # FAISS index
│   │   └── faiss_chunks.pkl      # Chunks and metadata
│   └── utils/
│       ├── arxiv_search.py       # arXiv API utilities
│       ├── text_extraction.py    # PDF text extraction
│       ├── text_chunking.py      # Text chunking
│       ├── embedding_generation.py # Embedding generation
│       └── faiss_indexing.py     # FAISS operations
├── requirements.txt              # Python dependencies
├── README.md                     # Project overview
└── RETRIEVAL_REPORT.md          # Performance report

```

---

## Next Steps (Week 5)

In Week 5, you'll evolve this into a hybrid database system:
- Add keyword search (BM25)
- Implement metadata filtering
- Build a more sophisticated reranking system
- Create a full-featured research assistant

---

## Resources

- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [arXiv API](https://arxiv.org/help/api)

---

## Questions?

If you encounter issues:
1. Check the error messages carefully
2. Verify all dependencies are installed
3. Ensure the FAISS index is built before running the API
4. Review the example code in this guide
