# RAG System with arXiv Papers - Instructions

## Overview
This project implements a Retrieval-Augmented Generation (RAG) system that indexes arXiv AI research papers and provides a search endpoint to retrieve relevant passages based on user queries.

## Prerequisites

### Required Python Packages
Install the following dependencies:

```bash
pip install pymupdf requests sentence-transformers faiss-cpu fastapi uvicorn numpy
```

Or if you have a GPU and want to use FAISS with GPU acceleration:
```bash
pip install pymupdf requests sentence-transformers faiss-gpu fastapi uvicorn numpy
```

### Python Version
- Python 3.8 or higher recommended

## Project Structure

```
.
├── main.py              # Main FastAPI application
├── instructions.md      # This file
├── faiss_index.bin      # Generated FAISS index (created on first run)
└── chunks.json          # Generated text chunks (created on first run)
```

## How It Works

### 1. Data Collection
The system queries the arXiv API for recent papers in the `cs.AI` category (Artificial Intelligence).

### 2. Text Extraction
Each PDF is downloaded and processed using PyMuPDF (fitz) to extract raw text from all pages.

### 3. Text Chunking
The extracted text is split into chunks of 500 tokens with a 50-token overlap to maintain context between chunks.

### 4. Embedding Generation
Each chunk is converted to a 384-dimensional vector using the `all-MiniLM-L6-v2` sentence transformer model.

### 5. FAISS Indexing
All embeddings are stored in a FAISS index (`IndexFlatL2`) for fast similarity search.

### 6. Search API
A FastAPI endpoint accepts queries, embeds them, and returns the top-3 most relevant passages.

## Running the Application

### Option 1: Direct Execution
```bash
python main.py
```

### Option 2: With Custom Configuration
You can set environment variables to customize the server:

```bash
HOST=127.0.0.1 PORT=8080 DEBUG=False python main.py
```

### Environment Variables
- `HOST`: Server host address (default: `0.0.0.0`)
- `PORT`: Server port (default: `8000`)
- `DEBUG`: Enable hot-reload during development (default: `True`)

## Using the Search Endpoint

Once the server is running, you can query it in several ways:

### 1. Browser
Navigate to:
```
http://localhost:8000/search?q=What%20are%20the%20latest%20advances%20in%20AI?
```

### 2. cURL
```bash
curl "http://localhost:8000/search?q=What%20are%20the%20latest%20advances%20in%20AI?"
```

### 3. Python Requests
```python
import requests

response = requests.get(
    "http://localhost:8000/search",
    params={"q": "What are the latest advances in AI?"}
)
print(response.json())
```

### 4. Interactive API Documentation
Visit `http://localhost:8000/docs` for an interactive Swagger UI where you can test the API directly.

## Response Format

The `/search` endpoint returns JSON with the following structure:

```json
{
  "query": "What are the latest advances in AI?",
  "results": [
    "First most relevant chunk text...",
    "Second most relevant chunk text...",
    "Third most relevant chunk text..."
  ]
}
```

## Performance Notes

### First Run
- The first time you run `main.py`, it will:
  - Download 10 arXiv papers (can take 1-5 minutes depending on network speed)
  - Extract text from PDFs
  - Generate embeddings (takes 10-30 seconds depending on hardware)
  - Build the FAISS index

### Subsequent Runs
- If you implement persistence (saving the index and chunks), subsequent runs will be much faster
- The system currently rebuilds the index on each startup

## Optimization Tips

### 1. Save and Load Index
To avoid rebuilding the index every time, add this code after creating the index:

```python
import json

# Save index and chunks
faiss.write_index(index, "faiss_index.bin")
with open("chunks.json", "w") as f:
    json.dump(chunks, f)
```

And load them at startup:

```python
import json

# Load index and chunks
if os.path.exists("faiss_index.bin") and os.path.exists("chunks.json"):
    index = faiss.read_index("faiss_index.bin")
    with open("chunks.json", "r") as f:
        chunks = json.load(f)
else:
    # Build index from scratch
    ...
```

### 2. Increase Paper Count
Change `max_results=10` to `max_results=50` in the `query_arxiv()` call to index more papers.

### 3. Adjust Chunk Size
Modify `max_tokens` and `overlap` parameters in `chunk_text()` for different retrieval characteristics:
- Smaller chunks (250 tokens): More precise but less context
- Larger chunks (512 tokens): More context but potentially less precise

### 4. Try Different Models
Replace `'all-MiniLM-L6-v2'` with other models:
- `'all-mpnet-base-v2'`: Better quality but slower
- `'paraphrase-MiniLM-L6-v2'`: Optimized for paraphrase detection

## Troubleshooting

### Port Already in Use
If port 8000 is already occupied:
```bash
PORT=8080 python main.py
```

### Out of Memory
Reduce the number of papers:
```python
arxiv_papers = query_arxiv('cs.AI', max_results=5)
```

### Slow PDF Downloads
arXiv servers can be slow. Consider:
- Reducing `max_results`
- Implementing caching
- Using a download timeout

### Module Import Errors
Ensure all packages are installed:
```bash
pip install pymupdf requests sentence-transformers faiss-cpu fastapi uvicorn numpy
```

## Development Mode

When `DEBUG=True` (default), the server runs with auto-reload enabled. This means any changes to `main.py` will automatically restart the server, but the index will be rebuilt each time.

For production, set `DEBUG=False` to disable auto-reload.

## Next Steps

1. Implement index persistence (save/load FAISS index and chunks)
2. Add metadata filtering (by paper date, author, etc.)
3. Implement reranking with a cross-encoder model
4. Add more query parameters (k for number of results, filters, etc.)
5. Create a web interface for easier interaction
6. Add logging and monitoring

## Contact & Support

For issues or questions about this homework assignment, please refer to the course materials or contact your instructor.
