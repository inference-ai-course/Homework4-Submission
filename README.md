# Week 4: Retrieval-Augmented Generation (RAG) with arXiv Papers

A complete RAG pipeline for semantic search over scientific research papers from arXiv.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Notebook
Open and run `src/assignment.ipynb` to:
- Download 50 arXiv cs.CL papers
- Extract text from PDFs
- Chunk documents
- Generate embeddings
- Build FAISS index
- Test retrieval with sample queries

### 3. Start the API Service
```bash
cd src
python main.py
```

Or using uvicorn:
```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Test the API
```bash
# Health check
curl http://localhost:8000/health

# Search query
curl "http://localhost:8000/search?q=transformer%20architecture&k=3"
```

## 📋 Project Structure

```
Homework4-Submission/
├── src/
│   ├── assignment.ipynb          # Main notebook with complete pipeline
│   ├── main.py                   # FastAPI service
│   ├── data/
│   │   ├── downloads/pdf/        # Downloaded PDFs (20 papers)
│   │   ├── scrapped/             # Paper metadata
│   │   ├── faiss_index.index     # FAISS index (generated)
│   │   └── faiss_chunks.pkl      # Chunks and metadata (generated)
│   └── utils/
│       ├── arxiv_search.py       # arXiv API utilities
│       ├── text_extraction.py    # PDF text extraction
│       ├── text_chunking.py      # Text chunking
│       ├── embedding_generation.py # Embedding generation
│       └── faiss_indexing.py     # FAISS operations
├── requirements.txt              # Python dependencies
├── STEP_BY_STEP_GUIDE.md        # Detailed implementation guide
└── RETRIEVAL_REPORT_TEMPLATE.md # Template for performance report
```

## 🔧 Pipeline Components

### 1. Data Collection
- Downloads arXiv cs.CL papers via API
- Saves PDFs and metadata

### 2. Text Extraction
- Extracts text from PDFs using PyMuPDF
- Threaded processing for performance

### 3. Text Chunking
- Splits documents into 512-token chunks
- 50-token overlap for context preservation

### 4. Embedding Generation
- Uses sentence-transformers (all-MiniLM-L6-v2)
- Generates 384-dimensional embeddings

### 5. FAISS Indexing
- Builds L2 distance index
- Enables fast similarity search

### 6. FastAPI Service
- REST API for semantic search
- Returns top-k relevant passages

## 📊 Current Status

- ✅ Data collection utilities implemented
- ✅ PDF text extraction (20 papers downloaded)
- ✅ Text chunking implemented
- ✅ Embedding generation (fixed bug)
- ✅ FAISS indexing utilities created
- ✅ FastAPI service implemented
- ⏳ Need to download 30 more papers (to reach 50)
- ⏳ Need to run complete pipeline in notebook
- ⏳ Need to create retrieval report

## 📝 Deliverables

1. **Code Notebook**: `src/assignment.ipynb` with complete pipeline
2. **FAISS Index**: Generated index files in `src/data/`
3. **FastAPI Service**: `src/main.py` with `/search` endpoint
4. **Retrieval Report**: Performance analysis with 5 example queries

## 🔍 API Endpoints

- `GET /` - API documentation
- `GET /health` - Health check
- `GET /stats` - Index statistics
- `GET /search?q={query}&k={num_results}` - Search for relevant passages

## 📚 Documentation

- **STEP_BY_STEP_GUIDE.md** - Complete implementation guide
- **RETRIEVAL_REPORT_TEMPLATE.md** - Template for performance report
- **Class 4 Homework.ipynb** - Original assignment requirements

## 🛠️ Technologies Used

- **FastAPI** - Web framework
- **sentence-transformers** - Embedding generation
- **FAISS** - Similarity search
- **PyMuPDF** - PDF text extraction
- **NumPy** - Numerical operations

## 🎯 Next Steps

1. Download remaining 30 papers (currently have 20)
2. Run complete pipeline in notebook
3. Generate and save FAISS index
4. Test API with various queries
5. Create retrieval report with 5 example queries
6. Document observations and improvements

## 💡 Tips

- Experiment with different chunk sizes (256, 512, 1024 tokens)
- Try different embedding models (all-mpnet-base-v2, paraphrase-MiniLM-L6-v2)
- Consider adding reranking with cross-encoder models
- Use metadata filtering for better relevance

## 📖 Resources

- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [arXiv API](https://arxiv.org/help/api)

## 📄 License

See LICENSE file for details.
