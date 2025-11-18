# 🧠 ResumeBrain - RAG-Powered Resume Intelligence

A sophisticated RAG (Retrieval-Augmented Generation) system that allows you to upload your resume and ask questions about it using AI. Built with LangChain, Ollama, and ChromaDB.

## ✨ Features

- **📄 PDF Resume Processing**: Upload and extract text from PDF resumes
- **🔍 Intelligent Chunking**: Smart text splitting with 10% overlap for context preservation
- **🎯 Semantic Search**: Find relevant information using BGE embeddings
- **💬 Natural Q&A**: Ask questions in natural language and get accurate answers
- **📚 Source Citations**: Every answer includes source chunks with similarity scores
- **🎨 User-Friendly UI**: Clean Gradio interface for easy interaction
- **⚡ Local & Fast**: Runs entirely on your machine using Ollama

## 🏗️ Architecture

```
ResumeBrain/
├── app.py                      # Main Gradio application
├── config/
│   └── config.yaml            # Configuration settings
├── modules/
│   ├── __init__.py
│   ├── document_loader.py     # PDF upload & text extraction
│   ├── text_processor.py      # Text chunking & preprocessing
│   ├── embedding_engine.py    # Ollama embedding wrapper
│   ├── vector_store.py        # ChromaDB vector store
│   └── retrieval_chain.py     # RAG orchestration with LangChain
├── data/
│   ├── uploads/               # Uploaded resumes
│   └── vector_db/             # ChromaDB persistence
├── logs/                      # Application logs
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 📋 Prerequisites

### 1. Install Ollama

**macOS/Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
Download from [ollama.com](https://ollama.com)

**Verify installation:**
```bash
ollama --version
```

### 2. Download Required Models

```bash
# Download embedding model (BGE-base)
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf

# Download LLM model (Llama 3.2 1B Instruct)
ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
```

**Verify models are downloaded:**
```bash
ollama list
```

You should see both models listed.

## 🚀 Installation

### Step 1: Clone or Create Project Directory

```bash
mkdir ResumeBrain
cd ResumeBrain
```

### Step 2: Create Project Structure

```bash
# Create directories
mkdir -p modules config data/uploads data/vector_db logs

# Create empty __init__.py for modules
touch modules/__init__.py
```

### Step 3: Install Python Dependencies

```bash
# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 4: Configure Application

Edit `config/config.yaml` to customize settings (or use defaults).

## ▶️ Running the Application

### Start Ollama Service

Make sure Ollama is running:

```bash
# Check if Ollama is running
curl http://localhost:11434/api/version

# If not running, start it (usually runs automatically)
ollama serve
```

### Launch ResumeBrain

```bash
python app.py
```

The application will be available at: **http://localhost:7860**

## 📖 How to Use

### 1. Upload Resume
- Click "Upload Resume (PDF)" button
- Select your PDF resume file (max 10MB)
- Click "🚀 Process Resume" button
- Wait for processing to complete

### 2. Ask Questions
Once processing is complete, you can ask questions like:

- "What is the resume owner's name?"
- "What is the domain field and how many years of experience?"
- "What major projects has the candidate contributed to?"
- "List the main technical skills."
- "Which projects included NLP or AI?"

### 3. Review Sources
Each answer includes:
- The AI-generated response
- Source chunks retrieved from the resume
- Similarity scores for each chunk

## ⚙️ Configuration

Edit `config/config.yaml` to customize:

### Model Settings
```yaml
models:
  embedding_model: "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"
  llm_model: "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"
```

### Chunking Parameters
```yaml
chunking:
  chunk_size: 800  # Characters (~500 tokens)
  chunk_overlap: 80  # 10% overlap
```

### Retrieval Settings
```yaml
retrieval:
  top_k: 3  # Number of chunks to retrieve
  similarity_threshold: 0.5
```

### Generation Settings
```yaml
generation:
  temperature: 0.1  # Lower = more factual
  max_tokens: 512
```

## 🔧 Troubleshooting

### Issue: "Cannot connect to Ollama"

**Solution:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/version

# Restart Ollama service
ollama serve
```

### Issue: "Model not found"

**Solution:**
```bash
# Re-download models
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF

# Verify installation
ollama list
```

### Issue: "PDF extraction failed"

**Solution:**
- Ensure PDF is not password-protected
- Try converting scanned PDFs to text-based PDFs
- Check file size (must be < 10MB)

### Issue: "ChromaDB errors"

**Solution:**
```bash
# Clear vector database
rm -rf data/vector_db/*

# Restart application
python app.py
```

### Issue: "Out of memory"

**Solution:**
- Reduce chunk_size in config.yaml (e.g., 600)
- Reduce top_k (e.g., 2)
- Use smaller LLM model if available

## 📊 Technical Details

### Chunking Strategy
- **Chunk Size**: 800 characters (~500 tokens)
- **Overlap**: 80 characters (10%)
- **Separators**: `["\n\n", "\n", ". ", " ", ""]`

### Embedding Model
- **Model**: BGE-base-en-v1.5 (GGUF)
- **Dimension**: 768
- **Type**: Dense embeddings optimized for retrieval

### LLM Model
- **Model**: Llama 3.2 1B Instruct (GGUF)
- **Size**: 1B parameters
- **Type**: Instruction-tuned for Q&A

### Vector Database
- **Database**: ChromaDB
- **Similarity**: Cosine similarity
- **Persistence**: Local disk storage

## 🎯 Performance Metrics

Expected performance on typical hardware:

| Operation | Time | Notes |
|-----------|------|-------|
| Resume Upload | < 1s | Depends on file size |
| Text Extraction | 1-3s | 5-page PDF |
| Embedding Generation | 2-5s | 15 chunks |
| Vector Indexing | < 1s | ChromaDB |
| Query + Answer | 3-7s | Including retrieval + generation |

## 🔐 Privacy & Security

- **100% Local**: All processing happens on your machine
- **No Cloud Calls**: No data sent to external services
- **Data Storage**: Resumes stored locally in `data/uploads/`
- **Vector DB**: Embeddings stored locally in `data/vector_db/`

## 🛠️ Development

### Project Structure Explained

```
modules/
├── document_loader.py      # Handles PDF upload, validation, text extraction
├── text_processor.py       # Chunks text using LangChain's RecursiveCharacterTextSplitter
├── embedding_engine.py     # Wraps Ollama embeddings API
├── vector_store.py         # Manages ChromaDB operations
└── retrieval_chain.py      # Orchestrates RAG pipeline with LangChain
```

### Adding New Features

**To add a new embedding model:**
1. Pull model with Ollama: `ollama pull <model-name>`
2. Update `config/config.yaml`:
   ```yaml
   models:
     embedding_model: "<model-name>"
   ```

**To add a new LLM:**
1. Pull model with Ollama: `ollama pull <model-name>`
2. Update `config/config.yaml`:
   ```yaml
   models:
     llm_model: "<model-name>"
   ```

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests (when test suite is added)
pytest tests/ -v
```

## 📚 Dependencies

Key dependencies:
- **langchain**: RAG orchestration framework
- **langchain-community**: Community integrations
- **ollama**: Python client for Ollama
- **chromadb**: Vector database
- **pdfplumber**: PDF text extraction
- **gradio**: Web UI framework
- **loguru**: Advanced logging

See `requirements.txt` for complete list.

## 🐛 Known Issues

1. **Large PDFs**: Files > 50 pages may take longer to process
2. **.page Files**: Not yet supported (PDF only for MVP)
3. **Scanned PDFs**: OCR not implemented yet
4. **Multi-language**: Optimized for English resumes

## 🗺️ Roadmap

- [ ] Add .page file support
- [ ] Implement OCR for scanned PDFs
- [ ] Add evaluation metrics dashboard
- [ ] Support multi-document Q&A
- [ ] Add conversation history
- [ ] Implement re-ranking
- [ ] Add GPU acceleration toggle
- [ ] Export Q&A sessions

## 🤝 Contributing

This is a learning project. Feel free to:
- Report bugs
- Suggest features
- Submit improvements

## 📄 License

This project is for educational purposes.

## 🙏 Acknowledgments

- **LangChain**: RAG framework
- **Ollama**: Local model inference
- **ChromaDB**: Vector database
- **Hugging Face**: Model hosting
- **Gradio**: UI framework

## 📞 Support

For issues or questions:
1. Check Troubleshooting section
2. Review logs in `logs/resumebrain.log`
3. Verify Ollama is running and models are downloaded

---

**Built with ❤️ for Week 4 of ML Engineer in Generative AI Era**
