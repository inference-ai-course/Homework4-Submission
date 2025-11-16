# Resume Chat Bot

A full-stack chatbot application with FastAPI backend and interactive web frontend that answers questions about resume and portfolio information using RAG (Retrieval Augmented Generation).

## Features

- 🤖 **AI-Powered Chat**: Uses Qwen2-0.5B-Instruct model with RAG for intelligent responses
- 💬 **Conversation History**: Maintains last 10 messages per session for context-aware conversations
- 🎨 **Modern UI**: Responsive chat interface with gradient design
- 📄 **Document Processing**: Supports PDF and TXT file processing from resume hub
- 🔍 **Semantic Search**: FAISS vector store with sentence transformers for retrieval
- 🚀 **FastAPI Backend**: RESTful API with CORS support

## Project Structure

```
ChatBotApp/
├── app.py                 # FastAPI backend server
├── main.py               # Original RAG implementation
├── static/
│   └── index.html        # Chat interface frontend
├── resume hub/           # Directory for PDF/TXT documents
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Prerequisites

- Python 3.8+
- pip

## Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd ChatBotApp
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare your documents**:
   - Place PDF and TXT files in the `resume hub` directory
   - The application will automatically load and process these documents

5. **Create .env file** (optional):
   ```bash
   # Add any environment variables if needed
   ```

## Usage

### Starting the Server

Run the FastAPI server:
```bash
python app.py
```

Or use uvicorn directly:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The server will start at `http://localhost:8000`

### Accessing the Chat Interface

Open your browser and navigate to:
```
http://localhost:8000
```

### Using the API

#### Send a chat message:
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the key skills?",
    "session_id": "user123"
  }'
```

#### Get conversation history:
```bash
curl "http://localhost:8000/history/user123"
```

#### Clear conversation history:
```bash
curl -X DELETE "http://localhost:8000/history/user123"
```

#### Health check:
```bash
curl "http://localhost:8000/health"
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Serve chat interface |
| POST | `/chat` | Send message and get response |
| GET | `/history/{session_id}` | Get conversation history |
| DELETE | `/history/{session_id}` | Clear conversation history |
| GET | `/health` | Health check endpoint |

## Features in Detail

### Conversation History
- Automatically maintains last 10 messages per session
- Uses session IDs to manage multiple user conversations
- History is included as context for more coherent responses

### RAG System
- **Document Loading**: Supports PDF and TXT files
- **Text Splitting**: Chunks documents with 100 character size and 10 character overlap
- **Embeddings**: Uses sentence-transformers/all-MiniLM-L6-v2
- **Vector Store**: FAISS for efficient similarity search
- **LLM**: Qwen2-0.5B-Instruct for response generation

### Frontend Features
- Real-time chat interface
- Message animations
- Loading indicators
- Error handling
- Session persistence (per browser session)
- Clear chat functionality
- Responsive design

## Configuration

### Modify Model Settings

Edit `app.py` to change:
- Model name: `model_name = "Qwen/Qwen2-0.5B-Instruct"`
- Chunk size: `chunk_size=100`
- History size: `MAX_HISTORY_SIZE = 10`
- Retriever k value: `search_kwargs={"k": 3}`

### Modify Server Settings

Change host/port in `app.py`:
```python
uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Troubleshooting

### No documents loaded
- Ensure `resume hub` directory exists
- Verify PDF/TXT files are in the directory
- Check file permissions

### Model loading issues
- Ensure you have sufficient RAM (2GB+ for Qwen2-0.5B)
- Check internet connection for first-time model download
- Models are cached in `~/.cache/huggingface/`

### Port already in use
- Change the port in `app.py` or kill the process using port 8000:
  ```bash
  lsof -ti:8000 | xargs kill -9
  ```

## Development

To modify the frontend:
1. Edit `static/index.html`
2. Refresh browser (no restart needed)

To modify the backend:
1. Edit `app.py`
2. Server auto-reloads if using `--reload` flag

## License

MIT License

## Acknowledgments

- FastAPI for the backend framework
- LangChain for RAG implementation
- Hugging Face for models and embeddings
- FAISS for vector search
