import os
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from collections import deque
from dotenv import load_dotenv
from langchain_classic.document_loaders import PyPDFLoader, TextLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.vectorstores import FAISS
from langchain_classic.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_classic.llms import HuggingFacePipeline

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Resume Chat Bot", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store conversation history (last 10 messages per session)
# In production, use a proper database and session management
conversation_histories: Dict[str, deque] = {}
MAX_HISTORY_SIZE = 10

# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatResponse(BaseModel):
    response: str
    history: List[ChatMessage]

# Initialize the RAG system
print("Loading documents and initializing RAG system...")

# 1. Load documents
docs = []
resume_hub_path = "resume hub"
if os.path.exists(resume_hub_path):
    for file in os.listdir(resume_hub_path):
        file_path = os.path.join(resume_hub_path, file)
        if file.endswith(".pdf"):
            try:
                resume = PyPDFLoader(file_path).load()
                docs.extend(resume)
            except Exception as e:
                print(f"Error loading PDF {file}: {e}")
        elif file.endswith(".txt"):
            try:
                extras = TextLoader(file_path).load()
                docs.extend(extras)
            except Exception as e:
                print(f"Error loading TXT {file}: {e}")

if not docs:
    print("Warning: No documents loaded. Make sure 'resume hub' directory exists and contains PDF/TXT files.")


# 2. Split documents into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=30)
chunks = splitter.split_documents(docs) if docs else []

# Save chunks as JSONL with 50 examples
import json
chunks_count = len(chunks)
print(f"Total chunks created: {chunks_count}")
with open("chunks examples/chunks_examples.json", "w") as f:
    for i in range(chunks_count):
        chunk = {"chunk_index": i, "chunk_text": chunks[i].page_content}
        json.dump(chunk, f)
        f.write("\n")


# 3. Create embeddings and vector store
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

if chunks:
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
else:
    retriever = None
    print("Warning: No chunks created, retriever is None")

# 4. Initialize LLM
print("Loading language model...")
model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=2048
)

llm = HuggingFacePipeline(pipeline=pipe)

# 5. Create Retrieval QA chain
if retriever:
    agent = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
else:
    agent = None

print("RAG system initialized successfully!")

# Helper functions
def get_conversation_history(session_id: str) -> deque:
    """Get or create conversation history for a session"""
    if session_id not in conversation_histories:
        conversation_histories[session_id] = deque(maxlen=MAX_HISTORY_SIZE)
    return conversation_histories[session_id]

def add_message_to_history(session_id: str, role: str, content: str):
    """Add a message to the conversation history"""
    history = get_conversation_history(session_id)
    history.append(ChatMessage(role=role, content=content))

def format_history_for_context(history: deque) -> str:
    """Format conversation history as context for the LLM"""
    if not history:
        return ""
    
    context = "\n\nPrevious conversation:\n"
    for msg in history:
        context += f"{msg.role.capitalize()}: {msg.content}\n"
    return context

# API Routes
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the chat interface"""
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat requests"""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    if not agent:
        raise HTTPException(status_code=500, detail="RAG system not properly initialized")
    
    # Get conversation history
    history = get_conversation_history(request.session_id)
    
    # Add user message to history
    add_message_to_history(request.session_id, "user", request.message)
    
    try:
        # Build context with conversation history
        context = format_history_for_context(history)
        query_with_context = f"{context}\n\nCurrent question: {request.message}"
        
        # Get response from RAG agent
        response = agent.run(query_with_context)
        
        # Add assistant response to history
        add_message_to_history(request.session_id, "assistant", response)
        
        # Get updated history
        updated_history = list(get_conversation_history(request.session_id))
        
        return ChatResponse(
            response=response,
            history=updated_history
        )
    
    except Exception as e:
        print(f"Error processing chat: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/history/{session_id}", response_model=List[ChatMessage])
async def get_history(session_id: str):
    """Get conversation history for a session"""
    history = get_conversation_history(session_id)
    return list(history)

@app.delete("/history/{session_id}")
async def clear_history(session_id: str):
    """Clear conversation history for a session"""
    if session_id in conversation_histories:
        conversation_histories[session_id].clear()
    return {"message": "History cleared successfully"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "rag_initialized": agent is not None,
        "documents_loaded": len(docs) if docs else 0,
        "chunks_created": len(chunks) if chunks else 0
    }

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
