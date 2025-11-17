# Task 7: FastAPI Service; Production-ready API for the RAG system

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from contextlib import asynccontextmanager

# Global variables for model and index
model = None
index = None
chunks = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load model and index on startup
    """
    global model, index, chunks
    
    print("Loading model and index...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    index = faiss.read_index("faiss_index.bin")
    
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    
    print(f"✓ Model loaded")
    print(f"✓ Index loaded: {index.ntotal} vectors")
    print(f"✓ Chunks loaded: {len(chunks)} chunks")
    
    yield
    
    # Cleanup
    print("Shutting down...")

# Initialize FastAPI app
app = FastAPI(
    title="arXiv cs.CL RAG API",
    description="Retrieval-Augmented Generation API for arXiv Computational Linguistics papers",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response models
class SearchResult(BaseModel):
    chunk_id: str
    source_paper: str
    text: str
    distance: float
    similarity_score: float
    chunk_index: int

class SearchResponse(BaseModel):
    query: str
    num_results: int
    results: List[SearchResult]

class HealthResponse(BaseModel):
    status: str
    model: str
    total_chunks: int
    total_vectors: int

# Endpoints
@app.get("/", response_model=dict)
async def root():
    """
    Root endpoint with API information
    """
    return {
        "message": "arXiv cs.CL RAG API",
        "version": "1.0.0",
        "endpoints": {
            "search": "/search?q=your_query&k=3",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint
    """
    if model is None or index is None or chunks is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return HealthResponse(
        status="healthy",
        model="sentence-transformers/all-MiniLM-L6-v2",
        total_chunks=len(chunks),
        total_vectors=index.ntotal
    )

@app.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., description="Search query", min_length=1),
    k: int = Query(3, description="Number of results to return", ge=1, le=20)
):
    """
    Search for relevant paper chunks
    
    Args:
        q: Query string
        k: Number of results (1-20)
        
    Returns:
        SearchResponse with top-k matching chunks
    """
    if model is None or index is None or chunks is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # Embed the query
        query_embedding = model.encode([q], convert_to_numpy=True)
        
        # Search the index
        distances, indices = index.search(query_embedding, k)
        
        # Collect results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(chunks):
                chunk = chunks[idx]
                results.append(SearchResult(
                    chunk_id=chunk['chunk_id'],
                    source_paper=chunk['source_id'],
                    text=chunk['text'],
                    distance=float(distance),
                    similarity_score=float(1 / (1 + distance)),
                    chunk_index=chunk['chunk_index']
                ))
        
        return SearchResponse(
            query=q,
            num_results=len(results),
            results=results
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/papers", response_model=dict)
async def list_papers():
    """
    List all indexed papers
    """
    if chunks is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # Get unique papers
    unique_sources = {}
    for chunk in chunks:
        source = chunk['source_id']
        if source not in unique_sources:
            unique_sources[source] = {'num_chunks': 0}
        unique_sources[source]['num_chunks'] += 1
    
    return {
        "total_papers": len(unique_sources),
        "papers": [
            {"paper_id": paper_id, **info}
            for paper_id, info in unique_sources.items()
        ]
    }

@app.get("/paper/{paper_id}", response_model=dict)
async def get_paper_chunks(
    paper_id: str,
    limit: Optional[int] = Query(None, description="Limit number of chunks returned")
):
    """
    Get all chunks for a specific paper
    """
    if chunks is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # Filter chunks for this paper
    paper_chunks = [c for c in chunks if c['source_id'] == paper_id]
    
    if not paper_chunks:
        raise HTTPException(status_code=404, detail=f"Paper {paper_id} not found")
    
    # Sort by chunk index
    paper_chunks.sort(key=lambda x: x['chunk_index'])
    
    if limit:
        paper_chunks = paper_chunks[:limit]
    
    return {
        "paper_id": paper_id,
        "total_chunks": len(paper_chunks),
        "chunks": paper_chunks
    }

# Optional: Batch search endpoint
class BatchSearchRequest(BaseModel):
    queries: List[str] = Field(..., min_items=1, max_items=10)
    k: int = Field(3, ge=1, le=20)

@app.post("/batch_search", response_model=List[SearchResponse])
async def batch_search(request: BatchSearchRequest):
    """
    Search multiple queries at once
    """
    if model is None or index is None or chunks is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    results = []
    for query in request.queries:
        # Embed query
        query_embedding = model.encode([query], convert_to_numpy=True)
        distances, indices = index.search(query_embedding, request.k)
        
        # Collect results for this query
        query_results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(chunks):
                chunk = chunks[idx]
                query_results.append(SearchResult(
                    chunk_id=chunk['chunk_id'],
                    source_paper=chunk['source_id'],
                    text=chunk['text'],
                    distance=float(distance),
                    similarity_score=float(1 / (1 + distance)),
                    chunk_index=chunk['chunk_index']
                ))
        
        results.append(SearchResponse(
            query=query,
            num_results=len(query_results),
            results=query_results
        ))
    
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
