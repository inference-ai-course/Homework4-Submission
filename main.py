"""
FastAPI Service for RAG Search
Provides REST API endpoint for searching the FAISS index
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from embedding_indexer import EmbeddingIndexer

# Initialize FastAPI app
app = FastAPI(
    title="arXiv RAG Search API",
    description="Search arXiv papers using semantic similarity",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize indexer (will be loaded on startup)
indexer = None


# Response models
class SearchResult(BaseModel):
    rank: int
    distance: float
    chunk_id: str
    paper_id: str
    text: str
    token_count: int


class SearchResponse(BaseModel):
    query: str
    num_results: int
    results: List[SearchResult]


class IndexStats(BaseModel):
    total_chunks: int
    dimension: int
    total_papers: int
    avg_chunk_tokens: float


# Startup event
@app.on_event("startup")
async def startup_event():
    """Load the FAISS index on startup"""
    global indexer
    print("Loading FAISS index and embeddings model...")
    
    indexer = EmbeddingIndexer(model_name='all-MiniLM-L6-v2')
    
    try:
        indexer.load_index("faiss_index.bin", "index_metadata.pkl")
        print("✅ Index loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading index: {str(e)}")
        print("Please run embedding_indexer.py first to build the index.")


# Routes
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "arXiv RAG Search API",
        "version": "1.0.0",
        "endpoints": {
            "/search": "Search for relevant passages (GET with ?q=query&k=3)",
            "/stats": "Get index statistics",
            "/health": "Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if indexer is None or indexer.index is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    
    return {
        "status": "healthy",
        "index_loaded": True,
        "total_chunks": indexer.index.ntotal
    }


@app.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., description="Search query", min_length=3),
    k: int = Query(3, description="Number of results to return", ge=1, le=20)
):
    """
    Search the index for relevant passages
    
    Args:
        q: Search query string
        k: Number of results to return (default: 3, max: 20)
        
    Returns:
        SearchResponse with ranked results
    """
    if indexer is None or indexer.index is None:
        raise HTTPException(
            status_code=503, 
            detail="Index not loaded. Please build the index first."
        )
    
    try:
        # Perform search
        results = indexer.search(q, k=k)
        
        return SearchResponse(
            query=q,
            num_results=len(results),
            results=results
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search error: {str(e)}"
        )


@app.get("/stats", response_model=IndexStats)
async def get_stats():
    """Get statistics about the index"""
    if indexer is None or indexer.index is None:
        raise HTTPException(
            status_code=503,
            detail="Index not loaded"
        )
    
    stats = indexer.get_stats()
    return IndexStats(**stats)


@app.get("/paper/{paper_id}")
async def get_paper_chunks(paper_id: str):
    """Get all chunks for a specific paper"""
    if indexer is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    
    # Filter chunks by paper_id
    paper_chunks = [
        {
            'chunk_id': meta['chunk_id'],
            'chunk_index': meta['chunk_index'],
            'text': indexer.chunks[i],
            'token_count': meta.get('token_count', 0)
        }
        for i, meta in enumerate(indexer.metadata)
        if meta['paper_id'] == paper_id
    ]
    
    if not paper_chunks:
        raise HTTPException(
            status_code=404,
            detail=f"No chunks found for paper_id: {paper_id}"
        )
    
    return {
        'paper_id': paper_id,
        'num_chunks': len(paper_chunks),
        'chunks': paper_chunks
    }


def main():
    """Run the FastAPI server"""
    print("=" * 60)
    print("Starting arXiv RAG Search API")
    print("=" * 60)
    print("\nMake sure you have:")
    print("1. Processed PDFs (run pdf_processor.py)")
    print("2. Built FAISS index (run embedding_indexer.py)")
    print("\nAPI will be available at: http://localhost:8000")
    print("Interactive docs at: http://localhost:8000/docs")
    print("=" * 60)
    print()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()