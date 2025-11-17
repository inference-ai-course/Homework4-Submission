from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np
import logging
from pathlib import Path

from utils.faiss_indexing import FAISSIndex
from utils.embedding_generation import get_single_embedding

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Search API",
    description="Retrieval-Augmented Generation API for arXiv papers",
    version="1.0.0"
)

# Global variables for index
faiss_index: FAISSIndex = None
INDEX_PATH = "src/data/faiss_index.index"
CHUNKS_PATH = "src/data/faiss_chunks.pkl"


class SearchResponse(BaseModel):
    """Response model for search endpoint"""
    query: str
    results: List[Dict[str, Any]]
    num_results: int


@app.on_event("startup")
async def load_index():
    """Load FAISS index on startup"""
    global faiss_index
    try:
        if Path(INDEX_PATH).exists() and Path(CHUNKS_PATH).exists():
            faiss_index = FAISSIndex()
            faiss_index.load(INDEX_PATH, CHUNKS_PATH)
            stats = faiss_index.get_stats()
            logger.info(f"Loaded FAISS index with {stats['total_chunks']} chunks")
        else:
            logger.warning("FAISS index files not found. Please build the index first.")
            faiss_index = None
    except Exception as e:
        logger.error(f"Error loading FAISS index: {e}")
        faiss_index = None


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RAG Search API",
        "endpoints": {
            "/search": "Search for relevant passages (GET with query parameter 'q')",
            "/stats": "Get index statistics",
            "/health": "Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if faiss_index is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": "FAISS index not loaded"}
        )
    return {"status": "healthy", "index_loaded": True}


@app.get("/stats")
async def get_stats():
    """Get index statistics"""
    if faiss_index is None:
        raise HTTPException(status_code=503, detail="FAISS index not loaded")
    
    stats = faiss_index.get_stats()
    return stats


@app.get("/search", response_model=SearchResponse)
async def search(q: str, k: int = 3):
    """
    Search for relevant passages given a query.
    
    Args:
        q: Query string
        k: Number of results to return (default: 3, max: 10)
        
    Returns:
        SearchResponse with query and top-k results
    """
    if faiss_index is None:
        raise HTTPException(
            status_code=503, 
            detail="FAISS index not loaded. Please build the index first."
        )
    
    if not q or len(q.strip()) == 0:
        raise HTTPException(status_code=400, detail="Query parameter 'q' is required")
    
    # Limit k to reasonable range
    k = min(max(1, k), 10)
    
    try:
        # Generate embedding for query
        logger.info(f"Processing query: {q}")
        query_embedding = get_single_embedding(q)
        
        if query_embedding is None:
            raise HTTPException(status_code=500, detail="Failed to generate query embedding")
        
        # Search FAISS index
        distances, indices, chunks, metadata = faiss_index.search(query_embedding, k=k)
        
        # Format results
        results = []
        for i, (dist, idx, chunk, meta) in enumerate(zip(distances[0], indices[0], chunks, metadata)):
            results.append({
                "rank": i + 1,
                "chunk_id": int(idx),
                "distance": float(dist),
                "text": chunk,
                "metadata": meta
            })
        
        logger.info(f"Found {len(results)} results for query")
        
        return SearchResponse(
            query=q,
            results=results,
            num_results=len(results)
        )
        
    except Exception as e:
        logger.error(f"Error during search: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
