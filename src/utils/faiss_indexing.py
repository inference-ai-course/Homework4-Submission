import faiss
import numpy as np
import pickle
from typing import List, Tuple, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FAISSIndex:
    """
    A wrapper class for FAISS indexing operations.
    """
    
    def __init__(self, dimension: int = 384):
        """
        Initialize FAISS index.
        
        Args:
            dimension: Dimension of the embeddings (default 384 for all-MiniLM-L6-v2)
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks = []
        self.metadata = []
        
    def add_embeddings(self, embeddings: np.ndarray, chunks: List[str], metadata: List[Dict] = None):
        """
        Add embeddings to the FAISS index.
        
        Args:
            embeddings: numpy array of shape (num_chunks, dimension)
            chunks: List of text chunks corresponding to embeddings
            metadata: Optional list of metadata dicts for each chunk
        """
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} doesn't match index dimension {self.dimension}")
        
        # Ensure embeddings are float32
        embeddings = embeddings.astype('float32')
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store chunks and metadata
        self.chunks.extend(chunks)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(chunks))
            
        logger.info(f"Added {len(chunks)} chunks to index. Total chunks: {len(self.chunks)}")
        
    def search(self, query_embedding: np.ndarray, k: int = 3) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Search for top-k similar chunks.
        
        Args:
            query_embedding: Query embedding vector of shape (dimension,) or (1, dimension)
            k: Number of results to return
            
        Returns:
            Tuple of (distances, indices, chunks)
        """
        # Ensure query is 2D and float32
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # Get corresponding chunks
        result_chunks = [self.chunks[idx] for idx in indices[0]]
        result_metadata = [self.metadata[idx] for idx in indices[0]]
        
        return distances, indices, result_chunks, result_metadata
    
    def save(self, index_path: str, chunks_path: str):
        """
        Save FAISS index and chunks to disk.
        
        Args:
            index_path: Path to save FAISS index (.index file)
            chunks_path: Path to save chunks and metadata (.pkl file)
        """
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save chunks and metadata
        with open(chunks_path, 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'metadata': self.metadata,
                'dimension': self.dimension
            }, f)
            
        logger.info(f"Saved index to {index_path} and chunks to {chunks_path}")
        
    def load(self, index_path: str, chunks_path: str):
        """
        Load FAISS index and chunks from disk.
        
        Args:
            index_path: Path to FAISS index file
            chunks_path: Path to chunks pickle file
        """
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load chunks and metadata
        with open(chunks_path, 'rb') as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.metadata = data['metadata']
            self.dimension = data['dimension']
            
        logger.info(f"Loaded index from {index_path} with {len(self.chunks)} chunks")
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the index.
        
        Returns:
            Dictionary with index statistics
        """
        return {
            'total_chunks': len(self.chunks),
            'dimension': self.dimension,
            'index_size': self.index.ntotal
        }


def create_index_from_chunks(chunks: List[str], embeddings: np.ndarray, 
                             metadata: List[Dict] = None, 
                             dimension: int = 384) -> FAISSIndex:
    """
    Helper function to create a FAISS index from chunks and embeddings.
    
    Args:
        chunks: List of text chunks
        embeddings: numpy array of embeddings
        metadata: Optional metadata for each chunk
        dimension: Embedding dimension
        
    Returns:
        FAISSIndex object
    """
    index = FAISSIndex(dimension=dimension)
    index.add_embeddings(embeddings, chunks, metadata)
    return index
