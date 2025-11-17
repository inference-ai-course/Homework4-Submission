# Task 4 & 5: Embedding Generation and FAISS Indexing: Generate embeddings and build searchable index

import numpy as np
import faiss
import pickle
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class EmbeddingIndexer:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize embedding model and FAISS index
        """
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.chunks = None
        
    def generate_embeddings(self, chunks: List[Dict], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for all chunks
        
        Args:
            chunks: List of chunk dictionaries with 'text' field
            batch_size: Batch size for encoding
            
        Returns:
            Numpy array of embeddings (num_chunks x dimension)
        """
        texts = [chunk['text'] for chunk in chunks]
        
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def build_index(self, embeddings: np.ndarray, index_type: str = "flat"):
        """
        Build FAISS index from embeddings
        
        Args:
            embeddings: Numpy array of embeddings
            index_type: Type of index ("flat", "ivf", or "hnsw")
        """
        print(f"Building FAISS index (type: {index_type})...")
        
        if index_type == "flat":
            # Simple flat L2 index (exact search)
            self.index = faiss.IndexFlatL2(self.dimension)
            
        elif index_type == "ivf":
            # IVF index for faster approximate search on larger datasets
            nlist = min(100, len(embeddings) // 10)  # Number of clusters
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            
            # Train the index
            print("Training IVF index...")
            self.index.train(embeddings)
            
        elif index_type == "hnsw":
            # HNSW index for very fast approximate search
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Add embeddings to index
        self.index.add(embeddings)
        print(f"Added {self.index.ntotal} vectors to index")
        
    def search(self, query: str, k: int = 3) -> List[Tuple[Dict, float]]:
        """
        Search for most similar chunks to query
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of (chunk_dict, distance) tuples
        """
        if self.index is None or self.chunks is None:
            raise ValueError("Index not built. Call build_index first.")
        
        # Embed query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Search index
        distances, indices = self.index.search(query_embedding, k)
        
        # Get corresponding chunks
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(distance)))
        
        return results
    
    def save(self, index_path: str, chunks_path: str):
        """
        Save FAISS index and chunks to disk
        """
        faiss.write_index(self.index, index_path)
        with open(chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        print(f"Saved index to {index_path} and chunks to {chunks_path}")
    
    def load(self, index_path: str, chunks_path: str):
        """
        Load FAISS index and chunks from disk
        """
        self.index = faiss.read_index(index_path)
        with open(chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)
        print(f"Loaded index with {self.index.ntotal} vectors")
    
    def build_from_chunks(self, chunks: List[Dict], index_type: str = "flat"):
        """
        Complete pipeline: embed chunks and build index
        """
        self.chunks = chunks
        embeddings = self.generate_embeddings(chunks)
        self.build_index(embeddings, index_type)


# Usage example
if __name__ == "__main__":
    import json
    
    # Load chunks
    print("Loading chunks...")
    with open("paper_chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    # Initialize indexer
    indexer = EmbeddingIndexer()
    
    # Build index
    indexer.build_from_chunks(chunks, index_type="flat")
    
    # Save index
    indexer.save("faiss_index.bin", "chunks.pkl")
    
    # Test search
    print("\n" + "="*50)
    print("Testing search...")
    query = "What are transformer models?"
    results = indexer.search(query, k=3)
    
    print(f"\nQuery: {query}")
    print(f"Top {len(results)} results:\n")
    
    for i, (chunk, distance) in enumerate(results, 1):
        print(f"{i}. Distance: {distance:.4f}")
        print(f"   Source: {chunk['source_id']}")
        print(f"   Text: {chunk['text'][:200]}...")
        print()
