"""
Embedding and FAISS Indexing Module
Generates embeddings for chunks and creates searchable FAISS index
"""

import numpy as np
import faiss
import pickle
import json
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from pathlib import Path


class EmbeddingIndexer:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the embedding model and indexer
        
        Args:
            model_name: Sentence-transformers model name
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.chunks = []
        self.metadata = []
        
        print(f"Model loaded. Embedding dimension: {self.dimension}")
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            
        Returns:
            Numpy array of embeddings
        """
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings
    
    def build_index(self, chunks_data: List[Dict]) -> faiss.Index:
        """
        Build FAISS index from chunk data
        
        Args:
            chunks_data: List of chunk dictionaries with 'text' field
            
        Returns:
            FAISS index
        """
        # Extract texts
        texts = [chunk['text'] for chunk in chunks_data]
        self.chunks = texts
        self.metadata = chunks_data
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Create FAISS index
        print(f"Building FAISS index (dimension: {self.dimension})...")
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))
        
        print(f"Index built with {self.index.ntotal} vectors")
        return self.index
    
    def search(self, query: str, k: int = 3) -> List[Dict]:
        """
        Search the index for relevant chunks
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of dictionaries containing results
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Embed the query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Search
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Package results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            result = {
                'rank': i + 1,
                'distance': float(dist),
                'chunk_id': self.metadata[idx]['chunk_id'],
                'paper_id': self.metadata[idx]['paper_id'],
                'text': self.chunks[idx],
                'token_count': self.metadata[idx].get('token_count', 0)
            }
            results.append(result)
        
        return results
    
    def save_index(self, index_path: str = "faiss_index.bin", 
                   metadata_path: str = "index_metadata.pkl"):
        """
        Save FAISS index and metadata to disk
        
        Args:
            index_path: Path to save FAISS index
            metadata_path: Path to save metadata
        """
        if self.index is None:
            raise ValueError("No index to save. Build index first.")
        
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        print(f"FAISS index saved to: {index_path}")
        
        # Save metadata (chunks and metadata)
        metadata_dict = {
            'chunks': self.chunks,
            'metadata': self.metadata,
            'dimension': self.dimension
        }
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata_dict, f)
        print(f"Metadata saved to: {metadata_path}")
    
    def load_index(self, index_path: str = "faiss_index.bin",
                   metadata_path: str = "index_metadata.pkl"):
        """
        Load FAISS index and metadata from disk
        
        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to metadata file
        """
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        print(f"FAISS index loaded from: {index_path}")
        print(f"Index contains {self.index.ntotal} vectors")
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            metadata_dict = pickle.load(f)
        
        self.chunks = metadata_dict['chunks']
        self.metadata = metadata_dict['metadata']
        self.dimension = metadata_dict['dimension']
        print(f"Metadata loaded from: {metadata_path}")
    
    def get_stats(self) -> Dict:
        """Get statistics about the index"""
        if self.index is None:
            return {"error": "No index built"}
        
        return {
            'total_chunks': self.index.ntotal,
            'dimension': self.dimension,
            'total_papers': len(set(m['paper_id'] for m in self.metadata)),
            'avg_chunk_tokens': np.mean([m.get('token_count', 0) for m in self.metadata])
        }


def main():
    """Example usage"""
    # Initialize indexer
    indexer = EmbeddingIndexer(model_name='all-MiniLM-L6-v2')
    
    # Load processed chunks
    with open('processed_chunks.json', 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
    
    print(f"Loaded {len(chunks_data)} chunks")
    
    # Build index
    indexer.build_index(chunks_data)
    
    # Save index
    indexer.save_index("faiss_index.bin", "index_metadata.pkl")
    
    # Test search
    print("\n" + "=" * 60)
    print("Testing Search")
    print("=" * 60)
    
    test_queries = [
        "What are attention mechanisms in transformers?",
        "How do language models perform few-shot learning?",
        "What is the role of pre-training in NLP?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = indexer.search(query, k=3)
        print(f"Top result (distance={results[0]['distance']:.4f}):")
        print(f"  Paper: {results[0]['paper_id']}")
        print(f"  Text: {results[0]['text'][:150]}...")
    
    # Print stats
    print("\n" + "=" * 60)
    print("Index Statistics")
    print("=" * 60)
    stats = indexer.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()