"""
Embedding Engine Module
Handles text embedding generation using Ollama.
"""

from typing import List
from langchain_community.embeddings import OllamaEmbeddings
from loguru import logger
import time


class EmbeddingEngine:
    """Wraps Ollama embedding model for text vectorization."""
    
    def __init__(self, model_name: str = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"):
        """
        Initialize EmbeddingEngine.
        
        Args:
            model_name: Name of the Ollama embedding model
        """
        self.model_name = model_name
        
        try:
            self.embeddings = OllamaEmbeddings(
                model=model_name,
                # base_url="http://localhost:11434"  # Default Ollama URL
            )
            logger.info(f"EmbeddingEngine initialized with model: {model_name}")
            
            # Test the connection
            self._test_connection()
            
        except Exception as e:
            logger.error(f"Error initializing EmbeddingEngine: {e}")
            raise
    
    def _test_connection(self):
        """Test connection to Ollama and model availability."""
        try:
            # Try to embed a simple test string
            test_embedding = self.embeddings.embed_query("test")
            logger.info(f"Ollama connection successful. Embedding dimension: {len(test_embedding)}")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama or model not available: {e}")
            raise ConnectionError(
                f"Cannot connect to Ollama. Please ensure:\n"
                f"1. Ollama is installed and running\n"
                f"2. Model '{self.model_name}' is downloaded\n"
                f"Run: ollama pull {self.model_name}"
            )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            if not texts:
                logger.warning("Empty text list provided for embedding")
                return []
            
            start_time = time.time()
            embeddings = self.embeddings.embed_documents(texts)
            elapsed_time = time.time() - start_time
            
            logger.info(f"Generated {len(embeddings)} embeddings in {elapsed_time:.2f}s "
                       f"({len(embeddings)/elapsed_time:.1f} docs/sec)")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating document embeddings: {e}")
            raise
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query text to embed
            
        Returns:
            Embedding vector
        """
        try:
            if not query or not query.strip():
                logger.warning("Empty query provided for embedding")
                return []
            
            start_time = time.time()
            embedding = self.embeddings.embed_query(query)
            elapsed_time = time.time() - start_time
            
            logger.debug(f"Generated query embedding in {elapsed_time:.3f}s")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        
        Returns:
            Embedding dimension
        """
        try:
            test_embedding = self.embed_query("test")
            return len(test_embedding)
        except Exception as e:
            logger.error(f"Error getting embedding dimension: {e}")
            return 0
