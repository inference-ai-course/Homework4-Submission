"""
Vector Store Module
Handles vector database operations using ChromaDB.
"""

from typing import List, Optional
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from loguru import logger
import shutil


class VectorStore:
    """Manages ChromaDB vector store for document retrieval."""
    
    def __init__(
        self, 
        persist_directory: str = "./data/vector_db",
        collection_name: str = "resume_chunks"
    ):
        """
        Initialize VectorStore.
        
        Args:
            persist_directory: Directory for ChromaDB persistence
            collection_name: Name of the collection
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.vectorstore = None
        
        # Create directory if it doesn't exist
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"VectorStore initialized: {self.persist_directory} / {collection_name}")
    
    def create_vectorstore(self, documents: List[Document], embeddings) -> bool:
        """
        Create a new vector store from documents.
        
        Args:
            documents: List of Document objects to index
            embeddings: Embedding function to use
            
        Returns:
            Success status
        """
        try:
            if not documents:
                logger.warning("No documents provided to create vector store")
                return False
            
            logger.info(f"Creating vector store with {len(documents)} documents...")
            
            # Create Chroma vector store
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                collection_name=self.collection_name,
                persist_directory=str(self.persist_directory)
            )
            
            # Persist the vector store
            self.vectorstore.persist()
            
            logger.info(f"Vector store created successfully with {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            return False
    
    def load_vectorstore(self, embeddings) -> bool:
        """
        Load existing vector store from disk.
        
        Args:
            embeddings: Embedding function to use
            
        Returns:
            Success status
        """
        try:
            if not self.persist_directory.exists():
                logger.warning(f"Persist directory does not exist: {self.persist_directory}")
                return False
            
            logger.info("Loading existing vector store...")
            
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=embeddings,
                persist_directory=str(self.persist_directory)
            )
            
            # Check if collection has any documents
            collection_count = self.vectorstore._collection.count()
            
            if collection_count == 0:
                logger.warning("Loaded vector store is empty")
                return False
            
            logger.info(f"Vector store loaded successfully with {collection_count} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return False
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 3,
        score_threshold: float = 0.0
    ) -> List[tuple]:
        """
        Perform similarity search for a query.
        
        Args:
            query: Query text
            k: Number of results to return
            score_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of (Document, score) tuples
        """
        try:
            if self.vectorstore is None:
                logger.error("Vector store not initialized")
                return []
            
            if not query or not query.strip():
                logger.warning("Empty query provided for similarity search")
                return []
            
            logger.info(f"Performing similarity search for: '{query[:50]}...' (top {k})")
            
            # Perform similarity search with scores
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k
            )
            
            # Filter by score threshold
            filtered_results = [
                (doc, score) for doc, score in results 
                if score >= score_threshold
            ]
            
            logger.info(f"Found {len(filtered_results)} relevant chunks")
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {e}")
            return []
    
    def get_retriever(self, k: int = 3):
        """
        Get a retriever interface for the vector store.
        
        Args:
            k: Number of documents to retrieve
            
        Returns:
            Retriever object
        """
        if self.vectorstore is None:
            logger.error("Vector store not initialized")
            return None
        
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
    
    def add_documents(self, documents: List[Document]) -> bool:
        """
        Add new documents to existing vector store.
        
        Args:
            documents: List of documents to add
            
        Returns:
            Success status
        """
        try:
            if self.vectorstore is None:
                logger.error("Vector store not initialized")
                return False
            
            if not documents:
                logger.warning("No documents provided to add")
                return False
            
            logger.info(f"Adding {len(documents)} documents to vector store...")
            
            self.vectorstore.add_documents(documents)
            self.vectorstore.persist()
            
            logger.info("Documents added successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False
    
    def clear_collection(self) -> bool:
        """
        Clear the vector store collection (for new resume uploads).
        
        Returns:
            Success status
        """
        try:
            logger.info("Clearing vector store collection...")
            
            # Delete the persist directory
            if self.persist_directory.exists():
                shutil.rmtree(self.persist_directory)
                logger.info("Persist directory removed")
            
            # Recreate the directory
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            
            # Reset vectorstore
            self.vectorstore = None
            
            logger.info("Vector store cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            return False
    
    def get_collection_stats(self) -> dict:
        """
        Get statistics about the current collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            if self.vectorstore is None:
                return {
                    'initialized': False,
                    'document_count': 0
                }
            
            document_count = self.vectorstore._collection.count()
            
            return {
                'initialized': True,
                'document_count': document_count,
                'collection_name': self.collection_name,
                'persist_directory': str(self.persist_directory)
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'initialized': False, 'error': str(e)}
