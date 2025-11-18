"""
Text Processor Module
Handles text chunking and preprocessing for RAG pipeline.
"""

from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from loguru import logger


class TextProcessor:
    """Handles text chunking and preprocessing."""
    
    def __init__(
        self, 
        chunk_size: int = 800, 
        chunk_overlap: int = 80,
        separators: List[str] = None
    ):
        """
        Initialize TextProcessor.
        
        Args:
            chunk_size: Size of each text chunk in characters
            chunk_overlap: Overlap between chunks in characters
            separators: List of separators for splitting text
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if separators is None:
            separators = ["\n\n", "\n", ". ", " ", ""]
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
        )
        
        logger.info(f"TextProcessor initialized: chunk_size={chunk_size}, "
                   f"overlap={chunk_overlap}")
    
    def chunk_text(self, text: str, metadata: dict = None) -> List[Document]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of Document objects with chunked text
        """
        try:
            if not text or not text.strip():
                logger.warning("Empty text provided for chunking")
                return []
            
            # Create chunks
            chunks = self.splitter.split_text(text)
            
            # Create Document objects with metadata
            documents = []
            for i, chunk in enumerate(chunks):
                doc_metadata = {
                    'chunk_index': i,
                    'chunk_size': len(chunk),
                    'total_chunks': len(chunks)
                }
                
                # Merge with provided metadata
                if metadata:
                    doc_metadata.update(metadata)
                
                # Try to extract page number from chunk if it contains page markers
                page_num = self._extract_page_number(chunk)
                if page_num:
                    doc_metadata['page_num'] = page_num
                
                documents.append(Document(
                    page_content=chunk,
                    metadata=doc_metadata
                ))
            
            logger.info(f"Created {len(documents)} chunks from text "
                       f"(avg size: {sum(len(d.page_content) for d in documents) // len(documents)} chars)")
            
            return documents
            
        except Exception as e:
            logger.error(f"Error chunking text: {e}")
            return []
    
    def _extract_page_number(self, chunk: str) -> int:
        """
        Extract page number from chunk if it contains page markers.
        
        Args:
            chunk: Text chunk
            
        Returns:
            Page number or None
        """
        import re
        
        # Look for page markers like "--- Page 1 ---"
        match = re.search(r'--- Page (\d+) ---', chunk)
        if match:
            return int(match.group(1))
        
        return None
    
    def preprocess_text(self, text: str) -> str:
        """
        Additional preprocessing for text before chunking.
        
        Args:
            text: Raw text
            
        Returns:
            Preprocessed text
        """
        try:
            # Remove excessive whitespace
            import re
            text = re.sub(r'\s+', ' ', text)
            
            # Remove special characters that might interfere with embeddings
            # (keeping most punctuation for context)
            text = re.sub(r'[^\w\s\.,!?;:\-\(\)\'\"/@#$%&*+=\[\]{}|\\<>~`]', '', text)
            
            # Normalize line endings
            text = text.replace('\r\n', '\n').replace('\r', '\n')
            
            logger.debug("Text preprocessing completed")
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error preprocessing text: {e}")
            return text
    
    def get_chunk_statistics(self, documents: List[Document]) -> dict:
        """
        Calculate statistics about chunks.
        
        Args:
            documents: List of Document objects
            
        Returns:
            Dictionary with statistics
        """
        if not documents:
            return {
                'total_chunks': 0,
                'avg_chunk_size': 0,
                'min_chunk_size': 0,
                'max_chunk_size': 0,
                'total_characters': 0
            }
        
        chunk_sizes = [len(doc.page_content) for doc in documents]
        
        stats = {
            'total_chunks': len(documents),
            'avg_chunk_size': sum(chunk_sizes) // len(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'total_characters': sum(chunk_sizes),
            'estimated_tokens': sum(chunk_sizes) // 4  # Rough estimate
        }
        
        return stats
