from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from typing import List, Any

load_dotenv()


# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(document_list: List) -> Any:
    """
    Generate embeddings for a list of text chunks.
    
    Args:
        document_list: List of text strings to embed
        
    Returns:
        numpy array of embeddings with shape (num_chunks, embedding_dim)
    """
    if isinstance(document_list, list) and len(document_list) > 0:
        # Encode all chunks at once - more efficient
        embeddings = model.encode(document_list, show_progress_bar=True)
        return embeddings
    else:
        return None


def get_single_embedding(text: str) -> Any:
    """
    Generate embedding for a single text string.
    
    Args:
        text: Single text string to embed
        
    Returns:
        numpy array of embedding with shape (embedding_dim,)
    """
    if isinstance(text, str) and len(text) > 0:
        embedding = model.encode(text)
        return embedding
    else:
        return None
