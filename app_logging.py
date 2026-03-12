"""
Embedding generation module using Sentence Transformers.
Provides text-to-vector conversion for semantic search.
"""
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from logger import setup_logger
from config import settings

logger = setup_logger(__name__)

_embed_model = None


def get_model() -> SentenceTransformer:
    """
    Get or initialize the sentence transformer model (lazy loading).
    
    Returns:
        Initialized SentenceTransformer model
    """
    global _embed_model
    if _embed_model is None:
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        try:
            _embed_model = SentenceTransformer(settings.embedding_model)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    return _embed_model


def get_embedding(text: str) -> np.ndarray:
    """
    Generate embedding vector for a given text.
    
    Args:
        text: Input text to embed
        
    Returns:
        Embedding vector as numpy array
        
    Raises:
        ValueError: If text is empty
        RuntimeError: If embedding generation fails
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")
    
    try:
        model = get_model()
        embedding = model.encode([text.strip()])[0]
        logger.debug(f"Generated embedding for text length {len(text)}")
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise RuntimeError(f"Failed to generate embedding: {e}") from e


def get_embeddings_batch(texts: List[str]) -> List[np.ndarray]:
    """
    Generate embeddings for multiple texts efficiently (batch processing).
    
    Args:
        texts: List of texts to embed
        
    Returns:
        List of embedding vectors
    """
    if not texts:
        raise ValueError("Texts list cannot be empty")
    
    valid_texts = [t.strip() for t in texts if t and t.strip()]
    
    if not valid_texts:
        raise ValueError("All texts are empty")
    
    try:
        model = get_model()
        embeddings = model.encode(valid_texts)
        logger.info(f"Generated {len(embeddings)} embeddings in batch")
        return embeddings
    except Exception as e:
        logger.error(f"Error generating batch embeddings: {e}")
        raise RuntimeError(f"Failed to generate batch embeddings: {e}") from e