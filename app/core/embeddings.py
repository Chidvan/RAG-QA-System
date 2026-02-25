"""
Embedding generation module using HuggingFace local embeddings.
"""

from functools import lru_cache
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Create and cache HuggingFace embedding model instance.

    Returns:
        HuggingFaceEmbeddings instance
    """
    settings = get_settings()

    logger.info(
        f"Initializing HuggingFace embeddings model: {settings.embedding_model}"
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},  # change to "cuda" if GPU available
        encode_kwargs={"normalize_embeddings": True},
    )

    logger.info("HuggingFace embeddings initialized successfully")
    return embeddings


class EmbeddingService:
    """
    Service wrapper for embedding generation.
    """

    def __init__(self):
        settings = get_settings()
        self.model_name = settings.embedding_model
        self.embeddings = get_embeddings()

        logger.info(f"EmbeddingService ready using model: {self.model_name}")

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query.

        Args:
            text: Query string

        Returns:
            Embedding vector
        """
        logger.debug(f"Embedding query (first 50 chars): {text[:50]}")
        return self.embeddings.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents.

        Args:
            texts: List of text strings

        Returns:
            List of embedding vectors
        """
        logger.debug(f"Embedding {len(texts)} documents")
        return self.embeddings.embed_documents(texts)
