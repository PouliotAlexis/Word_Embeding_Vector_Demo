"""Backend package for Word Embedding Demo."""

from .embeddings import EmbeddingManager, get_manager
from .server import app

__all__ = ["EmbeddingManager", "get_manager", "app"]
