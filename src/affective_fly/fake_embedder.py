"""
Fake embedder for tests and demos.

Provides a simple deterministic embedder that doesn't require
any real model or network access.
"""

import hashlib
from typing import List


class FakeEmbedder:
    """
    Fake embedder for offline tests and demos.

    Creates deterministic 384-dimensional embeddings based on text hash.
    Implements the emotional-memory Embedder protocol.
    """

    def __init__(self, dimension: int = 384):
        """
        Initialize fake embedder.

        Args:
            dimension: Embedding dimension (default 384)
        """
        self.dimension = dimension

    def embed(self, text: str) -> List[float]:
        """
        Convert text to fake embedding vector.

        Args:
            text: Input text

        Returns:
            List of floats representing the embedding
        """
        # Hash the text to get deterministic pseudo-random values
        hash_obj = hashlib.sha256(text.encode("utf-8"))
        hash_bytes = hash_obj.digest()

        # Expand hash to dimension size by repeating and XORing
        embedding = []
        for i in range(self.dimension):
            # Use hash byte with wraparound
            byte_val = hash_bytes[i % len(hash_bytes)]
            # XOR with position for variation
            byte_val ^= (i // len(hash_bytes)) & 0xFF
            # Normalize to [-1, 1] range
            embedding.append((byte_val / 127.5) - 1.0)

        return embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Convert batch of texts to embeddings.

        Args:
            texts: List of input texts

        Returns:
            List of embeddings
        """
        return [self.embed(text) for text in texts]
