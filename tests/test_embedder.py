"""Optional sentence-transformers extra. Skips unless the extra is installed.

Does not construct SentenceTransformerEmbedder (that downloads MiniLM).
"""

import pytest


def test_sentence_transformers_importable():
    pytest.importorskip("sentence_transformers")
    from emotional_memory import SentenceTransformerEmbedder

    assert SentenceTransformerEmbedder is not None


def test_retrieve_memory_fields():
    """Memory objects from retrieve() have content/id/embedding/tag/metadata, not .similarity."""
    from emotional_memory import EmotionalMemory, InMemoryStore

    from affective_fly import FakeEmbedder

    em = EmotionalMemory(store=InMemoryStore(), embedder=FakeEmbedder())
    em.encode("test content", metadata={"context": "journal"})
    hits = em.retrieve("test", top_k=1)

    assert len(hits) == 1
    hit = hits[0]

    # Memory has these fields
    assert hasattr(hit, "content")
    assert hasattr(hit, "id")
    assert hasattr(hit, "embedding")
    assert hasattr(hit, "tag")
    assert hasattr(hit, "metadata")

    # Memory does NOT have .similarity
    assert not hasattr(hit, "similarity")

    # Validate content is accessible
    assert isinstance(hit.content, str)
    assert "test content" in hit.content
