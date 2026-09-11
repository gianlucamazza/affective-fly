"""Optional sentence-transformers extra. Skips unless the extra is installed.

Does not construct SentenceTransformerEmbedder (that downloads MiniLM).
"""

import pytest


def test_sentence_transformers_importable():
    pytest.importorskip("sentence_transformers")
    from emotional_memory import SentenceTransformerEmbedder

    assert SentenceTransformerEmbedder is not None
