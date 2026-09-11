"""emotional-memory resonance links form on retrieve (library default on)."""

from emotional_memory import CoreAffect, EmotionalMemory, InMemoryStore

from affective_fly import FakeEmbedder


def test_resonance_links_after_related_encodes():
    em = EmotionalMemory(store=InMemoryStore(), embedder=FakeEmbedder())
    em.set_affect(CoreAffect(valence=-0.5, arousal=0.3))
    em.encode("chart MEME crash")
    em.set_affect(CoreAffect(valence=-0.4, arousal=0.2))
    em.encode("chart MEME dump")
    em.retrieve("MEME", top_k=2)
    linked = [m for m in em._store.list_all() if m.tag.resonance_links]
    assert linked, "expected at least one resonance link"
    assert all(lk.strength > 0 for m in linked for lk in m.tag.resonance_links)
