"""Tests for labile-window reconsolidation."""

from datetime import UTC, datetime, timedelta

from emotional_memory import CoreAffect, EmotionalMemory, InMemoryStore

from affective_fly import FakeEmbedder, Reconsolidator, stimulus_key


def _em():
    return EmotionalMemory(store=InMemoryStore(), embedder=FakeEmbedder())


def test_stimulus_key_prefers_ticker():
    assert stimulus_key({"page": "chart", "ticker": "MEME", "query": "crash"}) == "ticker:MEME"
    assert stimulus_key({"event": "Market crash"}) == "event:Market crash"
    assert stimulus_key({"page": "x"}) == "page:x"
    assert stimulus_key({"sentiment": 0.1}) is None


def test_reconsolidate_same_ticker_does_not_duplicate():
    em = _em()
    rec = Reconsolidator(labile_window_seconds=600.0, blend=0.4)
    em.set_affect(CoreAffect(valence=0.8, arousal=0.3))
    first = em.encode(
        "frame 0",
        metadata={"ticker": "MEME", "valence": 0.8, "arousal": 0.3, "approach": 0.5},
    )
    match = rec.find_match(em, {"ticker": "MEME"})
    assert match is not None
    assert match.id == first.id
    updated = rec.update(
        em,
        match,
        "frame 1 crash",
        {"ticker": "MEME", "valence": -0.5, "arousal": 0.6, "approach": -0.4},
        valence=-0.5,
        arousal=0.6,
        approach=-0.4,
    )
    assert len(em._store.list_all()) == 1
    assert updated.tag.reconsolidation_count == 1
    assert updated.tag.core_affect.valence < 0.8
    assert updated.tag.core_affect.valence > -0.5
    assert updated.metadata["reconsolidation_count"] == 1


def test_different_ticker_is_new_memory():
    em = _em()
    rec = Reconsolidator()
    em.set_affect(CoreAffect(valence=0.2, arousal=0.1))
    em.encode("a", metadata={"ticker": "MEME", "valence": 0.2})
    assert rec.find_match(em, {"ticker": "PEPE"}) is None


def test_expired_window_does_not_match():
    em = _em()
    rec = Reconsolidator(labile_window_seconds=10.0)
    em.set_affect(CoreAffect(valence=0.1, arousal=0.0))
    mem = em.encode("a", metadata={"ticker": "MEME"})
    old = datetime.now(tz=UTC) - timedelta(seconds=30)
    em._store.update(mem.model_copy(update={"tag": mem.tag.model_copy(update={"timestamp": old})}))
    assert rec.find_match(em, {"ticker": "MEME"}) is None


def test_repeated_neutral_exposure_reduces_valence():
    """Extinction analogue: same ticker, neutral updates pull valence toward 0."""
    em = _em()
    rec = Reconsolidator(labile_window_seconds=10_000.0, blend=0.5)
    em.set_affect(CoreAffect(valence=0.9, arousal=0.4))
    mem = em.encode(
        "crash",
        metadata={"ticker": "MEME", "valence": 0.9, "arousal": 0.4, "approach": 0.8},
    )
    peak = mem.tag.core_affect.valence
    for i in range(6):
        match = rec.find_match(em, {"ticker": "MEME"})
        assert match is not None
        mem = rec.update(
            em,
            match,
            f"neutral {i}",
            {"ticker": "MEME", "valence": 0.0, "arousal": 0.0, "approach": 0.0},
            valence=0.0,
            arousal=0.0,
            approach=0.0,
        )
    assert mem.tag.core_affect.valence < peak
    assert abs(mem.tag.core_affect.valence) < 0.05


def test_disabled_window_never_matches():
    em = _em()
    rec = Reconsolidator(labile_window_seconds=0)
    em.set_affect(CoreAffect(valence=0.1, arousal=0.0))
    em.encode("a", metadata={"ticker": "MEME"})
    assert rec.find_match(em, {"ticker": "MEME"}) is None
