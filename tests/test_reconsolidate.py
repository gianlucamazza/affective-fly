"""Tests for labile-window reconsolidation."""

from datetime import UTC, datetime, timedelta

from emotional_memory import CoreAffect, EmotionalMemory, InMemoryStore

from affective_fly import FakeEmbedder, Reconsolidator, stimulus_key


def _em():
    store = InMemoryStore()
    embedder = FakeEmbedder()
    return EmotionalMemory(store=store, embedder=embedder), store, embedder


def test_stimulus_key_prefers_note_id():
    assert stimulus_key({"context": "journal", "note_id": "exp-001", "query": "success"}) == "note_id:exp-001"
    assert stimulus_key({"page": "chart", "ticker": "MEME", "query": "crash"}) == "ticker:MEME"
    assert stimulus_key({"note_id": "exp-001", "ticker": "MEME"}) == "note_id:exp-001"
    assert stimulus_key({"event": "Market crash"}) == "event:Market crash"
    assert stimulus_key({"page": "x"}) == "page:x"
    assert stimulus_key({"sentiment": 0.1}) is None


def test_reconsolidate_same_note_id_does_not_duplicate():
    em, store, embedder = _em()
    rec = Reconsolidator(labile_window_seconds=600.0, blend=0.4)
    em.set_affect(CoreAffect(valence=0.8, arousal=0.3))
    first = em.encode(
        "frame 0",
        metadata={"note_id": "exp-001", "valence": 0.8, "arousal": 0.3, "approach": 0.5},
    )
    match = rec.find_match(store, {"note_id": "exp-001"})
    assert match is not None
    assert match.id == first.id
    updated = rec.update(
        store,
        embedder,
        match,
        "frame 1 failed",
        {"note_id": "exp-001", "valence": -0.5, "arousal": 0.6, "approach": -0.4},
        valence=-0.5,
        arousal=0.6,
        approach=-0.4,
    )
    assert len(store.list_all()) == 1
    assert updated.tag.reconsolidation_count == 1
    assert updated.tag.core_affect.valence < 0.8
    assert updated.tag.core_affect.valence > -0.5
    assert updated.metadata["reconsolidation_count"] == 1


def test_different_note_id_is_new_memory():
    em, store, _ = _em()
    rec = Reconsolidator()
    em.set_affect(CoreAffect(valence=0.2, arousal=0.1))
    em.encode("a", metadata={"note_id": "exp-001", "valence": 0.2})
    assert rec.find_match(store, {"note_id": "exp-002"}) is None


def test_expired_window_does_not_match():
    em, store, _ = _em()
    rec = Reconsolidator(labile_window_seconds=10.0)
    em.set_affect(CoreAffect(valence=0.1, arousal=0.0))
    mem = em.encode("a", metadata={"note_id": "exp-001"})
    old = datetime.now(tz=UTC) - timedelta(seconds=30)
    store.update(mem.model_copy(update={"tag": mem.tag.model_copy(update={"timestamp": old})}))
    assert rec.find_match(store, {"note_id": "exp-001"}) is None


def test_repeated_neutral_exposure_reduces_valence():
    """Extinction analogue: same note_id, neutral updates pull valence toward 0."""
    em, store, embedder = _em()
    rec = Reconsolidator(labile_window_seconds=10_000.0, blend=0.5)
    em.set_affect(CoreAffect(valence=0.9, arousal=0.4))
    mem = em.encode(
        "failed",
        metadata={"note_id": "exp-001", "valence": 0.9, "arousal": 0.4, "approach": 0.8},
    )
    peak = mem.tag.core_affect.valence
    for i in range(6):
        match = rec.find_match(store, {"note_id": "exp-001"})
        assert match is not None
        mem = rec.update(
            store,
            embedder,
            match,
            f"neutral {i}",
            {"note_id": "exp-001", "valence": 0.0, "arousal": 0.0, "approach": 0.0},
            valence=0.0,
            arousal=0.0,
            approach=0.0,
        )
    assert mem.tag.core_affect.valence < peak
    assert abs(mem.tag.core_affect.valence) < 0.05


def test_disabled_window_never_matches():
    em, store, _ = _em()
    rec = Reconsolidator(labile_window_seconds=0)
    em.set_affect(CoreAffect(valence=0.1, arousal=0.0))
    em.encode("a", metadata={"note_id": "exp-001"})
    assert rec.find_match(store, {"note_id": "exp-001"}) is None
