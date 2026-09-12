"""Tests for dual-path appraisal (slow path does not overwrite circuit affect)."""

import pytest
from emotional_memory import CoreAffect, EmotionalMemory, InMemoryStore

from affective_fly import DualPathEncoder, FakeEmbedder, HeuristicAppraisalEngine


def _memory_with_circuit_affect(valence: float = 0.8, arousal: float = 0.2):
    store = InMemoryStore()
    em = EmotionalMemory(store=store, embedder=FakeEmbedder())
    em.set_affect(CoreAffect(valence=valence, arousal=arousal))
    mem = em.encode("frame: successful experiment", metadata={"note_id": "exp-001", "sentiment": 1.0})
    return em, mem


def test_heuristic_crash_is_aversive():
    engine = HeuristicAppraisalEngine()
    crash = engine.appraise("experiment failed", {"note_id": "exp-001", "sentiment": -0.8, "context": "review"})
    launch = engine.appraise(
        "successful experiment", {"note_id": "exp-002", "sentiment": 0.8, "context": "journal"}
    )
    assert crash.norm_congruence < launch.norm_congruence
    assert crash.coping_potential < launch.coping_potential
    assert crash.goal_relevance < launch.goal_relevance


def test_heuristic_novelty_drops_on_repeat():
    engine = HeuristicAppraisalEngine()
    first = engine.appraise("successful experiment", {"note_id": "exp-001"})
    second = engine.appraise("successful experiment", {"note_id": "exp-001"})
    assert first.novelty > second.novelty


def test_heuristic_wallet_is_self_relevant():
    engine = HeuristicAppraisalEngine()
    wallet = engine.appraise("review my results", {"context": "review", "note_id": "exp-001"})
    other = engine.appraise("browse", {"context": "journal"})
    assert wallet.self_relevance > other.self_relevance


def test_attach_preserves_circuit_core_affect():
    encoder = DualPathEncoder()
    em, mem = _memory_with_circuit_affect(valence=0.8, arousal=0.2)
    appraisal = encoder.appraise("successful experiment", {"note_id": "exp-001", "sentiment": 1.0})
    updated = encoder.attach(em, mem, appraisal)
    assert updated.tag.core_affect.valence == 0.8
    assert updated.tag.core_affect.arousal == 0.2
    assert updated.tag.appraisal is not None
    assert updated.tag.appraisal.novelty == appraisal.novelty
    assert updated.tag.pending_appraisal is False


def test_from_llm_uses_callable_not_network():
    import json

    calls = {"n": 0}

    def fake_llm(prompt: str, schema: dict) -> str:
        calls["n"] += 1
        return json.dumps(
            {
                "novelty": 0.4,
                "goal_relevance": 0.2,
                "coping_potential": 0.6,
                "norm_congruence": -0.1,
                "self_relevance": 0.5,
            }
        )

    encoder = DualPathEncoder.from_llm(fake_llm, fallback_on_error=False)
    vector = encoder.appraise("successful experiment", {"note_id": "exp-001"})
    assert calls["n"] == 1
    assert vector.novelty == pytest.approx(0.4)
    assert vector.self_relevance == pytest.approx(0.5)


def test_dual_path_cache_hit_does_not_consume_novelty():
    encoder = DualPathEncoder()
    ctx = {"note_id": "exp-001", "query": "successful experiment"}
    a = encoder.appraise("successful experiment", ctx)
    b = encoder.appraise("successful experiment", ctx)
    assert a.novelty == b.novelty
    # a different query with same note_id still counts as a repeat at the engine
    # only when it misses the cache
    c = encoder.appraise("successful experiment again", ctx)
    assert c.novelty < a.novelty
