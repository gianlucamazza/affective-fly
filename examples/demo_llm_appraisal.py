#!/usr/bin/env python3
"""
LLM-driven Scherer appraisal via DualPathEncoder.from_llm().

Requires environment variables:
- EMOTIONAL_MEMORY_LLM_API_KEY (or OPENAI_API_KEY)
- EMOTIONAL_MEMORY_LLM_MODEL (optional, default: gpt-4o-mini)
- EMOTIONAL_MEMORY_LLM_BASE_URL (optional, default: https://api.openai.com/v1)

Skips cleanly if keys are missing (no crash, exit 0).
"""

from emotional_memory import EmotionalMemory, InMemoryStore

from affective_fly import AffectiveLoop, DualPathEncoder, FakeEmbedder, LIFCircuit, SensoryFrame
from affective_fly.llm_env import build_llm_client


def main() -> None:
    llm_client = build_llm_client()

    if llm_client is None:
        print("LLM appraisal demo: skipping (no API key)")
        print("Set EMOTIONAL_MEMORY_LLM_API_KEY or OPENAI_API_KEY to enable")
        print("Optional: EMOTIONAL_MEMORY_LLM_MODEL, EMOTIONAL_MEMORY_LLM_BASE_URL")
        return

    print("=== LLM-driven appraisal (live OpenAI-compatible API call) ===")

    encoder = DualPathEncoder.from_llm(llm_client, fallback_on_error=True)

    store = InMemoryStore()
    em = EmotionalMemory(store=store, embedder=FakeEmbedder())

    loop = AffectiveLoop(
        fly_circuit=LIFCircuit(n_kc=500, n_dan=20, n_mbon=34, seed=42),
        emotional_memory=em,
    )

    frame = SensoryFrame.from_dict({
        "context": "journal",
        "note_id": "exp-session-042",
        "query": "successful experiment session notes",
        "sentiment": 0.8,
    })

    em.set_affect(loop.fly_circuit.core_affect(frame.sensory))
    mem = em.encode(frame.event_text(), metadata=frame.context)

    print(f"Event: {frame.event_text()}")
    print(f"Context: {frame.context}")

    appraisal = encoder.appraise(frame.event_text(), frame.context)
    updated = encoder.attach(em, mem, appraisal)

    print(f"\nLLM Appraisal:")
    print(f"  novelty:          {appraisal.novelty:+.3f}")
    print(f"  goal_relevance:   {appraisal.goal_relevance:+.3f}")
    print(f"  coping_potential: {appraisal.coping_potential:.3f}")
    print(f"  norm_congruence:  {appraisal.norm_congruence:+.3f}")
    print(f"  self_relevance:   {appraisal.self_relevance:.3f}")

    print(f"\nCircuit CoreAffect (preserved):")
    print(f"  valence: {updated.tag.core_affect.valence:+.3f}")
    print(f"  arousal: {updated.tag.core_affect.arousal:.3f}")


if __name__ == "__main__":
    main()
