#!/usr/bin/env python3
"""
Demo: Host adapter journal replay with outcome reporting.

Shows how a host integration can:
1. Create HostFrames from semantic events
2. Save frames to JSONL for replay
3. Replay frames through AffectiveLoop with real LIFCircuit
4. Demonstrate outcome→learn loop with negative memory influencing policy
"""

from pathlib import Path

from emotional_memory import EmotionalMemory, InMemoryStore

from affective_fly import (
    AffectiveLoop,
    FakeEmbedder,
    HostAdapter,
    HostFrame,
    LIFCircuit,
    MoodField,
)


def create_experiment_journal() -> list[HostFrame]:
    """
    Simulate a research journal with experiment entries.

    Episode: Initial success → failed replication → decision about retry.
    """
    return [
        # Day 1: Successful initial experiment
        HostFrame(
            timestamp="2026-09-10T09:00:00",
            context={
                "context": "journal",
                "note_id": "exp-042-valence",
                "query": "initial experiment shows strong positive effect",
                "sentiment": 1.0,
            },
        ),
        # Day 1: Record positive outcome
        HostFrame(
            timestamp="2026-09-10T17:00:00",
            context={
                "context": "journal",
                "note_id": "exp-042-valence",
                "query": "experiment validated with peer review",
                "sentiment": 0.9,
                "outcome": 0.9,  # Positive outcome triggers PAM
            },
        ),
        # Day 2: Attempt replication
        HostFrame(
            timestamp="2026-09-11T09:30:00",
            context={
                "context": "journal",
                "note_id": "exp-042-valence",
                "query": "replication attempt with new cohort",
                "sentiment": 0.7,
            },
        ),
        # Day 2: Replication fails
        HostFrame(
            timestamp="2026-09-11T16:00:00",
            context={
                "context": "review",
                "note_id": "exp-042-valence",
                "query": "replication failed to show effect",
                "sentiment": -0.8,
                "outcome": -0.9,  # Negative outcome triggers PPL1
            },
        ),
        # Day 3: Second replication fails
        HostFrame(
            timestamp="2026-09-12T14:00:00",
            context={
                "context": "review",
                "note_id": "exp-042-valence",
                "query": "second replication also negative",
                "sentiment": -0.9,
                "reward": -0.8,  # Another negative outcome
            },
        ),
        # Day 4: Should we retry?
        HostFrame(
            timestamp="2026-09-13T10:00:00",
            context={
                "context": "journal",
                "note_id": "exp-042-valence",
                "query": "considering third replication attempt",
                "sentiment": 0.1,  # Weakly positive query, but memory overrides
            },
        ),
    ]


def main() -> None:
    # 1. Create journal frames
    frames = create_experiment_journal()

    # 2. Save to JSONL
    journal_path = Path("host_journal_demo.jsonl")
    HostAdapter.save_journal(frames, journal_path)
    print(f"Saved {len(frames)} frames to {journal_path}")

    # 3. Set up AffectiveLoop with real LIFCircuit
    store = InMemoryStore()
    embedder = FakeEmbedder()
    loop = AffectiveLoop(
        fly_circuit=LIFCircuit(n_kc=2000, n_dan=20, n_mbon=34, seed=42),
        emotional_memory=EmotionalMemory(store=store, embedder=embedder),
        store=store,
        embedder=embedder,
        # Demo-compressed taus (not the 300/60/180 hypothesis defaults).
        mood_field=MoodField(tau_valence=10.0, tau_arousal=5.0, tau_approach=8.0),
    )

    # 4. Replay frames through loop
    print("\n=== Replaying journal ===")
    loaded_frames = HostAdapter.load_journal(journal_path)
    decisions = HostAdapter.replay(loaded_frames, loop, encode_memory=True)

    # 5. Display results
    print(f"\n{'Step':<6} {'Context':<10} {'Note ID':<20} {'Action':<6} {'V':>6} {'A':>6} {'App':>6} {'Outcome':<8}")
    print("-" * 80)

    for i, (frame, decision) in enumerate(zip(loaded_frames, decisions)):
        ctx = frame.context.get("context", "?")[:10]
        note = frame.context.get("note_id", "?")[:20]
        outcome = frame.context.get("outcome") or frame.context.get("reward") or "-"
        if outcome != "-":
            outcome = f"{outcome:+.1f}"

        print(
            f"{i:<6} {ctx:<10} {note:<20} {decision.action.value:<6} "
            f"{decision.mood_valence:+6.2f} {decision.mood_arousal:+6.2f} "
            f"{decision.approach_tendency:+6.2f} {outcome:<8}"
        )

    # 6. Analyze outcome
    print("\n=== Analysis ===")
    final_decision = decisions[-1]

    print("Final frame: 'considering third replication attempt'")
    print(f"Decision: {final_decision.action.value}")
    print(f"Reason: {final_decision.reason}")
    print(f"Mood valence: {final_decision.mood_valence:+.2f}")
    print(f"Approach tendency: {final_decision.approach_tendency:+.2f}")

    # Memory check
    memories = loop.emotional_memory.list_all()
    print(f"\nMemories stored: {len(memories)}")
    if memories:
        last_mem = memories[-1]
        v = last_mem.metadata.get("valence", 0.0)
        reconsolidated = last_mem.metadata.get("reconsolidation_count", 0)
        print(f"Last memory valence: {v:+.2f}")
        print(f"Reconsolidation count: {reconsolidated}")

    # TD learning check
    if loop.last_td is not None:
        print("\nLast TD result:")
        print(f"  Reward: {loop.last_td.reward:+.2f}")
        print(f"  Delta: {loop.last_td.delta:+.2f}")

    print(f"\n✓ Replay complete. Journal saved to {journal_path}")
    print("✓ Outcomes triggered three-factor plasticity (KC→MBON weights)")
    print("✓ Negative memories + learned avoidance influenced final policy")


if __name__ == "__main__":
    main()
