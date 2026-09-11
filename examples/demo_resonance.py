#!/usr/bin/env python3
"""Resonance links from emotional-memory after two related encodes."""

from emotional_memory import CoreAffect, EmotionalMemory, InMemoryStore

from affective_fly import FakeEmbedder


def main() -> None:
    em = EmotionalMemory(store=InMemoryStore(), embedder=FakeEmbedder())
    em.set_affect(CoreAffect(valence=-0.6, arousal=0.4))
    em.encode("chart MEME crash")
    em.set_affect(CoreAffect(valence=-0.5, arousal=0.3))
    em.encode("chart MEME dump")
    em.retrieve("MEME crash", top_k=2)
    print(f"{'id':<10}  {'n_links':>7}  links")
    for mem in em._store.list_all():
        links = mem.tag.resonance_links or []
        print(
            f"{mem.id[:8]:<10}  {len(links):7d}  {[(lk.link_type, round(lk.strength, 3)) for lk in links]}"
        )


if __name__ == "__main__":
    main()
