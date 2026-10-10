---
title: Affective Fly — Demo Loop
emoji: 🪰
colorFrom: green
colorTo: indigo
sdk: gradio
sdk_version: "6.30.0"
python_version: "3.11"
app_file: app.py
license: mit
short_description: Toy fly-circuit affect loop on the circumplex
suggested_hardware: cpu-basic
tags:
  - affective-computing
  - drosophila
  - research
  - gradio
---

# Affective Fly — demo loop

Interactive, **cpu-basic** Gradio overlay for
[`affective-fly`](https://github.com/gianlucamazza/affective-fly).
It wraps the same numpy path that `make demo` uses to write `demo_loop.png`:
`AffectiveLoop` + `MockFlyCircuit` or `LIFCircuit` + `FakeEmbedder` +
`viz.plot_journal`.

Pick a preset (or type a short sequence of text events), choose the circuit,
and you get the two-panel figure — circumplex trajectory on the left, mood
valence / approach and the **action gate** on the right — plus a small
per-step table.

Intended for the **Research & demos** collection next to
[emotional-memory-demo](https://huggingface.co/spaces/homen3/emotional-memory-demo).

## What this is (and is not)

This is a **toy affect loop**, not a model of a real fly and not a model of
human emotion. Human-emotion words elsewhere in the library are circumplex
labels, not claims about fly experience.

**Without a real connectome the KC→MBON weights are random**, as the
[repo README](https://github.com/gianlucamazza/affective-fly) states. The LIF
option is a seeded numpy leaky-integrate-and-fire circuit with those random
weights; it is not MaleCNS anatomy.

The embedder is a **fake hashed stand-in** (`FakeEmbedder`). Nothing is
downloaded at runtime. There is no MiniLM, no `sentence-transformers`, no
Brian2, and no C++ compiler. The default path is numpy only.

The **action gate** only opens after several consecutive ticks of sufficient
approach and valence, then the policy may act (type/click) instead of wait.
Failed reviews can pull valence down; retrieved negative affect can yield
`skip`. That is a demo of the loop, not a trading or product signal.

## Local run

From a clone of this repository (library already on the numpy / viz extras):

```bash
uv sync --extra viz
uv pip install 'gradio==6.30.0'
uv run python spaces/demo/app.py
```

Or, matching the Space overlay:

```bash
python -m pip install -r spaces/demo/requirements.txt
python -m pip install 'gradio==6.30.0'
python spaces/demo/app.py
```

The smoke test imports the app and runs one callback; it does not start a server:

```bash
uv run pytest tests/test_space_demo.py -v
```

## Links

- **GitHub**: [gianlucamazza/affective-fly](https://github.com/gianlucamazza/affective-fly)
- **Sibling Space**: [homen3/emotional-memory-demo](https://huggingface.co/spaces/homen3/emotional-memory-demo)
- **emotional-memory**: [gianlucamazza/emotional-memory](https://github.com/gianlucamazza/emotional-memory)

## License

MIT, matching the repository. See
[LICENSE](https://github.com/gianlucamazza/affective-fly/blob/main/LICENSE).
