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
pinned: true
tags:
  - affective-computing
  - drosophila
  - research
  - gradio
---

# Affective Fly — demo loop

An interactive Gradio demo of
[`affective-fly`](https://github.com/gianlucamazza/affective-fly), running on
**cpu-basic**. It wraps the same numpy path that `make demo` uses to write
`demo_loop.png`: `AffectiveLoop` + `MockFlyCircuit` or `LIFCircuit` +
`FakeEmbedder` + `viz.plot_journal`.

Pick a preset or type a short sequence of text events, then choose the circuit.
You get a two-panel figure and a small per-step table. The left panel shows the
trajectory on the circumplex. The right panel shows mood valence, approach and
the **action gate**.

## What this is (and is not)

This is a **toy affect loop**, not a model of a real fly and not a model of
human emotion. Human-emotion words elsewhere in the library are circumplex
labels, not claims about fly experience.

**Without a real connectome the KC→MBON weights are random**, as the
[repo README](https://github.com/gianlucamazza/affective-fly) states. The LIF
option is a seeded numpy leaky-integrate-and-fire circuit with those random
weights. It is not MaleCNS anatomy.

The embedder is a **fake hashed stand-in** (`FakeEmbedder`). No model is downloaded at runtime, and the app switches off Gradio analytics and Hugging Face Hub telemetry at start-up.
There is no MiniLM, no `sentence-transformers`, no Brian2 and no C++ compiler.
The default path is numpy only.

The **action gate** opens only after three consecutive ticks of sufficient
approach and valence. Then the policy may act (type or click) instead of
waiting. Failed reviews can pull valence down, and retrieved negative affect
can lead to `skip`. That is a demo of the loop, not a signal for real-world
decisions.

## Local run

From a clone of the [affective-fly GitHub repository](https://github.com/gianlucamazza/affective-fly):

```bash
uv sync --extra viz
uv pip install 'gradio==6.30.0'
uv run python spaces/demo/app.py
```

Or, with the same requirements as this Space:

```bash
python -m pip install -r spaces/demo/requirements.txt
python -m pip install 'gradio==6.30.0'
python spaces/demo/app.py
```

The smoke test imports the app and runs its callbacks. It does not start a server:

```bash
uv run pytest tests/test_space_demo.py -v
```

## Links

- **GitHub**: [gianlucamazza/affective-fly](https://github.com/gianlucamazza/affective-fly)
- **Sibling Space**: [gianlucamazza/emotional-memory-demo](https://huggingface.co/spaces/gianlucamazza/emotional-memory-demo)
- **emotional-memory**: [gianlucamazza/emotional-memory](https://github.com/gianlucamazza/emotional-memory)

## License

MIT, matching the repository. See
[LICENSE](https://github.com/gianlucamazza/affective-fly/blob/main/LICENSE).

**Author**: [Gianluca Mazza](https://gianlucamazza.it). I take LLM systems into production: state, recovery, eval, cost.
