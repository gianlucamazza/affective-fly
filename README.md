# Affective Fly

**Drosophila mushroom-body valence circuit bridged to emotional-memory (Affective Field Theory) for persistent mood agents.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What is this?

**Affective Fly** is NOT another FLYBRAIN fork. It is a computational agent with **persistent mood**:

1. **Sensory frame** (image features, ticker symbol, message, virtual odor) enters fly receptors
2. From **mushroom-body (MB) readout** (MBON/DAN firing rates) extract a short vector: **valence, arousal, approach/avoid**
3. That becomes `CoreAffect` + optional `AppraisalVector` and is `encode()`'d into **EmotionalMemory** (Gianluca Mazza's AFT library)
4. Next turn retrieval is weighted by the fly's **current mood**; avoidance pulls aversive memories and skips actions; approach consolidates and repeats

### Honesty Layer

The fly does **NOT** "feel human sadness". It senses **approach/avoid drives + arousal**. Human emotion labels (fear, joy) are optional readouts, explicitly declared as interpretive approximations—not ground truth.

---

## Why it fits: Fly MB + Affective Field Theory

The Drosophila **mushroom-body learning circuit** is already valence-coded:

- **Kenyon cells (KC)**: Sparse stimulus representation (~2000 neurons, ~5% active)
- **Dopaminergic neurons (DAN)**: Reinforcement/outcome signal (PAM for reward, PPL1 for punishment)
- **Mushroom-body output neurons (MBON)**: Approach vs avoid output

`emotional-memory` ([PyPI](https://pypi.org/project/emotional-memory/), [GitHub](https://github.com/gianlucamazza/emotional-memory)) implements **Affective Field Theory** with 5 layers:

1. **CoreAffect**: Valence–arousal circumplex
2. **Momentum**: Short-term affective persistence
3. **MoodField**: Slow background mood (EMA over CoreAffect)
4. **Appraisal**: Optional LLM/heuristic appraisal of content
5. **Resonance**: Hebbian links between affective memories

Retrieval is **not text-cosine alone**—it's mood-congruent, decay-weighted, and resonance-amplified.

---

## Non-trivial Features (v1)

1. **MoodField as slow background**: EMA over MBON outputs so a market dump leaves mood for tens of minutes, not a spike
2. **Dual-path encoding (LeDoux-style)**: Fast path = CoreAffect from circuit only; slow path = optional LLM appraisal
3. **Reconsolidation**: Re-encountering similar stimulus inside labile window updates affective tag instead of duplicating
4. **Resonance graph hooks**: Hebbian links between contrasting cues (dark vs light, loss vs gain)
5. **Honesty layer**: Approach/avoid + arousal primary; human labels optional and labeled

---

## Quick Start

### Installation

```bash
# With uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .

# Optional: Brian2 for spiking network (not required for v1 mock/LIF)
uv pip install -e ".[brian2]"

# Development dependencies
uv pip install -e ".[dev]"
```

### Run Demos

```bash
# Basic affective loop
python examples/demo_loop.py

# Mood-conditioned launch gate
python examples/demo_mood_launch.py

# 8-agent swarm with shared memory
python examples/demo_swarm.py
```

### Run Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src/affective_fly --cov-report=term-missing

# Or via Makefile
make test
```

---

## Architecture

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for detailed design.

**Key components:**

- `fly_circuit.py`: MockFlyCircuit (deterministic) and LIFFlyCircuit (spiking stub)
- `affect_bridge.py`: MBON/DAN rates → CoreAffect (valence, arousal) in [-1, 1]
- `mood_field.py`: Slow EMA wrapper for persistent background mood
- `loop.py`: Sensory → fly → encode → retrieve → policy
- `policy.py`: Click/skip/type based on approach/avoid + retrieved memories
- `journal.py`: Action log with circumplex coordinates for visualization
- `launch_gate.py`: Mood-conditioned action approval (requires N sustained approach steps)
- `swarm.py`: N=8 agents, shared EmotionalMemory, resonance as "swarm culture"
- `honesty.py`: Optional human-label readout, clearly marked as interpretive

---

## Three Deliverable Surfaces

### 1. Fly Journal

Every action annotated with circumplex + mood. Export to JSON for AFT viz:

```python
from affective_fly import AffectiveLoop, SensoryFrame
import numpy as np

loop = AffectiveLoop(agent_id="fly-0")

for step in range(100):
    frame = SensoryFrame(
        features=np.random.randn(10),
        context=f"ticker: DOGE",
        metadata={"pnl": step * 10}
    )
    decision = loop.step(frame)

loop.journal.print_summary()
loop.journal.export_json("journal.json")
```

### 2. Mood-Conditioned Launch

Action only fires if MoodField approach is above threshold for N consecutive ticks:

```python
from affective_fly import AffectiveLoop, LaunchGate, SensoryFrame

loop = AffectiveLoop(agent_id="launch-fly")
gate = LaunchGate(required_steps=10, approach_threshold=0.2)

for step in range(100):
    frame = SensoryFrame(...)
    decision = loop.step(frame)
    mood = loop.get_mood()
    status = gate.check(mood)
    
    if status == MoodGateStatus.APPROVED:
        print("🚀 Launch approved!")
        # Execute high-stakes action
```

### 3. Swarm with Shared Memory

N=8 fly brains, one shared EmotionalMemory; resonances become swarm "culture":

```python
from affective_fly import FlySwarm, SwarmConfig, SensoryFrame

swarm = FlySwarm(config=SwarmConfig(n_agents=8))

for step in range(100):
    frame = SensoryFrame(...)
    decisions = swarm.step_all(frame)
    consensus = swarm.get_consensus_decision(decisions)

swarm.print_summary()
```

---

## MBON/DAN → CoreAffect Mapping

See [`docs/MAPPING_MBON_DAN.md`](docs/MAPPING_MBON_DAN.md) for explicit mapping.

**Summary:**

```
valence = tanh(k_approach * mbon_approach 
               - k_avoid * mbon_avoid 
               + k_dan * dan_reinforcement)

arousal = tanh(k_arousal * arousal_signal / baseline)
```

Default coefficients are heuristic estimates for mock/LIF circuits. For real MaleCNS v1.0 connectome data, coefficients should be fitted from recordings.

---

## What is Stub vs Real (v1)

| Component | Status | Notes |
|-----------|--------|-------|
| MockFlyCircuit | ✅ Real | Deterministic, CI-green, no network/LLM needed |
| LIFFlyCircuit | 🟡 Stub | Simplified LIF dynamics, placeholder for Brian2 |
| AffectBridge | ✅ Real | Explicit linear/affine mapping, testable |
| MoodField EMA | ✅ Real | Exponential moving average, configurable half-life |
| EmotionalMemory | ✅ Real | Uses Gianluca Mazza's published library (v0.18.0+) |
| Reconsolidation | 🟡 Stub | Simple context string match; TODO: semantic similarity |
| Resonance | ✅ Real | Delegated to emotional-memory library |
| MaleCNS neuron IDs | ❌ Placeholder | Clearly marked TODOs for PAM/PPL1/MBON-specific IDs |

---

## Roadmap

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for detailed plan.

**v1 (current)**: Mock/LIF circuits, InMemoryStore, 3 demos, pytest green  
**v2**: Real MaleCNS v1.0 neuron IDs, Brian2 full dynamics  
**v3**: Qdrant/Redis backend, distributed swarm, Hebbian plasticity  
**v4**: On-chain integration, real market/form data, GUI dashboard  

---

## Contributing

This is an experimental research scaffold. Contributions welcome, but:

1. **Maintain honesty layer**: Don't claim biological fidelity beyond stubs
2. **Test coverage**: All new code must have tests
3. **Type hints**: Use Python 3.11+ type annotations
4. **Ruff-compliant**: Run `ruff check src/` before committing

---

## Citation

If you use Affective Fly in research, please cite:

```bibtex
@software{affective_fly,
  title = {Affective Fly: Drosophila MB valence circuit bridged to Affective Field Theory},
  year = {2026},
  url = {https://github.com/YOUR_ORG/affective-fly}
}
```

And cite Gianluca Mazza's emotional-memory:

```bibtex
@software{emotional_memory,
  author = {Mazza, Gianluca},
  title = {emotional-memory: Affective Field Theory for LLM memory},
  year = {2024-2026},
  url = {https://github.com/gianlucamazza/emotional-memory}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **Gianluca Mazza**: emotional-memory (Affective Field Theory implementation)
- **Drosophila connectomics community**: MaleCNS, Janelia FlyEM, Hemibrain project
- **Brian2 developers**: Spiking neural network simulator

---

**Status**: v0.1.0 — Scaffold complete, real runnable demos, CI-green tests.
