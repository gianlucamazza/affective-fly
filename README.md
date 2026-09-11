# Affective Fly

**Drosophila mushroom-body valence circuit bridged to Affective Field Theory for persistent mood agents**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

Affective Fly bridges neuroscience-inspired valence circuits from *Drosophila* (fruit fly) mushroom body with the [`emotional-memory`](https://github.com/gianlucamazza/emotional-memory) library (Affective Field Theory / AFT). This creates **organisms with persistent mood** that learn and make decisions based on affective state, not just semantic similarity.

### What This Is NOT

- **Not another FLYBRAIN fork**: We don't simulate the entire fly nervous system
- **Not a human emotion simulator**: Flies sense approach/avoid + arousal, not human "sadness" or "joy"
- **Not a black-box ML model**: All mappings from neural activity to affect are explicit and testable

### What This IS

An **affective agent architecture** where:

1. **Sensory input** (visual, virtual odor, context) enters simplified fly receptors
2. **Mushroom body circuits** (Kenyon cells + DANs + MBONs) generate valence/arousal readouts
3. **Affect bridge** maps MBON/DAN firing rates to `CoreAffect` (valence, arousal in [-1, 1])
4. **MoodField** maintains persistent mood via exponential moving average (EMA)
5. **EmotionalMemory** encodes events with affective tags and retrieves mood-congruently
6. **Policy** decides actions based on approach/avoid + retrieved memories

## Key Features

### 1. **Persistent Mood (MoodField)**
Mood decays slowly (5-minute time constant for valence) so a market crash or failed form leaves mood depressed for tens of minutes, not just a spike-and-gone.

### 2. **Dual-Path Encoding** (LeDoux-style)
- **Fast path**: CoreAffect directly from fly circuit (approach/avoid + arousal)
- **Slow path**: Optional LLM/heuristic appraisal for cognitive dimensions (novelty, controllability, goal relevance)

### 3. **Reconsolidation**
Re-encountering the same stimulus (ticker / event) within a labile window updates the affective tag instead of duplicating the memory (computational analogue of extinction/reconsolidation).

### 4. **Mood-Conditioned Launch Gate**
Actions only trigger when mood criteria (approach + valence) hold for N consecutive ticks. Prevents impulsive decisions during avoidant states.

### 5. **NeuroSwarm × AFT**
Multiple fly brains (N=8 default) with independent circuits share one `EmotionalMemory`. Resonances and memory become swarm "culture".

### 6. **Honesty Layer**
Human emotion labels (fear, joy, sadness) are **optional interpretive readouts**, explicitly marked as heuristic projections. The primary observables are approach/avoid + arousal.

### 7. **TD Plasticity**
Outcomes (`reward` / `outcome` / `pnl` in the sensory context) update KC→MBON weights with a three-factor rule: KC eligibility × DAN gate × prediction error `δ = r − V`.

## Installation

### Using `uv` (recommended)

```bash
# Clone the repository
git clone https://github.com/gianlucamazza/affective-fly.git
cd affective-fly

# Install with uv
uv sync

# Or with all extras (Brian2, viz)
uv sync --all-extras
# Brian2 2.9 on Python 3.11 needs numpy 1.x (pinned in pyproject).
```

### Using pip

```bash
pip install -e .

# With optional dependencies
pip install -e ".[dev,brian,viz]"
```

## Quick Start

### Run the Demos

```bash
# Complete affective loop
make demo
# Or individually:
uv run python examples/demo_loop.py
uv run python examples/demo_mood_launch.py
uv run python examples/demo_swarm.py
uv run python examples/demo_td.py
```

### Run Tests

**Important**: Tests require dev dependencies. Install with `uv sync --all-extras` or `make install` first.

```bash
# Install all dependencies (required before first test run)
uv sync --all-extras
# Or:
make install

# Then run tests
make test
# Or:
uv run pytest tests/ -v
```

### Basic Usage

```python
from emotional_memory import EmotionalMemory, InMemoryStore
from affective_fly import (
    FakeEmbedder,
    MockFlyCircuit,
    AffectiveLoop,
    SensoryFrame,
)

# Initialize components
fly_circuit = MockFlyCircuit(seed=42)
embedder = FakeEmbedder()  # For offline demos/tests
emotional_memory = EmotionalMemory(store=InMemoryStore(), embedder=embedder)

# Create affective loop
loop = AffectiveLoop(
    fly_circuit=fly_circuit,
    emotional_memory=emotional_memory,
)

# Step through sensory frames
frame = SensoryFrame.from_dict({
    "page": "launchpad",
    "ticker": "MEME",
    "sentiment": 0.8,
    "query": "launch token"
})

decision = loop.step(frame, encode_memory=True)

print(f"Action: {decision.action}")
print(f"Mood: V={decision.mood_valence:.2f}, A={decision.mood_arousal:.2f}")
print(f"Reason: {decision.reason}")
```

## Architecture

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for detailed system design.

### Component Overview

```
┌──────────────┐
│ SensoryFrame │  (visual vector + context dict)
└──────┬───────┘
       │
       v
┌──────────────┐
│ FlyCircuit   │  (KC → DAN/MBON)
│ - MockFly    │
│ - LIFCircuit │
│ - Brian2     │
│ - MaleCNS    │  (Aso 2014 names; random weights)
└──────┬───────┘
       │
       v
┌──────────────┐
│  TD learn()  │  (reward → δ → Δw KC→MBON)
└──────┬───────┘
       │
       v
┌──────────────┐
│ AffectBridge │  (MBON/DAN rates → CoreAffect)
└──────┬───────┘
       │
       v
┌──────────────┐
│  MoodField   │  (slow EMA: 5min valence, 1min arousal)
└──────┬───────┘
       │
       v
┌──────────────┐
│EmotionalMem. │  (encode or reconsolidate; mood-weighted retrieve)
│ DualPath     │  (slow Scherer appraisal attached, not blended)
└──────┬───────┘
       │
       v
┌──────────────┐
│   Policy     │  (approach/avoid + memories → action)
└──────┬───────┘
       │
       v
┌──────────────┐
│  LaunchGate  │  (CLICK/TYPE only after N ticks of approach+valence)
└──────┬───────┘
       │
       v
┌──────────────┐
│   Journal    │  (log decisions with circumplex coords)
└──────────────┘
```

## Circuit → Affect Mapping

The mapping from MBON/DAN firing rates to `CoreAffect` is **explicit and testable**:

### Valence
```
valence = (approach - avoid) / (approach + avoid + ε)
```
- Pure approach (80 Hz approach, 10 Hz avoid) → +1.0
- Pure avoid (10 Hz approach, 80 Hz avoid) → -1.0
- Balanced → 0.0

### Arousal
```
arousal = (DAN_rate - baseline) / (max - baseline)
```
- Low DAN activity (≤5 Hz) → -1.0 (calm)
- High DAN activity (≥80 Hz) → +1.0 (excited)

See [`docs/MAPPING_MBON_DAN.md`](docs/MAPPING_MBON_DAN.md) for biological grounding and MaleCNS integration roadmap.

## Documentation

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)**: System design and component interactions
- **[MAPPING_MBON_DAN.md](docs/MAPPING_MBON_DAN.md)**: Neural circuit to affect mapping
- **[ROADMAP.md](docs/ROADMAP.md)**: Current status and future plans

## Testing

Comprehensive test suite with pytest:

```bash
# Run all tests
make test

# With coverage
uv run pytest tests/ --cov=affective_fly --cov-report=html

# Specific test file
uv run pytest tests/test_fly_circuit.py -v
```

All tests run **offline with mocks** (no network, no real LLM required).

## Honesty & Scientific Integrity

### What We Claim

- Fly MB circuits implement **valence-coded learning** (approach vs. avoid)
- DANs signal **reinforcement** (reward/punishment)
- MBONs output **behavioral tendency** (approach vs. avoid)
- This maps cleanly to **circumplex affect** (valence + arousal)

### What We DON'T Claim

- Flies experience **human emotions** (they don't feel "sad" or "joyful")
- Our circuits are **biologically accurate** (this is a functional abstraction)
- Current circuits use **MaleCNS/Schlegel body IDs or connectome weights** (v0.2 labels published Aso types; synapses stay random)

Human emotion labels (`honesty.py`) are **interpretive readouts only**, clearly marked as heuristic projections from the circumplex.

## Future Work

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for details:

- [x] Published Aso cell-type labels (`MaleCNSCircuit`; connectome weights still pending)
- [x] Brian2-based spiking implementation (`uv sync --extra brian`)
- [x] Reconsolidation during labile window
- [x] Dual-path heuristic appraisal (LLM optional later)
- [x] TD learning (outcome → DAN → KC→MBON; `learn()` / context `reward`)
- [ ] MaleCNS/Schlegel connectome weights (HDF5/JSON; no invented IDs)
- [ ] Hebbian resonance graph hooks
- [ ] Concrete LLM client (OpenAI / Ollama) on `DualPathEncoder.from_llm`
- [ ] On-chain integration (token launches, PnL tracking)

## Contributing

Contributions welcome! Please:

1. Follow conventional commits style
2. Add tests for new features
3. Update docs as needed
4. Run `make lint` and `make test` before submitting

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use Affective Fly in research, please cite:

- Mazza, G. (2024). *Emotional Memory: Affective Field Theory*. [GitHub](https://github.com/gianlucamazza/emotional-memory)
- Russell, J. A. (1980). A circumplex model of affect. *Journal of Personality and Social Psychology*, 39(6), 1161-1178.
- Aso, Y., et al. (2014). The neuronal architecture of the mushroom body provides a logic for associative learning. *eLife*, 3, e04577.

---

**Status**: v0.2.0 — Dual-path, reconsolidation, Brian2, Aso labels, TD bandit. Connectome weights pending.

**Disclaimer**: This is experimental research software. Fly circuits are simplified abstractions. Human emotion labels are interpretive, not ground truth.
