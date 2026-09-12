# Host Integration Guide

This document explains how to integrate Affective Fly with a host system (browser extension, API service, journal app) using the Phase 2 host adapter infrastructure.

## Overview

A host integration consists of three parts:

1. **Host-side frame schema** — Structured JSON frames with semantic fields and outcomes
2. **Journal replay** — Ability to replay recorded frames through the affective loop
3. **Outcome reporting** — Host supplies reward/outcome/pnl so `learn()` runs on honest signals

**No NullHost stubs**: A host integration must be based on real frames and real outcomes, not invented data.

## HostFrame Schema (Version 1.0)

`HostFrame` is the stable contract between a host system and Affective Fly. It is JSON-serializable and can be logged, streamed, or replayed.

### Required Fields

```json
{
  "schema_version": "1.0",
  "timestamp": "2026-09-12T12:30:00.000Z",
  "context": {
    "context": "journal"
  }
}
```

- `schema_version`: Must be `"1.0"` for this version of the schema
- `timestamp`: ISO 8601 timestamp of when the frame was created
- `context`: Dictionary of semantic fields (see below)

### Context Fields

The `context` dict provides semantic information about the frame:

#### Required
- `context`: str — Semantic context category (e.g., `"journal"`, `"review"`, `"form"`, `"page"`)

#### Stimulus Identifiers (at least one recommended)
- `note_id`: str — Identifier for journal entries, notes, or documents
- `ticker`: str — Identifier for financial instruments or tracked entities
- `event`: str — Event name or identifier
- `page`: str — Page identifier in a web context

#### Content and Query
- `query`: str — Retrieval query or question (used for memory retrieval)
- `sentiment`: float in [-1, 1] — Declared sentiment bias (optional but recommended)

#### Outcomes (triggers plasticity when present)
- `reward`: float in [-1, 1] — Generic reward signal
- `outcome`: float in [-1, 1] — Task outcome signal
- `pnl`: float in [-1, 1] — Profit/loss or numeric outcome (any domain)

**Important**: When any of `reward`, `outcome`, or `pnl` is present, the frame triggers three-factor KC→MBON plasticity (`learn()`). The host must provide honest outcome signals — no invented rewards.

### Visual Input (optional)

```json
{
  "visual_hash": "a1b2c3d4",
  "visual_data": {
    "dom_structure": "...",
    "screenshot_url": "..."
  }
}
```

- `visual_hash`: str — Seed for deterministic visual vector generation when set by the host at frame creation. Used as the hash input to `to_sensory_frame()`. Important: `sensory_frame_to_host_frame()` stores a fingerprint of the actual visual vector and does NOT round-trip to the same visual via that fingerprint — exact replay requires logging the host-chosen `visual_hash` at frame creation (or logging context that seeds the vector).
- `visual_data`: dict — Host-specific visual encoding (e.g., DOM structure, screenshot metadata)

If neither `visual_hash` nor `visual_data` is provided, the visual vector is generated from the `context` dict hash.

### Example Frames

#### Research Journal Entry
```json
{
  "schema_version": "1.0",
  "timestamp": "2026-09-12T10:00:00",
  "context": {
    "context": "journal",
    "note_id": "exp-042-replication",
    "query": "successful replication of experiment 042",
    "sentiment": 1.0
  }
}
```

#### Failed Replication with Outcome
```json
{
  "schema_version": "1.0",
  "timestamp": "2026-09-12T14:30:00",
  "context": {
    "context": "review",
    "note_id": "exp-042-replication",
    "query": "replication failed validation",
    "sentiment": -0.9,
    "outcome": -0.8
  }
}
```

## Host Adapter API

### Creating and Converting Frames

```python
from affective_fly import HostFrame, HostAdapter

# Create a HostFrame
frame = HostFrame(
    context={
        "context": "journal",
        "note_id": "exp-001",
        "sentiment": 0.8,
        "query": "successful experiment"
    }
)

# Convert to SensoryFrame for loop.step()
sensory_frame = frame.to_sensory_frame(dim=64)

# Run through affective loop
decision = loop.step(sensory_frame, encode_memory=True)
```

### Saving and Loading Journals

```python
from pathlib import Path
from affective_fly import HostAdapter, HostFrame

# Create frames
frames = [
    HostFrame(context={"context": "journal", "note_id": "exp-001", "sentiment": 1.0}),
    HostFrame(context={"context": "review", "note_id": "exp-001", "outcome": -0.8}),
]

# Save to JSONL
HostAdapter.save_journal(frames, "frames.jsonl")

# Load from JSONL
loaded_frames = HostAdapter.load_journal("frames.jsonl")
```

### Replaying Journals

```python
from emotional_memory import EmotionalMemory, SQLiteStore
from affective_fly import (
    AffectiveLoop,
    LIFCircuit,
    FakeEmbedder,
    HostAdapter,
)

# Set up loop (store/embedder are injected so reconsolidation and appraisal
# attach persist through the same instances the engine uses)
store = SQLiteStore("affective_fly.db")
embedder = FakeEmbedder()
loop = AffectiveLoop(
    fly_circuit=LIFCircuit(n_kc=2000, n_dan=20, n_mbon=34, seed=42),
    emotional_memory=EmotionalMemory(store=store, embedder=embedder),
    store=store,
    embedder=embedder,
)

# Load and replay frames
frames = HostAdapter.load_journal("recorded_frames.jsonl")
decisions = HostAdapter.replay(
    frames,
    loop,
    encode_memory=True,
    retrieve_top_k=5
)

# Analyze decisions
for frame, decision in zip(frames, decisions):
    print(f"{frame.context.get('note_id')}: {decision.action.value} "
          f"(V={decision.mood_valence:.2f})")
```

## Integration Patterns

### Pattern 1: Browser Extension

A browser extension can create HostFrames from user interactions:

```python
def on_page_view(dom_hash: str, page_title: str, sentiment: float):
    """User viewed a page."""
    frame = HostFrame(
        visual_hash=dom_hash,
        context={
            "context": "page",
            "page": page_title,
            "sentiment": sentiment,
            "query": f"viewing {page_title}"
        }
    )
    # Stream to affective loop or log for replay
    return frame

def on_task_completed(task_id: str, success: bool):
    """User completed a task."""
    frame = HostFrame(
        context={
            "context": "task",
            "event": task_id,
            "outcome": 0.8 if success else -0.6,
            "query": f"completed task {task_id}"
        }
    )
    return frame
```

### Pattern 2: Research Journal

A research journal app can log entries with outcomes:

```python
def log_experiment_result(note_id: str, content: str, outcome_score: float):
    """Log an experiment result."""
    frame = HostFrame(
        context={
            "context": "journal",
            "note_id": note_id,
            "query": content,
            "outcome": outcome_score,
            "sentiment": outcome_score  # Match outcome to sentiment
        }
    )
    # Append to journal
    HostAdapter.save_journal([frame], "journal.jsonl")
    
    # Also run through loop
    sf = frame.to_sensory_frame()
    decision = loop.step(sf, encode_memory=True)
    return decision
```

### Pattern 3: API Event Stream

An API service can stream events as HostFrames:

```python
def handle_api_event(event_type: str, payload: dict):
    """Process an API event."""
    # Extract sentiment and outcome from payload
    sentiment = payload.get("sentiment", 0.0)
    outcome = payload.get("outcome")  # None if no outcome
    
    frame = HostFrame(
        context={
            "context": "api",
            "event": event_type,
            "query": payload.get("query", event_type),
            "sentiment": sentiment,
            **({"outcome": outcome} if outcome is not None else {}),
        }
    )
    
    # Process through loop
    sf = frame.to_sensory_frame()
    decision = loop.step(sf, encode_memory=True)
    
    # Return decision to API caller
    return {
        "action": decision.action.value,
        "confidence": decision.confidence,
        "mood_valence": decision.mood_valence,
    }
```

## Outcome Reporting

### When to Report Outcomes

Report an outcome when:
- A task completes (success/failure)
- A prediction can be validated (correct/incorrect)
- A trade settles (profit/loss)
- An experiment finishes (positive/negative result)
- User provides feedback (satisfied/dissatisfied)

### Outcome Signal Guidelines

- Use `reward` for generic positive/negative signals
- Use `outcome` for task results (1.0 = complete success, -1.0 = complete failure)
- Use `pnl` for financial profit/loss (normalized to [-1, 1])
- Clip all outcomes to [-1, 1]
- **Never invent outcomes** — if you don't have an outcome, omit the field

### Sequential Learning

To enable sequential learning (delayed US), set `td_sequential=True` when creating the loop:

```python
loop = AffectiveLoop(
    fly_circuit=circuit,
    emotional_memory=memory,
    store=store,          # same instances backing `memory`
    embedder=embedder,
    td_sequential=True,  # Delayed US writes onto previous frame's eligibility
)
```

This is useful for classical conditioning scenarios where the outcome (US) arrives after the stimulus (CS).

## Schema Versioning

The current schema version is **1.0**. Future versions may add fields but will maintain backward compatibility for required fields.

When loading journals, the adapter warns if `schema_version` does not match the current version but still attempts to parse the frame.

### Forward Compatibility

To ensure forward compatibility:
1. Always set `schema_version` when creating frames
2. When reading frames, check `schema_version` and handle unknown versions gracefully
3. Ignore unknown fields in the `context` dict
4. Required fields (`schema_version`, `timestamp`, `context.context`) must always be present

## Testing Host Integrations

Use the provided test utilities:

```python
from affective_fly import HostAdapter, HostFrame, AffectiveLoop, LIFCircuit
from emotional_memory import EmotionalMemory, InMemoryStore

# Create test frames
frames = [
    HostFrame(context={"context": "test", "note_id": "t1", "sentiment": 1.0}),
    HostFrame(context={"context": "test", "note_id": "t1", "outcome": -0.8}),
]

# Replay through real circuit
store = InMemoryStore()
embedder = FakeEmbedder()
loop = AffectiveLoop(
    fly_circuit=LIFCircuit(n_kc=200, n_dan=10, n_mbon=20, seed=42),
    emotional_memory=EmotionalMemory(store=store, embedder=embedder),
    store=store,
    embedder=embedder,
)

decisions = HostAdapter.replay(frames, loop, encode_memory=True)

# Verify outcome triggered learning
assert loop.last_td is not None
assert loop.last_td.reward == -0.8
```

See `tests/test_host_adapter.py` for comprehensive test examples.

## No NullHost Pattern

**Do not create a NullHost** that invents outcomes or generates synthetic frames without a real data source. A host integration must:

1. Derive frames from real events (user actions, API calls, file reads)
2. Report real outcomes when available (not invented or randomized)
3. Use journal replay with recorded frames, not generated sequences

If your integration doesn't have outcomes yet, omit the outcome fields — don't generate fake ones.

## Migration from Direct SensoryFrame Usage

If you were previously using `SensoryFrame.from_dict()` directly, migrate to HostFrame:

### Before (v0.2.5)
```python
frame = SensoryFrame.from_dict({
    "context": "journal",
    "note_id": "exp-001",
    "sentiment": 0.8
})
decision = loop.step(frame)
```

### After (Phase 2)
```python
host_frame = HostFrame(context={
    "context": "journal",
    "note_id": "exp-001",
    "sentiment": 0.8
})
sensory_frame = host_frame.to_sensory_frame()
decision = loop.step(sensory_frame)

# Can now also save for replay
HostAdapter.save_journal([host_frame], "journal.jsonl")
```

`SensoryFrame.from_dict()` remains supported but HostFrame is the recommended integration path.

## See Also

- `examples/demo_host_replay.py` — Full replay demonstration
- `tests/test_host_adapter.py` — Comprehensive test suite
- `docs/ARCHITECTURE_COMPLETE.md` — System architecture
- `docs/ROADMAP.md` — Development phases

## Phase 3: LLM and Embedder Configuration

### Environment Variables for LLM Appraisal

`DualPathEncoder.from_llm()` supports OpenAI-compatible LLM providers via environment variables. Set these before running `demo_llm_appraisal.py` or any code using LLM appraisal:

**Required:**
- `EMOTIONAL_MEMORY_LLM_API_KEY` (or `OPENAI_API_KEY` as fallback) — API key for your LLM provider

**Optional:**
- `EMOTIONAL_MEMORY_LLM_BASE_URL` — Base URL for API (default: `https://api.openai.com/v1`)
- `EMOTIONAL_MEMORY_LLM_MODEL` — Model name (default: `gpt-4o-mini`)

**Example:**
```bash
export EMOTIONAL_MEMORY_LLM_API_KEY="sk-..."
export EMOTIONAL_MEMORY_LLM_MODEL="gpt-5-mini"
python examples/demo_llm_appraisal.py
```

If keys are missing, demos skip cleanly with a message (exit 0, no crash). This ensures CI stays green without secrets.

### Embedder Configuration

The semantic embedder (`SentenceTransformerEmbedder`) requires the `embed` extra:

```bash
uv sync --extra embed
python examples/demo_embedder.py
```

**Optional:**
- `HF_TOKEN` — HuggingFace token for private model downloads (rarely needed for public models)

If the extra is not installed or model download fails, the demo skips cleanly.

### Integration with Host Systems

When integrating LLM appraisal in production:

1. Set environment variables in your deployment (Kubernetes secrets, .env file, etc.)
2. **Never commit secrets to the repository**
3. Use `affective_fly.llm_env.build_llm_client()` to get a client (returns `None` if keys missing)
4. Fall back to `HeuristicAppraisalEngine` when client is `None`

```python
from affective_fly import DualPathEncoder, HeuristicAppraisalEngine
from affective_fly.llm_env import build_llm_client

llm_client = build_llm_client()
if llm_client is not None:
    encoder = DualPathEncoder.from_llm(llm_client, fallback_on_error=True)
else:
    encoder = DualPathEncoder(engine=HeuristicAppraisalEngine())
```

This pattern ensures your code works offline (CI, development without keys) and uses LLM when available (production with credentials).
