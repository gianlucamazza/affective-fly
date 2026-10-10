"""Space-ready Gradio wrapper around the numpy demo loop.

Same path as ``examples/demo_loop.py`` / ``make demo`` → ``demo_loop.png``:
``AffectiveLoop`` + ``MockFlyCircuit`` or ``LIFCircuit`` + ``FakeEmbedder``
+ ``viz.plot_journal``.

No Brian2, no C++ compiler, no MiniLM / sentence-transformers download.
"""

from __future__ import annotations

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")

import re  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
from emotional_memory import EmotionalMemory, InMemoryStore  # noqa: E402

from affective_fly import (  # noqa: E402
    ActionGate,
    ActionJournal,
    AffectiveLoop,
    FakeEmbedder,
    LIFCircuit,
    MockFlyCircuit,
    SensoryFrame,
    lab_mood_field,
)
from affective_fly.viz import plot_journal  # noqa: E402

matplotlib.use("Agg")

DEFAULT_SEED = 42
MAX_EVENTS = 32
CIRCUIT_MOCK = "Mock"
CIRCUIT_LIF = "LIF"
CIRCUIT_CHOICES = [CIRCUIT_MOCK, CIRCUIT_LIF]

# Exact ``examples/demo_loop.py`` episode (six journal successes, four failed reviews).
DEMO_LOOP_SCENARIOS: list[dict] = [
    *[
        {"context": "journal", "note_id": "exp-042", "sentiment": 1.0, "query": "success"}
        for _ in range(6)
    ],
    *[
        {"context": "review", "note_id": "exp-042", "sentiment": -1.0, "query": "failed"}
        for _ in range(4)
    ],
]

FAILED_REVIEWS_SCENARIOS: list[dict] = [
    {"context": "review", "note_id": "exp-042", "sentiment": -1.0, "query": "failed"}
    for _ in range(6)
]

SUSTAINED_SUCCESS_SCENARIOS: list[dict] = [
    {"context": "journal", "note_id": "exp-042", "sentiment": 1.0, "query": "success"}
    for _ in range(8)
]

# Pipe format round-trips the structured demo dicts so the Mock preset
# matches ``demo_loop.png``. Custom lines may also use ``+1.0 text``.
_PIPE_LINE = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*(.+?)\s*$"
)
_SENTIMENT_LINE = re.compile(r"^\s*([+-](?:\d+(?:\.\d+)?|\.\d+))\s+(.+?)\s*$")

TABLE_HEADERS = ["step", "event", "action", "valence", "arousal", "approach", "gate"]

INTRO_MD = """
# Affective Fly — demo loop

Toy affect loop: a reduced *Drosophila* mushroom-body circuit (Kenyon cells → DANs → MBONs)
turns a short sequence of text events into valence, arousal, and approach/avoid on Russell's
circumplex, then a discrete decision (wait / act / skip). The figure is the same two-panel
readout that `make demo` writes as `demo_loop.png` (circumplex trajectory + mood / action gate).

This is **not** a model of a real fly or of human emotion. Without a real connectome the
KC→MBON weights are **random**. The embedder is a **fake hashed stand-in** (`FakeEmbedder`) —
nothing is downloaded, and there is no MiniLM or other network model. The circuit path is
numpy only (no Brian2, no compiler).

Sibling demo: [emotional-memory](https://huggingface.co/spaces/homen3/emotional-memory-demo)
(the memory engine this library is the affect source for). Source:
[gianlucamazza/affective-fly](https://github.com/gianlucamazza/affective-fly).
"""

EVENTS_INFO = (
    "One event per line. Structured lines match the numpy demo exactly: "
    "`sentiment | context | note_id | query`. "
    "Or write `+1.0 success` / `-1.0 failed` / plain text (plain text uses sentiment 0)."
)

CIRCUIT_INFO = (
    "Mock is the linear heuristic `make demo` uses. "
    "LIF is the numpy leaky-integrate-and-fire circuit (default size, seeded). "
    "Both use random KC→MBON weights — there is no published connectome in this demo."
)


def _temp_path(suffix: str) -> Path:
    """Create a closed temp path so ActionJournal / matplotlib can reopen it."""
    fd, name = tempfile.mkstemp(prefix="afly-", suffix=suffix)
    os.close(fd)
    return Path(name)


def scenarios_to_text(scenarios: list[dict]) -> str:
    """Serialize structured events to the pipe format the parser accepts."""
    lines = []
    for row in scenarios:
        sentiment = float(row["sentiment"])
        lines.append(f"{sentiment:+.1f} | {row['context']} | {row['note_id']} | {row['query']}")
    return "\n".join(lines)


PRESETS: dict[str, list[dict]] = {
    "Demo loop (journal → review)": DEMO_LOOP_SCENARIOS,
    "Failed reviews only": FAILED_REVIEWS_SCENARIOS,
    "Sustained journal success": SUSTAINED_SUCCESS_SCENARIOS,
}

DEFAULT_PRESET = "Demo loop (journal → review)"
DEFAULT_EVENTS_TEXT = scenarios_to_text(DEMO_LOOP_SCENARIOS)


def parse_events(text: str) -> list[dict]:
    """Parse the events textbox into SensoryFrame-compatible dicts."""
    events: list[dict] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        pipe = _PIPE_LINE.match(line)
        if pipe:
            sentiment, context, note_id, query = pipe.groups()
            events.append(
                {
                    "context": context.strip(),
                    "note_id": note_id.strip(),
                    "sentiment": float(sentiment),
                    "query": query.strip(),
                }
            )
            continue
        signed = _SENTIMENT_LINE.match(line)
        if signed:
            sentiment, query = signed.groups()
            events.append(
                {
                    "context": "event",
                    "note_id": "user-seq",
                    "sentiment": float(sentiment),
                    "query": query.strip(),
                }
            )
            continue
        events.append(
            {
                "context": "event",
                "note_id": "user-seq",
                "sentiment": 0.0,
                "query": line,
            }
        )
    if not events:
        raise ValueError("Enter at least one event (one per line), or pick a preset.")
    if len(events) > MAX_EVENTS:
        raise ValueError(f"At most {MAX_EVENTS} events on cpu-basic; got {len(events)}.")
    return events


def _make_circuit(circuit_name: str, seed: int):
    name = circuit_name.strip()
    if name == CIRCUIT_MOCK:
        return MockFlyCircuit(seed=seed)
    if name == CIRCUIT_LIF:
        return LIFCircuit(seed=seed)
    raise ValueError(f"Unknown circuit {circuit_name!r}; choose Mock or LIF.")


def _display_reason(reason: str) -> str:
    """Pass through the library reason (already uses 'Action gate')."""
    return reason


def run_sequence(
    events_text: str,
    circuit_name: str,
    seed: float | int,
) -> tuple[str, list[list[object]], str]:
    """Run one seeded episode. Gradio callback — does not launch a server."""
    seed_i = int(seed)
    events = parse_events(events_text)
    journal = ActionJournal(filepath=_temp_path(".jsonl"))
    store = InMemoryStore()
    embedder = FakeEmbedder()
    loop = AffectiveLoop(
        fly_circuit=_make_circuit(circuit_name, seed_i),
        emotional_memory=EmotionalMemory(store=store, embedder=embedder),
        store=store,
        embedder=embedder,
        mood_field=lab_mood_field(),
        action_gate=ActionGate(required_ticks=3),
        journal=journal,
    )

    rows: list[list[object]] = []
    gate_opened_at: int | None = None
    for i, scenario in enumerate(events):
        loop.step(SensoryFrame.from_dict(scenario), encode_memory=True)
        entry = journal.entries[-1]
        gate_open = bool(entry.gate_open)
        if gate_open and gate_opened_at is None:
            gate_opened_at = i
        rows.append(
            [
                i,
                str(scenario.get("query", "")),
                entry.action,
                round(float(entry.mood_valence), 3),
                round(float(entry.mood_arousal), 3),
                round(float(entry.approach_tendency), 3),
                "open" if gate_open else "closed",
            ]
        )

    png = _temp_path(".png")
    plot_journal(journal.entries, png)
    opened = (
        f"gate opened at step {gate_opened_at}"
        if gate_opened_at is not None
        else "gate stayed closed"
    )
    last_reason = _display_reason(journal.entries[-1].reason) if journal.entries else ""
    summary = (
        f"{len(rows)} steps · {circuit_name} · seed {seed_i} · {opened}. "
        f"Last decision: {journal.entries[-1].action} ({last_reason})"
    )
    return str(png), rows, summary


def load_preset(name: str) -> str:
    """Fill the events box from a named preset."""
    scenarios = PRESETS.get(name)
    if scenarios is None:
        raise ValueError(f"Unknown preset {name!r}.")
    return scenarios_to_text(scenarios)


def build_interface():
    """Build the Gradio Blocks UI. Imported only when a server is needed."""
    import gradio as gr

    with gr.Blocks(title="Affective Fly — demo loop") as demo:
        gr.Markdown(INTRO_MD)
        with gr.Row():
            with gr.Column(scale=1):
                preset = gr.Dropdown(
                    choices=list(PRESETS.keys()),
                    value=DEFAULT_PRESET,
                    label="Preset",
                    info="The first preset is the exact `make demo` / demo_loop.png episode.",
                )
                events = gr.Textbox(
                    value=DEFAULT_EVENTS_TEXT,
                    lines=12,
                    label="Text events",
                    info=EVENTS_INFO,
                )
                circuit = gr.Radio(
                    choices=CIRCUIT_CHOICES,
                    value=CIRCUIT_MOCK,
                    label="Circuit",
                    info=CIRCUIT_INFO,
                )
                seed = gr.Number(value=DEFAULT_SEED, precision=0, label="Seed")
                run_btn = gr.Button("Run loop", variant="primary")
            with gr.Column(scale=2):
                plot = gr.Image(label="Circumplex + mood / action gate", type="filepath")
                table = gr.Dataframe(headers=TABLE_HEADERS, label="Per-step values")
                summary = gr.Markdown()
        inputs = [events, circuit, seed]
        outputs = [plot, table, summary]
        preset.change(fn=load_preset, inputs=[preset], outputs=[events])
        run_btn.click(fn=run_sequence, inputs=inputs, outputs=outputs)
        demo.load(fn=run_sequence, inputs=inputs, outputs=outputs)
    return demo


def main() -> None:
    """Launch the Space / local Gradio server. Not used by the smoke test."""
    demo = build_interface()
    demo.launch(show_error=True, ssr_mode=False)


if __name__ == "__main__":
    main()
