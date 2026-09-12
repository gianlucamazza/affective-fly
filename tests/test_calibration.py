"""Circuit rates stay inside the band documented in docs/MAPPING_MBON_DAN.md.

These are the regression tests for the v0.2.4 calibration pass: before it,
LIFCircuit ran an order of magnitude below the band and Brian2Circuit an order
above it, so Policy and LaunchGate were only ever exercised by MockFlyCircuit.
"""

from pathlib import Path

import numpy as np
import pytest

from affective_fly import AffectBridge, MaleCNSCircuit, MockFlyCircuit
from affective_fly.fly_circuit import (
    SYN_GAIN_EXPONENT,
    SYN_GAIN_PREFACTOR,
    LIFCircuit,
    MBONDanState,
    default_syn_gain,
    mean_column_fan_in,
)

PUBLISHED_CONNECTOME = Path("data/malecns/kc_mbon_connectivity.feather")
TOP200_CONNECTOME = Path("tests/fixtures/malecns_kc_mbon_real_top200.json")

MBON_MAX = 100.0
DAN_MAX = 80.0


def _drive(circuit, n=15, dim=64, seed=0):
    """Untrained rates over varied stimuli, skipping the spike-window warmup."""
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n):
        s = rng.randn(dim) * 0.5 + rng.choice([-0.8, -0.3, 0.3, 0.8])
        state = circuit.step(s, dt=0.05)
        if i >= 3:
            rows.append(
                (
                    state.mbon_approach_rate + state.mbon_avoid_rate,
                    state.dan_reinforcement_rate,
                )
            )
    return np.array(rows)


@pytest.mark.parametrize("n_kc,n_dan,n_mbon", [(40, 6, 8), (80, 8, 16), (200, 20, 34), (2000, 20, 34)])
def test_lif_rates_in_band_at_every_population_size(n_kc, n_dan, n_mbon):
    """syn_gain is derived from n_kc, so a small circuit is not a different model."""
    rates = _drive(LIFCircuit(n_kc=n_kc, n_dan=n_dan, n_mbon=n_mbon, seed=1))
    mbon_total, dan = rates[:, 0].mean(), rates[:, 1].mean()
    assert 0.0 < mbon_total <= MBON_MAX, f"MBON {mbon_total:.1f} Hz outside band"
    assert dan <= DAN_MAX, f"DAN {dan:.1f} Hz outside band"


def test_mock_rates_in_band():
    rates = _drive(MockFlyCircuit(seed=42))
    assert 0.0 < rates[:, 0].mean() <= MBON_MAX
    assert rates[:, 1].mean() <= DAN_MAX


def test_saturated_lif_stays_under_mbon_ceiling():
    """Weight bounds must keep a fully trained circuit inside the band."""
    circuit = LIFCircuit(n_kc=2000, seed=1)
    odor = np.random.RandomState(7).randn(64) * 0.5 + 0.8
    for _ in range(60):
        circuit.step(odor, dt=0.05)
        circuit.learn(1.0)
    state = circuit.step(odor, dt=0.05)
    assert state.mbon_approach_rate <= MBON_MAX
    assert state.mbon_avoid_rate <= MBON_MAX


def test_learning_potentiates_before_it_clips():
    """Initial weights must sit under w_max, or the first learn() only clips."""
    circuit = LIFCircuit(n_kc=200, seed=3)
    assert circuit.w_kc_mbon.max() <= circuit.w_max
    assert circuit.w_kc_mbon.min() >= circuit.w_min
    circuit.step(np.ones(16) * 0.7, dt=0.05)
    before = float(circuit.w_kc_mbon[:, : circuit.n_approach].mean())
    circuit.learn(1.0)
    assert float(circuit.w_kc_mbon[:, : circuit.n_approach].mean()) > before


def test_substepping_lifts_the_rate_ceiling():
    """One Euler step of the caller's dt would cap every rate at 1/dt."""
    circuit = LIFCircuit(n_kc=200, seed=1)
    assert circuit.dt_sim < 0.05
    odor = np.random.RandomState(7).randn(64) * 0.5 + 0.8
    for _ in range(40):
        circuit.step(odor, dt=0.05)
        circuit.learn(1.0)
    state = circuit.step(odor, dt=0.05)
    assert state.mbon_approach_rate > 1.0 / 0.05


def test_arousal_never_leaves_core_affect_range():
    """CoreAffect defines arousal on [0, 1]; a negative value would be clamped."""
    bridge = AffectBridge()
    for dan in (0.0, 2.5, 5.0, 40.0, 200.0):
        _, arousal, _ = bridge.readout_to_tuple(MBONDanState(10.0, 10.0, dan, dan))
        assert 0.0 <= arousal <= 1.0
    assert bridge.readout_to_tuple(MBONDanState(10.0, 10.0, 0.0, 0.0))[1] == 0.0
    assert bridge.readout_to_tuple(MBONDanState(10.0, 10.0, 200.0, 200.0))[1] == 1.0


def test_single_spike_on_silent_population_is_not_full_avoidance():
    """0 vs 0.59 Hz used to read as valence = -1.0."""
    bridge = AffectBridge()
    valence, _, approach = bridge.readout_to_tuple(MBONDanState(0.0, 0.59, 1.0, 1.0))
    assert -0.3 < valence < 0.0
    assert -0.1 < approach < 0.0
    assert bridge.readout_to_tuple(MBONDanState(0.0, 0.0, 0.0, 0.0))[0] == 0.0


def test_valence_and_approach_are_not_the_same_axis():
    """Same ratio at different rates: equal valence, different net drive."""
    bridge = AffectBridge()
    weak_v, _, weak_a = bridge.readout_to_tuple(MBONDanState(12.0, 8.0, 20.0, 20.0))
    strong_v, _, strong_a = bridge.readout_to_tuple(MBONDanState(60.0, 40.0, 20.0, 20.0))
    assert weak_v == pytest.approx(strong_v, abs=1e-6)
    assert strong_a > weak_a + 0.5


def _run(circuit, n=40, sentiment=0.8, reward=None):
    from emotional_memory import EmotionalMemory, InMemoryStore

    from affective_fly import AffectiveLoop, FakeEmbedder, SensoryFrame
    from affective_fly.mood_field import MoodField

    store = InMemoryStore()
    embedder = FakeEmbedder()
    loop = AffectiveLoop(
        fly_circuit=circuit,
        emotional_memory=EmotionalMemory(store=store, embedder=embedder),
        store=store,
        embedder=embedder,
        mood_field=MoodField(tau_valence=8.0, tau_arousal=4.0, tau_approach=5.0),
    )
    actions = []
    for _ in range(n):
        context = {"context": "journal", "note_id": "exp-001", "sentiment": sentiment, "query": "experiment"}
        if reward is not None:
            context["reward"] = reward
        actions.append(loop.step(SensoryFrame.from_dict(context)).action)
    return actions


def test_real_circuit_closes_the_loop():
    """Phase 1 exit criterion: a spiking circuit must steer Policy, not just Mock.

    Before the calibration pass LIFCircuit produced WAIT for every input.
    """
    from affective_fly.policy import Action

    rewarded = _run(LIFCircuit(n_kc=2000, seed=1), reward=1.0)
    punished = _run(LIFCircuit(n_kc=2000, seed=1), sentiment=-0.6, reward=-1.0)

    assert rewarded.count(Action.TYPE) > 10, "reward should eventually open the gate"
    assert punished.count(Action.SKIP) > 10, "punishment should drive avoidance"
    assert set(rewarded) != set(punished)


def test_untrained_circuit_does_not_act():
    """Random KC->MBON weights carry no valence: the honest answer is WAIT."""
    from affective_fly.policy import Action

    actions = _run(LIFCircuit(n_kc=2000, seed=1))
    assert Action.TYPE not in actions
    assert Action.CLICK not in actions


def test_rates_do_not_depend_on_the_caller_step_size():
    """Rates are per-second, so the same simulated span must read the same.

    Accumulated float time used to make the window cutoff keep one extra event
    at dt=0.05 — the step size AffectiveLoop uses — inflating rates by 1.5x
    exactly at the operating point the calibration was measured at. Compared
    across dt rather than against a fixed number, so a recalibration of the
    gain does not turn this into a failing magic constant.
    """
    odor = np.random.RandomState(7).randn(64) * 0.5 + 0.8
    seen = {}
    for dt in (0.005, 0.01, 0.02, 0.05, 0.1):
        circuit = LIFCircuit(n_kc=2000, seed=1)
        for _ in range(int(round(0.5 / dt))):
            state = circuit.step(odor, dt=dt)
        seen[dt] = (state.mbon_approach_rate, state.dan_reinforcement_rate)

    reference = seen[0.01]
    assert reference[0] > 0.0
    for dt, (mbon, dan) in seen.items():
        assert mbon == pytest.approx(reference[0], rel=0.15), f"dt={dt}: {mbon:.2f} Hz"
        assert dan == pytest.approx(reference[1], rel=0.15), f"dt={dt}: {dan:.2f} Hz"


def test_gain_law_keeps_every_seed_in_band():
    """The n_kc gain law is a fit, so hold it against seeds it was not fit on."""
    for seed in range(1, 9):
        rates = _drive(LIFCircuit(n_kc=2000, seed=seed), seed=seed)
        mbon_total = rates[:, 0].mean()
        assert 1.0 < mbon_total <= MBON_MAX, f"seed {seed}: {mbon_total:.1f} Hz"
        assert rates[:, 1].mean() <= DAN_MAX


def test_default_syn_gain_recovers_nkc_law_on_random_weights():
    """Uniform [0, w_max] fan-in recovers 950 · n_kc^(−0.70) within seed noise."""
    for n_kc in (80, 200, 2000):
        circuit = LIFCircuit(n_kc=n_kc, seed=1)
        g0 = SYN_GAIN_PREFACTOR * float(n_kc) ** SYN_GAIN_EXPONENT
        assert circuit.syn_gain == pytest.approx(g0, rel=0.02)
        assert default_syn_gain(n_kc) == pytest.approx(g0)
        expected = n_kc * circuit.w_max / 2.0
        assert mean_column_fan_in(circuit.w_kc_mbon) == pytest.approx(expected, rel=0.02)


def test_syn_gain_override_is_honored():
    circuit = LIFCircuit(n_kc=200, seed=1, syn_gain=1.25)
    assert circuit.syn_gain == 1.25
    assert circuit.dan_syn_gain == 1.25
    circuit.w_kc_mbon[:] = 10.0
    circuit.refresh_default_syn_gain()
    assert circuit.syn_gain == 1.25


def test_refresh_default_syn_gain_tracks_replaced_weights():
    circuit = LIFCircuit(n_kc=80, n_dan=4, n_mbon=7, seed=1)
    before = circuit.syn_gain
    circuit.w_kc_mbon[:] = 8.0
    circuit.refresh_default_syn_gain()
    assert circuit.syn_gain < before / 10.0
    assert circuit.syn_gain == pytest.approx(
        default_syn_gain(80, weights=circuit.w_kc_mbon, w_max=circuit.w_max)
    )


def test_mbon_min_active_still_guards_below_operating_point():
    """5 Hz stays a silence floor: untrained LIF sits above it, 0.59 Hz does not."""
    bridge = AffectBridge()
    assert bridge.mbon_min_active == 5.0
    rates = _drive(LIFCircuit(n_kc=2000, seed=1))
    assert rates[:, 0].mean() > bridge.mbon_min_active
    silent_v, _, _ = bridge.readout_to_tuple(MBONDanState(0.0, 0.59, 1.0, 1.0))
    assert -0.3 < silent_v < 0.0


@pytest.mark.skipif(not TOP200_CONNECTOME.exists(), reason="top-200 fixture missing")
def test_top200_published_fanout_stays_in_band():
    circuit = MaleCNSCircuit(
        backend="lif", n_kc=185, seed=42, connectivity_path=TOP200_CONNECTOME
    )
    g0 = default_syn_gain(185)
    assert circuit.backend.syn_gain < g0
    rates = _drive(circuit, seed=42)
    mbon_total, dan = rates[:, 0].mean(), rates[:, 1].mean()
    assert 1.0 < mbon_total <= MBON_MAX, f"MBON {mbon_total:.1f} Hz outside band"
    assert dan <= DAN_MAX, f"DAN {dan:.1f} Hz outside band"


def _published_pyarrow_missing() -> bool:
    if not PUBLISHED_CONNECTOME.exists():
        return True
    import importlib.util

    return importlib.util.find_spec("pyarrow") is None


@pytest.mark.skipif(_published_pyarrow_missing(), reason="published feather or pyarrow missing")
def test_published_malecns_fanout_stays_in_band():
    """Item 1.3: real KC→MBON fan-out must land in the Policy/LaunchGate band."""
    circuit = MaleCNSCircuit(
        backend="lif", n_kc=4063, seed=1, connectivity_path=PUBLISHED_CONNECTOME
    )
    g0 = default_syn_gain(4063)
    assert circuit.backend.syn_gain == pytest.approx(
        default_syn_gain(
            4063, weights=circuit.backend.w_kc_mbon, w_max=circuit.backend.w_max
        )
    )
    assert circuit.backend.syn_gain < g0 / 10.0
    # DAN weights stay random, so they keep the n_kc-scale gain.
    assert circuit.backend.dan_syn_gain == pytest.approx(g0, rel=0.05)
    rates = _drive(circuit, seed=1)
    mbon_total, dan = rates[:, 0].mean(), rates[:, 1].mean()
    assert 10.0 <= mbon_total <= MBON_MAX, f"MBON {mbon_total:.1f} Hz outside band"
    assert 0.0 < dan <= DAN_MAX, f"DAN {dan:.1f} Hz outside band"
    assert mbon_total > AffectBridge().mbon_min_active


@pytest.mark.skipif(_published_pyarrow_missing(), reason="published feather or pyarrow missing")
def test_published_syn_gain_override_survives_connectome_load():
    circuit = MaleCNSCircuit(
        backend="lif",
        n_kc=4063,
        seed=1,
        connectivity_path=PUBLISHED_CONNECTOME,
        syn_gain=0.5,
    )
    assert circuit.backend.syn_gain == 0.5
    assert circuit.backend.dan_syn_gain == 0.5
