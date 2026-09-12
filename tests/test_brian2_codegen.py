"""Phase 5: selectable Brian2 codegen target and skip-clean C++ path."""

import numpy as np
import pytest

pytest.importorskip("brian2", reason="needs the 'brian' extra")

from affective_fly.brian2_circuit import (  # noqa: E402
    CODEGEN_CPP_STANDALONE,
    Brian2Circuit,
    CppStandaloneUnavailableError,
    activate_runtime_numpy,
    has_cpp_compiler,
)

MBON_MAX = 100.0
DAN_MAX = 80.0


@pytest.fixture(autouse=True)
def _restore_runtime_device():
    yield
    activate_runtime_numpy()


def test_has_cpp_compiler_returns_bool():
    assert isinstance(has_cpp_compiler(), bool)


def test_cpp_standalone_raises_without_compiler(monkeypatch):
    monkeypatch.setattr("affective_fly.brian2_circuit.has_cpp_compiler", lambda: False)
    with pytest.raises(CppStandaloneUnavailableError, match="C\\+\\+ compiler"):
        Brian2Circuit(
            n_kc=8,
            n_dan=2,
            n_mbon=4,
            seed=1,
            codegen_target=CODEGEN_CPP_STANDALONE,
        )


@pytest.mark.skipif(not has_cpp_compiler(), reason="no C++ toolchain")
def test_cpp_standalone_step_nonnegative(tmp_path):
    circuit = Brian2Circuit(
        n_kc=40,
        n_dan=6,
        n_mbon=8,
        seed=2,
        codegen_target=CODEGEN_CPP_STANDALONE,
        build_dir=tmp_path / "build",
    )
    state = circuit.step(np.ones(12) * 0.8, dt=0.005)
    assert state.mbon_approach_rate >= 0
    assert state.mbon_avoid_rate >= 0
    assert state.dan_reinforcement_rate >= 0
    assert circuit.last_kc_driven_frac == pytest.approx(0.05)
    assert circuit._standalone_built


@pytest.mark.skipif(not has_cpp_compiler(), reason="no C++ toolchain")
def test_cpp_standalone_sparse_reset_and_learn(tmp_path):
    circuit = Brian2Circuit(
        n_kc=40,
        n_dan=6,
        n_mbon=8,
        seed=3,
        codegen_target=CODEGEN_CPP_STANDALONE,
        build_dir=tmp_path / "build",
    )
    sensory = np.linspace(-0.2, 1.0, 10)
    circuit.step(sensory, dt=0.002)
    before = float(circuit.w_kc_mbon[:, : circuit.n_approach].mean())
    result = circuit.learn(1.0)
    assert result is not None
    assert float(circuit.w_kc_mbon[:, : circuit.n_approach].mean()) >= before
    circuit.reset()
    assert circuit.time == 0.0
    assert circuit._spike_events == []
    state = circuit.step(sensory, dt=0.002)
    assert state.mbon_approach_rate >= 0


@pytest.mark.skipif(not has_cpp_compiler(), reason="no C++ toolchain")
def test_cpp_standalone_same_seed_same_first_step(tmp_path):
    sensory = np.ones(8) * 0.6
    a = Brian2Circuit(
        n_kc=30,
        n_dan=4,
        n_mbon=6,
        seed=9,
        codegen_target=CODEGEN_CPP_STANDALONE,
        build_dir=tmp_path / "a",
    )
    sa = a.step(sensory, dt=0.003)
    b = Brian2Circuit(
        n_kc=30,
        n_dan=4,
        n_mbon=6,
        seed=9,
        codegen_target=CODEGEN_CPP_STANDALONE,
        build_dir=tmp_path / "b",
    )
    sb = b.step(sensory, dt=0.003)
    assert sa.mbon_approach_rate == pytest.approx(sb.mbon_approach_rate, rel=1e-6, abs=1e-9)
    assert sa.mbon_avoid_rate == pytest.approx(sb.mbon_avoid_rate, rel=1e-6, abs=1e-9)


@pytest.mark.skipif(not has_cpp_compiler(), reason="no C++ toolchain")
def test_cpp_standalone_rejects_dt_change_after_build(tmp_path):
    circuit = Brian2Circuit(
        n_kc=20,
        n_dan=4,
        n_mbon=6,
        seed=1,
        codegen_target=CODEGEN_CPP_STANDALONE,
        build_dir=tmp_path / "build",
    )
    circuit.step(np.ones(8) * 0.5, dt=0.002)
    with pytest.raises(ValueError, match="dt"):
        circuit.step(np.ones(8) * 0.5, dt=0.005)


@pytest.mark.skipif(not has_cpp_compiler(), reason="no C++ toolchain")
def test_cpp_standalone_untrained_rates_in_band(tmp_path):
    circuit = Brian2Circuit(
        n_kc=200,
        seed=1,
        codegen_target=CODEGEN_CPP_STANDALONE,
        build_dir=tmp_path / "build",
    )
    rng = np.random.RandomState(0)
    rows = []
    for i in range(12):
        s = rng.randn(64) * 0.5 + rng.choice([-0.8, -0.3, 0.3, 0.8])
        state = circuit.step(s, dt=0.05)
        if i >= 3:
            rows.append(
                (
                    state.mbon_approach_rate + state.mbon_avoid_rate,
                    state.dan_reinforcement_rate,
                )
            )
    rates = np.array(rows)
    assert 0.0 < rates[:, 0].mean() <= MBON_MAX
    assert rates[:, 1].mean() <= DAN_MAX
