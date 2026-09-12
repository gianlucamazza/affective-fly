"""Circuit registry: one name → Mock / LIF / Brian2 / MaleCNS."""

import pytest

from affective_fly import CIRCUIT_NAMES, LIFCircuit, MockFlyCircuit, get_circuit
from affective_fly.malecns import MaleCNSCircuit


def test_get_circuit_mock_and_lif():
    mock = get_circuit("mock", seed=7)
    lif = get_circuit("LIF", n_kc=40, n_dan=4, n_mbon=8, seed=7)
    assert isinstance(mock, MockFlyCircuit)
    assert mock.seed == 7
    assert isinstance(lif, LIFCircuit)
    assert lif.n_kc == 40


def test_get_circuit_malecns():
    circuit = get_circuit("malecns", n_kc=40, seed=1)
    assert isinstance(circuit, MaleCNSCircuit)
    assert isinstance(circuit.backend, LIFCircuit)
    assert circuit.backend.n_kc == 40


def test_get_circuit_unknown_name():
    with pytest.raises(ValueError, match="Unknown circuit"):
        get_circuit("spiking-gpu")


def test_circuit_names_are_the_four_backends():
    assert CIRCUIT_NAMES == ("mock", "lif", "brian2", "malecns")


def test_get_circuit_brian2_or_skip():
    try:
        circuit = get_circuit("brian2", n_kc=20, n_dan=4, n_mbon=8, seed=1)
    except ImportError:
        pytest.skip("brian2 extra not installed")
    from affective_fly.brian2_circuit import Brian2Circuit

    assert isinstance(circuit, Brian2Circuit)
