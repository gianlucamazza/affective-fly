"""Tests for the published Aso catalog and named MaleCNSCircuit."""

import os
from pathlib import Path

import numpy as np
import pytest

from affective_fly import ASO_CATALOG, MaleCNSCircuit


def test_catalog_has_only_published_names():
    assert ASO_CATALOG.n_mbon == 7
    assert ASO_CATALOG.n_dan == 4
    assert "MBON-gamma5beta'2a" in ASO_CATALOG.mbon_names
    assert "PPL1-gamma1pedc" in ASO_CATALOG.dan_names
    # No invented numeric body IDs
    for cell in ASO_CATALOG.approach_mbons + ASO_CATALOG.avoid_mbons:
        assert cell.source.startswith("Aso")
        assert not cell.name[0].isdigit()


def test_gamma1pedc_is_avoid():
    names = {c.name: c.valence for c in ASO_CATALOG.avoid_mbons}
    assert names["MBON-gamma1pedc>alpha/beta"] == "avoid"


def test_malecns_circuit_labels_and_step():
    circuit = MaleCNSCircuit(backend="lif", n_kc=80, seed=1)
    assert circuit.backend.n_mbon == ASO_CATALOG.n_mbon
    assert circuit.backend.n_approach == ASO_CATALOG.n_approach
    state = circuit.step(np.ones(16) * 0.5, dt=0.001)
    assert state.mbon_approach_rate >= 0
    rates = circuit.named_rates()
    assert set(rates) == set(circuit.mbon_names + circuit.dan_names)


def _find_connectivity_file() -> Path | None:
    """Find MaleCNS connectivity file from env var or standard location."""
    # Check environment variable first
    env_path = os.environ.get("MALECNS_CONNECTIVITY_PATH")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path

    # Check standard locations
    standard_paths = [
        Path("data/malecns/kc_mbon_connectivity.feather"),
        Path("data/malecns/kc_mbon_connectivity.parquet"),
        Path("data/malecns/kc_mbon_connectivity.json"),
    ]
    for path in standard_paths:
        if path.exists():
            return path

    return None


@pytest.mark.skipif(
    _find_connectivity_file() is None,
    reason="MaleCNS connectivity data not available (set MALECNS_CONNECTIVITY_PATH or place file in data/malecns/)",
)
def test_malecns_real_weights_differ_from_random():
    """E2E test: real weights loaded from connectivity file differ from random init.
    
    This test validates Phase 4: REAL published KC→MBON weights are loaded
    and differ from the random fallback. Skips cleanly when data is absent.
    """
    connectivity_path = _find_connectivity_file()
    assert connectivity_path is not None  # Should not reach here if skip condition met

    # Create circuit with real weights
    circuit_real = MaleCNSCircuit(
        backend="lif",
        n_kc=80,
        seed=42,
        connectivity_path=connectivity_path,
    )

    # Create circuit with random weights (same seed)
    circuit_random = MaleCNSCircuit(
        backend="lif",
        n_kc=80,
        seed=42,
        connectivity_path=None,  # Force random fallback
    )

    # Extract weight matrices
    w_real = circuit_real.backend.w_kc_mbon
    w_random = circuit_random.backend.w_kc_mbon

    assert w_real.shape == w_random.shape
    assert w_real.shape == (80, ASO_CATALOG.n_mbon)

    # Real weights should differ from random initialization
    # (unless by extreme coincidence all real weights matched random, which is ~impossible)
    diff_norm = np.linalg.norm(w_real - w_random)
    print(f"\nWeight matrix difference norm: {diff_norm:.4f}")

    # If connectivity was successfully loaded, difference should be substantial
    # A tiny difference would suggest real weights weren't actually loaded
    assert diff_norm > 0.01, "Real weights should differ significantly from random initialization"

    # Verify both circuits can step
    sensory = np.random.rand(16)
    state_real = circuit_real.step(sensory, dt=0.05)
    state_random = circuit_random.step(sensory, dt=0.05)

    assert state_real.mbon_approach_rate >= 0
    assert state_random.mbon_approach_rate >= 0

    print(f"Real circuit approach rate: {state_real.mbon_approach_rate:.2f} Hz")
    print(f"Random circuit approach rate: {state_random.mbon_approach_rate:.2f} Hz")
