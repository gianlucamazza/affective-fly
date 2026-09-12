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


def test_malecns_real_weights_committed_fixture():
    """Test with committed fixture of REAL published KC→MBON weights.

    Uses tests/fixtures/malecns_kc_mbon_real_top200.json: top 200 KC→MBON edges
    by weight from Janelia MaleCNS v1.0 (Schlegel et al. 2023, CC-BY 4.0).
    Weights range [41, 152], mean≈49.70, 185 KCs, 18 MBONs — NO SYNTHETIC DATA.
    """
    fixture_path = Path("tests/fixtures/malecns_kc_mbon_real_top200.json")
    assert fixture_path.exists(), f"Fixture missing: {fixture_path}"

    # Create circuit with real weights from committed fixture
    circuit_real = MaleCNSCircuit(
        backend="lif",
        n_kc=80,
        seed=42,
        connectivity_path=fixture_path,
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
    diff_norm = np.linalg.norm(w_real - w_random)
    print(f"\nWeight matrix difference norm: {diff_norm:.4f}")
    assert diff_norm > 0.01, "Real weights should differ significantly from random initialization"

    # Verify both circuits can step
    sensory = np.random.rand(16)
    state_real = circuit_real.step(sensory, dt=0.05)
    state_random = circuit_random.step(sensory, dt=0.05)

    assert state_real.mbon_approach_rate >= 0
    assert state_random.mbon_approach_rate >= 0

    print(f"Real circuit approach rate: {state_real.mbon_approach_rate:.2f} Hz")
    print(f"Random circuit approach rate: {state_random.mbon_approach_rate:.2f} Hz")


@pytest.mark.skipif(
    _find_connectivity_file() is None,
    reason="Full MaleCNS connectivity data not available (set MALECNS_CONNECTIVITY_PATH or place file in data/malecns/)",
)
def test_malecns_real_weights_full_data():
    """Optional test with full KC→MBON connectivity (61,210 edges).

    Skips when data/malecns/ files are absent (CI). Passes on Lenovo/local
    with full downloaded dataset.
    """
    connectivity_path = _find_connectivity_file()
    assert connectivity_path is not None

    circuit = MaleCNSCircuit(
        backend="lif",
        n_kc=80,
        seed=42,
        connectivity_path=connectivity_path,
    )

    # Verify circuit can step with full real weights
    sensory = np.random.rand(16)
    state = circuit.step(sensory, dt=0.05)
    assert state.mbon_approach_rate >= 0
    print(f"Full data circuit approach rate: {state.mbon_approach_rate:.2f} Hz")
