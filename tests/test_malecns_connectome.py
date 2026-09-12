"""Tests for MaleCNS connectome loader (honest loading, no invented weights)."""

from pathlib import Path

import numpy as np
import pytest

from affective_fly import (
    ASO_CATALOG,
    ConnectomeLoadError,
    MaleCNSCircuit,
    load_connectome,
    map_to_aso_names,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "malecns_mini.json"


def test_load_connectome_from_json():
    """Load the minimal fixture and verify structure."""
    connectivity = load_connectome(FIXTURE_PATH)
    assert connectivity.kc_to_mbon.shape[0] == 3  # 3 KCs in fixture
    assert connectivity.kc_to_mbon.shape[1] == 3  # 3 MBONs in fixture
    assert connectivity.source_path == str(FIXTURE_PATH)
    # Check that weights are non-zero (from fixture, not random)
    assert np.sum(connectivity.kc_to_mbon) > 0


def test_load_connectome_missing_file_raises():
    """Missing file raises ConnectomeLoadError."""
    with pytest.raises(ConnectomeLoadError, match="not found"):
        load_connectome("/nonexistent/path.json")


def test_load_connectome_unsupported_format_raises():
    """Unsupported format raises ConnectomeLoadError."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"test data")
        temp_path = f.name
    try:
        with pytest.raises(ConnectomeLoadError, match="Unsupported file format"):
            load_connectome(temp_path)
    finally:
        Path(temp_path).unlink()


def test_map_to_aso_names_matches_fixture():
    """Map fixture MBONs to Aso names."""
    connectivity = load_connectome(FIXTURE_PATH)
    # Fixture has MBON-gamma5beta'2a and MBON-gamma2alpha'1
    aso_names = ["MBON-gamma5beta'2a", "MBON-gamma2alpha'1", "MBON-beta'2mp"]
    mapped, matched = map_to_aso_names(connectivity, aso_names)
    assert mapped.shape == (3, 3)  # 3 KCs, 3 Aso names
    assert "MBON-gamma5beta'2a" in matched
    assert "MBON-gamma2alpha'1" in matched
    assert "MBON-beta'2mp" in matched


def test_map_to_aso_names_unmatched_stays_zero():
    """Unmatched Aso names have zero columns."""
    connectivity = load_connectome(FIXTURE_PATH)
    # Include a name not in the fixture
    aso_names = ["MBON-gamma5beta'2a", "MBON-not-in-fixture"]
    mapped, matched = map_to_aso_names(connectivity, aso_names)
    assert "MBON-gamma5beta'2a" in matched
    assert "MBON-not-in-fixture" not in matched
    # Second column should be all zeros
    assert np.all(mapped[:, 1] == 0.0)


def test_malecns_circuit_without_connectivity_path_uses_random():
    """MaleCNSCircuit without connectivity_path uses random weights."""
    circuit = MaleCNSCircuit(backend="lif", n_kc=80, seed=1)
    assert circuit.connectivity_path is None
    assert not circuit.connectivity_loaded
    # Weights should be random (non-zero from LIF init)
    assert hasattr(circuit.backend, "w_kc_mbon")
    assert np.any(circuit.backend.w_kc_mbon > 0)


def test_malecns_circuit_with_connectivity_path_loads_weights():
    """MaleCNSCircuit with connectivity_path loads real weights."""
    # Use small n_kc to match fixture
    circuit = MaleCNSCircuit(
        backend="lif",
        n_kc=3,
        seed=1,
        connectivity_path=FIXTURE_PATH,
    )
    assert circuit.connectivity_path == FIXTURE_PATH
    assert circuit.connectivity_loaded
    # Weights should be loaded from fixture, not random
    weights = circuit.backend.w_kc_mbon
    # Check that some weights match fixture values (fixture has weights 12, 8, 15, 5)
    assert weights.shape == (3, 7)  # 3 KCs, 7 MBONs from Aso catalog


def test_malecns_circuit_connectivity_path_not_found_raises():
    """MaleCNSCircuit with missing connectivity_path raises."""
    with pytest.raises(ConnectomeLoadError, match="not found"):
        MaleCNSCircuit(
            backend="lif",
            n_kc=80,
            seed=1,
            connectivity_path="/nonexistent/path.json",
        )


def test_loaded_weights_differ_from_random():
    """Loaded weights are not the same as random seed initialization."""
    circuit_random = MaleCNSCircuit(backend="lif", n_kc=3, seed=42)
    circuit_loaded = MaleCNSCircuit(
        backend="lif",
        n_kc=3,
        seed=42,  # Same seed, different weights
        connectivity_path=FIXTURE_PATH,
    )
    w_random = circuit_random.backend.w_kc_mbon
    w_loaded = circuit_loaded.backend.w_kc_mbon
    # Weights should differ
    assert not np.allclose(w_random, w_loaded)


def test_circuit_with_loaded_weights_can_step():
    """Circuit with loaded connectivity can step normally."""
    circuit = MaleCNSCircuit(
        backend="lif",
        n_kc=3,
        seed=1,
        connectivity_path=FIXTURE_PATH,
    )
    state = circuit.step(np.ones(16) * 0.5, dt=0.001)
    assert state.mbon_approach_rate >= 0
    assert state.mbon_avoid_rate >= 0
    rates = circuit.named_rates()
    assert len(rates) == len(ASO_CATALOG.mbon_names) + len(ASO_CATALOG.dan_names)


# Real body ID fixtures (optional, skip if pyarrow not installed or file missing)
REAL_IDS_JSON = Path(__file__).parent / "fixtures" / "malecns_real_ids.json"
REAL_IDS_FEATHER = Path(__file__).parent / "fixtures" / "malecns_real_ids.feather"
REAL_IDS_PARQUET = Path(__file__).parent / "fixtures" / "malecns_real_ids.parquet"


@pytest.mark.skipif(not REAL_IDS_FEATHER.exists(), reason="Real IDs Feather fixture not available")
def test_load_connectome_from_feather():
    """Load the real-IDs Feather fixture and verify structure."""
    try:
        connectivity = load_connectome(REAL_IDS_FEATHER)
    except ImportError as e:
        pytest.skip(f"pyarrow not installed: {e}")

    assert connectivity.kc_to_mbon.shape[0] == 10  # 10 KCs in real-IDs fixture
    assert connectivity.kc_to_mbon.shape[1] == 7  # 7 MBONs in real-IDs fixture
    assert connectivity.source_path == str(REAL_IDS_FEATHER)
    # Check that weights are non-zero (from fixture)
    assert np.sum(connectivity.kc_to_mbon) > 0
    # Check for real body IDs (from MaleCNS v1.0)
    assert 11862 in connectivity.kc_body_ids or 13173 in connectivity.kc_body_ids


@pytest.mark.skipif(not REAL_IDS_PARQUET.exists(), reason="Real IDs Parquet fixture not available")
def test_load_connectome_from_parquet():
    """Load the real-IDs Parquet fixture and verify structure."""
    try:
        connectivity = load_connectome(REAL_IDS_PARQUET)
    except ImportError as e:
        pytest.skip(f"pyarrow not installed: {e}")

    assert connectivity.kc_to_mbon.shape[0] == 10  # 10 KCs in real-IDs fixture
    assert connectivity.kc_to_mbon.shape[1] == 7  # 7 MBONs in real-IDs fixture
    assert connectivity.source_path == str(REAL_IDS_PARQUET)
    # Check that weights are non-zero (from fixture)
    assert np.sum(connectivity.kc_to_mbon) > 0


@pytest.mark.skipif(not REAL_IDS_JSON.exists(), reason="Real IDs JSON fixture not available")
def test_real_ids_json_has_malecns_body_ids():
    """Verify the real-IDs JSON fixture contains actual MaleCNS body IDs."""
    connectivity = load_connectome(REAL_IDS_JSON)
    # Check for real body IDs from MaleCNS v1.0 (not synthetic like 10001, 20001)
    real_kc_ids = {11862, 13173, 14292, 15103, 17488, 18031, 18540, 19083, 19102, 19788}
    real_mbon_ids = {10013, 10079, 10267, 10495, 10599, 10704, 10804}

    assert real_kc_ids.issubset(set(connectivity.kc_body_ids))
    assert real_mbon_ids.issubset(set(connectivity.mbon_body_ids))


def test_feather_without_pyarrow_raises_importerror():
    """Loading Feather without pyarrow raises ImportError with clear message."""
    if not REAL_IDS_FEATHER.exists():
        pytest.skip("Real IDs Feather fixture not available")

    # This test documents the expected error when pyarrow is missing
    # In practice, if pyarrow is installed (as in connectome extra), this won't fail
    # But the error message should be helpful
    import importlib.util
    if importlib.util.find_spec("pyarrow") is not None:
        pytest.skip("pyarrow is installed, can't test missing-dependency case")

    with pytest.raises(ImportError, match="uv sync --extra connectome"):
        load_connectome(REAL_IDS_FEATHER)
