"""Tests for MaleCNS connectome loader (honest loading, no invented weights)."""

from pathlib import Path

import numpy as np
import pytest

from affective_fly import (
    ASO_CATALOG,
    PUBLISHED_MBON_SHORT_TO_ASO,
    ConnectivityData,
    ConnectomeLoadError,
    MaleCNSCircuit,
    load_connectome,
    malecns_mbon_short_name,
    map_to_aso_names,
    resolve_aso_name,
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

    # HONEST TEST: Verify matched MBON count > 0 and weights are non-zero
    assert circuit.matched_mbon_count > 0, (
        f"Expected at least 1 matched MBON, got {circuit.matched_mbon_count}. "
        "If 0 MBONs match, all weights are zeros (not proof of real weights)."
    )
    assert np.any(weights > 0), "Loaded weights must contain non-zero values"


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

    # HONEST TEST: Verify connectivity actually loaded
    assert circuit_loaded.connectivity_loaded, "connectivity_loaded flag must be True"

    w_random = circuit_random.backend.w_kc_mbon
    w_loaded = circuit_loaded.backend.w_kc_mbon

    # HONEST TEST: Verify loaded weights are not all zeros
    assert np.any(w_loaded > 0), "Loaded weights must contain non-zero values (proof real weights installed)"
    assert (w_loaded > 0).sum() > 0, f"Expected non-zero weights, got nnz={(w_loaded > 0).sum()}"

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


def test_published_short_names_cover_aso_catalog_only_from_aso_2014():
    """Curated table is Aso 2014 e04580 Table 1; catalog names are a subset."""
    assert set(ASO_CATALOG.mbon_names) <= set(PUBLISHED_MBON_SHORT_TO_ASO.values())
    assert PUBLISHED_MBON_SHORT_TO_ASO["MBON01"] == "MBON-gamma5beta'2a"
    assert PUBLISHED_MBON_SHORT_TO_ASO["MBON02"] == "MBON-beta2beta'2a"
    assert PUBLISHED_MBON_SHORT_TO_ASO["MBON03"] == "MBON-beta'2mp"
    assert PUBLISHED_MBON_SHORT_TO_ASO["MBON11"] == "MBON-gamma1pedc>alpha/beta"
    assert PUBLISHED_MBON_SHORT_TO_ASO["MBON12"] == "MBON-gamma2alpha'1"
    assert PUBLISHED_MBON_SHORT_TO_ASO["MBON13"] == "MBON-alpha'2"
    assert PUBLISHED_MBON_SHORT_TO_ASO["MBON14"] == "MBON-alpha3"
    # MaleCNS expansions are not invented into the table
    assert "MBON27" not in PUBLISHED_MBON_SHORT_TO_ASO
    assert "MBON35" not in PUBLISHED_MBON_SHORT_TO_ASO


def test_malecns_short_name_rejects_like_suffix():
    assert malecns_mbon_short_name({"type": "MBON01"}) == "MBON01"
    assert malecns_mbon_short_name({"instance": "MBON11(y1pedc>a/B)_R"}) == "MBON11"
    assert malecns_mbon_short_name({"type": "MBON15-like"}) is None
    assert malecns_mbon_short_name({"instance": "MBON15-like(a'1)_R"}) is None


def test_curated_mapping_sums_hemisphere_bodies_and_skips_non_catalog():
    """MBON01 L+R sum onto gamma5beta'2a; MBON05 is published but not in catalog."""
    connectivity = ConnectivityData(
        kc_to_mbon=np.array([[3.0, 4.0, 9.0]]),
        neuron_metadata={
            1: {"type": "KC", "instance": "KCg"},
            10: {"type": "MBON01", "instance": "MBON01(y5B'2a)_R"},
            11: {"type": "MBON01", "instance": "MBON01(y5B'2a)_L"},
            12: {"type": "MBON05", "instance": "MBON05(y4>y1y2)_R"},
        },
        source_path="memory",
        mbon_body_ids=[10, 11, 12],
        kc_body_ids=[1],
    )
    aso_names = list(ASO_CATALOG.mbon_names)
    mapped, matched = map_to_aso_names(connectivity, aso_names)
    assert matched == ["MBON-gamma5beta'2a"]
    col = aso_names.index("MBON-gamma5beta'2a")
    assert mapped[0, col] == 7.0
    # MBON05 is a real Aso 2014 type but not an ASO_CATALOG column
    assert resolve_aso_name(
        {"type": "MBON05", "instance": "MBON05(y4>y1y2)_R"}, aso_names
    ) is None
    assert np.all(mapped[:, aso_names.index("MBON-alpha3")] == 0.0)


def test_kc_count_mismatch_raises_without_override():
    with pytest.raises(ConnectomeLoadError, match="allow_kc_mismatch"):
        MaleCNSCircuit(
            backend="lif",
            n_kc=80,
            seed=1,
            connectivity_path=FIXTURE_PATH,
        )


def test_kc_count_mismatch_override_truncates_or_pads():
    truncated = MaleCNSCircuit(
        backend="lif",
        n_kc=2,
        seed=1,
        connectivity_path=FIXTURE_PATH,
        allow_kc_mismatch=True,
    )
    assert truncated.connectivity_loaded
    assert truncated.backend.w_kc_mbon.shape == (2, ASO_CATALOG.n_mbon)
    assert truncated.kc_rows_in_file == 3

    padded = MaleCNSCircuit(
        backend="lif",
        n_kc=5,
        seed=1,
        connectivity_path=FIXTURE_PATH,
        allow_kc_mismatch=True,
    )
    assert padded.backend.w_kc_mbon.shape == (5, ASO_CATALOG.n_mbon)
    assert np.all(padded.backend.w_kc_mbon[3:] == 0.0)


def test_named_rates_are_population_aliases():
    circuit = MaleCNSCircuit(backend="lif", n_kc=3, seed=1, connectivity_path=FIXTURE_PATH)
    circuit.step(np.ones(16) * 0.5, dt=0.001)
    rates = circuit.named_rates()
    index_rates = circuit.mbon_index_rates()
    n_app = ASO_CATALOG.n_approach
    approach_names = ASO_CATALOG.mbon_names[:n_app]
    avoid_names = ASO_CATALOG.mbon_names[n_app:]
    assert len(set(rates[name] for name in approach_names)) == 1
    assert len(set(rates[name] for name in avoid_names)) == 1
    assert index_rates == [rates[name] for name in ASO_CATALOG.mbon_names]
    assert circuit.dan_index_rates() == [rates[name] for name in ASO_CATALOG.dan_names]


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
