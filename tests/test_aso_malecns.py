"""Tests for the published Aso catalog and named MaleCNSCircuit."""

import numpy as np

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
