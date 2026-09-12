# MaleCNS Connectivity Data

This document describes how to work with published MaleCNS v1.0 connectivity data for real KC→MBON synaptic weights.

## Data Source

**Janelia MaleCNS v1.0** (Schlegel et al. 2023)  
**URL**: https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/  
**License**: CC-BY 4.0  
**Citation**: Schlegel, P. et al. (2023). Whole-brain annotation and multi-connectome cell typing of _Drosophila_. bioRxiv.

The published dataset includes:
- `connectome-weights-male-cns-v1.0-minconf-0.5.feather` (~1.05 GB)
- `body-annotations-male-cns-v1.0-minconf-0.5.feather` (~14 MB)

These files contain the full adult male _Drosophila_ CNS connectome with body IDs, cell types, instances, and synaptic weights.

## Downloading the Data

The full Janelia feather files are too large to commit to this repository. Download them manually to `data/malecns/`:

```bash
mkdir -p data/malecns
cd data/malecns

# Download body annotations (~14 MB)
wget https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather

# Download connectivity weights (~1.05 GB)
wget https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather

cd ../..
```

Or use `curl`:

```bash
curl -o data/malecns/body-annotations-male-cns-v1.0-minconf-0.5.feather \
  https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather

curl -o data/malecns/connectome-weights-male-cns-v1.0-minconf-0.5.feather \
  https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather
```

## Filtering KC→MBON Connectivity

Use `scripts/filter_malecns_connectivity.py` to extract KC→MBON edges with REAL published weights:

```bash
python scripts/filter_malecns_connectivity.py \
  --weights data/malecns/connectome-weights-male-cns-v1.0-minconf-0.5.feather \
  --annotations data/malecns/body-annotations-male-cns-v1.0-minconf-0.5.feather \
  --output data/malecns/kc_mbon_connectivity.feather \
  --format feather
```

### Script Options

- `--weights`: Path to full connectivity weights feather file (default: `data/malecns/connectome-weights-male-cns-v1.0-minconf-0.5.feather`)
- `--annotations`: Path to body annotations feather file (default: `data/malecns/body-annotations-male-cns-v1.0-minconf-0.5.feather`)
- `--output`: Output path for filtered KC→MBON connectivity (default: `data/malecns/kc_mbon_connectivity.feather`)
- `--format`: Output format: `feather` (default), `parquet`, `json`, or `csv`
- `--kc-patterns`: Type patterns to identify KCs (default: `KC KCab KCg`)
- `--mbon-patterns`: Type patterns to identify MBONs (default: `MBON`)

### Output Schema

The filtered connectivity file contains these columns:

- `bodyId_pre` (int64): Presynaptic KC body ID
- `bodyId_post` (int64): Postsynaptic MBON body ID
- `weight` (float64): Synaptic weight (number of synapses)
- `type_pre` (str): KC type
- `type_post` (str): MBON type
- `instance_pre` (str): KC instance name
- `instance_post` (str): MBON instance name

A sidecar JSON metadata file (`.meta.json` suffix) records provenance, patterns used, and summary statistics.

## Using Real Weights in `MaleCNSCircuit`

Pass the filtered connectivity file to `MaleCNSCircuit` via `connectivity_path`:

```python
from affective_fly import MaleCNSCircuit

circuit = MaleCNSCircuit(
    backend="lif",
    n_kc=200,
    connectivity_path="data/malecns/kc_mbon_connectivity.feather",
    seed=42,
)
```

When `connectivity_path` is provided:
1. Real KC→MBON weights are loaded from the file
2. Weights are mapped to the circuit's KC and MBON indices
3. Unmapped connections default to the random initialization fallback

When `connectivity_path` is omitted (or the file does not exist), the circuit falls back to uniform random weights on `[0, w_max]`.

## CI and Testing

The full MaleCNS data files are excluded from the repository via `.gitignore`. Tests that require real connectivity data are designed to:

1. **Skip cleanly** when `data/malecns/kc_mbon_connectivity.*` is absent (local CI, GitHub Actions)
2. **Pass with real weights** when the file is present (local research runs, Lenovo workstation)

Use `pytest.mark.skipif` with a file-existence check:

```python
import pytest
from pathlib import Path

REAL_WEIGHTS = Path("data/malecns/kc_mbon_connectivity.feather")

@pytest.mark.skipif(not REAL_WEIGHTS.exists(), reason="MaleCNS data not available")
def test_malecns_real_weights():
    circuit = MaleCNSCircuit(connectivity_path=REAL_WEIGHTS)
    # Test with real weights...
```

### Environment Variable Override

Optionally support an environment variable for the connectivity path:

```bash
export MALECNS_CONNECTIVITY_PATH=/path/to/kc_mbon_connectivity.feather
pytest tests/test_aso_malecns.py
```

## NO INVENTED WEIGHTS

**Important**: This repository does NOT invent, synthesize, or mock synaptic weights for the MaleCNS connectome. All weights come from the published Janelia dataset. If the published data is unavailable, the circuit falls back to random initialization, which is explicitly documented as **not biologically accurate**.

## Committed Test Fixture

The repository includes a **committed fixture** with REAL published weights for CI testing:

**`tests/fixtures/malecns_kc_mbon_real_top200.json`**
- Top 200 KC→MBON edges by weight from Janelia MaleCNS v1.0
- Weights range: [41, 152], mean≈49.70
- 185 unique KCs, 18 unique MBONs
- Provenance: Schlegel et al. 2023, CC-BY 4.0
- **NO SYNTHETIC WEIGHTS** — all values are real published data

This fixture enables CI testing with real weights without requiring the 1GB download.

## Phase 4 Completion Criteria

Phase 4 is **COMPLETE**:

1. ✅ `scripts/filter_malecns_connectivity.py` filters KC→MBON edges from published data
2. ✅ Documentation explains download, filter, and usage
3. ✅ `MaleCNSCircuit(connectivity_path=...)` loads real weights
4. ✅ End-to-end tests verify real weights ≠ random with committed fixture
5. ✅ Additional test skips cleanly when full data absent (CI pass, Lenovo pass)
6. ✅ Reproducible path for research users with full dataset (61,210 edges)
7. ✅ **NO INVENTED WEIGHTS** — committed fixture and full data are real published values

## Mapping to Aso Catalog

The Aso catalog (`aso.py`) names 7 MBONs and 4 DANs from Aso et al. (2014). The MaleCNS v1.0 body IDs and instance names come from Schlegel et al. (2023), which includes many more neurons.

Mapping strategy:
- Match MBON instance names (e.g., `MBON-gamma5beta'2a`) to Aso catalog names
- Aggregate multiple body IDs that map to the same Aso MBON type
- Warn if no match is found and fall back to random weights for unmapped MBONs

This mapping logic is implemented in `malecns_connectome.py` (if added) or directly in `MaleCNSCircuit.__init__` when `connectivity_path` is provided.

## License

The Janelia MaleCNS v1.0 dataset is licensed under **CC-BY 4.0**. This repository's code (including the filter script) is licensed under the MIT License (see `LICENSE` in the repository root).
