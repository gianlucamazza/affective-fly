# MaleCNS Connectivity Data

## Overview

The `affective-fly` package implements loaders for published MaleCNS (male CNS) connectome data from Janelia FlyEM. This document describes how to obtain and use real connectivity data.

## Data Sources

### Official MaleCNS Data

The male-cns:v1.0 dataset is available from:

- **Website**: https://male-cns.janelia.org/download/
- **Google Cloud Storage**: `gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/`
- **License**: CC-BY 4.0 (Janelia FlyEM)

### Key Files

1. **body-annotations-male-cns-v1.0-minconf-0.5.feather** (13 MB)
   - Neuron annotations (types, instances, body IDs)
   - Contains 211,577 neurons including 4,064 KCs and 97 MBONs

2. **connectome-weights-male-cns-v1.0-minconf-0.5.feather** (1.1 GB)
   - Full connectivity matrix for all neurons
   - Segment-to-segment connection strengths

3. **neuPrint API** (requires authentication)
   - Interactive queries at https://neuprint.janelia.org
   - Dataset: `male-cns:v1.0`
   - Requires free token from Janelia

## Using MaleCNS Connectivity

### Install Optional Dependencies

```bash
# Install pyarrow for Feather/Parquet support
uv sync --extra connectome

# Or with pip
pip install affective-fly[connectome]
```

### Option 1: Committed published KC→MBON weights

`data/malecns/kc_mbon_connectivity.feather` is every KC→MBON edge from MaleCNS v1.0 minconf-0.5 (61,210 edges, 4,063 KCs, 97 MBONs, weights [1, 152], CC-BY 4.0). Requires `uv sync --extra connectome`.

```python
from affective_fly import MaleCNSCircuit, load_connectome

path = "data/malecns/kc_mbon_connectivity.feather"
n_kc = load_connectome(path).kc_to_mbon.shape[0]
circuit = MaleCNSCircuit(backend="lif", n_kc=n_kc, connectivity_path=path)
```

`n_kc` must equal the file's KC count. A mismatch raises `ConnectomeLoadError` unless you pass `allow_kc_mismatch=True` (truncate extra rows or pad with zeros).

Mapping onto the 7-name `ASO_CATALOG` uses the curated Aso 2014 (eLife e04580 Table 1) short-name table. Types outside that table, and `*-like` MaleCNS labels, stay unmatched — we do not invent Aso identities. `named_rates()` is a **population alias** (approach/avoid/DAN rate repeated per name), not a per-cell readout.

Published synapse counts sit far above the LIF plasticity ceiling (`w_max` = 0.15). At load, `MaleCNSCircuit` applies a **homogeneous** scale (`scale_published_weights_to_band`) so the peak count maps to `w_max`. Relative anatomy is preserved; `learn()` can then potentiate without flattening every published edge to the ceiling. Assigning raw counts onto `w_kc_mbon` and calling `learn()` raises `ValueError`. This is not a Hige depression table and does not invent body IDs. `circuit.anatomy_scale` and `circuit.published_weight_peak` record the choice.

### Option 2: Test fixtures (synthetic weights or top-200)

`tests/fixtures/malecns_real_ids.{json,feather,parquet}` have **REAL body IDs** from MaleCNS v1.0 and **SYNTHETIC weights**.

`tests/fixtures/malecns_kc_mbon_real_top200.json` has **REAL published weights** for the top 200 edges. Aso coverage there is **partial** (see `MALECNS_TOP200_MAPPING.md`). Use `n_kc=185`.

```python
from affective_fly import MaleCNSCircuit

circuit = MaleCNSCircuit(
    backend="lif",
    n_kc=10,
    connectivity_path="tests/fixtures/malecns_real_ids.json",  # or .feather, .parquet
)
```

### Option 3: Extract Real Connectivity (rebuild from Janelia)

#### Using neuPrint API

```python
from neuprint import Client, NeuronCriteria as NC, fetch_adjacencies
import json

# Get free token from https://neuprint.janelia.org
client = Client('https://neuprint.janelia.org', dataset='male-cns:v1.0', token=YOUR_TOKEN)

# Query KC→MBON connections
adj, neurons = fetch_adjacencies(
    sources=NC(type='KC', regex=False),
    targets=NC(type='MBON.*', regex=True),
    min_total_weight=1,
    client=client
)

# Save for affective-fly
# ... (convert to affective-fly JSON schema)
```

#### Using the Extraction Script

```bash
# Install neuprint-python
pip install neuprint-python

# Set token
export NEUPRINT_APPLICATION_CREDENTIALS=your_token_here

# Extract sample
python scripts/extract_malecns_sample.py \
    --output my_real_connectivity.json \
    --format json \
    --max-edges 500
```

#### Manual Download (for offline use)

```bash
# Download body annotations (13 MB)
curl -O https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather

# Download full connectivity (1.1 GB, takes ~5-10 minutes)
curl -O https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather

# Filter to KC→MBON connections (expected: 61,210 edges)
python scripts/filter_malecns_connectivity.py \
    --weights data/malecns/connectome-weights-male-cns-v1.0-minconf-0.5.feather \
    --annotations data/malecns/body-annotations-male-cns-v1.0-minconf-0.5.feather \
    --output data/malecns/kc_mbon_connectivity.feather \
    --format feather
```

### Option 4: CI / offline (no 1.1 GB download)

CI installs `--extra connectome` and loads the committed filtered feather plus JSON fixtures. Raw Janelia files are gitignored. Tests skip only if a named fixture file is absent or pyarrow is missing.

## File Formats

The loader supports:

- **JSON**: `{"edges": [...], "neurons": [...]}`
- **Feather**: Apache Arrow format (requires `pyarrow`)
- **Parquet**: Apache Parquet format (requires `pyarrow`)
- **HDF5**: Not yet implemented (would require `h5py`)

## Schema

### JSON Format

```json
{
  "_comment": "Optional metadata",
  "_source": "https://male-cns.janelia.org/download/",
  "_license": "CC-BY 4.0",
  "edges": [
    {
      "bodyId_pre": 13173,
      "bodyId_post": 10013,
      "weight": 9.5,
      "type": "KC_to_MBON"
    }
  ],
  "neurons": [
    {
      "bodyId": 13173,
      "type": "KC",
      "instance": "KC_alpha_R"
    },
    {
      "bodyId": 10013,
      "type": "MBON01",
      "instance": "MBON01(y5B'2a)_R"
    }
  ]
}
```

### Feather/Parquet Format

Columns:
- `bodyId_pre`: Pre-synaptic neuron body ID
- `bodyId_post`: Post-synaptic neuron body ID
- `weight`: Connection strength (synapse count or normalized)
- `type_pre`: Pre-synaptic neuron type (e.g., "KC")
- `type_post`: Post-synaptic neuron type (e.g., "MBON01")
- `instance_pre`: Pre-synaptic neuron instance name
- `instance_post`: Post-synaptic neuron instance name

## References

- Li, F., et al. (2020). "The connectome of the adult Drosophila mushroom body provides insights into function." eLife. https://doi.org/10.7554/eLife.62576
- Aso, Y., et al. (2014). "The neuronal architecture of the mushroom body provides a logic for associative learning." eLife. https://doi.org/10.7554/eLife.04577
- MaleCNS v1.0 dataset: https://male-cns.janelia.org/

## Troubleshooting

### ImportError: pyarrow not found

```bash
uv sync --extra connectome
# or
pip install pyarrow>=14.0.0
```

### ConnectomeLoadError: File not found

- Check that the path points to an existing file
- For test fixtures, ensure you're running from the repository root
- For custom files, verify the download completed successfully

### neuPrint authentication failed

- Request a free token at https://neuprint.janelia.org
- Set `NEUPRINT_APPLICATION_CREDENTIALS` environment variable
- Or pass `token=YOUR_TOKEN` to the `Client()` constructor

## Phase 4 Status

**Partial.** Published KC→MBON weights load end-to-end from `data/malecns/kc_mbon_connectivity.feather` (or a user rebuild). Mapping is the curated Aso 2014 short-name table onto the 7-name catalog. KC-count mismatch fails unless overridden. Counts are scaled into LIF `w_max` before `learn()`. `syn_gain` is refit from the scaled KC→MBON fan-in (`default_syn_gain`); pass `syn_gain` to override.

See `ROADMAP.md` and `MALECNS_TOP200_MAPPING.md`.
