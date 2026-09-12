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

### Option 1: Use Provided Test Fixtures (Synthetic Weights)

The repository includes test fixtures with **REAL body IDs** from MaleCNS v1.0 but **SYNTHETIC weights** for testing:

```python
from affective_fly import MaleCNSCircuit

circuit = MaleCNSCircuit(
    backend="lif",
    n_kc=10,
    connectivity_path="tests/fixtures/malecns_real_ids.json",  # or .feather, .parquet
)
```

**Note**: These fixtures have real MaleCNS body IDs (e.g., KC: 11862, 13173; MBON: 10013, 10079) but connection weights are synthetic for test purposes only.

### Option 2: Extract Real Connectivity (Recommended for Research)

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

# Filter to KC→MBON connections with Python
python scripts/filter_malecns_connectivity.py \
    --input connectome-weights-male-cns-v1.0-minconf-0.5.feather \
    --output kc_mbon_connectivity.feather \
    --source-type KC \
    --target-type MBON
```

### Option 3: CI/Development (No Download)

For CI or offline development, tests automatically skip when fixtures are missing:

```python
# Tests with @pytest.mark.skipif will skip gracefully
pytest tests/test_malecns_connectome.py
```

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

**Current**: Loader implemented for JSON/Feather/Parquet; real body IDs documented; connection weights require user-provided connectome export.

**Complete Phase 4 requires**: Real KC→MBON connectivity matrix with published weights loaded into a `MaleCNSCircuit` for research use.

See `ROADMAP.md` for development phases.
