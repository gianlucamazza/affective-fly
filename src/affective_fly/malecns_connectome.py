"""
MaleCNS connectome loader for published connectivity data.

Loads KC→MBON and DAN→MBON edges from local files (JSON, Feather, Parquet, HDF5).
Does NOT invent weights. If the connectivity file is missing or unreadable,
fails clearly or allows tests to skip.

Public MaleCNS data is available from Janelia:
    https://male-cns.janelia.org/download/
    gs://flyem-male-cns/v1.0/...

See also:
    - Li et al. eLife 2020 (hemibrain MB)
    - Aso et al. 2014 (published MBON/DAN types)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ConnectivityData:
    """Parsed connectivity matrix for KC→MBON synapses.

    All weights are from published data, not random initialization.
    """

    kc_to_mbon: np.ndarray  # shape (n_kc, n_mbon), synapse counts or weights
    neuron_metadata: dict[int, dict[str, Any]]  # bodyId → {type, instance, ...}
    source_path: str
    mbon_body_ids: list[int]  # Ordered body IDs matching columns
    kc_body_ids: list[int]  # Ordered body IDs matching rows


class ConnectomeLoadError(Exception):
    """Raised when connectome file cannot be loaded."""

    pass


def load_connectome(path: str | Path) -> ConnectivityData:
    """
    Load MaleCNS connectivity from a local file.

    Supports:
        - JSON: {"edges": [...], "neurons": [...]}
        - Feather/Parquet: requires pyarrow (install via `uv sync --extra connectome`)
        - HDF5: future (requires h5py)

    Args:
        path: Local filesystem path to connectivity export.

    Returns:
        ConnectivityData with KC→MBON weights from the file.

    Raises:
        ConnectomeLoadError: If file is missing, unreadable, or malformed.
        ImportError: If required optional dependency is not installed.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise ConnectomeLoadError(f"Connectome file not found: {path}")

    suffix = path_obj.suffix.lower()

    if suffix == ".json":
        return _load_json(path_obj)
    elif suffix in (".feather", ".parquet"):
        return _load_arrow(path_obj)
    elif suffix in (".h5", ".hdf5"):
        raise ConnectomeLoadError(
            f"HDF5 loading not yet implemented. "
            f"Install h5py and extend _load_hdf5() to support {suffix}."
        )
    else:
        raise ConnectomeLoadError(f"Unsupported file format: {suffix}")


def _load_json(path: Path) -> ConnectivityData:
    """Load JSON connectivity export."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise ConnectomeLoadError(f"Failed to read JSON from {path}: {e}") from e

    if "edges" not in data or "neurons" not in data:
        raise ConnectomeLoadError(
            f"JSON must contain 'edges' and 'neurons' keys. Found: {list(data.keys())}"
        )

    # Build neuron metadata map
    neurons = {n["bodyId"]: n for n in data["neurons"]}

    # Collect KC and MBON body IDs
    kc_ids = sorted([bid for bid, meta in neurons.items() if meta.get("type") == "KC"])
    mbon_ids = sorted([
        bid for bid, meta in neurons.items()
        if meta.get("type", "").startswith("MBON") or meta.get("type") == "MBON"
    ])

    if not kc_ids or not mbon_ids:
        raise ConnectomeLoadError("No KC or MBON neurons found in connectivity data.")

    # Build index maps
    kc_idx = {bid: i for i, bid in enumerate(kc_ids)}
    mbon_idx = {bid: i for i, bid in enumerate(mbon_ids)}

    # Initialize weight matrix
    weights = np.zeros((len(kc_ids), len(mbon_ids)), dtype=float)

    # Populate from edges
    for edge in data["edges"]:
        pre_id = edge.get("bodyId_pre")
        post_id = edge.get("bodyId_post")
        weight = edge.get("weight", 1.0)
        edge_type = edge.get("type", "")

        # Only process KC→MBON edges
        if "KC_to_MBON" in edge_type or (pre_id in kc_idx and post_id in mbon_idx):
            i = kc_idx.get(pre_id)
            j = mbon_idx.get(post_id)
            if i is not None and j is not None:
                weights[i, j] += weight

    return ConnectivityData(
        kc_to_mbon=weights,
        neuron_metadata=neurons,
        source_path=str(path),
        mbon_body_ids=mbon_ids,
        kc_body_ids=kc_ids,
    )


def _load_arrow(path: Path) -> ConnectivityData:
    """Load Feather or Parquet connectivity export using pyarrow."""
    try:
        import pyarrow.feather as feather
        import pyarrow.parquet as parquet
    except ImportError as e:
        raise ImportError(
            f"pyarrow is required to load {path.suffix} files. "
            f"Install with: uv sync --extra connectome"
        ) from e

    try:
        suffix = path.suffix.lower()
        if suffix == ".feather":
            table = feather.read_table(path)
        elif suffix == ".parquet":
            table = parquet.read_table(path)
        else:
            raise ConnectomeLoadError(f"Unexpected arrow format: {suffix}")
    except Exception as e:
        raise ConnectomeLoadError(f"Failed to read {path.suffix} from {path}: {e}") from e

    # Convert to pandas for easier manipulation
    try:
        df = table.to_pandas()
    except Exception as e:
        raise ConnectomeLoadError(f"Failed to convert arrow table to pandas: {e}") from e

    # Expect columns: bodyId_pre, bodyId_post, weight, type (and optionally neuron metadata)
    required_cols = {"bodyId_pre", "bodyId_post"}
    if not required_cols.issubset(df.columns):
        raise ConnectomeLoadError(
            f"Arrow file must contain columns {required_cols}. Found: {list(df.columns)}"
        )

    # If neuron metadata columns exist (type_pre, type_post, instance_pre, instance_post),
    # build the neurons dict. Otherwise, infer from the edge types.
    neurons: dict[int, dict[str, Any]] = {}

    # Collect unique pre and post body IDs
    all_pre = df["bodyId_pre"].unique()
    all_post = df["bodyId_post"].unique()
    all_body_ids = set(all_pre) | set(all_post)

    # Build neuron metadata from columns if available
    if "type_pre" in df.columns and "type_post" in df.columns:
        for _, row in df.iterrows():
            pre_id = row["bodyId_pre"]
            post_id = row["bodyId_post"]
            if pre_id not in neurons:
                neurons[pre_id] = {
                    "bodyId": int(pre_id),
                    "type": row.get("type_pre", ""),
                    "instance": row.get("instance_pre", ""),
                }
            if post_id not in neurons:
                neurons[post_id] = {
                    "bodyId": int(post_id),
                    "type": row.get("type_post", ""),
                    "instance": row.get("instance_post", ""),
                }
    else:
        # Infer types from connection patterns or explicit type column
        # Assume KC→MBON edges: pre are KC, post are MBON
        for body_id in all_body_ids:
            if body_id not in neurons:
                # Heuristic: if it appears as pre more often, likely KC; as post, likely MBON
                inferred_type = "KC" if body_id in all_pre and body_id not in all_post else "MBON"
                neurons[int(body_id)] = {
                    "bodyId": int(body_id),
                    "type": inferred_type,
                    "instance": "",
                }

    # Collect KC and MBON body IDs
    kc_ids = sorted([bid for bid, meta in neurons.items() if meta.get("type") == "KC"])
    mbon_ids = sorted([
        bid for bid, meta in neurons.items()
        if meta.get("type", "").startswith("MBON") or meta.get("type") == "MBON"
    ])

    if not kc_ids or not mbon_ids:
        raise ConnectomeLoadError("No KC or MBON neurons found in arrow connectivity data.")

    # Build index maps
    kc_idx = {bid: i for i, bid in enumerate(kc_ids)}
    mbon_idx = {bid: i for i, bid in enumerate(mbon_ids)}

    # Initialize weight matrix
    weights = np.zeros((len(kc_ids), len(mbon_ids)), dtype=float)

    # Populate from edges
    weight_col = "weight" if "weight" in df.columns else None
    for _, row in df.iterrows():
        pre_id = int(row["bodyId_pre"])
        post_id = int(row["bodyId_post"])
        weight = float(row[weight_col]) if weight_col else 1.0

        # Only process KC→MBON edges
        if pre_id in kc_idx and post_id in mbon_idx:
            i = kc_idx[pre_id]
            j = mbon_idx[post_id]
            weights[i, j] += weight

    return ConnectivityData(
        kc_to_mbon=weights,
        neuron_metadata=neurons,
        source_path=str(path),
        mbon_body_ids=mbon_ids,
        kc_body_ids=kc_ids,
    )


def map_to_aso_names(
    connectivity: ConnectivityData,
    aso_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """
    Map loaded connectivity onto Aso-named MBON populations.

    Args:
        connectivity: Loaded ConnectivityData
        aso_names: Ordered list of Aso MBON names (e.g. from ASO_CATALOG.mbon_names)

    Returns:
        (mapped_weights, matched_names) where:
            - mapped_weights: (n_kc, len(aso_names)) array; unmatched columns are zero
            - matched_names: list of Aso names that were successfully matched

    Note:
        MaleCNS body IDs and Aso catalog names do not have 1:1 correspondence.
        This function matches by instance name when available; unmatched MBONs
        are documented as mismatches.
    """
    n_kc = connectivity.kc_to_mbon.shape[0]
    n_aso = len(aso_names)
    mapped = np.zeros((n_kc, n_aso), dtype=float)
    matched = []

    # Try to match each Aso name to a loaded MBON
    for j, aso_name in enumerate(aso_names):
        # Look for matching instance name in neuron metadata
        for mbon_idx, mbon_body_id in enumerate(connectivity.mbon_body_ids):
            meta = connectivity.neuron_metadata.get(mbon_body_id, {})
            instance_name = meta.get("instance", "")
            if instance_name == aso_name:
                mapped[:, j] = connectivity.kc_to_mbon[:, mbon_idx]
                matched.append(aso_name)
                break

    return mapped, matched
