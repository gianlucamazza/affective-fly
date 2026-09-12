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
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Aso et al. 2014 eLife e04580 Table 1 short names → compartment names.
# ASCII forms match ASO_CATALOG (gamma/beta/alpha, prime as ').
# Types 23+ and "*-like" MaleCNS labels are intentionally absent: we do not
# invent Aso identities for them.
PUBLISHED_MBON_SHORT_TO_ASO: dict[str, str] = {
    "MBON01": "MBON-gamma5beta'2a",
    "MBON02": "MBON-beta2beta'2a",
    "MBON03": "MBON-beta'2mp",
    "MBON04": "MBON-beta'2mp_bilateral",
    "MBON05": "MBON-gamma4>gamma1gamma2",
    "MBON06": "MBON-beta1>alpha",
    "MBON07": "MBON-alpha1",
    "MBON08": "MBON-gamma3",
    "MBON09": "MBON-gamma3beta'1",
    "MBON10": "MBON-beta'1",
    "MBON11": "MBON-gamma1pedc>alpha/beta",
    "MBON12": "MBON-gamma2alpha'1",
    "MBON13": "MBON-alpha'2",
    "MBON14": "MBON-alpha3",
    "MBON15": "MBON-alpha'1",
    "MBON16": "MBON-alpha'3ap",
    "MBON17": "MBON-alpha'3m",
    "MBON18": "MBON-alpha2sc",
    "MBON19": "MBON-alpha2p3p",
    "MBON20": "MBON-gamma1gamma2",
    "MBON21": "MBON-gamma4gamma5",
    "MBON22": "MBON-calyx",
}

_MBON_EXACT_TYPE = re.compile(r"^MBON-?(\d+)$", re.IGNORECASE)
_MBON_INSTANCE_TYPE = re.compile(r"^MBON-?(\d+)(?!\d)(?:\(|_|$)", re.IGNORECASE)


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
    """Load JSON connectivity export.

    Supports two formats:
    1. Dict with 'edges' and 'neurons' keys
    2. List of edge dicts (infer neuron metadata from edge metadata columns)
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise ConnectomeLoadError(f"Failed to read JSON from {path}: {e}") from e

    # Handle edge-list format (list of dicts)
    if isinstance(data, list):
        edges_list = data
        # Infer neuron metadata from edges
        neurons: dict[int, dict[str, Any]] = {}
        for edge in edges_list:
            pre_id = edge["bodyId_pre"]
            post_id = edge["bodyId_post"]
            if pre_id not in neurons:
                neurons[pre_id] = {
                    "bodyId": pre_id,
                    "type": edge.get("type_pre", "KC"),
                    "instance": edge.get("instance_pre", ""),
                }
            if post_id not in neurons:
                neurons[post_id] = {
                    "bodyId": post_id,
                    "type": edge.get("type_post", "MBON"),
                    "instance": edge.get("instance_post", ""),
                }
    # Handle dict format with 'edges' and 'neurons'
    elif isinstance(data, dict):
        if "edges" not in data or "neurons" not in data:
            raise ConnectomeLoadError(
                f"JSON dict must contain 'edges' and 'neurons' keys. Found: {list(data.keys())}"
            )
        edges_list = data["edges"]
        neurons = {n["bodyId"]: n for n in data["neurons"]}
    else:
        raise ConnectomeLoadError(
            f"JSON must be a dict (with 'edges'/'neurons') or a list of edges. Got: {type(data)}"
        )

    # Collect KC and MBON body IDs
    # KC types can be "KC", "KCab-s", "KCg-s2", etc.
    # MBON types can be "MBON", "MBON01", "MBON05", etc.
    kc_ids = sorted([bid for bid, meta in neurons.items() if meta.get("type", "").startswith("KC")])
    mbon_ids = sorted([
        bid for bid, meta in neurons.items()
        if meta.get("type", "").startswith("MBON")
    ])

    if not kc_ids or not mbon_ids:
        raise ConnectomeLoadError("No KC or MBON neurons found in connectivity data.")

    # Build index maps
    kc_idx = {bid: i for i, bid in enumerate(kc_ids)}
    mbon_idx = {bid: i for i, bid in enumerate(mbon_ids)}

    # Initialize weight matrix
    weights = np.zeros((len(kc_ids), len(mbon_ids)), dtype=float)

    # Populate from edges
    for edge in edges_list:
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
    """Load Feather or Parquet connectivity export using pyarrow (pandas-free)."""
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

    # Expect columns: bodyId_pre, bodyId_post, weight (and optionally neuron metadata)
    required_cols = {"bodyId_pre", "bodyId_post"}
    if not required_cols.issubset(table.column_names):
        raise ConnectomeLoadError(
            f"Arrow file must contain columns {required_cols}. Found: {list(table.column_names)}"
        )

    # Access columns directly via pyarrow (no pandas conversion)
    body_id_pre_col = table.column("bodyId_pre")
    body_id_post_col = table.column("bodyId_post")
    weight_col = table.column("weight") if "weight" in table.column_names else None
    type_pre_col = table.column("type_pre") if "type_pre" in table.column_names else None
    type_post_col = table.column("type_post") if "type_post" in table.column_names else None
    instance_pre_col = table.column("instance_pre") if "instance_pre" in table.column_names else None
    instance_post_col = table.column("instance_post") if "instance_post" in table.column_names else None

    # Build neuron metadata from edge metadata columns
    neurons: dict[int, dict[str, Any]] = {}
    all_pre = set()
    all_post = set()

    for i in range(table.num_rows):
        pre_id = int(body_id_pre_col[i].as_py())
        post_id = int(body_id_post_col[i].as_py())
        all_pre.add(pre_id)
        all_post.add(post_id)

        if type_pre_col and pre_id not in neurons:
            neurons[pre_id] = {
                "bodyId": pre_id,
                "type": type_pre_col[i].as_py() if type_pre_col else "",
                "instance": instance_pre_col[i].as_py() if instance_pre_col else "",
            }
        if type_post_col and post_id not in neurons:
            neurons[post_id] = {
                "bodyId": post_id,
                "type": type_post_col[i].as_py() if type_post_col else "",
                "instance": instance_post_col[i].as_py() if instance_post_col else "",
            }

    # If no type columns, infer from connection patterns
    if not type_pre_col:
        for body_id in all_pre | all_post:
            if body_id not in neurons:
                inferred_type = "KC" if body_id in all_pre and body_id not in all_post else "MBON"
                neurons[body_id] = {
                    "bodyId": body_id,
                    "type": inferred_type,
                    "instance": "",
                }

    # Collect KC and MBON body IDs
    # KC types can be "KC", "KCab-s", "KCg-s2", etc.
    # MBON types can be "MBON", "MBON01", "MBON05", etc.
    kc_ids = sorted([bid for bid, meta in neurons.items() if meta.get("type", "").startswith("KC")])
    mbon_ids = sorted([
        bid for bid, meta in neurons.items()
        if meta.get("type", "").startswith("MBON")
    ])

    if not kc_ids or not mbon_ids:
        raise ConnectomeLoadError("No KC or MBON neurons found in arrow connectivity data.")

    # Build index maps
    kc_idx = {bid: i for i, bid in enumerate(kc_ids)}
    mbon_idx = {bid: i for i, bid in enumerate(mbon_ids)}

    # Initialize weight matrix
    weights = np.zeros((len(kc_ids), len(mbon_ids)), dtype=float)

    # Populate from edges (process all rows)
    for i in range(table.num_rows):
        pre_id = int(body_id_pre_col[i].as_py())
        post_id = int(body_id_post_col[i].as_py())
        weight = float(weight_col[i].as_py()) if weight_col else 1.0

        # Only process KC→MBON edges
        if pre_id in kc_idx and post_id in mbon_idx:
            row_i = kc_idx[pre_id]
            col_j = mbon_idx[post_id]
            weights[row_i, col_j] += weight

    return ConnectivityData(
        kc_to_mbon=weights,
        neuron_metadata=neurons,
        source_path=str(path),
        mbon_body_ids=mbon_ids,
        kc_body_ids=kc_ids,
    )


def _normalize_mbon_name(name: str) -> str:
    """
    Normalize MaleCNS MBON instance names to Aso-style names.

    MaleCNS uses abbreviated lobe names in instance strings:
        - y → gamma
        - B → beta (uppercase to distinguish from lowercase 'b')
        - a → alpha (only as a standalone segment, not within 'gamma')

    Examples:
        - "MBON01(y5B'2a)_R" → "MBON-gamma5beta'2a"
        - "MBON14(a3)_R" → "MBON-alpha3"
        - "MBON11(y1pedc>a/B)_L" → "MBON-gamma1pedc>alpha/beta"

    Args:
        name: Instance name from MaleCNS data (may include MBON##(...) wrapper)

    Returns:
        Normalized name matching Aso catalog style
    """
    # Extract the part in parentheses if present (e.g., "MBON01(y5B'2a)_R" → "y5B'2a")
    if "(" in name and ")" in name:
        start = name.index("(") + 1
        end = name.index(")")
        core = name[start:end]
    else:
        core = name

    # Apply abbreviation expansions in order
    # Start with less ambiguous replacements
    normalized = core

    # Replace y with gamma (Greek gamma lobe)
    # But be careful: 'y' can appear alone (y1, y2) or in sequences (y1y2)
    # We want: y1 → gamma1, but not gamma → galphamma
    normalized = normalized.replace("y", "gamma")

    # Replace B (uppercase) with beta
    normalized = normalized.replace("B", "beta")

    # Replace 'a' with 'alpha' only in specific contexts:
    # - At the start: a3 → alpha3
    # - After '>' or '/': >a/B → >alpha/beta
    # - But NOT in the middle of 'gamma' or at the end of lobe descriptors like '2a'
    # Strategy: Only replace standalone 'a' followed by a digit or at boundaries
    # Match 'a' at start of string followed by digit
    normalized = re.sub(r'^a(\d)', r'alpha\1', normalized)
    # Match 'a' after '>' or '/'
    normalized = re.sub(r'([>/])a([/\d]|$)', r'\1alpha\2', normalized)

    # Add MBON- prefix if not present
    if not normalized.startswith("MBON"):
        normalized = "MBON-" + normalized

    return normalized


def malecns_mbon_short_name(meta: dict[str, Any]) -> str | None:
    """Return published Aso short name (MBON01…) from MaleCNS type/instance.

    Exact types like ``MBON01`` match. Instance wrappers like
    ``MBON01(y5B'2a)_R`` match. ``MBON15-like`` does not (not in Aso 2014).
    """
    for raw in (str(meta.get("type") or ""), str(meta.get("instance") or "")):
        raw = raw.strip()
        if not raw:
            continue
        exact = _MBON_EXACT_TYPE.match(raw)
        if exact:
            return f"MBON{int(exact.group(1)):02d}"
        wrapped = _MBON_INSTANCE_TYPE.match(raw)
        if wrapped:
            return f"MBON{int(wrapped.group(1)):02d}"
    return None


def resolve_aso_name(meta: dict[str, Any], aso_names: list[str]) -> str | None:
    """Resolve one MaleCNS MBON onto a catalog name, or None if unmatched.

    Order: curated Aso 2014 short-name table, then exact instance, then
    heuristic instance normalization (for fixtures that already use Aso names).
    A published short name that is not in ``aso_names`` is unmatched — we do
    not invent a catalog column for it.
    """
    aso_set = set(aso_names)
    short = malecns_mbon_short_name(meta)
    if short is not None:
        published = PUBLISHED_MBON_SHORT_TO_ASO.get(short)
        if published is None:
            return None
        return published if published in aso_set else None

    instance_name = str(meta.get("instance") or "")
    if instance_name in aso_set:
        return instance_name
    if instance_name:
        normalized = _normalize_mbon_name(instance_name)
        if normalized in aso_set:
            return normalized
    return None


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
            - matched_names: unique Aso names that received at least one body

    Note:
        MaleCNS often has several bodies per Aso type (left/right). Their
        KC→MBON weights are **summed** onto the catalog column. Mapping uses
        the curated Aso 2014 short-name table first; heuristic instance
        normalization is fallback only. Unmatched MBONs stay zero.
    """
    n_kc = connectivity.kc_to_mbon.shape[0]
    n_aso = len(aso_names)
    mapped = np.zeros((n_kc, n_aso), dtype=float)
    aso_index = {name: j for j, name in enumerate(aso_names)}
    matched: list[str] = []

    for mbon_idx, mbon_body_id in enumerate(connectivity.mbon_body_ids):
        meta = connectivity.neuron_metadata.get(mbon_body_id, {})
        aso_name = resolve_aso_name(meta, aso_names)
        if aso_name is None:
            continue
        j = aso_index[aso_name]
        mapped[:, j] += connectivity.kc_to_mbon[:, mbon_idx]
        if aso_name not in matched:
            matched.append(aso_name)

    return mapped, matched
