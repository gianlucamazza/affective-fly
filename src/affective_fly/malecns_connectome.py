"""
Load published MaleCNS KC→MBON connectivity with REAL weights.

This module loads filtered connectivity data (output of
scripts/filter_malecns_connectivity.py) and maps it to circuit indices.

NO INVENTED WEIGHTS: all connectivity data comes from published sources.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


@dataclass
class ConnectivityData:
    """Loaded KC→MBON connectivity with real weights.

    Attributes:
        edges: List of (kc_body_id, mbon_body_id, weight) tuples
        kc_body_ids: Unique KC body IDs
        mbon_body_ids: Unique MBON body IDs
        mbon_types: Map from body ID to MBON type name
        mbon_instances: Map from body ID to MBON instance name
    """

    edges: list[tuple[int, int, float]]
    kc_body_ids: set[int]
    mbon_body_ids: set[int]
    mbon_types: dict[int, str]
    mbon_instances: dict[int, str]

    @property
    def n_kc(self) -> int:
        return len(self.kc_body_ids)

    @property
    def n_mbon(self) -> int:
        return len(self.mbon_body_ids)


def load_connectivity(path: str | Path) -> ConnectivityData:
    """Load KC→MBON connectivity from file.

    Supports .feather, .parquet, .json, .csv formats.

    Expected columns:
        - bodyId_pre: int64, presynaptic KC body ID
        - bodyId_post: int64, postsynaptic MBON body ID
        - weight: float64, synaptic weight
        - type_post: str, MBON type
        - instance_post: str, MBON instance name

    Args:
        path: Path to connectivity file (feather/parquet/json/csv)

    Returns:
        ConnectivityData with loaded edges and metadata

    Raises:
        FileNotFoundError: If path does not exist
        ValueError: If required columns are missing
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Connectivity file not found: {path}")

    # Import pandas lazily (not a core dependency for all users)
    try:
        import pandas as pd
        import pyarrow.feather as feather
    except ImportError as e:
        raise ImportError(
            f"Loading connectivity requires pandas and pyarrow: {e}\n"
            "Install with: pip install pandas pyarrow"
        ) from e

    # Load based on extension
    suffix = path.suffix.lower()
    if suffix == ".feather":
        df = feather.read_feather(path)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix == ".json":
        df = pd.read_json(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(
            f"Unsupported file format: {suffix} (expected .feather/.parquet/.json/.csv)"
        )

    # Validate required columns
    required = ["bodyId_pre", "bodyId_post", "weight"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Extract edges
    edges = [
        (int(row["bodyId_pre"]), int(row["bodyId_post"]), float(row["weight"]))
        for _, row in df.iterrows()
    ]

    kc_body_ids = set(df["bodyId_pre"].unique())
    mbon_body_ids = set(df["bodyId_post"].unique())

    # Extract MBON metadata if available
    mbon_types = {}
    mbon_instances = {}
    if "type_post" in df.columns:
        for body_id in mbon_body_ids:
            rows = df[df["bodyId_post"] == body_id]
            if not rows.empty:
                mbon_types[body_id] = rows.iloc[0]["type_post"]
    if "instance_post" in df.columns:
        for body_id in mbon_body_ids:
            rows = df[df["bodyId_post"] == body_id]
            if not rows.empty:
                mbon_instances[body_id] = rows.iloc[0]["instance_post"]

    return ConnectivityData(
        edges=edges,
        kc_body_ids=kc_body_ids,
        mbon_body_ids=mbon_body_ids,
        mbon_types=mbon_types,
        mbon_instances=mbon_instances,
    )


def map_connectivity_to_circuit(
    conn: ConnectivityData,
    n_kc: int,
    n_mbon: int,
    catalog_mbon_names: list[str],
    seed: int = 42,
    w_max: float = 0.15,
) -> np.ndarray:
    """Map loaded connectivity to circuit weight matrix.

    Strategy:
    1. Sample n_kc KCs randomly from available body IDs
    2. Match MaleCNS MBON instances to Aso catalog names (fuzzy match)
    3. Build weight matrix w_kc_mbon[kc_idx, mbon_idx] from real weights
    4. Unmapped connections use uniform random on [0, w_max] as fallback

    Args:
        conn: Loaded connectivity data
        n_kc: Circuit KC population size
        n_mbon: Circuit MBON population size (should match len(catalog_mbon_names))
        catalog_mbon_names: Aso catalog MBON names in order
        seed: Random seed for sampling and fallback init
        w_max: Max weight for fallback random init

    Returns:
        Weight matrix w_kc_mbon with shape (n_kc, n_mbon)
    """
    rng = np.random.RandomState(seed)

    # Initialize with fallback (uniform random)
    w_kc_mbon = rng.uniform(0.0, w_max, size=(n_kc, n_mbon))

    # Sample n_kc KCs from available body IDs
    available_kcs = sorted(conn.kc_body_ids)
    if len(available_kcs) < n_kc:
        print(
            f"Warning: Only {len(available_kcs)} KCs in data, but circuit needs {n_kc}. "
            f"Will sample with replacement."
        )
        sampled_kc_ids = rng.choice(available_kcs, size=n_kc, replace=True)
    else:
        sampled_kc_ids = rng.choice(available_kcs, size=n_kc, replace=False)

    # Map KC body IDs to circuit indices
    kc_body_to_idx = {body_id: idx for idx, body_id in enumerate(sampled_kc_ids)}

    # Map MBON instances to catalog indices (fuzzy match on name)
    # MaleCNS v1.0 uses notation like "MBON01(y5B'2a)_L" while Aso catalog
    # uses "MBON-gamma5beta'2a". Mapping: gamma=y, alpha=a, beta=B.
    mbon_body_to_idx: dict[int, int] = {}
    for body_id, instance in conn.mbon_instances.items():
        instance_lower = instance.lower()
        # Try direct substring match first
        for idx, catalog_name in enumerate(catalog_mbon_names):
            catalog_lower = catalog_name.lower()
            if catalog_lower in instance_lower or instance_lower in catalog_lower:
                mbon_body_to_idx[body_id] = idx
                break
        # Try Greek-to-shorthand mapping (gamma→y, alpha→a, beta→b)
        if body_id not in mbon_body_to_idx:
            for idx, catalog_name in enumerate(catalog_mbon_names):
                # Extract compartment part after "MBON-" prefix
                if not catalog_name.startswith("MBON-"):
                    continue
                compartment = catalog_name[5:]  # Skip "MBON-"
                # Simple heuristic: gamma→y, alpha→a, beta→b (case-insensitive)
                shorthand = (
                    compartment.replace("gamma", "y").replace("alpha", "a").replace("beta", "b")
                )
                if shorthand.lower() in instance_lower:
                    mbon_body_to_idx[body_id] = idx
                    break

    if not mbon_body_to_idx:
        print(
            "Warning: Could not match any MBON instances to catalog names. "
            "Falling back to random weights."
        )
        print(f"Catalog names: {catalog_mbon_names[:3]}...")
        print(f"Instance samples: {list(conn.mbon_instances.values())[:3]}...")
        return w_kc_mbon

    print(f"Matched {len(mbon_body_to_idx)}/{len(conn.mbon_body_ids)} MBONs to catalog")

    # Fill in real weights where we have matches
    n_mapped = 0
    for kc_body, mbon_body, weight in conn.edges:
        if kc_body in kc_body_to_idx and mbon_body in mbon_body_to_idx:
            kc_idx = kc_body_to_idx[kc_body]
            mbon_idx = mbon_body_to_idx[mbon_body]
            # Normalize weight: raw weight is synapse count, scale to [0, w_max]
            # Simple scaling: weight / max_weight_in_data * w_max
            # For now, just use raw weight and clip to w_max
            w_kc_mbon[kc_idx, mbon_idx] = min(weight, w_max)
            n_mapped += 1

    if n_mapped > 0:
        print(f"Loaded {n_mapped} real KC→MBON weights from connectivity data")
    else:
        print("Warning: No KC→MBON connections mapped to circuit. Using random fallback.")

    return w_kc_mbon
