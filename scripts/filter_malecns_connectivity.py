#!/usr/bin/env python3
"""
Filter KC→MBON connectivity from published MaleCNS v1.0 data.

Reads the full connectome-weights-male-cns-v1.0-minconf-0.5.feather and
body-annotations-male-cns-v1.0-minconf-0.5.feather files, filters edges where
presynaptic neurons are Kenyon Cells (KCs) and postsynaptic neurons are
Mushroom Body Output Neurons (MBONs), and writes the filtered connectivity
matrix with REAL published weights.

Data source:
    Janelia MaleCNS v1.0 (Schlegel et al. 2023)
    https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/
    License: CC-BY 4.0

Output schema matches expected loader format:
    - bodyId_pre: int64, presynaptic KC body ID
    - bodyId_post: int64, postsynaptic MBON body ID
    - weight: float64, synaptic weight (number of synapses)
    - type_pre: str, KC type
    - type_post: str, MBON type
    - instance_pre: str, KC instance name
    - instance_post: str, MBON instance name

Usage:
    python scripts/filter_malecns_connectivity.py \\
        --weights data/malecns/connectome-weights-male-cns-v1.0-minconf-0.5.feather \\
        --annotations data/malecns/body-annotations-male-cns-v1.0-minconf-0.5.feather \\
        --output data/malecns/kc_mbon_connectivity.feather \\
        --format feather

NO INVENTED WEIGHTS. This script only extracts published data.

MaleCNS v1.0 minconf-0.5 yields 61,210 KC→MBON edges (4,063 KCs, 97 MBONs).
The committed output is data/malecns/kc_mbon_connectivity.feather.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import pandas as pd
    import pyarrow.feather as feather
except ImportError as e:
    print(f"Error: {e}")
    print("Install required packages: pip install pandas pyarrow")
    raise


def load_annotations(annotations_path: Path) -> pd.DataFrame:
    """Load body annotations from feather file.

    Expected columns: bodyId, type, instance, ...
    """
    print(f"Loading annotations from {annotations_path}...")
    df = feather.read_feather(annotations_path)
    print(f"Loaded {len(df):,} body annotations")

    # Normalize column name: bodyId or body_id → bodyId
    if "body_id" in df.columns and "bodyId" not in df.columns:
        df = df.rename(columns={"body_id": "bodyId"})

    if "bodyId" not in df.columns:
        raise ValueError(
            f"Expected 'bodyId' or 'body_id' column in annotations, got: {df.columns.tolist()}"
        )
    return df


def load_weights(weights_path: Path) -> pd.DataFrame:
    """Load connectivity weights from feather file.

    Expected columns: body_pre, body_post, weight (MaleCNS v1.0 flat connectome)
    """
    print(f"Loading connectivity weights from {weights_path}...")
    df = feather.read_feather(weights_path)
    print(f"Loaded {len(df):,} connectivity edges")

    # Normalize column names: body_pre/body_post → bodyId_pre/bodyId_post for consistency
    if "body_pre" in df.columns and "bodyId_pre" not in df.columns:
        df = df.rename(columns={"body_pre": "bodyId_pre", "body_post": "bodyId_post"})

    required = ["bodyId_pre", "bodyId_post", "weight"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Got: {df.columns.tolist()}")
    return df


def filter_kc_mbon_connectivity(
    weights: pd.DataFrame,
    annotations: pd.DataFrame,
    kc_patterns: list[str] | None = None,
    mbon_patterns: list[str] | None = None,
) -> pd.DataFrame:
    """Filter KC→MBON edges from full connectome.

    Args:
        weights: DataFrame with bodyId_pre, bodyId_post, weight
        annotations: DataFrame with bodyId, type, instance
        kc_patterns: Type patterns to identify KCs (default: ['KC', 'KCab', 'KCg'])
        mbon_patterns: Type patterns to identify MBONs (default: ['MBON'])

    Returns:
        Filtered DataFrame with KC→MBON edges including type/instance names
    """
    if kc_patterns is None:
        kc_patterns = ["KC", "KCab", "KCg"]
    if mbon_patterns is None:
        mbon_patterns = ["MBON"]

    print("\nFiltering KC→MBON connectivity...")

    # Identify KC and MBON body IDs
    kc_mask = annotations["type"].str.contains("|".join(kc_patterns), case=False, na=False)
    mbon_mask = annotations["type"].str.contains("|".join(mbon_patterns), case=False, na=False)

    kc_bodies = annotations[kc_mask][["bodyId", "type", "instance"]].rename(
        columns={"bodyId": "bodyId_pre", "type": "type_pre", "instance": "instance_pre"}
    )
    mbon_bodies = annotations[mbon_mask][["bodyId", "type", "instance"]].rename(
        columns={"bodyId": "bodyId_post", "type": "type_post", "instance": "instance_post"}
    )

    print(f"Found {len(kc_bodies):,} KC bodies")
    print(f"Found {len(mbon_bodies):,} MBON bodies")

    # Filter edges: pre must be KC, post must be MBON
    kc_mbon = weights.merge(kc_bodies, on="bodyId_pre", how="inner")
    kc_mbon = kc_mbon.merge(mbon_bodies, on="bodyId_post", how="inner")

    print(f"Filtered {len(kc_mbon):,} KC→MBON edges")

    # Select and order output columns
    out_cols = [
        "bodyId_pre",
        "bodyId_post",
        "weight",
        "type_pre",
        "type_post",
        "instance_pre",
        "instance_post",
    ]
    return kc_mbon[out_cols]


def save_connectivity(
    df: pd.DataFrame,
    output_path: Path,
    format: str = "feather",
    metadata: dict | None = None,
) -> None:
    """Save filtered connectivity to file.

    Args:
        df: Filtered connectivity DataFrame
        output_path: Output file path
        format: Output format ('feather', 'parquet', 'json', 'csv')
        metadata: Optional metadata dict to embed
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if metadata:
        meta_str = json.dumps(metadata, indent=2)
        print(f"\nMetadata:\n{meta_str}")

    if format == "feather":
        feather.write_feather(df, output_path)
    elif format == "parquet":
        df.to_parquet(output_path, index=False)
    elif format == "json":
        df.to_json(output_path, orient="records", indent=2)
    elif format == "csv":
        df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")

    print(f"\nWrote {len(df):,} edges to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Write metadata as sidecar JSON
    if metadata:
        meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Wrote metadata to {meta_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter KC→MBON connectivity from MaleCNS v1.0 data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("data/malecns/connectome-weights-male-cns-v1.0-minconf-0.5.feather"),
        help="Path to connectivity weights feather file (default: %(default)s)",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/malecns/body-annotations-male-cns-v1.0-minconf-0.5.feather"),
        help="Path to body annotations feather file (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/malecns/kc_mbon_connectivity.feather"),
        help="Output path for filtered connectivity (default: %(default)s)",
    )
    parser.add_argument(
        "--format",
        choices=["feather", "parquet", "json", "csv"],
        default="feather",
        help="Output format (default: %(default)s)",
    )
    parser.add_argument(
        "--kc-patterns",
        nargs="+",
        default=["KC", "KCab", "KCg", "KCa'b'"],
        help="Type patterns to identify KCs (default: %(default)s)",
    )
    parser.add_argument(
        "--mbon-patterns",
        nargs="+",
        default=["MBON"],
        help="Type patterns to identify MBONs (default: %(default)s)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Keep only top N edges by weight (default: keep all)",
    )

    args = parser.parse_args()

    # Load data
    annotations = load_annotations(args.annotations)
    weights = load_weights(args.weights)

    # Filter KC→MBON
    kc_mbon = filter_kc_mbon_connectivity(
        weights,
        annotations,
        kc_patterns=args.kc_patterns,
        mbon_patterns=args.mbon_patterns,
    )

    if len(kc_mbon) == 0:
        print("\nWARNING: No KC→MBON edges found! Check patterns and data.")
        return 1

    # Keep top N by weight if requested
    if args.top_n is not None and args.top_n < len(kc_mbon):
        print(f"\nKeeping top {args.top_n} edges by weight...")
        kc_mbon = kc_mbon.nlargest(args.top_n, "weight")

    # Provenance metadata
    metadata = {
        "source": "Janelia MaleCNS v1.0 (Schlegel et al. 2023)",
        "url": "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/",
        "license": "CC-BY 4.0",
        "description": "KC→MBON connectivity filtered from full MaleCNS connectome",
        "weights_file": str(args.weights.name),
        "annotations_file": str(args.annotations.name),
        "kc_patterns": args.kc_patterns,
        "mbon_patterns": args.mbon_patterns,
        "n_edges": len(kc_mbon),
        "n_kc": kc_mbon["bodyId_pre"].nunique(),
        "n_mbon": kc_mbon["bodyId_post"].nunique(),
    }

    # Save
    save_connectivity(kc_mbon, args.output, format=args.format, metadata=metadata)

    print("\nSummary:")
    print(f"  Unique KCs: {metadata['n_kc']:,}")
    print(f"  Unique MBONs: {metadata['n_mbon']:,}")
    print(f"  Total edges: {metadata['n_edges']:,}")
    print(f"  Weight range: [{kc_mbon['weight'].min():.1f}, {kc_mbon['weight'].max():.1f}]")
    print(f"  Mean weight: {kc_mbon['weight'].mean():.2f}")

    return 0


if __name__ == "__main__":
    exit(main())
