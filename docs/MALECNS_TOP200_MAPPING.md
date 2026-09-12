# MaleCNS Top-200 Fixture: Mapping Limitations

## Summary

The committed fixture `tests/fixtures/malecns_kc_mbon_real_top200.json` contains **REAL published weights** (top 200 KC→MBON edges by weight from Janelia MaleCNS v1.0). MBON mapping onto the reduced Aso catalog is **partial**.

Phase 4 is **partial**. This fixture is not a complete production connectome.

## Mapping Results

- **Source**: Top 200 edges, weights [41, 152], mean≈49.70
- **MBONs in fixture**: 18 unique bodies (types MBON01, MBON03, MBON05, MBON09, MBON11, MBON27, MBON30, MBON31, MBON32)
- **Aso catalog**: 7 MBONs (gamma5beta'2a, beta'2mp, beta2beta'2a, gamma2alpha'1, alpha3, alpha'2, gamma1pedc>alpha/beta)
- **Catalog types present in the top-200**: MBON01, MBON03, MBON11 (3/7)
- **Unmapped fixture types**: published Aso 2014 types not in this catalog (e.g. MBON05, MBON09) and MaleCNS expansions (MBON27, MBON30–32) for which we do **not** invent Aso identities

Mapping uses the curated Aso 2014 eLife e04580 Table 1 short-name table in `malecns_connectome.PUBLISHED_MBON_SHORT_TO_ASO`. Heuristic instance normalization is fallback only. Left/right bodies of the same type are summed onto one catalog column.

## Why Partial?

1. **Catalog is a 7-type subset**: MaleCNS v1.0 has 97 MBON bodies / 37 type labels. Only the seven `ASO_CATALOG` names become circuit columns.
2. **Top-200 coverage**: Highest-weight edges miss MBON02, MBON12, MBON13, MBON14.
3. **No invented identities**: Types 23+ and `*-like` labels stay unmatched.

## Impact

- **Tests pass**: Real weights differ from random initialization (diff norm > 0.01) when `n_kc` matches the file (185).
- **Circuit runs**: Mapped columns carry published synapse counts; unmatched catalog columns stay zero.
- **Honest documentation**: We do **not** claim full Aso coverage from the top-200 fixture, and we do **not** mark Phase 4 complete.

## Full Dataset

A filtered export of **all** 61,210 published KC→MBON edges is at `data/malecns/kc_mbon_connectivity.feather` (4,063 KCs, 97 MBONs, CC-BY 4.0). That file covers all seven catalog types. Load it with matching `n_kc`:

```python
from affective_fly import MaleCNSCircuit, load_connectome

path = "data/malecns/kc_mbon_connectivity.feather"
n_kc = load_connectome(path).kc_to_mbon.shape[0]  # 4063
circuit = MaleCNSCircuit(backend="lif", n_kc=n_kc, connectivity_path=path)
```

`n_kc` mismatch raises `ConnectomeLoadError` unless `allow_kc_mismatch=True` (truncate/pad).

To rebuild the file from Janelia GCS, see `docs/MALECNS_DATA.md`.

## Phase 4 Status

**Partial**:

- REAL published weights: top-200 fixture + committed full KC→MBON feather
- Curated Aso 2014 short-name table (not heuristic-only)
- Loader fails on KC-count mismatch unless overridden
- Full biological coverage of 34/97 MBON types is out of scope for the reduced catalog
- Gain-law refit after real fan-out is a later item (not done here)
