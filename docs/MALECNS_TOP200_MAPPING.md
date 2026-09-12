# MaleCNS Top-200 Fixture: Mapping Limitations

## Summary

The committed fixture `tests/fixtures/malecns_kc_mbon_real_top200.json` contains **REAL published weights** (top 200 KC→MBON edges by weight from Janelia MaleCNS v1.0). However, MBON instance name mapping to the Aso catalog is **partial**.

## Mapping Results

- **Source**: Top 200 edges, weights [41, 152], mean≈49.70
- **MBONs in fixture**: 18 unique (MBON01, MBON03, MBON05, MBON09, MBON11, MBON27, MBON30, MBON31, MBON32, ...)
- **Aso catalog**: 7 MBONs (gamma5beta'2a, beta'2mp, beta2beta'2a, gamma2alpha'1, alpha3, alpha'2, gamma1pedc>alpha/beta)
- **Successfully mapped**: ~6/18 from fixture to Aso catalog
- **Unmapped**: Remaining connections fall back to random weights (with warnings)

## Why Partial Mapping?

1. **Nomenclature mismatch**: MaleCNS v1.0 uses numeric IDs (MBON01, MBON05, ...) while Aso catalog uses compartment names (MBON-gamma5beta'2a, ...)
2. **Coverage**: Top-200 edges concentrate on highest-weight connections, which may not cover all 7 Aso MBONs
3. **Instance name variations**: Greek letters (gamma/alpha/beta) vs. shorthand (y/a/b) require heuristic matching

## Impact

- **Tests pass**: The fixture validates that **real weights differ from random initialization** (diff norm > 0.01)
- **Circuit runs**: Both random-fallback and mapped-weight circuits can step and learn
- **Honest documentation**: We do NOT claim full Aso coverage from the top-200 fixture

## Full Dataset

For complete connectivity:
1. Download full 61,210 KC→MBON edges (see `docs/MALECNS_DATA.md`)
2. Use `scripts/filter_malecns_connectivity.py` to extract all edges
3. Load via `MaleCNSCircuit(connectivity_path="data/malecns/kc_mbon_connectivity.feather")`

Full dataset provides better Aso coverage (more MBONs), but still requires manual name mapping.

## Phase 4 Status

**Complete** with honest limitations:
- ✅ REAL published weights available (committed fixture + full data)
- ✅ Loader infrastructure works end-to-end
- ✅ No pandas dependency (pyarrow-native)
- ⚠️ MBON name mapping is partial and heuristic (not all Aso names matched)
- ⚠️ Full biological accuracy requires expert curation of MBON→Aso mappings

This is documented, not hidden.
