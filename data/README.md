# Required analysis inputs

Large inputs are not committed to the code repository. Deposit them separately
or provide stable source links and preparation metadata.

## Minimum prepared inputs

| Input group | Expected project location | Purpose |
|---|---|---|
| Peatland samples and predictor stack | `zoigepeatland/data/` | Boundary validation |
| Fixed peatland mask | `zoigepeatland/output_basic_boundary/` | All spatial analyses |
| AGB calibration features | `agb_ecological_weighted_validation/outputs/` | Spatial-block PLSR validation |
| Frozen AGB model bundle | `agb_ecological_weighted_validation/models/` | 2025 independent validation |
| Annual AGB rasters, 1988–2024 | `agb_dual_grid_2013_2024/rasters_90m/` | Long-term, drought and driver analyses |
| 2025 AGB raster and field table | `agb_2025_independent_validation/` | Independent validation and 2025 extension |
| Annual FVC rasters | `agb_resilience_section_3_3_1988_2025/rasters/fvc_90m/` | Four-event drought analysis |
| LAI/FAPAR and annual-state panel | `agb_lai_resilience_breakpoint_main95_1988_2024/` | Long-term and drought analyses |
| 5-km grid definition | `agb_driver_xgboost_shap_v2/outputs/analysis_grid.json` | Driver and drought analyses |
| Climate and ancillary products | `climate_data/input/` and `AGB_COLLECT_DATA_ROOT` | Event screening and XGBoost–SHAP |
| Annual Landsat composites | `LANDSAT_COMPOSITE_ROOT` | 2025 FVC/AGB support where rebuilding is needed |

## Metadata required for deposition

For each released dataset, provide title, creators, version, DOI/URL, license,
units, missing-value codes, coordinate reference system, spatial/temporal
coverage and processing provenance.

If field coordinates cannot be public, deposit generalized coordinates or
provide controlled access while retaining non-identifying observations and
validation predictions where possible.

