# Zoige peatland manuscript code

This is the simplified public-repository edition of the local analysis code
for the manuscript:

> Long term biomass trajectories, vegetation drought resilience and
> hydroclimatic associations in the Zoige peatlands

It begins with prepared annual rasters, field/validation tables and
environmental datasets. Satellite-download scripts, exploratory models,
obsolete workflows, group-meeting figures and manuscript-editing utilities are
not included.

## Repository structure

```text
scripts/
  validation/   peatland and AGB validation
  long_term/    1988–2025 AGB/FVC/LAI analyses
  drought/      four-event drought metrics
  drivers/      reduced-factor XGBoost–SHAP analysis
  figures/      manuscript Figs. 1–11
  _internal/    three helpers required by validation/annual-state processing
tests/          drought-analysis and map-output checks
config/         external path example
run_all.py      one public workflow entry point
```

There are 13 manuscript-facing analysis/figure scripts, three internal helper
modules, two tests and one launcher. The internal modules are implementation
dependencies and do not represent additional manuscript analyses.

## Figure and analysis map

| Manuscript output | Analysis | Figure code |
|---|---|---|
| Fig. 1 | prepared spatial layers | `scripts/figures/fig01_study_area.py` |
| Fig. 2 | `scripts/validation/` | `scripts/figures/fig02_validation.py` |
| Figs. 3–4 | `scripts/long_term/long_term_dynamics.py` | `scripts/figures/fig03_04_long_term.py` |
| Figs. 5–6 | `scripts/drought/run_drought_analysis.py` | `scripts/figures/fig05_06_drought.py` |
| Fig. 7 | four-event 90-m responses | `scripts/figures/fig07_drought_maps.py` |
| Figs. 8–11 | `scripts/drivers/xgboost_shap_reduced.py` | `scripts/figures/fig08_11_trajectory_shap.py` |

The reduced-factor workflow is the source of the reported 88.1% summed
absolute held-out SHAP contribution for climate variables.

## Installation

```bash
conda env create -f environment.yml
conda activate zoige-peatland
```

Set the external data locations using the variables shown in
`config/paths.env.example`:

- `AGB_COLLECT_DATA_ROOT`
- `LANDSAT_COMPOSITE_ROOT`

## Running the workflows

Run individual stages:

```bash
python run_all.py study-area
python run_all.py validation
python run_all.py long-term
python run_all.py drivers
python run_all.py drought
```

Run everything in dependency order:

```bash
python run_all.py all
```

The drought preparation uses the trajectory panel written by the driver stage,
so `drivers` precedes `drought` in the full run.

## Data boundary

The repository intentionally does not contain raw field data, third-party
datasets or large raster products. Required inputs and suggested repository
treatment are listed in `data/README.md`. A complete public release should
link this code to a versioned data deposit with stable identifiers.

## Independent 2025 validation

The underlying workflow contains a complete 16-record temporal test and a
14-record result after two pre-flagged very high-AGB records were excluded.
The manuscript currently reports 14 quality-controlled observations. The
Methods, deposited record-level audit and code outputs must describe this
choice consistently.

## Before publication

1. Select a software license; see `LICENSE_OPTIONS.md`.
2. Add the data and software DOI(s) to this README and `CITATION.cff`.
3. Run the package in a clean environment with the deposited inputs.
4. Confirm the regenerated headline values: 2.53, 88.0%, 50.2% and 88.1%.

