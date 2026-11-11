# Simplified code-package QA

## Package scope

- 13 manuscript-facing analysis and figure scripts.
- 3 internal helper modules.
- 3 validation/test scripts.
- 1 top-level workflow launcher.
- No Google Earth Engine or other JavaScript files.

## Completed checks

- Parsed all 20 Python files successfully.
- Ran 9 synthetic drought-analysis and independent-validation unit tests; all passed.
- Confirmed that the top-level launcher, independent-validation script and
  reduced-factor XGBoost–SHAP script expose working command-line help.
- Found no hard-coded local drive paths in the Python source.
- Found no likely API keys, passwords, private keys, service-account strings or
  client-secret strings.

## Not run

The complete scientific workflow was not executed because raw field data,
third-party products, annual rasters and analysis-ready panels are not included
in this code-only package. A clean-environment numerical rerun remains required
after the linked data deposit is assembled.

## Author checks before publication

1. Select a software license.
2. Add the final data and software DOI(s).
3. Verify the regenerated headline values: 2.53, 88.0%, 50.2% and 88.1%.

The independent-validation code now defines the 14 records marked
`pass_basic_checks` as the primary set and retains all 16 candidates as a
sensitivity analysis.

