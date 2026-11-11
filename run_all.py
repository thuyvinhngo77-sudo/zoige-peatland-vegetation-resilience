#!/usr/bin/env python
"""Run the manuscript workflows from one public entry point."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

STAGES = {
    "study-area": [
        ["scripts/figures/fig01_study_area.py"],
    ],
    "validation": [
        ["scripts/validation/boundary_spatial_cv.py"],
        ["scripts/validation/agb_spatial_cv.py"],
        ["scripts/validation/independent_2025_validation.py"],
        ["scripts/figures/fig02_validation.py"],
    ],
    "long-term": [
        ["scripts/long_term/long_term_dynamics.py"],
        ["scripts/figures/fig03_04_long_term.py"],
    ],
    "drivers": [
        ["scripts/drivers/xgboost_shap_reduced.py", "--rebuild-panel"],
        ["scripts/figures/fig08_11_trajectory_shap.py"],
    ],
    "drought": [
        ["scripts/drought/run_drought_analysis.py"],
        ["scripts/figures/fig05_06_drought.py"],
        ["scripts/figures/fig07_drought_maps.py"],
    ],
}

ALL_ORDER = ["study-area", "validation", "long-term", "drivers", "drought"]


def run_stage(stage: str) -> None:
    for command in STAGES[stage]:
        script = ROOT / command[0]
        print(f"\n[{stage}] {script.relative_to(ROOT)}", flush=True)
        subprocess.run(
            [sys.executable, str(script), *command[1:]],
            cwd=ROOT,
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one manuscript workflow or all workflows in dependency order."
    )
    parser.add_argument(
        "stage",
        choices=[*STAGES, "all"],
        help="Workflow stage to run.",
    )
    args = parser.parse_args()
    stages = ALL_ORDER if args.stage == "all" else [args.stage]
    for stage in stages:
        run_stage(stage)


if __name__ == "__main__":
    main()

