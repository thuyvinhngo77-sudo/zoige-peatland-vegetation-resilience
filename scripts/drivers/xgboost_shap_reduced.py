#!/usr/bin/env python
"""Run the reduced-factor XGBoost-SHAP experiment for Section 3.4.

The response is Peak-only AGB aggregated from the 90 m production grid inside
the strict main95 peatland mask. Driver modelling uses the common 2001-2024
window and a 5 km analysis grid. All SHAP values are calculated on held-out
fold rows only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import shap
from matplotlib.colors import Normalize
from matplotlib.transforms import Bbox
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window, from_bounds
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[2]
SOURCE_OUT_ROOT = ROOT / "agb_driver_xgboost_shap_full_2001_2024"
OUT_ROOT = ROOT / "agb_driver_xgboost_shap_reduced_2001_2024"
OUTPUTS = OUT_ROOT / "outputs"
FIGURES = OUT_ROOT / "figures"
PANELS = FIGURES / "panels"
CACHE = SOURCE_OUT_ROOT / "cache"
FEATURE_MANIFEST = SOURCE_OUT_ROOT / "outputs" / "feature_manifest.json"

BASE_PANEL = (
    ROOT
    / "agb_lai_resilience_breakpoint_main95_1988_2024"
    / "cache"
    / "annual_state_panel_5km.pkl.gz"
)
GRID_JSON = ROOT / "agb_driver_xgboost_shap_v2" / "outputs" / "analysis_grid.json"
AGB_DIR = ROOT / "agb_dual_grid_2013_2024" / "rasters_90m"
STRICT_MASK = (
    ROOT
    / "zoigepeatland"
    / "output_basic_boundary"
    / "Zoige_RF_basic_analysis_mask_main95_fill400.tif"
)
TERRACLIMATE = ROOT / "climate_data" / "input" / "Zoige_TerraClimate_GS_1985_2024_stack.tif"
ERA5 = ROOT / "climate_data" / "input" / "Zoige_ERA5Land_Climate_2000_2025_stack.tif"

COLLECT = Path(
    os.environ.get("AGB_COLLECT_DATA_ROOT", str(ROOT / "external_data"))
)
HFP_DIR = COLLECT / "人类足迹数据" / "16571064"

START_YEAR = 2001
END_YEAR = 2024
GRID_M = 5000
SPATIAL_BLOCK_M = 25000
MIN_MASK_FRACTION = 0.10
MIN_VALID_FRACTION = 0.50
MIN_CELL_YEARS = 15
SEED = 20260820

CRU_RAW = {
    "ai": "cru_ai_z",
    "pre_annual_mm": "cru_pre_annual_z",
    "pre_amj_mm": "cru_pre_amj_z",
    "pre_jas_mm": "cru_pre_jas_z",
    "tmp_annual_c": "cru_tmp_annual_z",
    "tmp_amj_c": "cru_tmp_amj_z",
    "tmp_jas_c": "cru_tmp_jas_z",
}
TC_VARIABLES = {
    "GST_C": "tc_gst_z",
    "GSP_mm": "tc_gsp_z",
    "VPD_kPa": "tc_vpd_z",
    "SR_W_m2": "tc_solar_z",
    "PET_mm": "tc_pet_z",
    "P_minus_PET_mm": "tc_water_balance_z",
}
ERA_VARIABLES = {
    "tmean_c_ann": "era_tmean_ann_z",
    "tmin_c_ann": "era_tmin_ann_z",
    "tmax_c_ann": "era_tmax_ann_z",
    "precip_mm_ann": "era_precip_ann_z",
    "pet_mm_ann": "era_pet_ann_z",
    "water_balance_mm_ann": "era_water_balance_ann_z",
    "vpd_kpa_ann": "era_vpd_ann_z",
    "soil_water_l1_ann": "era_soil_water_ann_z",
    "solar_mj_m2_ann": "era_solar_ann_z",
    "snowfall_mm_ann": "era_snowfall_ann_z",
    "tmean_c_gs": "era_tmean_gs_z",
    "tmin_c_gs": "era_tmin_gs_z",
    "tmax_c_gs": "era_tmax_gs_z",
    "precip_mm_gs": "era_precip_gs_z",
    "pet_mm_gs": "era_pet_gs_z",
    "water_balance_mm_gs": "era_water_balance_gs_z",
    "vpd_kpa_gs": "era_vpd_gs_z",
    "soil_water_l1_gs": "era_soil_water_gs_z",
    "solar_mj_m2_gs": "era_solar_gs_z",
    "snowfall_mm_gs": "era_snowfall_gs_z",
}
STATIC_FEATURES = [
    "elevation_m",
    "slope_deg",
    "aspect_sin",
    "aspect_cos",
    "soc_dg_kg",
    "nitrogen_cg_kg",
    "ph_h2o",
    "sand_pct",
    "clay_pct",
]

FEATURE_LABELS = {
    "cru_ai_z": "CRU aridity index",
    "cru_pre_annual_z": "CRU annual precipitation",
    "cru_pre_amj_z": "CRU Apr-Jun precipitation",
    "cru_pre_jas_z": "CRU Jul-Sep precipitation",
    "cru_tmp_annual_z": "CRU annual temperature",
    "cru_tmp_amj_z": "CRU Apr-Jun temperature",
    "cru_tmp_jas_z": "CRU Jul-Sep temperature",
    "tc_gst_z": "TerraClimate growing-season temperature",
    "tc_gsp_z": "TerraClimate growing-season precipitation",
    "tc_vpd_z": "TerraClimate vapor-pressure deficit",
    "tc_solar_z": "TerraClimate solar radiation",
    "tc_pet_z": "TerraClimate potential evapotranspiration",
    "tc_water_balance_z": "TerraClimate climatic water balance",
    "era_tmean_ann_z": "ERA5-Land annual mean temperature",
    "era_tmin_ann_z": "ERA5-Land annual minimum temperature",
    "era_tmax_ann_z": "ERA5-Land annual maximum temperature",
    "era_precip_ann_z": "ERA5-Land annual precipitation",
    "era_pet_ann_z": "ERA5-Land annual potential evapotranspiration",
    "era_water_balance_ann_z": "ERA5-Land annual water balance",
    "era_vpd_ann_z": "ERA5-Land annual vapor-pressure deficit",
    "era_soil_water_ann_z": "ERA5-Land annual surface soil water",
    "era_solar_ann_z": "ERA5-Land annual solar radiation",
    "era_snowfall_ann_z": "ERA5-Land annual snowfall",
    "era_tmean_gs_z": "ERA5-Land growing-season mean temperature",
    "era_tmin_gs_z": "ERA5-Land growing-season minimum temperature",
    "era_tmax_gs_z": "ERA5-Land growing-season maximum temperature",
    "era_precip_gs_z": "ERA5-Land growing-season precipitation",
    "era_pet_gs_z": "ERA5-Land growing-season potential evapotranspiration",
    "era_water_balance_gs_z": "ERA5-Land growing-season water balance",
    "era_vpd_gs_z": "ERA5-Land growing-season vapor-pressure deficit",
    "era_soil_water_gs_z": "ERA5-Land growing-season surface soil water",
    "era_solar_gs_z": "ERA5-Land growing-season solar radiation",
    "era_snowfall_gs_z": "ERA5-Land growing-season snowfall",
    "lai_era_z": "LAI state",
    "fapar_era_z": "FAPAR state",
    "fvc_z": "FVC state",
    "lhgi_z": "Grazing intensity",
    "hfp_z": "Human footprint",
    "elevation_m": "Elevation",
    "slope_deg": "Slope",
    "aspect_sin": "Aspect sine",
    "aspect_cos": "Aspect cosine",
    "soc_dg_kg": "Soil organic carbon",
    "nitrogen_cg_kg": "Soil nitrogen",
    "ph_h2o": "Soil pH",
    "sand_pct": "Sand",
    "clay_pct": "Clay",
}


def ensure_dirs() -> None:
    for path in (OUTPUTS, FIGURES, CACHE):
        path.mkdir(parents=True, exist_ok=True)


def load_grid() -> dict:
    spec = json.loads(GRID_JSON.read_text(encoding="utf-8"))
    left, bottom, right, top = spec["bounds"]
    return {
        **spec,
        "transform": from_origin(left, top, spec["grid_m"], spec["grid_m"]),
    }


def grid_frame(grid: dict) -> pd.DataFrame:
    rows, cols = np.indices((grid["height"], grid["width"]))
    xs = grid["transform"].c + (cols + 0.5) * GRID_M
    ys = grid["transform"].f - (rows + 0.5) * GRID_M
    return pd.DataFrame({
        "row": rows.ravel(),
        "col": cols.ravel(),
        "x": xs.ravel(),
        "y": ys.ravel(),
    })


def warp_array(
    values: np.ndarray,
    src_transform,
    src_crs,
    grid: dict,
    resampling: Resampling = Resampling.average,
) -> np.ndarray:
    destination = np.full((grid["height"], grid["width"]), np.nan, dtype="float32")
    reproject(
        source=values.astype("float32"),
        destination=destination,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=np.nan,
        dst_transform=grid["transform"],
        dst_crs=grid["crs"],
        dst_nodata=np.nan,
        resampling=resampling,
    )
    return destination


def strict_mask_on_grid(src: rasterio.DatasetReader) -> np.ndarray:
    destination = np.zeros((src.height, src.width), dtype="uint8")
    with rasterio.open(STRICT_MASK) as mask_src:
        reproject(
            source=rasterio.band(mask_src, 1),
            destination=destination,
            src_transform=mask_src.transform,
            src_crs=mask_src.crs,
            src_nodata=mask_src.nodata,
            dst_transform=src.transform,
            dst_crs=src.crs,
            dst_nodata=0,
            resampling=Resampling.nearest,
        )
    return destination > 0


def aggregate_peak_only(grid: dict, rebuild: bool) -> pd.DataFrame:
    cache_path = CACHE / "strict_peak_only_5km_2001_2024.pkl.gz"
    if cache_path.exists() and not rebuild:
        return pd.read_pickle(cache_path, compression="gzip")
    base = grid_frame(grid)[["row", "col"]]
    frames = []
    for year in range(START_YEAR, END_YEAR + 1):
        path = AGB_DIR / f"Zoige_AGB_GuardedConsensus_PLSR_90m_{year}.tif"
        if not path.exists():
            raise FileNotFoundError(path)
        with rasterio.open(path) as src:
            peak = src.read(3).astype("float32")
            valid_count = src.read(8).astype("float32")
            if src.nodata is not None:
                peak[peak == src.nodata] = np.nan
                valid_count[valid_count == src.nodata] = np.nan
            strict = strict_mask_on_grid(src)
            valid = strict & np.isfinite(peak) & np.isfinite(valid_count) & (valid_count > 0)
            strict_area = warp_array(strict.astype("float32"), src.transform, src.crs, grid)
            valid_area = warp_array(valid.astype("float32"), src.transform, src.crs, grid)
            peak_mean = warp_array(np.where(valid, peak, np.nan), src.transform, src.crs, grid)
        valid_fraction = np.divide(
            valid_area,
            strict_area,
            out=np.full_like(valid_area, np.nan),
            where=strict_area > 0,
        )
        part = base.copy()
        part["year"] = year
        part["peak_agb"] = peak_mean.ravel()
        part["strict_area_fraction"] = strict_area.ravel()
        part["strict_valid_fraction"] = valid_fraction.ravel()
        frames.append(part)
        print(f"Peak-only strict aggregation: {year}", flush=True)
    result = pd.concat(frames, ignore_index=True)
    result.to_pickle(cache_path, compression="gzip")
    return result


def stack_panel(
    path: Path,
    variables: dict[str, str],
    grid: dict,
    source_prefix: str,
    rebuild: bool,
) -> pd.DataFrame:
    cache_path = CACHE / f"{source_prefix}_5km_2001_2024.pkl.gz"
    if cache_path.exists() and not rebuild:
        return pd.read_pickle(cache_path, compression="gzip")
    base = grid_frame(grid)[["row", "col"]]
    frames = []
    with rasterio.open(path) as src:
        descriptions = {name: i for i, name in enumerate(src.descriptions, 1)}
        for year in range(START_YEAR, END_YEAR + 1):
            part = base.copy()
            part["year"] = year
            for suffix, output in variables.items():
                name = f"{year}_{suffix}"
                if name not in descriptions:
                    raise KeyError(f"Missing band {name} in {path}")
                values = src.read(descriptions[name]).astype("float32")
                if src.nodata is not None:
                    values[values == src.nodata] = np.nan
                values[~np.isfinite(values)] = np.nan
                part[output.replace("_z", "_raw")] = warp_array(
                    values, src.transform, src.crs, grid
                ).ravel()
            frames.append(part)
            print(f"{source_prefix} extraction: {year}", flush=True)
    result = pd.concat(frames, ignore_index=True)
    result.to_pickle(cache_path, compression="gzip")
    return result


def hfp_panel(grid: dict, rebuild: bool) -> pd.DataFrame:
    cache_path = CACHE / "hfp_5km_2001_2024.pkl.gz"
    if cache_path.exists() and not rebuild:
        return pd.read_pickle(cache_path, compression="gzip")
    base = grid_frame(grid)[["row", "col"]]
    frames = []
    for year in range(START_YEAR, END_YEAR + 1):
        zip_path = HFP_DIR / f"hfp{year}.zip"
        if not zip_path.exists():
            raise FileNotFoundError(zip_path)
        vsi_path = f"/vsizip/{zip_path.as_posix()}/hfp{year}.tif"
        with rasterio.open(vsi_path) as src:
            source_bounds = transform_bounds(grid["crs"], src.crs, *grid["bounds"])
            window = from_bounds(*source_bounds, src.transform).round_offsets().round_lengths()
            window = window.intersection(Window(0, 0, src.width, src.height))
            values = src.read(1, window=window).astype("float32")
            values[~np.isfinite(values)] = np.nan
            hfp = warp_array(
                values,
                src.window_transform(window),
                src.crs,
                grid,
            )
        part = base.copy()
        part["year"] = year
        part["hfp_raw"] = hfp.ravel()
        frames.append(part)
        print(f"HFP extraction: {year}", flush=True)
    result = pd.concat(frames, ignore_index=True)
    result.to_pickle(cache_path, compression="gzip")
    return result


def within_cell_z(panel: pd.DataFrame, source: str, output: str) -> None:
    grouped = panel.groupby("cell_id")[source]
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0, np.nan)
    panel[output] = (panel[source] - mean) / std


def within_cell_era_z(panel: pd.DataFrame, source: str, output: str) -> None:
    era = np.where(panel["year"] <= 2013, "AVHRR", "VIIRS")
    key = pd.Series(panel["cell_id"].astype(str) + "_" + era, index=panel.index)
    mean = panel[source].groupby(key).transform("mean")
    std = panel[source].groupby(key).transform("std").replace(0, np.nan)
    panel[output] = (panel[source] - mean) / std


def add_calendar_lags(panel: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    previous = panel[["cell_id", "year"] + features].copy()
    previous["year"] = previous["year"] + 1
    previous = previous.rename(columns={name: f"{name}_lag1" for name in features})
    return panel.merge(previous, on=["cell_id", "year"], how="left", validate="one_to_one")


def detrend_response(panel: pd.DataFrame) -> pd.Series:
    result = pd.Series(np.nan, index=panel.index, dtype="float64")
    for _, group in panel.groupby("cell_id"):
        valid = group[["year", "peak_agb"]].dropna()
        if len(valid) < MIN_CELL_YEARS:
            continue
        slope, intercept = np.polyfit(valid["year"], valid["peak_agb"], 1)
        result.loc[valid.index] = valid["peak_agb"] - (
            intercept + slope * valid["year"]
        )
    return result


def build_panel(grid: dict, rebuild: bool) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    cache_path = CACHE / "full_factor_panel_2001_2024.pkl.gz"
    manifest_path = FEATURE_MANIFEST
    if cache_path.exists() and manifest_path.exists() and not rebuild:
        return (
            pd.read_pickle(cache_path, compression="gzip"),
            json.loads(manifest_path.read_text(encoding="utf-8")),
        )

    base = pd.read_pickle(BASE_PANEL, compression="gzip")
    base = base[base["year"].between(START_YEAR, END_YEAR)].copy()
    peak = aggregate_peak_only(grid, rebuild)
    tc = stack_panel(TERRACLIMATE, TC_VARIABLES, grid, "terraclimate", rebuild)
    era = stack_panel(ERA5, ERA_VARIABLES, grid, "era5land", rebuild)
    hfp = hfp_panel(grid, rebuild)
    panel = base.drop(columns=["peat_agb"], errors="ignore").merge(
        peak,
        on=["year", "row", "col"],
        how="left",
        validate="one_to_one",
    )
    panel = panel.merge(tc, on=["year", "row", "col"], how="left", validate="one_to_one")
    panel = panel.merge(era, on=["year", "row", "col"], how="left", validate="one_to_one")
    panel = panel.merge(hfp, on=["year", "row", "col"], how="left", validate="one_to_one")
    panel = panel[
        (panel["strict_area_fraction"] >= MIN_MASK_FRACTION)
        & (panel["strict_valid_fraction"] >= MIN_VALID_FRACTION)
    ].copy()
    panel["cell_id"] = panel["row"].astype(str) + "_" + panel["col"].astype(str)
    panel["spatial_group"] = (
        (panel["x"] // SPATIAL_BLOCK_M).astype(int).astype(str)
        + "_"
        + (panel["y"] // SPATIAL_BLOCK_M).astype(int).astype(str)
    )
    radians = np.deg2rad(panel["aspect_deg"])
    panel["aspect_sin"] = np.sin(radians)
    panel["aspect_cos"] = np.cos(radians)

    cru_features = []
    for source, output in CRU_RAW.items():
        within_cell_z(panel, source, output)
        cru_features.append(output)
    tc_features = []
    for output in TC_VARIABLES.values():
        source = output.replace("_z", "_raw")
        within_cell_z(panel, source, output)
        tc_features.append(output)
    era_features = []
    for output in ERA_VARIABLES.values():
        source = output.replace("_z", "_raw")
        within_cell_z(panel, source, output)
        era_features.append(output)

    within_cell_era_z(panel, "lai", "lai_era_z")
    within_cell_era_z(panel, "fapar", "fapar_era_z")
    within_cell_z(panel, "fvc", "fvc_z")
    within_cell_z(panel, "lhgi", "lhgi_z")
    within_cell_z(panel, "hfp_raw", "hfp_z")
    vegetation_features = ["lai_era_z", "fapar_era_z"]
    fvc_features = ["fvc_z"]
    human_features = ["lhgi_z", "hfp_z"]

    dynamic = cru_features + tc_features + era_features + vegetation_features + fvc_features + human_features
    panel = add_calendar_lags(panel, dynamic)
    panel["agb_detrended_anomaly"] = detrend_response(panel)

    groups = {
        "CRU climate": cru_features + [f"{x}_lag1" for x in cru_features],
        "TerraClimate": tc_features + [f"{x}_lag1" for x in tc_features],
        "ERA5-Land": era_features + [f"{x}_lag1" for x in era_features],
        "Vegetation state": vegetation_features + [f"{x}_lag1" for x in vegetation_features],
        "FVC sensitivity": fvc_features + [f"{x}_lag1" for x in fvc_features],
        "Human activity": human_features + [f"{x}_lag1" for x in human_features],
        "Soil and terrain": STATIC_FEATURES,
    }
    panel.to_pickle(cache_path, compression="gzip")
    manifest_path.write_text(json.dumps(groups, indent=2), encoding="utf-8")
    return panel, groups


def reduced_climate_features(groups: dict[str, list[str]]) -> list[str]:
    """Retain one nonredundant predictor for each major climate process."""
    current = [
        "tc_gsp_z",                 # growing-season water supply
        "tc_gst_z",                 # growing-season thermal state
        "tc_vpd_z",                 # atmospheric moisture demand
        "era_soil_water_gs_z",      # near-surface soil water availability
    ]
    available = set(
        groups["CRU climate"] + groups["TerraClimate"] + groups["ERA5-Land"]
    )
    selected = current + [f"{feature}_lag1" for feature in current]
    missing = [feature for feature in selected if feature not in available]
    if missing:
        raise KeyError(f"Reduced climate features missing from panel: {missing}")
    return selected


def model_sets(groups: dict[str, list[str]]) -> dict[str, list[str]]:
    climate_reduced = reduced_climate_features(groups)
    vegetation = groups["Vegetation state"]
    human = groups["Human activity"]
    static = groups["Soil and terrain"]
    return {
        "Human + controls": human + static,
        "Reduced climate": climate_reduced + static,
        "Reduced climate + vegetation": climate_reduced + vegetation + static,
        "Reduced climate + human": climate_reduced + human + static,
        "Reduced full": climate_reduced + vegetation + human + static,
        "Reduced full + FVC sensitivity": (
            climate_reduced + vegetation + human + groups["FVC sensitivity"] + static
        ),
    }


def interpretation_groups(groups: dict[str, list[str]]) -> dict[str, list[str]]:
    return {
        "Climate": reduced_climate_features(groups),
        "Vegetation state": groups["Vegetation state"],
        "FVC sensitivity": groups["FVC sensitivity"],
        "Human activity": groups["Human activity"],
        "Soil and terrain": groups["Soil and terrain"],
    }


def regression_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict:
    mask = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[mask]
    predicted = predicted[mask]
    return {
        "n": int(mask.sum()),
        "pearson_r": float(np.corrcoef(observed, predicted)[0, 1]) if mask.sum() > 2 else np.nan,
        "r2": float(r2_score(observed, predicted)),
        "rmse_g_m2": float(np.sqrt(mean_squared_error(observed, predicted))),
        "mae_g_m2": float(mean_absolute_error(observed, predicted)),
        "bias_g_m2": float(np.mean(predicted - observed)),
    }


def make_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=350,
        max_depth=3,
        learning_rate=0.035,
        min_child_weight=8,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=0.10,
        reg_lambda=2.0,
        objective="reg:squarederror",
        random_state=SEED,
        n_jobs=-1,
        tree_method="hist",
    )


def temporal_groups(years: pd.Series) -> pd.Series:
    chunks = np.array_split(np.array(sorted(years.unique())), 5)
    mapping = {int(year): i for i, chunk in enumerate(chunks) for year in chunk}
    return years.map(mapping).astype(int)


def run_oof(
    frame: pd.DataFrame,
    features: list[str],
    model_name: str,
    validation: str,
    calculate_shap: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    required = features + [
        "agb_detrended_anomaly", "spatial_group", "year", "cell_id", "x", "y"
    ]
    work = frame[required].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if validation == "spatial block":
        splitter = GroupKFold(n_splits=5)
        groups = work["spatial_group"].to_numpy()
    else:
        splitter = LeaveOneGroupOut()
        groups = temporal_groups(work["year"]).to_numpy()
    splits = splitter.split(work, groups=groups)

    predictions = []
    metrics = []
    shap_parts = []
    for fold, (train_index, test_index) in enumerate(splits, 1):
        train = work.iloc[train_index]
        test = work.iloc[test_index]
        model = make_model()
        model.fit(train[features], train["agb_detrended_anomaly"])
        predicted = model.predict(test[features])
        pred = test[["year", "cell_id", "x", "y"]].copy()
        pred["observed"] = test["agb_detrended_anomaly"].to_numpy()
        pred["predicted"] = predicted
        pred["fold"] = fold
        pred["model"] = model_name
        pred["validation"] = validation
        predictions.append(pred)
        metric = regression_metrics(pred["observed"].to_numpy(), predicted)
        metric.update({
            "model": model_name,
            "validation": validation,
            "fold": str(fold),
            "test_year_min": int(test["year"].min()),
            "test_year_max": int(test["year"].max()),
            "n_features": len(features),
        })
        metrics.append(metric)
        if calculate_shap:
            values = shap.TreeExplainer(model).shap_values(test[features])
            part = test[["year", "cell_id"]].copy().reset_index(drop=True)
            part["fold"] = fold
            part["model"] = model_name
            part["validation"] = validation
            for index, feature in enumerate(features):
                part[f"value__{feature}"] = test[feature].to_numpy()
                part[f"shap__{feature}"] = values[:, index]
            shap_parts.append(part)
        print(f"OOF {model_name} {validation} fold {fold}", flush=True)

    prediction_frame = pd.concat(predictions, ignore_index=True)
    overall = regression_metrics(
        prediction_frame["observed"].to_numpy(),
        prediction_frame["predicted"].to_numpy(),
    )
    overall.update({
        "model": model_name,
        "validation": validation,
        "fold": "overall_oof",
        "test_year_min": int(work["year"].min()),
        "test_year_max": int(work["year"].max()),
        "n_features": len(features),
    })
    metric_frame = pd.concat([pd.DataFrame(metrics), pd.DataFrame([overall])], ignore_index=True)
    shap_frame = pd.concat(shap_parts, ignore_index=True) if shap_parts else None
    return prediction_frame, metric_frame, shap_frame


def feature_group_lookup(groups: dict[str, list[str]]) -> dict[str, str]:
    return {feature: group for group, features in groups.items() for feature in features}


def summarize_shap(shap_frame: pd.DataFrame, groups: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    lookup = feature_group_lookup(groups)
    rows = []
    for column in shap_frame.columns:
        if not column.startswith("shap__"):
            continue
        if not shap_frame[column].notna().any():
            continue
        feature = column.replace("shap__", "", 1)
        rows.append({
            "model": shap_frame["model"].iloc[0],
            "validation": shap_frame["validation"].iloc[0],
            "feature": feature,
            "feature_label": label_feature(feature),
            "factor_group": lookup.get(feature, "Other"),
            "mean_abs_shap": float(shap_frame[column].abs().mean()),
            "mean_shap": float(shap_frame[column].mean()),
            "n": int(shap_frame[column].notna().sum()),
        })
    importance = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)
    importance["rank"] = np.arange(1, len(importance) + 1)
    grouped = importance.groupby("factor_group", as_index=False).agg(
        mean_abs_shap=("mean_abs_shap", "sum"),
        n_features=("feature", "size"),
    )
    grouped["share_pct"] = 100 * grouped["mean_abs_shap"] / grouped["mean_abs_shap"].sum()
    grouped["mean_abs_shap_per_feature"] = (
        grouped["mean_abs_shap"] / grouped["n_features"]
    )
    grouped = grouped.sort_values("mean_abs_shap", ascending=False)
    return importance, grouped


def label_feature(feature: str) -> str:
    lagged = feature.endswith("_lag1")
    base = feature[:-5] if lagged else feature
    if base in FEATURE_LABELS:
        label = FEATURE_LABELS[base]
    elif base.startswith("tc_"):
        label = "TerraClimate " + base[3:].replace("_z", "").replace("_", " ")
    elif base.startswith("era_"):
        label = "ERA5-Land " + base[4:].replace("_z", "").replace("_", " ")
    else:
        label = base.replace("_z", "").replace("_", " ")
    return label + (", lag 1" if lagged else "")


def binned_dependence(shap_frame: pd.DataFrame, importance: pd.DataFrame, top_n: int = 4) -> pd.DataFrame:
    frames = []
    for feature in importance.head(top_n)["feature"]:
        data = pd.DataFrame({
            "value": shap_frame[f"value__{feature}"],
            "shap": shap_frame[f"shap__{feature}"],
        }).dropna()
        data["bin"] = pd.qcut(data["value"], q=min(18, data["value"].nunique()), duplicates="drop")
        summary = data.groupby("bin", observed=True).agg(
            feature_value=("value", "median"),
            shap_median=("shap", "median"),
            shap_p25=("shap", lambda x: np.quantile(x, 0.25)),
            shap_p75=("shap", lambda x: np.quantile(x, 0.75)),
            n=("shap", "size"),
        ).reset_index(drop=True)
        summary.insert(0, "feature_label", label_feature(feature))
        summary.insert(0, "feature", feature)
        frames.append(summary)
    return pd.concat(frames, ignore_index=True)


def figure_style() -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.10, 1.03, f"({label})", transform=ax.transAxes,
        ha="left", va="bottom", fontsize=9.5, fontweight="bold",
    )


def add_panel_subtitle(ax: plt.Axes, text: str, y: float = -0.24) -> None:
    ax.text(
        0.5, y, text, transform=ax.transAxes,
        ha="center", va="top", fontsize=9,
    )


def style_boxed_axis(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("0.25")
        spine.set_linewidth(0.8)
    ax.tick_params(direction="in", top=True, right=True)


def save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            FIGURES / f"{stem}.{suffix}",
            dpi=400 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    fig.savefig(
        FIGURES / f"{stem}.tif",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def save_panel_crop(
    fig: plt.Figure,
    axes: list[plt.Axes],
    stem: str,
    pad_fraction: float = 0.04,
) -> None:
    """Export a panel crop from the same figure used for the composite."""
    PANELS.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = Bbox.union([axis.get_tightbbox(renderer) for axis in axes])
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    bbox = bbox.expanded(1 + pad_fraction, 1 + pad_fraction)
    for suffix, kwargs in (
        ("png", {"dpi": 450}),
        ("tif", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
        ("pdf", {}),
        ("svg", {}),
    ):
        fig.savefig(
            PANELS / f"{stem}.{suffix}",
            bbox_inches=bbox,
            facecolor="white",
            **kwargs,
        )


def plot_model_comparison(metrics: pd.DataFrame, group_imp: pd.DataFrame, importance: pd.DataFrame) -> None:
    overall = metrics[metrics["fold"] == "overall_oof"].copy()
    order = list(model_sets(json.loads(FEATURE_MANIFEST.read_text())) .keys())
    colors = {"spatial block": "#2F6FB0", "temporal block": "#D65F4A"}
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.6))
    fig.subplots_adjust(left=0.10, right=0.985, top=0.97, bottom=0.10, hspace=0.68, wspace=0.66)
    ax = axes[0, 0]
    width = 0.36
    x = np.arange(len(order))
    for offset, validation in ((-width / 2, "spatial block"), (width / 2, "temporal block")):
        values = overall.set_index(["model", "validation"]).reindex(
            pd.MultiIndex.from_product([order, [validation]])
        )["r2"].to_numpy()
        ax.bar(x + offset, values, width, color=colors[validation], label=validation.capitalize())
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_ylabel("Out-of-fold predictive $R^2$")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=28, ha="right")
    ax.legend(frameon=False, fontsize=8)
    add_panel_label(ax, "a")
    add_panel_subtitle(ax, "Blocked predictive performance", y=-0.34)

    ax = axes[0, 1]
    for offset, validation in ((-width / 2, "spatial block"), (width / 2, "temporal block")):
        values = overall.set_index(["model", "validation"]).reindex(
            pd.MultiIndex.from_product([order, [validation]])
        )["rmse_g_m2"].to_numpy()
        ax.bar(x + offset, values, width, color=colors[validation])
    ax.set_ylabel("Out-of-fold RMSE (g m$^{-2}$)")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=28, ha="right")
    add_panel_label(ax, "b")
    add_panel_subtitle(ax, "Blocked prediction error", y=-0.34)

    ax = axes[1, 0]
    ordered_groups = group_imp.sort_values("share_pct")
    palette = {
        "CRU climate": "#4C78A8",
        "TerraClimate": "#72B7B2",
        "ERA5-Land": "#59A14F",
        "Climate": "#4C78A8",
        "Vegetation state": "#F2CF5B",
        "FVC sensitivity": "#EDC948",
        "Human activity": "#B279A2",
        "Soil and terrain": "#9D9D9D",
    }
    ax.barh(
        ordered_groups["factor_group"],
        ordered_groups["share_pct"],
        color=[palette.get(x, "#777777") for x in ordered_groups["factor_group"]],
    )
    ax.set_xlabel("Grouped held-out SHAP contribution (%)")
    add_panel_label(ax, "c")
    add_panel_subtitle(ax, "Grouped SHAP contribution")

    ax = axes[1, 1]
    top = importance.head(15).sort_values("mean_abs_shap")
    ax.barh(
        top["feature_label"],
        top["mean_abs_shap"],
        color=[palette.get(x, "#777777") for x in top["factor_group"]],
    )
    ax.set_xlabel("Mean |held-out SHAP| (g m$^{-2}$)")
    ax.tick_params(axis="y", labelsize=7.3)
    add_panel_label(ax, "d")
    add_panel_subtitle(ax, "Leading predictor contributions")
    for axis in axes.ravel():
        style_boxed_axis(axis)
    panel_names = (
        "Fig_10a_blocked_validation_r2",
        "Fig_10b_blocked_validation_rmse",
        "Fig_10c_grouped_shap_contribution",
        "Fig_10d_feature_shap_importance",
    )
    for axis, stem in zip(axes.ravel(), panel_names):
        save_panel_crop(fig, [axis], stem)
    save_figure(fig, "Fig_10_XGBoost_validation_and_SHAP_importance")


def plot_beeswarm_and_dependence(
    shap_frame: pd.DataFrame,
    importance: pd.DataFrame,
    dependence: pd.DataFrame,
) -> None:
    compact_response_labels = {
        "tc_wb_z": "Climatic water balance\n(standardized)",
        "tc_pet_z": "Potential evapotranspiration\n(standardized)",
        "era_tmax_gs_z": "GS maximum temperature\n(standardized)",
    }
    top_features = importance.head(14)["feature"].tolist()
    fig = plt.figure(figsize=(10.8, 9.2))
    fig.subplots_adjust(left=0.12, right=0.95, top=0.97, bottom=0.10, hspace=0.60, wspace=0.38)
    grid = fig.add_gridspec(2, 3, height_ratios=[1.35, 1])
    ax = fig.add_subplot(grid[0, :])
    rng = np.random.default_rng(SEED)
    cmap = plt.get_cmap("coolwarm")
    for y_pos, feature in enumerate(reversed(top_features)):
        values = shap_frame[f"value__{feature}"].to_numpy()
        shap_values = shap_frame[f"shap__{feature}"].to_numpy()
        valid = np.isfinite(values) & np.isfinite(shap_values)
        values, shap_values = values[valid], shap_values[valid]
        if len(values) > 2500:
            chosen = rng.choice(len(values), 2500, replace=False)
            values, shap_values = values[chosen], shap_values[chosen]
        low, high = np.nanpercentile(values, [5, 95])
        norm_values = np.clip((values - low) / (high - low + 1e-12), 0, 1)
        jitter = rng.normal(0, 0.10, len(values))
        ax.scatter(
            shap_values,
            np.full(len(values), y_pos) + jitter,
            c=cmap(norm_values),
            s=4,
            alpha=0.50,
            linewidths=0,
        )
    ax.axvline(0, color="#666666", linewidth=0.8)
    ax.set_yticks(np.arange(len(top_features)))
    ax.set_yticklabels([label_feature(x) for x in reversed(top_features)])
    ax.set_xlabel("Held-out SHAP value (impact on AGB anomaly, g m$^{-2}$)")
    add_panel_label(ax, "a")
    add_panel_subtitle(ax, "Held-out SHAP distribution", y=-0.13)
    style_boxed_axis(ax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(0, 1))
    colorbar = fig.colorbar(sm, ax=ax, fraction=0.018, pad=0.015)
    colorbar.set_ticks([0, 1])
    colorbar.set_ticklabels(["Low", "High"])
    colorbar.set_label("Feature value")

    response_axes = []
    response_features = importance.head(3)["feature"].tolist()
    for index, feature in enumerate(response_features):
        axis = fig.add_subplot(grid[1, index])
        response_axes.append(axis)
        part = dependence[dependence["feature"] == feature].sort_values("feature_value")
        axis.fill_between(
            part["feature_value"].to_numpy(),
            part["shap_p25"].to_numpy(),
            part["shap_p75"].to_numpy(),
            color="#8FB7D8",
            alpha=0.35,
            linewidth=0,
        )
        axis.plot(part["feature_value"], part["shap_median"], color="#2166AC", linewidth=1.8)
        axis.axhline(0, color="#666666", linewidth=0.8)
        axis.set_xlabel(
            compact_response_labels.get(
                feature,
                label_feature(feature) + "\n(standardized)",
            )
        )
        axis.set_ylabel("Median held-out SHAP\n(g m$^{-2}$)")
        add_panel_label(axis, chr(ord("b") + index))
        add_panel_subtitle(axis, label_feature(feature), y=-0.34)
        style_boxed_axis(axis)
    save_panel_crop(fig, [ax, colorbar.ax], "Fig_11a_heldout_SHAP_beeswarm")
    for index, (axis, feature) in enumerate(zip(response_axes, response_features)):
        safe_feature = feature.replace("_", "-")
        save_panel_crop(
            fig,
            [axis],
            f"Fig_11{chr(ord('b') + index)}_{safe_feature}_SHAP_response",
        )
    save_figure(fig, "Fig_11_heldout_SHAP_responses")


def markdown_table(frame: pd.DataFrame, digits: int = 3) -> str:
    def fmt(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.{digits}f}" if np.isfinite(value) else "NA"
        return str(value)
    columns = [str(x) for x in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(fmt(x) for x in row) + " |")
    return "\n".join(lines)


def write_report(
    panel: pd.DataFrame,
    model_features: dict[str, list[str]],
    metrics: pd.DataFrame,
    importance: pd.DataFrame,
    group_imp: pd.DataFrame,
) -> None:
    overall = metrics[metrics["fold"] == "overall_oof"].copy()
    model_table = overall[[
        "model", "validation", "n", "n_features", "pearson_r", "r2",
        "rmse_g_m2", "mae_g_m2", "bias_g_m2",
    ]]
    top_table = importance.head(15)[[
        "rank", "feature_label", "factor_group", "mean_abs_shap",
    ]]
    group_table = group_imp[["factor_group", "mean_abs_shap", "share_pct"]]
    full = overall.set_index(["model", "validation"])
    spatial = full.loc[("Reduced full", "spatial block")]
    temporal = full.loc[("Reduced full", "temporal block")]
    fvc_spatial = full.loc[("Reduced full + FVC sensitivity", "spatial block")]
    top_names = ", ".join(importance.head(5)["feature_label"].tolist())
    report = f"""# Section 3.4 reduced-factor XGBoost-SHAP experiment

## Analysis contract

- Spatial domain: strict Zoige peatland mask, approximately 4673.4 km2.
- Time window: {START_YEAR}-{END_YEAR}; no 2025 driver values were fabricated.
- Response: within-cell linearly detrended Peak-only AGB anomaly.
- Spatial support: 90 m AGB was masked first and area-averaged to a 5 km driver grid.
- Validation: five 25 km spatial folds and five contiguous temporal folds.
- Interpretation: SHAP was calculated only for held-out spatial-fold rows.
- Climate predictors were reduced a priori to growing-season precipitation,
  temperature, vapour-pressure deficit and surface soil water, each represented
  by current-year and one-year-lagged values.
- FVC was isolated as a sensitivity extension because it is derived from the Landsat input used by the AGB inversion.

Eligible panel: {len(panel):,} rows from {panel['cell_id'].nunique()} grid cells.

## Main numerical results

The reduced full model produced spatially blocked r={spatial['pearson_r']:.3f}, R2={spatial['r2']:.3f}, and RMSE={spatial['rmse_g_m2']:.2f} g m-2. Its temporally blocked performance was r={temporal['pearson_r']:.3f}, R2={temporal['r2']:.3f}, and RMSE={temporal['rmse_g_m2']:.2f} g m-2. Adding FVC changed the spatially blocked R2 to {fvc_spatial['r2']:.3f}. The leading held-out predictors were {top_names}.

These results quantify predictive associations. They do not establish causality, and the temporally blocked result determines whether the model can support cross-period generalization claims.

## Model comparison

{markdown_table(model_table)}

## Grouped held-out SHAP contribution

{markdown_table(group_table)}

## Leading held-out SHAP features

{markdown_table(top_table, digits=4)}

## Feature-set sizes

{markdown_table(pd.DataFrame([{'model': key, 'n_features': len(value)} for key, value in model_features.items()]))}
"""
    (OUT_ROOT / "report_section_3_4_experiments.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-panel", action="store_true")
    parser.add_argument("--panel-only", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    figure_style()
    grid = load_grid()
    panel, groups = build_panel(grid, args.rebuild_panel)
    feature_sets = model_sets(groups)
    # Freeze the original 85-feature complete-case sample so reduced and full
    # experiments remain directly comparable row for row.
    common_features = list(dict.fromkeys(
        feature for features in groups.values() for feature in features
    ))
    common_required = common_features + ["agb_detrended_anomaly"]
    model_panel = panel.replace([np.inf, -np.inf], np.nan).dropna(
        subset=common_required
    ).copy()
    pd.DataFrame([
        {"model": model, "n_features": len(features), "features": ";".join(features)}
        for model, features in feature_sets.items()
    ]).to_csv(OUTPUTS / "model_feature_sets.csv", index=False, encoding="utf-8-sig")
    if args.panel_only:
        print(f"Panel complete: {len(panel):,} rows", flush=True)
        return

    predictions_all = []
    metrics_all = []
    shap_all = []
    for model_name, features in feature_sets.items():
        for validation in ("spatial block", "temporal block"):
            calculate_shap = (
                validation == "spatial block"
                and model_name in {"Reduced full", "Reduced full + FVC sensitivity"}
            )
            predictions, metrics, shap_frame = run_oof(
                model_panel, features, model_name, validation, calculate_shap
            )
            predictions_all.append(predictions)
            metrics_all.append(metrics)
            if shap_frame is not None:
                shap_all.append(shap_frame)

    predictions = pd.concat(predictions_all, ignore_index=True)
    metrics = pd.concat(metrics_all, ignore_index=True)
    shap_values = pd.concat(shap_all, ignore_index=True)
    predictions.to_csv(OUTPUTS / "oof_predictions.csv.gz", index=False, compression="gzip")
    metrics.to_csv(OUTPUTS / "blocked_validation_metrics.csv", index=False, encoding="utf-8-sig")
    shap_values.to_pickle(OUTPUTS / "heldout_shap_values.pkl.gz", compression="gzip")

    primary_shap = shap_values[shap_values["model"] == "Reduced full"].copy()
    shap_groups = interpretation_groups(groups)
    importance, group_imp = summarize_shap(primary_shap, shap_groups)
    dependence = binned_dependence(primary_shap, importance, top_n=4)
    importance.to_csv(OUTPUTS / "heldout_shap_importance_reduced_full.csv", index=False, encoding="utf-8-sig")
    group_imp.to_csv(OUTPUTS / "heldout_shap_group_importance.csv", index=False, encoding="utf-8-sig")
    dependence.to_csv(OUTPUTS / "heldout_shap_dependence_binned.csv", index=False, encoding="utf-8-sig")

    fvc_shap = shap_values[shap_values["model"] == "Reduced full + FVC sensitivity"].copy()
    fvc_importance, fvc_groups = summarize_shap(fvc_shap, shap_groups)
    fvc_importance.to_csv(OUTPUTS / "heldout_shap_importance_fvc_sensitivity.csv", index=False, encoding="utf-8-sig")
    fvc_groups.to_csv(OUTPUTS / "heldout_shap_group_importance_fvc_sensitivity.csv", index=False, encoding="utf-8-sig")

    plot_model_comparison(metrics, group_imp, importance)
    plot_beeswarm_and_dependence(primary_shap, importance, dependence)
    write_report(model_panel, feature_sets, metrics, importance, group_imp)
    print(f"Section 3.4 analysis complete: {OUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
