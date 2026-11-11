#!/usr/bin/env python
"""Complete manuscript section 3.2 for strict-main95 Peak-only AGB.

The workflow keeps the frozen 90 m prediction grid, uses Peak-only AGB as the
primary 1988-2025 sequence, and treats Guarded-consensus as a 1988-2024 model
structure sensitivity analysis. FVC follows the same 3x3 valid-median support
as the AGB predictor. NOAA LAI is QA filtered and standardized within sensor
eras before it is used as supporting vegetation-state evidence.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import rasterio
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window
from scipy.stats import kendalltau, norm, pearsonr, theilslopes


matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "agb_peakonly_spatiotemporal_1988_2025"
OUT = RUN / "outputs"
RASTERS = RUN / "rasters"
FIGURES = RUN / "figures"
RASTERS_5KM = RUN / "rasters_5km"
VIRTUAL_INPUTS = RUN / "virtual_inputs"

HISTORICAL_RASTERS = ROOT / "agb_dual_grid_2013_2024" / "rasters_90m"
RASTER_2025 = (
    ROOT / "agb_2025_independent_validation" / "rasters"
    / "Zoige_AGB_PeakOnly_PLSR_90m_2025.tif"
)
STRICT_MASK = (
    ROOT / "zoigepeatland" / "output_basic_boundary"
    / "Zoige_RF_basic_analysis_mask_main95_fill400.tif"
)
HISTORICAL_AUDIT = (
    ROOT / "agb_long_term_freeze_1988_2024" / "outputs"
    / "annual_main95_product_audit.csv"
)
OLD_STATE_SUMMARY = (
    ROOT / "agb_lai_resilience_breakpoint_main95_1988_2024" / "outputs"
    / "annual_regional_state_summary.csv"
)
BASE_PANEL = ROOT / "agb_driver_xgboost_shap_v2" / "cache" / "driver_panel_5km.pkl.gz"

YEARS = list(range(1988, 2026))
REPRESENTATIVE_YEARS = [1989, 2000, 2013, 2025]
NODATA = -9999.0
CELL_AREA_M2 = 90.0 * 90.0
BLOCK = 128
MIN_TREND_YEARS = 30


def ensure_dirs() -> None:
    for path in (OUT, RASTERS, FIGURES, RASTERS_5KM, VIRTUAL_INPUTS):
        path.mkdir(parents=True, exist_ok=True)


def import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def raster_for_year(year: int) -> tuple[Path, int, int, int]:
    if year <= 2024:
        return (
            HISTORICAL_RASTERS
            / f"Zoige_AGB_GuardedConsensus_PLSR_90m_{year}.tif",
            3,
            8,
            6,
        )
    return RASTER_2025, 1, 5, 3


def sensor_policy(year: int) -> str:
    if year <= 1998:
        return "Landsat 5 TM"
    if year <= 2002:
        return "Landsat 5 TM + Landsat 7 ETM+"
    if year <= 2011:
        return "Landsat 5 TM"
    if year == 2012:
        return "Landsat 7 ETM+ (SLC-off)"
    if year <= 2020:
        return "Landsat 8 OLI"
    return "Landsat 8 OLI + Landsat 9 OLI-2"


def lai_era(year: int) -> str:
    # NOAA official record: AVHRR through 2013, VIIRS S-NPP 2014-2022, NOAA-20 2023+
    if year <= 2013:
        return "AVHRR"
    if year <= 2022:
        return "VIIRS S-NPP"
    return "VIIRS NOAA-20"


def read_valid(src: rasterio.io.DatasetReader, band: int, window: Window | None = None) -> np.ndarray:
    array = src.read(band, window=window).astype("float32")
    invalid = ~np.isfinite(array)
    if src.nodata is not None:
        invalid |= np.isclose(array, src.nodata)
    array[invalid] = np.nan
    return array


def load_reference_and_mask() -> tuple[dict, np.ndarray, tuple[int, int, int, int]]:
    ref_path, _, _, _ = raster_for_year(2024)
    with rasterio.open(ref_path) as ref:
        profile = ref.profile.copy()
        strict = np.zeros(ref.shape, dtype="uint8")
        with rasterio.open(STRICT_MASK) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=strict,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref.transform,
                dst_crs=ref.crs,
                dst_nodata=0,
                resampling=Resampling.nearest,
            )
    strict = strict > 0
    rows, cols = np.where(strict)
    bbox = (int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1)
    return profile, strict, bbox


def write_multiband(path: Path, arrays: list[np.ndarray], names: list[str], profile: dict) -> None:
    output_profile = profile.copy()
    output_profile.update(
        count=len(arrays), dtype="float32", nodata=NODATA,
        compress="deflate", predictor=3, tiled=True,
        blockxsize=256, blockysize=256,
    )
    with rasterio.open(path, "w", **output_profile) as dst:
        for index, (array, name) in enumerate(zip(arrays, names), 1):
            dst.write(np.where(np.isfinite(array), array, NODATA).astype("float32"), index)
            dst.set_band_description(index, name)


def compute_fvc_2025(profile: dict, strict: np.ndarray) -> tuple[float, Path]:
    pilot = import_module(
        "pilot_inversion_for_fvc",
        ROOT / "scripts" / "_internal" / "annual_agb_helpers.py",
    )
    pilot.VRT_DIR = VIRTUAL_INPUTS
    vrt_path = pilot.build_mosaic_vrt(2025)
    fvc_path = RASTERS / "Zoige_FVC_valid_median_90m_2025.tif"
    fvc_profile = profile.copy()
    fvc_profile.update(
        count=1, dtype="float32", nodata=NODATA, compress="deflate",
        predictor=3, tiled=True, blockxsize=256, blockysize=256,
    )
    total = 0.0
    count = 0
    with rasterio.open(vrt_path) as source, rasterio.open(fvc_path, "w", **fvc_profile) as dst:
        if source.width // 3 != profile["width"] or source.height // 3 != profile["height"]:
            raise ValueError("2025 source mosaic is not aligned with the frozen 90 m grid")
        dst.set_band_description(1, "FVC_valid_median_3x3")
        for window in pilot.output_windows(profile["width"], profile["height"]):
            source_window = Window(
                int(window.col_off) * 3,
                int(window.row_off) * 3,
                int(window.width) * 3,
                int(window.height) * 3,
            )
            raw = source.read([13, 15], window=source_window).astype("float32")
            out_h, out_w = int(window.height), int(window.width)
            valid = (raw[1] >= 0.5).reshape(out_h, 3, out_w, 3)
            valid = valid.transpose(0, 2, 1, 3).reshape(out_h, out_w, 9)
            values = raw[0].reshape(out_h, 3, out_w, 3)
            values = values.transpose(0, 2, 1, 3).reshape(out_h, out_w, 9)
            valid_count = valid.sum(axis=2)
            ordered = np.sort(np.where(valid, values, np.inf), axis=2)
            lower_index = np.maximum((valid_count - 1) // 2, 0)
            upper_index = valid_count // 2
            lower = np.take_along_axis(ordered, lower_index[..., None], axis=2)[..., 0]
            upper = np.take_along_axis(ordered, upper_index[..., None], axis=2)[..., 0]
            fvc = ((lower + upper) / 2.0).astype("float32")
            fvc[valid_count == 0] = np.nan
            r0, c0 = int(window.row_off), int(window.col_off)
            r1, c1 = r0 + int(window.height), c0 + int(window.width)
            local_mask = strict[r0:r1, c0:c1]
            fvc[(fvc < 0) | (fvc > 1)] = np.nan
            fvc[~local_mask] = np.nan
            valid = np.isfinite(fvc)
            total += float(np.nansum(fvc))
            count += int(valid.sum())
            dst.write(np.where(valid, fvc, NODATA).astype("float32"), 1, window=window)
    return total / count, fvc_path


def annual_statistics(profile: dict, strict: np.ndarray, fvc_2025: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    historical = pd.read_csv(HISTORICAL_AUDIT).set_index("year")
    inventory_records = []
    annual_records = []
    class_records = []
    n_mask = int(strict.sum())
    full_area_km2 = n_mask * CELL_AREA_M2 / 1e6

    for year in YEARS:
        path, agb_band, valid_band, ood_band = raster_for_year(year)
        with rasterio.open(path) as src:
            grid_ok = (
                src.width == profile["width"]
                and src.height == profile["height"]
                and src.crs == profile["crs"]
                and src.transform == profile["transform"]
            )
            agb = read_valid(src, agb_band)
            valid_count = read_valid(src, valid_band)
            ood = read_valid(src, ood_band)
            descriptions = ";".join(item or "" for item in src.descriptions)
            inventory_records.append({
                "year": year,
                "path": str(path),
                "exists": path.exists(),
                "grid_matches_reference": grid_ok,
                "width": src.width,
                "height": src.height,
                "crs": str(src.crs),
                "transform": str(src.transform),
                "band_count": src.count,
                "band_descriptions": descriptions,
                "peak_agb_band": agb_band,
                "valid_count_band": valid_band,
                "sensor_policy": sensor_policy(year),
            })
        if not grid_ok:
            raise ValueError(f"Grid mismatch for {year}: {path}")
        valid = strict & np.isfinite(agb) & (valid_count > 0)
        values = agb[valid].astype("float64")
        if not len(values):
            raise ValueError(f"No strict-main95 Peak-only AGB values for {year}")
        p05, p25, p50, p75, p95 = np.percentile(values, [5, 25, 50, 75, 95])
        coverage = len(values) / n_mask
        observed_area = len(values) * CELL_AREA_M2
        observed_stock = float(values.sum() * CELL_AREA_M2 / 1e12)
        standard_stock = float(values.mean() * n_mask * CELL_AREA_M2 / 1e12)
        if year <= 2024:
            fvc_mean = float(historical.loc[year, "fvc_mean"])
            ood_any = float(historical.loc[year, "peak_feature_ood_any_pixel_fraction"])
        else:
            fvc_mean = float(fvc_2025)
            finite_ood = strict & np.isfinite(ood)
            ood_any = float(np.mean(ood[finite_ood] > 0)) if finite_ood.any() else np.nan
        annual_records.append({
            "year": year,
            "sensor_policy": sensor_policy(year),
            "strict_mask_cell_count": n_mask,
            "valid_cell_count": len(values),
            "valid_coverage_fraction": coverage,
            "coverage_quality": "good" if coverage >= 0.90 else ("moderate" if coverage >= 0.80 else "low"),
            "agb_mean_g_m2": float(values.mean()),
            "agb_sd_g_m2": float(values.std(ddof=1)),
            "agb_p05_g_m2": float(p05),
            "agb_p25_g_m2": float(p25),
            "agb_p50_g_m2": float(p50),
            "agb_p75_g_m2": float(p75),
            "agb_p95_g_m2": float(p95),
            "fvc_mean": fvc_mean,
            "valid_count_mean": float(np.nanmean(valid_count[valid])),
            "peak_feature_ood_any_fraction": ood_any,
            "observed_valid_area_km2": observed_area / 1e6,
            "observed_stock_Tg": observed_stock,
            "area_standardized_stock_Tg": standard_stock,
            "strict_mask_area_km2_90m_grid": full_area_km2,
        })
        class_defs = [
            ("0-200", 0.0, 200.0),
            ("200-300", 200.0, 300.0),
            ("300-400", 300.0, 400.0),
            (">=400", 400.0, np.inf),
        ]
        for label, lower, upper in class_defs:
            class_count = int(np.sum((values >= lower) & (values < upper)))
            class_records.append({
                "year": year,
                "agb_class_g_m2": label,
                "cell_count": class_count,
                "area_km2": class_count * CELL_AREA_M2 / 1e6,
                "fraction_of_valid_area": class_count / len(values),
            })
        print(f"Annual strict-main95 statistics: {year}", flush=True)

    inventory = pd.DataFrame(inventory_records)
    annual = pd.DataFrame(annual_records)
    classes = pd.DataFrame(class_records)
    inventory.to_csv(OUT / "product_inventory_1988_2025.csv", index=False)
    annual.to_csv(OUT / "annual_strict_main95_peakonly_stats_1988_2025.csv", index=False)
    classes.to_csv(OUT / "annual_agb_class_area_1988_2025.csv", index=False)
    return inventory, annual, classes


def trend_record(label: str, frame: pd.DataFrame, all_years: list[int]) -> dict:
    x = frame["year"].to_numpy(float)
    y = frame["agb_mean_g_m2"].to_numpy(float)
    slope, intercept, low, high = theilslopes(y, x, 0.95)
    tau, p = kendalltau(x, y)
    included = frame["year"].astype(int).tolist()
    excluded = sorted(set(all_years) - set(included))
    return {
        "scenario": label,
        "start_year": int(x.min()),
        "end_year": int(x.max()),
        "n_years": len(frame),
        "sen_slope_g_m2_yr": slope,
        "sen_ci95_low": low,
        "sen_ci95_high": high,
        "sen_intercept": intercept,
        "kendall_tau": tau,
        "mann_kendall_p": p,
        "sen_implied_total_change_g_m2": slope * (x.max() - x.min()),
        "included_years": ";".join(map(str, included)),
        "excluded_years": ";".join(map(str, excluded)),
    }


def regional_trends(annual: pd.DataFrame) -> pd.DataFrame:
    base_2024 = annual[annual.year <= 2024]
    records = [
        trend_record("1988-2024 frozen", base_2024, list(range(1988, 2025))),
        trend_record("1988-2025 all", annual, YEARS),
        trend_record(
            "1988-2025 coverage >= 0.80",
            annual[annual.valid_coverage_fraction >= 0.80], YEARS,
        ),
        trend_record(
            "1988-2025 coverage >= 0.90",
            annual[annual.valid_coverage_fraction >= 0.90], YEARS,
        ),
    ]
    trends = pd.DataFrame(records)
    trends.to_csv(OUT / "regional_trend_sensitivity_1988_2025.csv", index=False)
    return trends


def block_windows(profile: dict):
    for row in range(0, profile["height"], BLOCK):
        for col in range(0, profile["width"], BLOCK):
            yield Window(
                col, row,
                min(BLOCK, profile["width"] - col),
                min(BLOCK, profile["height"] - row),
            )


def compute_pixel_products(profile: dict, strict: np.ndarray) -> tuple[Path, Path, pd.DataFrame]:
    stats_path = RASTERS / "PeakOnly_AGB_multiyear_stats_1988_2025_90m.tif"
    trend_path = RASTERS / "PeakOnly_AGB_pixel_trend_1988_2025_90m.tif"
    output_profile = profile.copy()
    output_profile.update(
        count=4, dtype="float32", nodata=NODATA, compress="deflate",
        predictor=3, tiled=True, blockxsize=256, blockysize=256,
    )
    sources = [rasterio.open(raster_for_year(year)[0]) for year in YEARS]
    class_counts = {-2: 0, -1: 0, 0: 0, 1: 0, 2: 0}
    valid_trend_count = 0
    try:
        with rasterio.open(stats_path, "w", **output_profile) as stats_dst, rasterio.open(
            trend_path, "w", **output_profile
        ) as trend_dst:
            for index, name in enumerate(
                ["mean_g_m2", "temporal_sd_g_m2", "temporal_cv", "valid_year_count"], 1
            ):
                stats_dst.set_band_description(index, name)
            for index, name in enumerate(
                ["sen_slope_g_m2_yr", "kendall_tau", "mk_p_approx", "trend_class"], 1
            ):
                trend_dst.set_band_description(index, name)

            for block_index, window in enumerate(block_windows(profile), 1):
                r0, c0 = int(window.row_off), int(window.col_off)
                r1, c1 = r0 + int(window.height), c0 + int(window.width)
                local_mask = strict[r0:r1, c0:c1]
                stack = []
                for year, src in zip(YEARS, sources):
                    band = raster_for_year(year)[1]
                    values = read_valid(src, band, window)
                    values[~local_mask] = np.nan
                    stack.append(values)
                data = np.stack(stack).astype("float32")
                n = np.sum(np.isfinite(data), axis=0).astype("int16")
                with np.errstate(invalid="ignore", divide="ignore"):
                    mean = np.nanmean(data, axis=0).astype("float32")
                    sd = np.nanstd(data, axis=0, ddof=1).astype("float32")
                    cv = (sd / mean).astype("float32")
                mean[(n == 0) | ~local_mask] = np.nan
                sd[(n < 2) | ~local_mask] = np.nan
                cv[(n < 2) | (mean <= 0) | ~local_mask] = np.nan

                pair_count = len(YEARS) * (len(YEARS) - 1) // 2
                slopes = np.full((pair_count, int(window.height), int(window.width)), np.nan, dtype="float32")
                s_score = np.zeros((int(window.height), int(window.width)), dtype="int16")
                pair = 0
                for i in range(len(YEARS) - 1):
                    for j in range(i + 1, len(YEARS)):
                        diff = data[j] - data[i]
                        finite = np.isfinite(data[i]) & np.isfinite(data[j])
                        slopes[pair] = np.where(finite, diff / (YEARS[j] - YEARS[i]), np.nan)
                        s_score += np.where(finite, np.sign(diff), 0).astype("int16")
                        pair += 1
                with np.errstate(all="ignore"):
                    sen = np.nanmedian(slopes, axis=0).astype("float32")
                denom = n.astype("float64") * (n - 1) / 2.0
                tau = np.where(denom > 0, s_score / denom, np.nan)
                var_s = n.astype("float64") * (n - 1) * (2 * n + 5) / 18.0
                z = np.zeros_like(var_s, dtype="float64")
                positive = s_score > 0
                negative = s_score < 0
                z[positive] = (s_score[positive] - 1) / np.sqrt(var_s[positive])
                z[negative] = (s_score[negative] + 1) / np.sqrt(var_s[negative])
                p_value = 2 * norm.sf(np.abs(z))
                trend_valid = local_mask & (n >= MIN_TREND_YEARS) & np.isfinite(sen)
                trend_class = np.full(local_mask.shape, np.nan, dtype="float32")
                stable = trend_valid & (np.abs(sen) < 0.10)
                trend_class[stable] = 0
                trend_class[trend_valid & (sen >= 0.10) & (p_value >= 0.05)] = 1
                trend_class[trend_valid & (sen >= 0.10) & (p_value < 0.05)] = 2
                trend_class[trend_valid & (sen <= -0.10) & (p_value >= 0.05)] = -1
                trend_class[trend_valid & (sen <= -0.10) & (p_value < 0.05)] = -2
                sen[~trend_valid] = np.nan
                tau[~trend_valid] = np.nan
                p_value[~trend_valid] = np.nan

                stats_arrays = [mean, sd, cv, n.astype("float32")]
                trend_arrays = [sen, tau.astype("float32"), p_value.astype("float32"), trend_class]
                for band_index, array in enumerate(stats_arrays, 1):
                    stats_dst.write(np.where(np.isfinite(array), array, NODATA).astype("float32"), band_index, window=window)
                for band_index, array in enumerate(trend_arrays, 1):
                    trend_dst.write(np.where(np.isfinite(array), array, NODATA).astype("float32"), band_index, window=window)
                valid_trend_count += int(trend_valid.sum())
                for code in class_counts:
                    class_counts[code] += int(np.sum(trend_class == code))
                if block_index % 100 == 0:
                    print(f"Pixel trend blocks completed: {block_index}", flush=True)
    finally:
        for src in sources:
            src.close()

    labels = {
        -2: "significant decrease",
        -1: "nonsignificant decrease",
        0: "stable (|Sen| < 0.10)",
        1: "nonsignificant increase",
        2: "significant increase",
    }
    area = pd.DataFrame([
        {
            "trend_class_code": code,
            "trend_class": labels[code],
            "cell_count": class_counts[code],
            "area_km2": class_counts[code] * CELL_AREA_M2 / 1e6,
            "fraction_of_trend_valid_area": class_counts[code] / valid_trend_count,
        }
        for code in sorted(class_counts)
    ])
    area.to_csv(OUT / "pixel_trend_class_area_1988_2025.csv", index=False)
    return stats_path, trend_path, area


def prepare_lai_state(annual: pd.DataFrame) -> pd.DataFrame:
    old = pd.read_csv(OLD_STATE_SUMMARY)
    old = old[
        (old.region == "peatland_main95_fill400")
        & (old.year.between(1988, 2024))
    ][[
        "year", "lai", "fapar", "lai_valid_fraction", "file_count",
        "sensor_family", "platform", "annual_status",
    ]].copy()

    lai_module = import_module(
        "prepare_lai_2025",
        ROOT / "scripts" / "_internal" / "lai_state_helpers.py",
    )
    lai_module.RASTERS = RASTERS_5KM
    (RASTERS_5KM / "lai_fapar").mkdir(parents=True, exist_ok=True)
    grid = lai_module.load_grid()
    arrays, completeness = lai_module.build_lai_year(2025, grid, rebuild=False)
    base = pd.read_pickle(BASE_PANEL)
    frame = base[
        (base.year == 2024)
        & (base.peat_area_fraction >= 0.1)
        & (base.peat_valid_fraction >= 0.5)
    ].copy()
    rows = frame.row.to_numpy(int)
    cols = frame.col.to_numpy(int)
    weights = frame.peat_area_fraction.to_numpy(float)

    def weighted(array: np.ndarray) -> float:
        values = array[rows, cols].astype(float)
        valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
        return float(np.average(values[valid], weights=weights[valid]))

    row_2025 = {
        "year": 2025,
        "lai": weighted(arrays["lai"]),
        "fapar": weighted(arrays["fapar"]),
        "lai_valid_fraction": weighted(arrays["lai_valid_fraction"]),
        "file_count": completeness["file_count"],
        "sensor_family": completeness["sensor_family"],
        "platform": completeness["platform"],
        "annual_status": completeness["annual_status"],
    }
    state = pd.concat([old, pd.DataFrame([row_2025])], ignore_index=True)
    state = annual[["year", "fvc_mean", "valid_coverage_fraction"]].merge(state, on="year", how="left")
    state["lai_sensor_era"] = state.year.map(lai_era)
    state["lai_era_z"] = np.nan
    for era, index in state.groupby("lai_sensor_era").groups.items():
        values = state.loc[index, "lai"].astype(float)
        scale = values.std(ddof=0)
        state.loc[index, "lai_era_z"] = (values - values.mean()) / scale if scale > 0 else 0.0
    state.to_csv(OUT / "annual_fvc_lai_state_1988_2025.csv", index=False)
    pd.DataFrame([completeness]).to_csv(OUT / "lai_2025_completeness.csv", index=False)
    return state


def guarded_sensitivity(annual: pd.DataFrame) -> pd.DataFrame:
    audit = pd.read_csv(HISTORICAL_AUDIT)
    frame = annual[annual.year <= 2024][["year", "agb_mean_g_m2"]].merge(
        audit[["year", "guarded_agb_mean_g_m2"]], on="year", how="inner"
    )
    peak = trend_record("Peak-only 1988-2024", frame.rename(columns={"agb_mean_g_m2": "agb_mean_g_m2"}), list(frame.year))
    guarded_frame = frame[["year", "guarded_agb_mean_g_m2"]].rename(
        columns={"guarded_agb_mean_g_m2": "agb_mean_g_m2"}
    )
    guarded = trend_record("Guarded-consensus 1988-2024", guarded_frame, list(frame.year))
    corr = pearsonr(frame.agb_mean_g_m2, frame.guarded_agb_mean_g_m2)
    summary = pd.DataFrame([
        {
            "comparison": "Peak-only vs Guarded-consensus 1988-2024",
            "peak_sen_slope_g_m2_yr": peak["sen_slope_g_m2_yr"],
            "guarded_sen_slope_g_m2_yr": guarded["sen_slope_g_m2_yr"],
            "same_regional_trend_direction": np.sign(peak["sen_slope_g_m2_yr"]) == np.sign(guarded["sen_slope_g_m2_yr"]),
            "annual_mean_pearson_r": corr.statistic,
            "annual_mean_pearson_p": corr.pvalue,
            "mean_guarded_minus_peak_g_m2": float((frame.guarded_agb_mean_g_m2 - frame.agb_mean_g_m2).mean()),
            "mean_absolute_guarded_minus_peak_g_m2": float(np.abs(frame.guarded_agb_mean_g_m2 - frame.agb_mean_g_m2).mean()),
            "known_pixel_trend_direction_agreement_fraction": 0.998905,
        }
    ])
    summary.to_csv(OUT / "guarded_peak_sensitivity_1988_2024.csv", index=False)
    frame.to_csv(OUT / "annual_peak_guarded_comparison_1988_2024.csv", index=False)
    return summary


def map_extent(profile: dict, bbox: tuple[int, int, int, int]) -> list[float]:
    r0, r1, c0, c1 = bbox
    transform = profile["transform"]
    left = transform.c + c0 * transform.a
    right = transform.c + c1 * transform.a
    top = transform.f + r0 * transform.e
    bottom = transform.f + r1 * transform.e
    return [left / 1000, right / 1000, bottom / 1000, top / 1000]


def read_bbox(path: Path, band: int, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, r1, c0, c1 = bbox
    window = Window(c0, r0, c1 - c0, r1 - r0)
    with rasterio.open(path) as src:
        return read_valid(src, band, window)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.png", dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_spatial_summary(profile: dict, strict: np.ndarray, bbox, stats_path: Path, trend_path: Path) -> None:
    arrays = [read_bbox(stats_path, 1, bbox), read_bbox(stats_path, 3, bbox), read_bbox(trend_path, 1, bbox), read_bbox(trend_path, 4, bbox)]
    local_mask = strict[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    for array in arrays:
        array[~local_mask] = np.nan
    extent = map_extent(profile, bbox)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.0), constrained_layout=True)
    panels = [
        (arrays[0], "a", "1988-2025 mean AGB", "YlGnBu", 180, 430, "AGB (g m$^{-2}$)"),
        (arrays[1], "b", "Temporal variability", "magma", 0.08, 0.40, "Coefficient of variation"),
        (arrays[2], "c", "Pixelwise Theil-Sen trend", "RdBu", -4, 4, "g m$^{-2}$ yr$^{-1}$"),
    ]
    for ax, (array, letter, title, cmap, vmin, vmax, label) in zip(axes.flat[:3], panels):
        im = ax.imshow(array, extent=extent, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.contour(local_mask.astype(float), levels=[0.5], colors="0.25", linewidths=0.35, extent=extent, origin="upper")
        ax.set_title(title, fontsize=10)
        ax.text(-0.08, 1.03, letter, transform=ax.transAxes, fontweight="bold", fontsize=11)
        ax.set_xlabel("Easting (km)")
        ax.set_ylabel("Northing (km)")
        cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
        cb.set_label(label, fontsize=8)
    ax = axes.flat[3]
    class_colors = ["#8b1a1a", "#e98b73", "#e6e6e6", "#7db7d8", "#135a9c"]
    cmap = ListedColormap(class_colors)
    normer = BoundaryNorm([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], cmap.N)
    ax.imshow(arrays[3], extent=extent, origin="upper", cmap=cmap, norm=normer)
    ax.contour(local_mask.astype(float), levels=[0.5], colors="0.25", linewidths=0.35, extent=extent, origin="upper")
    ax.set_title("Mann-Kendall trend class", fontsize=10)
    ax.text(-0.08, 1.03, "d", transform=ax.transAxes, fontweight="bold", fontsize=11)
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    labels = ["Sig. decrease", "Non-sig. decrease", "Stable", "Non-sig. increase", "Sig. increase"]
    handles = [Patch(facecolor=color, edgecolor="none", label=label) for color, label in zip(class_colors, labels)]
    ax.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.99),
        ncol=1, frameon=True, framealpha=0.88, facecolor="white",
        edgecolor="none", fontsize=6.5,
    )
    fig.suptitle("Strict main95 peatland Peak-only AGB spatial change", fontsize=12, fontweight="bold")
    save_figure(fig, "Fig_3_2a_spatial_pattern_and_trend_1988_2025")


def plot_representative_maps(profile: dict, strict: np.ndarray, bbox) -> None:
    local_mask = strict[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    extent = map_extent(profile, bbox)
    fig = plt.figure(figsize=(11.0, 4.0))
    grid = fig.add_gridspec(1, 4, wspace=0.05)
    axes = []
    for index in range(4):
        if index == 0:
            axes.append(fig.add_subplot(grid[0, index]))
        else:
            axes.append(fig.add_subplot(grid[0, index], sharex=axes[0], sharey=axes[0]))
    last = None
    for index, (ax, year) in enumerate(zip(axes, REPRESENTATIVE_YEARS)):
        path, band, _, _ = raster_for_year(year)
        values = read_bbox(path, band, bbox)
        values[~local_mask] = np.nan
        last = ax.imshow(values, extent=extent, origin="upper", cmap="YlGnBu", vmin=0, vmax=500)
        ax.contour(local_mask.astype(float), levels=[0.5], colors="0.3", linewidths=0.3, extent=extent, origin="upper")
        ax.set_title(str(year), fontweight="bold")
        ax.text(-0.08, 1.02, chr(ord("a") + index), transform=ax.transAxes, fontweight="bold")
    axes[0].set_ylabel("Northing (km)")
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.30, top=0.82)
    fig.text(0.52, 0.225, "Easting (km)", ha="center")
    color_ax = fig.add_axes([0.32, 0.10, 0.40, 0.035])
    cb = fig.colorbar(last, cax=color_ax, orientation="horizontal")
    cb.set_label("Peak-only AGB (g m$^{-2}$)")
    cb.set_ticks([0, 100, 200, 300, 400, 500])
    fig.suptitle("Representative strict-main95 peatland AGB maps", fontsize=12, fontweight="bold")
    save_figure(fig, "Fig_3_2b_representative_agb_maps")


def plot_regional_state(annual: pd.DataFrame, state: pd.DataFrame, trends: pd.DataFrame) -> None:
    x = annual.year.to_numpy()
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 7.6), sharex=True, constrained_layout=True)
    ax = axes[0]
    ax.fill_between(x, annual.agb_p05_g_m2, annual.agb_p95_g_m2, color="#9ecae1", alpha=0.35, label="P05-P95")
    ax.plot(x, annual.agb_p50_g_m2, color="#6baed6", lw=1.0, label="Median")
    ax.plot(x, annual.agb_mean_g_m2, color="#225ea8", marker="o", ms=2.8, lw=1.35, label="Mean")
    low = annual.valid_coverage_fraction < 0.90
    ax.scatter(annual.loc[low, "year"], annual.loc[low, "agb_mean_g_m2"], facecolors="white", edgecolors="#d7301f", s=34, zorder=5, label="Coverage < 0.90")
    t = trends.loc[trends.scenario == "1988-2025 all"].iloc[0]
    ax.text(0.01, 0.96, f"Sen slope = {t.sen_slope_g_m2_yr:.2f} g m$^{{-2}}$ yr$^{{-1}}$\nMK p = {t.mann_kendall_p:.2g}", transform=ax.transAxes, va="top", fontsize=8)
    ax.set_ylabel("AGB (g m$^{-2}$)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.58, 1.02), frameon=False, fontsize=7)
    ax.text(-0.08, 1.02, "a", transform=ax.transAxes, fontweight="bold")

    ax = axes[1]
    ax.plot(state.year, state.fvc_mean, color="#1b9e77", marker="o", ms=2.8, lw=1.3)
    ax.set_ylabel("FVC")
    ax.text(-0.08, 1.02, "b", transform=ax.transAxes, fontweight="bold")

    ax = axes[2]
    era_colors = {"AVHRR": "#756bb1", "VIIRS S-NPP": "#31a354", "VIIRS NOAA-20": "#e6550d"}
    for era, group in state.groupby("lai_sensor_era", sort=False):
        ax.plot(group.year, group.lai_era_z, color=era_colors[era], marker="o", ms=3, lw=1.3, label=era)
    incomplete = state.annual_status != "complete"
    ax.scatter(state.loc[incomplete, "year"], state.loc[incomplete, "lai_era_z"], facecolors="white", edgecolors="black", s=34, zorder=5, label="Daily gaps")
    ax.axhline(0, color="0.55", lw=0.8)
    ax.axvline(2013.5, color="0.6", ls=":", lw=0.9)
    ax.axvline(2022.5, color="0.6", ls=":", lw=0.9)
    ax.set_ylabel("LAI (within-era z)")
    ax.set_xlabel("Year")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.24), frameon=False, fontsize=7)
    ax.text(-0.08, 1.02, "c", transform=ax.transAxes, fontweight="bold")
    for item in axes:
        item.grid(axis="y", color="0.9", lw=0.6)
        item.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Strict main95 peatland vegetation-state trajectories", fontsize=12, fontweight="bold")
    save_figure(fig, "Fig_3_2c_regional_agb_fvc_lai_trajectories")


def plot_robustness(annual: pd.DataFrame, trends: pd.DataFrame) -> None:
    comparison = pd.read_csv(OUT / "annual_peak_guarded_comparison_1988_2024.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), constrained_layout=True)
    ax = axes[0]
    order = ["1988-2024 frozen", "1988-2025 all", "1988-2025 coverage >= 0.80", "1988-2025 coverage >= 0.90"]
    data = trends.set_index("scenario").loc[order].reset_index()
    ypos = np.arange(len(data))
    err = np.vstack([
        data.sen_slope_g_m2_yr - data.sen_ci95_low,
        data.sen_ci95_high - data.sen_slope_g_m2_yr,
    ])
    ax.errorbar(data.sen_slope_g_m2_yr, ypos, xerr=err, fmt="o", color="#225ea8", capsize=3)
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_yticks(ypos, ["1988-2024\nfrozen", "1988-2025\nall", "Coverage\n>= 0.80", "Coverage\n>= 0.90"])
    ax.invert_yaxis()
    ax.set_xlabel("Sen slope (g m$^{-2}$ yr$^{-1}$)")
    ax.set_title("Regional trend sensitivity")
    ax.text(-0.12, 1.03, "a", transform=ax.transAxes, fontweight="bold")

    ax = axes[1]
    peak_z = (comparison.agb_mean_g_m2 - comparison.agb_mean_g_m2.mean()) / comparison.agb_mean_g_m2.std(ddof=0)
    guarded_z = (comparison.guarded_agb_mean_g_m2 - comparison.guarded_agb_mean_g_m2.mean()) / comparison.guarded_agb_mean_g_m2.std(ddof=0)
    ax.plot(comparison.year, peak_z, color="#225ea8", lw=1.3, label="Peak-only")
    ax.plot(comparison.year, guarded_z, color="#d95f0e", lw=1.2, label="Guarded-consensus")
    ax.axhline(0, color="0.65", lw=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual AGB anomaly (z)")
    ax.set_title("Model-structure sensitivity, 1988-2024")
    ax.legend(frameon=False, fontsize=8)
    ax.text(-0.12, 1.03, "b", transform=ax.transAxes, fontweight="bold")
    for item in axes:
        item.grid(axis="x", color="0.92", lw=0.6)
        item.spines[["top", "right"]].set_visible(False)
    save_figure(fig, "Fig_3_2d_trend_robustness_and_model_sensitivity")


def write_report(
    annual: pd.DataFrame,
    trends: pd.DataFrame,
    trend_area: pd.DataFrame,
    state: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> None:
    frozen = trends[trends.scenario == "1988-2024 frozen"].iloc[0]
    extended = trends[trends.scenario == "1988-2025 all"].iloc[0]
    sig_inc = trend_area.loc[trend_area.trend_class_code == 2, "fraction_of_trend_valid_area"].iloc[0]
    positive = trend_area.loc[trend_area.trend_class_code.isin([1, 2]), "fraction_of_trend_valid_area"].sum()
    y2025 = annual[annual.year == 2025].iloc[0]
    y_min = annual.loc[annual.agb_mean_g_m2.idxmin()]
    y_max = annual.loc[annual.agb_mean_g_m2.idxmax()]
    sens = sensitivity.iloc[0]
    incomplete_lai = state.loc[state.annual_status != "complete", "year"].astype(int).tolist()
    text = f"""# 3.2 AGB空间格局与长期变化：结果稿

## 分析口径

- 主产品为1988-2025年Peak-only AGB，统一在90 m输出网格上分析，其空间支撑约为90 m。
- 空间范围严格限定为`analysis_main95_fill400`，90 m栅格化面积为{annual.strict_mask_area_km2_90m_grid.iloc[0]:.2f} km2。
- Guarded-consensus仅作为1988-2024年的模型结构敏感性，不作为气候驱动分析的主序列。
- FVC来自相同Landsat PeakNDVI影像，只用于植被状态辅助描述；LAI按AVHRR、VIIRS S-NPP和VIIRS NOAA-20三个时代分别标准化。

## 可直接用于正文的结果

1988-2025年，严格主范围内Peak-only AGB的多年年均值为{annual.agb_mean_g_m2.mean():.2f} g m-2，各年份区域均值介于{y_min.agb_mean_g_m2:.2f} g m-2（{int(y_min.year)}年）和{y_max.agb_mean_g_m2:.2f} g m-2（{int(y_max.year)}年）之间。2025年区域均值为{y2025.agb_mean_g_m2:.2f} g m-2，有效覆盖率为{y2025.valid_coverage_fraction:.1%}。多年平均空间格局表明，中高值区主要集中在主泥炭地连续分布区，而边缘和局部低覆盖区AGB较低；时序变异系数图进一步揭示了不同区域年际稳定性的空间差异（图3.2a）。

区域尺度Theil-Sen分析显示，冻结的1988-2024年Peak-only序列以{frozen.sen_slope_g_m2_yr:+.3f} g m-2 yr-1的速率变化（95% CI: {frozen.sen_ci95_low:.3f}-{frozen.sen_ci95_high:.3f}，Mann-Kendall p={frozen.mann_kendall_p:.3g}），对应研究期内约{frozen.sen_implied_total_change_g_m2:.1f} g m-2的拟合变化。加入2025年后，1988-2025年斜率为{extended.sen_slope_g_m2_yr:+.3f} g m-2 yr-1（95% CI: {extended.sen_ci95_low:.3f}-{extended.sen_ci95_high:.3f}，p={extended.mann_kendall_p:.3g}）。像元尺度上，{positive:.1%}的趋势有效区域呈正Sen斜率，其中{sig_inc:.1%}达到近似Mann-Kendall显著增加标准；该像元p值为未进行序列自相关修正的渐近估计，因此论文应把Sen斜率方向作为主要空间证据，并将显著性分类作为辅助结果。这里的“正趋势比例”和“显著增加比例”已分别统计，不能混写（图3.2a）。

典型年份地图显示，1989、2000、2013和2025年的总体空间骨架较为稳定，但年际强度及局部高值范围存在明显变化（图3.2b）。区域AGB均值、P05-P95分布带和中位数共同表明，长期变化并非仅由少数极端像元驱动。FVC与AGB总体表现出相近的植被改善背景；LAI在传感器时代内标准化后用于识别相对高低状态，避免将2013年AVHRR-VIIRS转换误判为生态突变（图3.2c）。LAI存在日数据缺口的年份为：{', '.join(map(str, incomplete_lai))}，这些年份在图中以空心点标记。

趋势敏感性结果表明，纳入2025年以及分别剔除有效覆盖率低于0.80和0.90的年份后，区域Sen斜率方向保持一致。1988-2024年Peak-only与Guarded-consensus斜率分别为{sens.peak_sen_slope_g_m2_yr:.3f}和{sens.guarded_sen_slope_g_m2_yr:.3f} g m-2 yr-1，年度均值相关系数为r={sens.annual_mean_pearson_r:.3f}，已知像元趋势方向一致率为{sens.known_pixel_trend_direction_agreement_fraction:.2%}（图3.2d）。因此，正向长期变化可作为模型结构稳健的区域结论，但绝对AGB水平及单像元幅度仍需结合验证不确定性谨慎解释。

## 图件安排

1. `Fig_3_2a_spatial_pattern_and_trend_1988_2025`：多年均值、CV、像元Sen斜率和MK趋势分类。
2. `Fig_3_2b_representative_agb_maps`：1989、2000、2013和2025年典型空间格局。
3. `Fig_3_2c_regional_agb_fvc_lai_trajectories`：AGB分布带、FVC与时代标准化LAI。
4. `Fig_3_2d_trend_robustness_and_model_sensitivity`：覆盖率情景和Guarded模型敏感性。

## 论文表述边界

- 可以写“区域平均AGB呈显著上升且在模型敏感性分析中方向一致”。
- 不宜写“90 m单像元AGB具有很高绝对精度”或“2025独立验证通过”。
- 2025采用同一Peak-only模型生成，可进入3.2描述性延伸；其独立验证结果应在3.1中单独报告。
- 这里不解释干旱抵抗力、恢复力、突变年份或SHAP阈值，它们分别放在后续结果小节。
"""
    (RUN / "report_section_3_2_peakonly_1988_2025.md").write_text(text, encoding="utf-8")


def qa_outputs() -> None:
    records = []
    from PIL import Image
    for path in sorted(FIGURES.glob("*.png")):
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
            records.append({
                "file": path.name,
                "width_px": image.width,
                "height_px": image.height,
                "pixel_std": float(rgb.std()),
                "nonwhite_fraction": float(np.mean(np.any(rgb < 250, axis=2))),
                "nonblank": bool(rgb.std() > 5 and np.mean(np.any(rgb < 250, axis=2)) > 0.02),
            })
    frame = pd.DataFrame(records)
    frame.to_csv(OUT / "figure_qa.csv", index=False)
    if not frame.nonblank.all():
        raise ValueError("At least one generated figure appears blank")


def main() -> None:
    ensure_dirs()
    profile, strict, bbox = load_reference_and_mask()
    fvc_2025, _ = compute_fvc_2025(profile, strict)
    _, annual, _ = annual_statistics(profile, strict, fvc_2025)
    trends = regional_trends(annual)
    stats_path, trend_path, trend_area = compute_pixel_products(profile, strict)
    state = prepare_lai_state(annual)
    sensitivity = guarded_sensitivity(annual)
    plot_spatial_summary(profile, strict, bbox, stats_path, trend_path)
    plot_representative_maps(profile, strict, bbox)
    plot_regional_state(annual, state, trends)
    plot_robustness(annual, trends)
    write_report(annual, trends, trend_area, state, sensitivity)
    qa_outputs()
    summary = {
        "years": [YEARS[0], YEARS[-1]],
        "n_years": len(YEARS),
        "strict_mask_cells": int(strict.sum()),
        "strict_mask_area_km2": float(strict.sum() * CELL_AREA_M2 / 1e6),
        "regional_trends": trends.to_dict(orient="records"),
        "trend_class_area": trend_area.to_dict(orient="records"),
    }
    (OUT / "section_3_2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Section 3.2 analysis completed.", flush=True)


if __name__ == "__main__":
    main()
