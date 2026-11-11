"""Produce the 2013-2024 map-ready Peak-only AGB series at 90 m support."""

from __future__ import annotations

import json
import sys
import warnings
import argparse
from pathlib import Path

import geopandas as gpd
import joblib
import matplotlib
import numpy as np
import pandas as pd
import rasterio
from matplotlib import pyplot as plt
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._internal import annual_agb_helpers as pilot  # noqa: E402


matplotlib.use("Agg")

RUN_DIR = ROOT / "agb_production_readiness_2013_2024"
OUTPUT_DIR = RUN_DIR / "outputs"
RASTER_DIR = RUN_DIR / "peak_only_rasters_2013_2024"
FIGURE_DIR = RUN_DIR / "figures"
MODEL_PATH = ROOT / "agb_ecological_weighted_validation" / "models" / "ecological_weighted_candidate.joblib"
TRAINING_FEATURES = ROOT / "agb_ecological_weighted_validation" / "outputs" / "climate_augmented_sample_features.csv"
YEARS = list(range(2013, 2025))
OUTPUT_BANDS = [
    "AGB_full_aoi_g_m2",
    "AGB_peatland_g_m2",
    "ood_feature_fraction",
    "peatland_probability",
    "valid_count",
]


def parse_years(value: str) -> list[int]:
    years: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            years.update(range(start, end + 1))
        else:
            years.add(int(part))
    result = sorted(years)
    invalid = [year for year in result if year < 1985 or year > 2024]
    if invalid:
        raise ValueError(f"Supported years are 1985-2024: {invalid}")
    return result


def ensure_dirs() -> None:
    for path in [OUTPUT_DIR, RASTER_DIR, FIGURE_DIR, pilot.VRT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def training_limits(features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(TRAINING_FEATURES)
    values = frame[features].to_numpy(float)
    return np.nanquantile(values, 0.01, axis=0), np.nanquantile(values, 0.99, axis=0)


def feature_arrays(dynamic: dict[str, np.ndarray], terrain: np.ndarray) -> dict[str, np.ndarray]:
    arrays = {
        f"pk_{name}": dynamic[name]
        for name in pilot.DYNAMIC_FEATURES
        if name not in {"NIRv", "NDTI"}
    }
    arrays["pk_NIRv"] = dynamic["NIR"] * dynamic["NDVI"]
    denominator = dynamic["SWIR1"] + dynamic["SWIR2"]
    arrays["pk_NDTI"] = np.divide(
        dynamic["SWIR1"] - dynamic["SWIR2"],
        denominator,
        out=np.full(dynamic["NIR"].shape, np.nan, dtype=np.float32),
        where=np.abs(denominator) > 1e-8,
    )
    for index, name in enumerate(pilot.TERRAIN_FEATURES):
        arrays[name] = terrain[index]
    return arrays


def run_year(
    year: int,
    model,
    features: list[str],
    lower: np.ndarray,
    upper: np.ndarray,
    geometries: list[dict],
) -> dict[str, float]:
    vrt_path = pilot.build_mosaic_vrt(year)
    sampled_full: list[np.ndarray] = []
    sampled_peat: list[np.ndarray] = []
    sampled_ood: list[np.ndarray] = []
    full_count = peat_count = high_ood_count = 0
    full_sum = peat_sum = 0.0

    with rasterio.open(vrt_path) as source, \
         rasterio.open(pilot.TERRAIN_STACK) as terrain_src, \
         rasterio.open(pilot.PEAT_PROBABILITY) as peat_src:
        width = source.width // pilot.AGGREGATION_FACTOR
        height = source.height // pilot.AGGREGATION_FACTOR
        transform = Affine(
            pilot.OUTPUT_RESOLUTION, 0.0, source.bounds.left,
            0.0, -pilot.OUTPUT_RESOLUTION, source.bounds.top,
        )
        aoi_full = geometry_mask(
            geometries,
            out_shape=(height, width),
            transform=transform,
            invert=True,
            all_touched=False,
        )
        descriptions = list(terrain_src.descriptions)
        terrain_indexes = [descriptions.index(name) + 1 for name in pilot.TERRAIN_FEATURES]
        vrt_options = {"crs": source.crs, "transform": transform, "width": width, "height": height}

        with WarpedVRT(terrain_src, resampling=Resampling.bilinear, **vrt_options) as terrain_vrt, \
             WarpedVRT(peat_src, resampling=Resampling.bilinear, **vrt_options) as peat_vrt:
            profile = source.profile.copy()
            profile.update(
                driver="GTiff", width=width, height=height, transform=transform,
                count=len(OUTPUT_BANDS), dtype="float32", nodata=pilot.NODATA,
                compress="DEFLATE", predictor=3, tiled=True,
                blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER",
            )
            output_path = RASTER_DIR / f"Zoige_AGB_PeakOnly_PLSR_90m_{year}.tif"
            with rasterio.open(output_path, "w", **profile) as dst:
                for index, name in enumerate(OUTPUT_BANDS, start=1):
                    dst.set_band_description(index, name)
                dst.update_tags(
                    model="Baseline_Peak_PLSR",
                    source_dummy="not_used",
                    climate_predictors="not_used",
                    support="valid-only 3x3 median from 30 m PeakNDVI Jul-Sep",
                    nominal_resolution_m=90,
                    peatland_probability_threshold=pilot.PEAT_THRESHOLD,
                )
                for window_index, window in enumerate(pilot.output_windows(width, height)):
                    row0, col0 = int(window.row_off), int(window.col_off)
                    row1, col1 = row0 + int(window.height), col0 + int(window.width)
                    aoi = aoi_full[row0:row1, col0:col1]
                    shape = aoi.shape
                    if not aoi.any():
                        dst.write(
                            np.full((len(OUTPUT_BANDS), *shape), pilot.NODATA, dtype=np.float32),
                            window=window,
                        )
                        continue

                    dynamic = pilot.read_valid_medians(source, window)
                    terrain = terrain_vrt.read(terrain_indexes, window=window).astype(np.float32)
                    peat = peat_vrt.read(1, window=window).astype(np.float32)
                    arrays = feature_arrays(dynamic, terrain)
                    matrix = np.column_stack([arrays[name].ravel() for name in features])
                    valid = dynamic["valid_mask"].ravel().astype(bool) & aoi.ravel()
                    valid &= np.isfinite(matrix).all(axis=1)

                    prediction = np.full(matrix.shape[0], pilot.NODATA, dtype=np.float32)
                    ood = np.full(matrix.shape[0], pilot.NODATA, dtype=np.float32)
                    if valid.any():
                        prediction[valid] = np.clip(
                            model.predict(matrix[valid]).reshape(-1), 0.0, 2000.0
                        ).astype(np.float32)
                        ood[valid] = np.mean(
                            (matrix[valid] < lower) | (matrix[valid] > upper), axis=1
                        ).astype(np.float32)

                    peat_flat = peat.ravel()
                    peat_valid = valid & np.isfinite(peat_flat) & (peat_flat >= pilot.PEAT_THRESHOLD)
                    peat_prediction = np.full(matrix.shape[0], pilot.NODATA, dtype=np.float32)
                    peat_prediction[peat_valid] = prediction[peat_valid]
                    peat_output = np.where(
                        aoi & np.isfinite(peat), np.clip(peat, 0.0, 1.0), pilot.NODATA
                    ).astype(np.float32)
                    valid_count = np.where(
                        dynamic["valid_mask"] & aoi,
                        dynamic["valid_count"],
                        pilot.NODATA,
                    ).astype(np.float32)
                    dst.write(
                        np.stack([
                            prediction.reshape(shape), peat_prediction.reshape(shape),
                            ood.reshape(shape), peat_output, valid_count,
                        ]),
                        window=window,
                    )

                    full_values = prediction[valid]
                    peat_values = prediction[peat_valid]
                    ood_values = ood[valid]
                    full_count += int(full_values.size)
                    peat_count += int(peat_values.size)
                    high_ood_count += int(np.sum(ood_values > 0.25))
                    full_sum += float(np.sum(full_values, dtype=np.float64))
                    peat_sum += float(np.sum(peat_values, dtype=np.float64))
                    sampled_full.append(full_values[::20])
                    sampled_peat.append(peat_values[::20])
                    sampled_ood.append(ood_values[::20])
                    if window_index % 40 == 0:
                        print(f"{year}: processed window {window_index}", flush=True)

    full = np.concatenate(sampled_full)
    peat_values = np.concatenate(sampled_peat)
    ood_values = np.concatenate(sampled_ood)
    return {
        "year": year,
        "output_path": str(output_path),
        "valid_pixel_count": full_count,
        "peatland_pixel_count": peat_count,
        "agb_full_mean_g_m2": full_sum / full_count,
        "agb_full_p05_g_m2": float(np.quantile(full, 0.05)),
        "agb_full_median_g_m2": float(np.median(full)),
        "agb_full_p95_g_m2": float(np.quantile(full, 0.95)),
        "agb_peat_mean_g_m2": peat_sum / peat_count,
        "agb_peat_p05_g_m2": float(np.quantile(peat_values, 0.05)),
        "agb_peat_median_g_m2": float(np.median(peat_values)),
        "agb_peat_p95_g_m2": float(np.quantile(peat_values, 0.95)),
        "ood_fraction_gt_0_25": high_ood_count / full_count,
        "ood_feature_fraction_median": float(np.median(ood_values)),
    }


def make_timeseries_figure(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    axes[0].plot(summary["year"], summary["agb_full_mean_g_m2"], marker="o", label="Zoige AOI")
    axes[0].plot(summary["year"], summary["agb_peat_mean_g_m2"], marker="o", label="Peatland mask")
    axes[0].set_ylabel("Mean AGB (g m$^{-2}$)")
    axes[0].legend(frameon=False)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[1].plot(summary["year"], 100 * summary["ood_fraction_gt_0_25"], marker="o", color="#B44C43")
    axes[1].axhline(25, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("High-OOD pixels (%)")
    axes[1].set_xlabel("Year")
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.savefig(FIGURE_DIR / "peak_only_annual_mean_and_ood_2013_2024.png", dpi=250)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", default="2013-2024")
    args = parser.parse_args()
    years = parse_years(args.years)
    year_label = args.years.replace(",", "_").replace("-", "_")
    ensure_dirs()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    features = list(bundle["features"])
    lower, upper = training_limits(features)
    aoi = gpd.read_file(pilot.AOI_PATH).to_crs("EPSG:32648")
    geometries = [geometry.__geo_interface__ for geometry in aoi.geometry if geometry is not None]
    rows = [run_year(year, model, features, lower, upper, geometries) for year in years]
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / f"peak_only_annual_summary_{year_label}.csv", index=False, encoding="utf-8-sig")
    make_timeseries_figure(summary)

    ood_gate = bool(summary["ood_fraction_gt_0_25"].le(0.25).all())
    range_gate = bool(summary["agb_peat_mean_g_m2"].between(100, 700).all())
    year_change = summary["agb_peat_mean_g_m2"].pct_change().abs()
    temporal_review_years = summary.loc[year_change.gt(0.35), "year"].astype(int).tolist()
    decision = {
        "model": bundle["experiment"],
        "years": [years[0], years[-1]],
        "source_dummy_used": False,
        "climate_predictors_used": False,
        "ood_gate_passed": ood_gate,
        "range_gate_passed": range_gate,
        "years_with_abs_mean_change_over_35_percent": temporal_review_years,
        "batch_numerical_qc_passed": ood_gate and range_gate,
        "independent_accuracy_claim": "5 km spatial-block OOF only",
    }
    (OUTPUT_DIR / f"peak_only_batch_decision_{year_label}.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = f"""# 2013-2024 Peak-only AGB批量反演

## 生产模型

- 模型：`Baseline_Peak_PLSR`，不含source dummy，不含气候预测变量。
- 动态输入：7-9月PeakNDVI合成，按有效像元做3×3中值，支持尺度90 m。
- 输出：全若尔盖AOI、泥炭地概率≥0.50、训练域外比例、泥炭地概率和有效观测数。
- 论文精度：5 km空间分块OOF，r=0.471，R²=0.221，RMSE=202.2 g/m²。

## 年度统计

{pilot.markdown_table(summary.drop(columns=['output_path']))}

## QC决定

- 域外门槛通过：{'是' if ood_gate else '否'}；
- 泥炭地年度均值100-700 g/m²门槛通过：{'是' if range_gate else '否'}；
- 年均值相邻年变化超过35%的年份：{temporal_review_years or '无'}。

BIO/QTP比较属于产品一致性分析；气候数据应在AGB产品生成后作为独立驱动变量使用。
"""
    (RUN_DIR / f"report_peak_only_batch_{year_label}.md").write_text(report, encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
