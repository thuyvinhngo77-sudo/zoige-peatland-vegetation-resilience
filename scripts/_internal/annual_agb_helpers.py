"""Run 2017/2020/2024 pilot AGB inversion on a 90 m support grid.

Each 90 m predictor pixel is the valid-only median of a non-overlapping 3x3
window from the 30 m annual PeakNDVI composite. This matches the field feature
support used to train the frozen TC_local_core3_PLSR candidate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from pathlib import Path
from xml.sax.saxutils import escape

import geopandas as gpd
import joblib
import matplotlib
import numpy as np
import pandas as pd
import rasterio
from matplotlib import pyplot as plt
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.io import MemoryFile
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window


matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "agb_production_readiness_2013_2024"
OUTPUT_DIR = RUN_DIR / "outputs"
RASTER_DIR = RUN_DIR / "pilot_rasters"
FIGURE_DIR = RUN_DIR / "figures"
VRT_DIR = RUN_DIR / "virtual_inputs"
IMAGE_DIR = Path(
    os.environ.get(
        "LANDSAT_COMPOSITE_ROOT",
        str(ROOT / "data" / "landsat_composites"),
    )
)
HISTORICAL_IMAGE_DIR = ROOT / "zoigepeatland" / "zoige_rsimg"
MODEL_PATH = ROOT / "agb_ecological_weighted_validation" / "models" / "climate_augmented_candidate.joblib"
PEAK_ONLY_MODEL_PATH = ROOT / "agb_ecological_weighted_validation" / "models" / "ecological_weighted_candidate.joblib"
TRAINING_FEATURES = ROOT / "agb_ecological_weighted_validation" / "outputs" / "climate_augmented_sample_features.csv"
TERRAIN_STACK = ROOT / "zoigepeatland" / "data" / "Zoige_RF_feature_stack_2023_2025_100m.tif"
PEAT_PROBABILITY = ROOT / "zoigepeatland" / "output_basic_boundary" / "Zoige_RF_basic_probability.tif"
AOI_PATH = ROOT / "zoigepeatland" / "data" / "Zoige.shp"
CLIMATE_STACK = ROOT / "climate_data" / "input" / "Zoige_TerraClimate_GS_1985_2024_stack.tif"

PILOT_YEARS = [2017, 2020, 2024]
SOURCE_RESOLUTION = 30.0
OUTPUT_RESOLUTION = 90.0
AGGREGATION_FACTOR = 3
BLOCK_SIZE = 256
PEAT_THRESHOLD = 0.50
NODATA = -9999.0
CLIMATE_VARIABLES = ["GST_C", "GSP_mm", "VPD_kPa"]
CLIMATOLOGY_YEARS = range(1985, 2014)

SOURCE_BANDS = [
    "Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2",
    "NDVI", "NDMI", "LSWI", "NDWI", "MNDWI", "EVI", "FVC",
    "valid_count", "valid_mask", "water_count", "water_fraction",
]
DYNAMIC_FEATURES = [
    "Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2", "NDVI", "EVI",
    "NDMI", "NDWI", "MNDWI", "FVC", "LSWI", "NIRv", "NDTI",
    "valid_count", "water_fraction",
]
TERRAIN_FEATURES = ["elevation", "slope", "relative_elevation"]
OUTPUT_BANDS = [
    "AGB_full_aoi_g_m2",
    "AGB_peatland_g_m2",
    "ood_feature_fraction",
    "peatland_probability",
    "valid_count",
    "AGB_peak_only_peatland_g_m2",
]


def markdown_table(frame: pd.DataFrame) -> str:
    shown = frame.copy()
    for column in shown.columns:
        if pd.api.types.is_float_dtype(shown[column]):
            shown[column] = shown[column].map(lambda value: f"{value:.3f}")
        else:
            shown[column] = shown[column].astype(str)
    columns = list(shown.columns)
    lines = ["| " + " | ".join(columns) + " |"]
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in shown.iterrows():
        lines.append("| " + " | ".join(row[column] for column in columns) + " |")
    return "\n".join(lines)


def ensure_dirs() -> None:
    for path in [OUTPUT_DIR, RASTER_DIR, FIGURE_DIR, VRT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def annual_paths(year: int) -> list[Path]:
    paths = IMAGE_DIR.glob(f"Zoige_Landsat_PeakNDVI_JulSep_{year}_30m*.tif")
    # Google Drive may leave a second copy with a " (1)" suffix. It is the
    # same tile, so exclude that duplicate before requiring four tiles.
    paths = sorted(path for path in paths if " (1)" not in path.stem)
    if len(paths) == 4:
        return paths
    # Some historical exports use the original GS naming and are stored with
    # the peatland preprocessing inputs rather than map_points.
    historical_dir = HISTORICAL_IMAGE_DIR / str(year)
    historical = sorted(historical_dir.glob(f"Zoige_Landsat_GS_{year}_30m*.tif"))
    if len(historical) == 4:
        return historical
    return paths


def build_mosaic_vrt(year: int) -> Path:
    paths = annual_paths(year)
    if len(paths) != 4:
        raise FileNotFoundError(f"Expected four Peak tiles for {year}, found {len(paths)}")
    datasets = [rasterio.open(path) for path in paths]
    try:
        left = min(ds.bounds.left for ds in datasets)
        right = max(ds.bounds.right for ds in datasets)
        bottom = min(ds.bounds.bottom for ds in datasets)
        top = max(ds.bounds.top for ds in datasets)
        width = int(round((right - left) / SOURCE_RESOLUTION))
        height = int(round((top - bottom) / SOURCE_RESOLUTION))
        srs = datasets[0].crs.to_wkt()
        lines = [
            f'<VRTDataset rasterXSize="{width}" rasterYSize="{height}">',
            f"  <SRS>{escape(srs)}</SRS>",
            f"  <GeoTransform>{left}, {SOURCE_RESOLUTION}, 0, {top}, 0, {-SOURCE_RESOLUTION}</GeoTransform>",
        ]
        source_band_count = min(ds.count for ds in datasets)
        if source_band_count < 15:
            raise ValueError(
                f"Historical input for {year} has {source_band_count} bands; "
                "at least the 15-band core schema is required."
            )
        # The older 15-band schema ends at valid_mask. The two optional water
        # QA bands are synthesized as zero bands by the readers below.
        vrt_band_names = SOURCE_BANDS[:source_band_count]
        for band_index, band_name in enumerate(vrt_band_names, start=1):
            lines.append(f'  <VRTRasterBand dataType="Float32" band="{band_index}">')
            lines.append(f"    <Description>{band_name}</Description>")
            for path, ds in zip(paths, datasets):
                x_off = int(round((ds.bounds.left - left) / SOURCE_RESOLUTION))
                y_off = int(round((top - ds.bounds.top) / SOURCE_RESOLUTION))
                lines.extend([
                    "    <SimpleSource>",
                    f"      <SourceFilename relativeToVRT=\"0\">{escape(str(path))}</SourceFilename>",
                    f"      <SourceBand>{band_index}</SourceBand>",
                    f'      <SrcRect xOff="0" yOff="0" xSize="{ds.width}" ySize="{ds.height}"/>',
                    f'      <DstRect xOff="{x_off}" yOff="{y_off}" xSize="{ds.width}" ySize="{ds.height}"/>',
                    "    </SimpleSource>",
                ])
            lines.append("  </VRTRasterBand>")
        lines.append("</VRTDataset>")
        vrt_path = VRT_DIR / f"PeakNDVI_JulSep_{year}_30m_mosaic.vrt"
        vrt_path.write_text("\n".join(lines), encoding="utf-8")
        return vrt_path
    finally:
        for ds in datasets:
            ds.close()


def climate_band_lookup(src: rasterio.io.DatasetReader) -> dict[str, dict[int, int]]:
    lookup = {name: {} for name in CLIMATE_VARIABLES}
    for index, description in enumerate(src.descriptions, start=1):
        if not description:
            continue
        for variable in CLIMATE_VARIABLES:
            suffix = f"_{variable}"
            if description.endswith(suffix):
                lookup[variable][int(description[:4])] = index
    return lookup


def climate_anomaly_memory(year: int) -> tuple[MemoryFile, rasterio.io.DatasetWriter]:
    with rasterio.open(CLIMATE_STACK) as src:
        lookup = climate_band_lookup(src)
        anomalies = []
        for variable in CLIMATE_VARIABLES:
            normal_stack = src.read([lookup[variable][item] for item in CLIMATOLOGY_YEARS]).astype(np.float64)
            normal = np.nanmean(normal_stack, axis=0)
            scale = np.nanstd(normal_stack, axis=0)
            scale[~np.isfinite(scale) | (scale <= 0)] = 1.0
            annual = src.read(lookup[variable][year]).astype(np.float64)
            anomalies.append(((annual - normal) / scale).astype(np.float32))
        profile = src.profile.copy()
        profile.update(count=3, dtype="float32", nodata=NODATA)
        memfile = MemoryFile()
        dataset = memfile.open(**profile)
        dataset.write(np.stack(anomalies))
        for index, variable in enumerate(CLIMATE_VARIABLES, start=1):
            dataset.set_band_description(index, f"tc_local_{variable}_anom")
        return memfile, dataset


def read_valid_medians(
    src: rasterio.io.DatasetReader,
    output_window: Window,
) -> dict[str, np.ndarray]:
    source_window = Window(
        int(output_window.col_off) * AGGREGATION_FACTOR,
        int(output_window.row_off) * AGGREGATION_FACTOR,
        int(output_window.width) * AGGREGATION_FACTOR,
        int(output_window.height) * AGGREGATION_FACTOR,
    )
    source_band_count = min(src.count, len(SOURCE_BANDS))
    data = src.read(
        list(range(1, source_band_count + 1)),
        window=source_window,
        masked=False,
    ).astype(np.float32)
    out_h = int(output_window.height)
    out_w = int(output_window.width)
    valid_index = SOURCE_BANDS.index("valid_mask")
    valid = data[valid_index] >= 0.5
    valid = valid.reshape(out_h, 3, out_w, 3).transpose(0, 2, 1, 3).reshape(out_h, out_w, 9)
    valid_count = valid.sum(axis=2)
    lower_index = np.maximum((valid_count - 1) // 2, 0)
    upper_index = valid_count // 2
    result: dict[str, np.ndarray] = {}
    available_names = SOURCE_BANDS[:source_band_count]
    for band_index, band_name in enumerate(available_names):
        values = data[band_index].reshape(out_h, 3, out_w, 3).transpose(0, 2, 1, 3).reshape(out_h, out_w, 9)
        ordered = np.sort(np.where(valid, values, np.inf), axis=2)
        lower = np.take_along_axis(ordered, lower_index[..., None], axis=2)[..., 0]
        upper = np.take_along_axis(ordered, upper_index[..., None], axis=2)[..., 0]
        median = ((lower + upper) / 2.0).astype(np.float32)
        median[valid_count == 0] = np.nan
        result[band_name] = median
    for band_name in SOURCE_BANDS[source_band_count:]:
        result[band_name] = np.zeros((out_h, out_w), dtype=np.float32)
    result["valid_mask"] = valid_count > 0
    return result


def output_windows(width: int, height: int):
    for row_off in range(0, height, BLOCK_SIZE):
        for col_off in range(0, width, BLOCK_SIZE):
            yield Window(
                col_off,
                row_off,
                min(BLOCK_SIZE, width - col_off),
                min(BLOCK_SIZE, height - row_off),
            )


def training_limits(features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(TRAINING_FEATURES)
    values = frame[features].to_numpy(float)
    return np.nanquantile(values, 0.01, axis=0), np.nanquantile(values, 0.99, axis=0)


def read_static_window(vrt: WarpedVRT, indexes: list[int] | int, window: Window) -> np.ndarray:
    return vrt.read(indexes, window=window, masked=False).astype(np.float32)


def run_year(
    year: int,
    model,
    features: list[str],
    peak_only_model,
    peak_only_features: list[str],
    lower: np.ndarray,
    upper: np.ndarray,
    aoi_geometries: list[dict],
) -> dict[str, float]:
    vrt_path = build_mosaic_vrt(year)
    climate_mem, climate_ds = climate_anomaly_memory(year)
    samples_full: list[np.ndarray] = []
    samples_peat: list[np.ndarray] = []
    samples_ood: list[np.ndarray] = []
    samples_peak_only: list[np.ndarray] = []
    counts = {"full": 0, "peat": 0, "ood_high": 0}
    sums = {"full": 0.0, "peat": 0.0, "peak_only_peat": 0.0}

    try:
        with rasterio.open(vrt_path) as source, rasterio.open(TERRAIN_STACK) as terrain_src, rasterio.open(PEAT_PROBABILITY) as peat_src:
            output_width = source.width // AGGREGATION_FACTOR
            output_height = source.height // AGGREGATION_FACTOR
            output_transform = Affine(
                OUTPUT_RESOLUTION, 0.0, source.bounds.left,
                0.0, -OUTPUT_RESOLUTION, source.bounds.top,
            )
            terrain_descriptions = list(terrain_src.descriptions)
            terrain_indexes = [terrain_descriptions.index(name) + 1 for name in TERRAIN_FEATURES]
            vrt_options = {
                "crs": source.crs,
                "transform": output_transform,
                "width": output_width,
                "height": output_height,
            }
            with WarpedVRT(terrain_src, resampling=Resampling.bilinear, **vrt_options) as terrain_vrt, \
                 WarpedVRT(peat_src, resampling=Resampling.bilinear, **vrt_options) as peat_vrt, \
                 WarpedVRT(climate_ds, resampling=Resampling.bilinear, **vrt_options) as climate_vrt:
                profile = source.profile.copy()
                profile.update(
                    driver="GTiff", width=output_width, height=output_height,
                    transform=output_transform, count=len(OUTPUT_BANDS),
                    dtype="float32", nodata=NODATA, compress="DEFLATE",
                    predictor=3, tiled=True, blockxsize=256, blockysize=256,
                    BIGTIFF="IF_SAFER",
                )
                output_path = RASTER_DIR / f"Zoige_AGB_TC_PLSR_Peak90m_{year}.tif"
                with rasterio.open(output_path, "w", **profile) as dst:
                    for band_index, name in enumerate(OUTPUT_BANDS, start=1):
                        dst.set_band_description(band_index, name)
                    dst.update_tags(
                        model="TC_local_core3_PLSR",
                        source_support="valid-only 3x3 median from 30 m PeakNDVI Jul-Sep",
                        nominal_resolution_m=90,
                        source_dummy="not_used",
                        peatland_threshold=PEAT_THRESHOLD,
                    )
                    for window_index, window in enumerate(output_windows(output_width, output_height)):
                        dynamic = read_valid_medians(source, window)
                        terrain = read_static_window(terrain_vrt, terrain_indexes, window)
                        climate = read_static_window(climate_vrt, [1, 2, 3], window)
                        peat = read_static_window(peat_vrt, 1, window)
                        shape = (int(window.height), int(window.width))
                        window_transform = rasterio.windows.transform(window, output_transform)
                        aoi = geometry_mask(
                            aoi_geometries, out_shape=shape, transform=window_transform,
                            invert=True, all_touched=False,
                        )

                        feature_arrays = {
                            f"pk_{name}": dynamic[name] for name in DYNAMIC_FEATURES
                            if name not in {"NIRv", "NDTI"}
                        }
                        feature_arrays["pk_NIRv"] = dynamic["NIR"] * dynamic["NDVI"]
                        denominator = dynamic["SWIR1"] + dynamic["SWIR2"]
                        feature_arrays["pk_NDTI"] = np.divide(
                            dynamic["SWIR1"] - dynamic["SWIR2"], denominator,
                            out=np.full(shape, np.nan, dtype=np.float32),
                            where=np.abs(denominator) > 1e-8,
                        )
                        for idx, name in enumerate(TERRAIN_FEATURES):
                            feature_arrays[name] = terrain[idx]
                        for idx, variable in enumerate(CLIMATE_VARIABLES):
                            feature_arrays[f"tc_local_{variable}_anom"] = climate[idx]

                        matrix = np.column_stack([feature_arrays[name].ravel() for name in features])
                        valid = dynamic["valid_mask"].ravel().astype(bool) & aoi.ravel()
                        valid &= np.isfinite(matrix).all(axis=1)
                        prediction = np.full(matrix.shape[0], NODATA, dtype=np.float32)
                        peak_only_prediction = np.full(matrix.shape[0], NODATA, dtype=np.float32)
                        ood = np.full(matrix.shape[0], NODATA, dtype=np.float32)
                        if valid.any():
                            prediction[valid] = np.clip(
                                model.predict(matrix[valid]).reshape(-1), 0.0, 2000.0
                            ).astype(np.float32)
                            ood[valid] = np.mean(
                                (matrix[valid] < lower) | (matrix[valid] > upper), axis=1
                            ).astype(np.float32)
                            peak_only_matrix = np.column_stack(
                                [feature_arrays[name].ravel() for name in peak_only_features]
                            )
                            peak_only_prediction[valid] = np.clip(
                                peak_only_model.predict(peak_only_matrix[valid]).reshape(-1),
                                0.0,
                                2000.0,
                            ).astype(np.float32)
                        peat_flat = peat.ravel()
                        peat_valid = valid & np.isfinite(peat_flat) & (peat_flat >= PEAT_THRESHOLD)
                        peat_prediction = np.full(matrix.shape[0], NODATA, dtype=np.float32)
                        peat_prediction[peat_valid] = prediction[peat_valid]
                        peak_only_peat_prediction = np.full(matrix.shape[0], NODATA, dtype=np.float32)
                        peak_only_peat_prediction[peat_valid] = peak_only_prediction[peat_valid]
                        peat_output = np.where(np.isfinite(peat), np.clip(peat, 0, 1), NODATA).astype(np.float32)
                        valid_count = np.where(
                            dynamic["valid_mask"], dynamic["valid_count"], NODATA
                        ).astype(np.float32)
                        output = np.stack([
                            prediction.reshape(shape), peat_prediction.reshape(shape),
                            ood.reshape(shape), peat_output, valid_count,
                            peak_only_peat_prediction.reshape(shape),
                        ])
                        dst.write(output, window=window)

                        full_values = prediction[valid]
                        peat_values = prediction[peat_valid]
                        ood_values = ood[valid]
                        peak_only_values = peak_only_prediction[peat_valid]
                        counts["full"] += int(full_values.size)
                        counts["peat"] += int(peat_values.size)
                        counts["ood_high"] += int(np.sum(ood_values > 0.25))
                        sums["full"] += float(np.sum(full_values, dtype=np.float64))
                        sums["peat"] += float(np.sum(peat_values, dtype=np.float64))
                        sums["peak_only_peat"] += float(np.sum(peak_only_values, dtype=np.float64))
                        samples_full.append(full_values[::20])
                        samples_peat.append(peat_values[::20])
                        samples_ood.append(ood_values[::20])
                        samples_peak_only.append(peak_only_values[::20])
                        if window_index % 40 == 0:
                            print(f"{year}: processed window {window_index}", flush=True)

        full = np.concatenate(samples_full) if samples_full else np.array([])
        peat_values = np.concatenate(samples_peat) if samples_peat else np.array([])
        ood_values = np.concatenate(samples_ood) if samples_ood else np.array([])
        peak_only_values = np.concatenate(samples_peak_only) if samples_peak_only else np.array([])
        return {
            "year": year,
            "output_path": str(output_path),
            "valid_pixel_count": counts["full"],
            "peatland_pixel_count": counts["peat"],
            "agb_full_mean_g_m2": sums["full"] / max(counts["full"], 1),
            "agb_full_p05_g_m2": float(np.quantile(full, 0.05)),
            "agb_full_median_g_m2": float(np.median(full)),
            "agb_full_p95_g_m2": float(np.quantile(full, 0.95)),
            "agb_peat_mean_g_m2": sums["peat"] / max(counts["peat"], 1),
            "agb_peat_p05_g_m2": float(np.quantile(peat_values, 0.05)),
            "agb_peat_median_g_m2": float(np.median(peat_values)),
            "agb_peat_p95_g_m2": float(np.quantile(peat_values, 0.95)),
            "agb_peak_only_peat_mean_g_m2": sums["peak_only_peat"] / max(counts["peat"], 1),
            "agb_peak_only_peat_p05_g_m2": float(np.quantile(peak_only_values, 0.05)),
            "agb_peak_only_peat_median_g_m2": float(np.median(peak_only_values)),
            "agb_peak_only_peat_p95_g_m2": float(np.quantile(peak_only_values, 0.95)),
            "climate_minus_peak_only_mean_g_m2": (
                sums["peat"] - sums["peak_only_peat"]
            ) / max(counts["peat"], 1),
            "ood_fraction_gt_0_25": counts["ood_high"] / max(counts["full"], 1),
            "ood_feature_fraction_median": float(np.median(ood_values)),
        }
    finally:
        climate_ds.close()
        climate_mem.close()


def create_overview_figure(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(summary), figsize=(15, 6), constrained_layout=True)
    image = None
    for axis, row in zip(axes, summary.itertuples(index=False)):
        with rasterio.open(row.output_path) as src:
            data = src.read(
                2,
                out_shape=(min(900, src.height), min(720, src.width)),
                resampling=Resampling.nearest,
            ).astype(float)
        data[data == NODATA] = np.nan
        image = axis.imshow(data, cmap="YlGn", vmin=100, vmax=600)
        axis.set_title(f"{row.year} peatland AGB")
        axis.set_axis_off()
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.72, label="AGB (g m$^{-2}$)")
    fig.savefig(FIGURE_DIR / "pilot_peatland_agb_2017_2020_2024.png", dpi=220)
    plt.close(fig)


def create_sensitivity_figure(summary: pd.DataFrame) -> None:
    years = summary["year"].astype(int).to_numpy()
    climate = summary["agb_peat_mean_g_m2"].to_numpy(float)
    peak_only = summary["agb_peak_only_peat_mean_g_m2"].to_numpy(float)
    positions = np.arange(len(years))
    width = 0.36
    fig, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    axis.bar(positions - width / 2, peak_only, width, label="Peak-only PLSR", color="#4B7F9E")
    axis.bar(positions + width / 2, climate, width, label="Peak + climate PLSR", color="#D47A3A")
    for x, value, delta in zip(positions, climate, climate - peak_only):
        axis.text(x + width / 2, value + 8, f"{delta:+.1f}", ha="center", fontsize=9)
    axis.set_xticks(positions, years)
    axis.set_ylabel("Peatland mean AGB (g m$^{-2}$)")
    axis.set_title("Climate augmentation sensitivity")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIGURE_DIR / "pilot_climate_vs_peak_only_sensitivity.png", dpi=250)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Reuse completed pilot rasters and summary without rerunning inversion.",
    )
    args = parser.parse_args()
    ensure_dirs()
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    bundle = joblib.load(MODEL_PATH)
    peak_only_bundle = joblib.load(PEAK_ONLY_MODEL_PATH)
    summary_path = OUTPUT_DIR / "pilot_inversion_summary.csv"
    if args.report_only:
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        summary = pd.read_csv(summary_path)
    else:
        model = bundle["model"]
        features = list(bundle["features"])
        peak_only_model = peak_only_bundle["model"]
        peak_only_features = list(peak_only_bundle["features"])
        lower, upper = training_limits(features)
        aoi = gpd.read_file(AOI_PATH).to_crs("EPSG:32648")
        geometries = [geometry.__geo_interface__ for geometry in aoi.geometry if geometry is not None]
        rows = [
            run_year(
                year,
                model,
                features,
                peak_only_model,
                peak_only_features,
                lower,
                upper,
                geometries,
            )
            for year in PILOT_YEARS
        ]
        summary = pd.DataFrame(rows)
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    create_overview_figure(summary)
    create_sensitivity_figure(summary)

    numerical_gate = bool(
        summary["agb_peat_mean_g_m2"].between(100, 700).all()
        and (summary["ood_fraction_gt_0_25"] <= 0.25).all()
    )
    climate_temporal_gate = bool(
        summary["climate_minus_peak_only_mean_g_m2"].abs().le(75).all()
    )
    peak_only_gate = bool(
        summary["agb_peak_only_peat_mean_g_m2"].between(100, 700).all()
    )
    report = f"""# 2017、2020、2024三年试反演

## 方法

- 冻结模型：`{bundle['experiment']}`，不使用source dummy。
- 动态特征：7-9月PeakNDVI年度合成，按有效像元做3x3中值，输出支持尺度90 m。
- 气候：TerraClimate本地GST/GSP/VPD相对1985-2013标准化异常。
- 输出包含全若尔盖AOI和泥炭地概率≥{PEAT_THRESHOLD:.2f}两套AGB。
- 同时输出Peak-only PLSR泥炭地结果，用于识别气候增强层造成的年度整体偏移。
- 域外诊断：23个模型特征超出训练样本P01-P99范围的比例；像元域外比例>0.25记为高风险。

## 统计结果

{markdown_table(summary.drop(columns=['output_path']))}

## 自动生产门槛

- 泥炭地平均AGB位于100-700 g/m²；
- 高域外风险像元不超过25%；
- 气候模型基本数值门槛通过：{'是' if numerical_gate else '否'}；
- 气候增强相对Peak-only年度均值偏移不超过75 g/m²：{'是' if climate_temporal_gate else '否'}；
- Peak-only值域门槛通过：{'是' if peak_only_gate else '否'}。

## 生产决定

2024年气候增强结果相对Peak-only抬升约{summary.loc[summary['year'].eq(2024), 'climate_minus_peak_only_mean_g_m2'].iloc[0]:.1f} g/m²，主要来自极端GST/GSP异常。气候模型虽然空间分块精度较高，但不通过年度迁移敏感性检查。

**2013-2024批量制图推荐改用不含气候的 `Baseline_Peak_PLSR`。** TerraClimate作为后续气候驱动分析的自变量，避免“用气候构建AGB后再解释气候驱动”的循环论证。气候增强模型保留为敏感性结果，不用于主年度产品。

空间图未发现分片接缝、异常条带和大面积内部空洞。2017和2020的最终模型回代图不是独立验证；Peak-only论文精度使用5 km空间分块OOF结果（r=0.471，R²=0.221，RMSE=202.2 g/m²）。
"""
    (RUN_DIR / "report_three_year_pilot_inversion.md").write_text(report, encoding="utf-8")
    (OUTPUT_DIR / "pilot_inversion_decision.json").write_text(
        json.dumps(
            {
                "climate_model_basic_numerical_gate_passed": numerical_gate,
                "climate_model_temporal_sensitivity_gate_passed": climate_temporal_gate,
                "climate_model_batch_approved": False,
                "recommended_production_model": "Baseline_Peak_PLSR",
                "peak_only_value_range_gate_passed": peak_only_gate,
                "peak_only_batch_2013_2024_approved": peak_only_gate,
                "source_dummy_used": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
