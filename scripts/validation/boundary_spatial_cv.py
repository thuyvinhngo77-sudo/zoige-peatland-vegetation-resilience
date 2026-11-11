"""Freeze peatland-boundary validation and downstream mask sensitivity.

This analysis does not rebuild the published boundary. It adds spatially
blocked out-of-fold validation for the existing training samples and tests
whether the long-term AGB conclusion changes across conservative mask choices.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Geod, Transformer
from rasterio.warp import Resampling, reproject
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "agb_paper_validation_freeze" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BOUNDARY_DIR = ROOT / "zoigepeatland" / "output_basic_boundary"
SAMPLE_FILE = BOUNDARY_DIR / "Zoige_RF_basic_samples_with_features.csv"
AGB_DIR = ROOT / "agb_dual_grid_2013_2024" / "rasters_90m"

YEARS = list(range(1988, 2025))
BLOCK_SIZE_M = 25_000
N_SPLITS = 5
RANDOM_STATE = 20260622

BAND_NAMES = [
    "NDVI_early", "NDVI_peak", "NDVI_late",
    "NDMI_early", "NDMI_peak", "NDMI_late",
    "NDWI_peak", "MNDWI_peak", "LSWI_peak", "EVI_peak",
    "NDVI_max", "NDVI_min", "NDVI_amp",
    "NDMI_max", "NDMI_min", "NDMI_amp",
    "VV_wet", "VH_wet", "VV_minus_VH_wet", "RVI_wet",
    "VV_dry", "VH_dry", "VV_minus_VH_dry", "RVI_dry",
    "elevation", "slope", "relative_elevation",
]

RF_PARAMS = {
    "n_estimators": 300,
    "max_features": "sqrt",
    "min_samples_leaf": 1,
    "bootstrap": True,
    "class_weight": None,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

MASKS = {
    "prob95_raw": BOUNDARY_DIR / "Zoige_RF_basic_main_mask_prob95.tif",
    "prob95_clean": BOUNDARY_DIR / "Zoige_RF_basic_main_mask_prob95_clean.tif",
    "analysis_main95_fill400": (
        BOUNDARY_DIR / "Zoige_RF_basic_analysis_mask_main95_fill400.tif"
    ),
    "core98_refined": BOUNDARY_DIR / "Zoige_RF_basic_core_mask_prob98_refined.tif",
}


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, prob=None) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    result = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
        "nonpeat_precision": float(precision[0]),
        "nonpeat_recall": float(recall[0]),
        "nonpeat_f1": float(f1[0]),
        "nonpeat_support": int(support[0]),
        "peat_precision": float(precision[1]),
        "peat_recall": float(recall[1]),
        "peat_f1": float(f1[1]),
        "peat_support": int(support[1]),
    }
    result["auc"] = (
        float(roc_auc_score(y_true, prob)) if prob is not None else np.nan
    )
    return result


def run_spatial_cv(samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32648", always_xy=True)
    east, north = transformer.transform(samples["lon"].values, samples["lat"].values)
    block_x = np.floor(np.asarray(east) / BLOCK_SIZE_M).astype(int)
    block_y = np.floor(np.asarray(north) / BLOCK_SIZE_M).astype(int)
    groups = np.array([f"{x}_{y}" for x, y in zip(block_x, block_y)])

    X = samples[BAND_NAMES].to_numpy(dtype="float32")
    y = samples["class"].to_numpy(dtype="int32")
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )

    oof_prob = np.full(len(samples), np.nan, dtype="float64")
    oof_pred = np.full(len(samples), -1, dtype="int16")
    fold_rows = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(X, y, groups=groups), start=1
    ):
        model = RandomForestClassifier(**RF_PARAMS)
        model.fit(X[train_idx], y[train_idx])
        class_index = list(model.classes_).index(1)
        prob = model.predict_proba(X[test_idx])[:, class_index]
        pred = (prob >= 0.5).astype("int16")
        oof_prob[test_idx] = prob
        oof_pred[test_idx] = pred
        metrics = classification_metrics(y[test_idx], pred, prob)
        metrics.update(
            {
                "validation": "spatial_block_25km",
                "fold": fold,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "n_train_blocks": int(len(np.unique(groups[train_idx]))),
                "n_test_blocks": int(len(np.unique(groups[test_idx]))),
            }
        )
        fold_rows.append(metrics)

    overall = classification_metrics(y, oof_pred, oof_prob)
    overall.update(
        {
            "validation": "spatial_block_25km_oof",
            "fold": "all",
            "n_train": np.nan,
            "n_test": int(len(y)),
            "n_train_blocks": np.nan,
            "n_test_blocks": int(len(np.unique(groups))),
        }
    )
    fold_rows.append(overall)

    source_column = "source" if "source" in samples.columns else "source_type"
    predictions = samples[
        ["sample_id", "lon", "lat", "class", source_column]
    ].copy()
    predictions = predictions.rename(columns={source_column: "source"})
    predictions["spatial_block_25km"] = groups
    predictions["oof_probability"] = oof_prob
    predictions["oof_class"] = oof_pred
    predictions["correct"] = predictions["class"] == predictions["oof_class"]
    return pd.DataFrame(fold_rows), predictions


def sample_mask_values(samples: pd.DataFrame, mask_path: Path) -> np.ndarray:
    with rasterio.open(mask_path) as src:
        xs = samples["lon"].to_numpy(dtype=float)
        ys = samples["lat"].to_numpy(dtype=float)
        if src.crs and src.crs.to_string() != "EPSG:4326":
            transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            xs, ys = transformer.transform(xs, ys)
        values = np.array([v[0] for v in src.sample(zip(xs, ys))], dtype=float)
        if src.nodata is not None:
            values[values == src.nodata] = 0
    return (values > 0).astype("int16")


def mask_area_km2(mask_path: Path) -> float:
    with rasterio.open(mask_path) as src:
        data = src.read(1, masked=True).filled(0) > 0
        if src.crs and src.crs.is_projected:
            pixel_area = abs(src.transform.a * src.transform.e)
            return float(data.sum() * pixel_area / 1e6)
        if src.crs and src.crs.is_geographic:
            geod = Geod(ellps="WGS84")
            total_m2 = 0.0
            for row in range(src.height):
                count = int(data[row].sum())
                if count == 0:
                    continue
                x0 = src.transform.c
                x1 = x0 + src.transform.a
                y0 = src.transform.f + src.transform.e * row
                y1 = src.transform.f + src.transform.e * (row + 1)
                area_m2, _ = geod.polygon_area_perimeter(
                    [x0, x1, x1, x0], [y0, y0, y1, y1]
                )
                total_m2 += count * abs(area_m2)
            return float(total_m2 / 1e6)
        raise ValueError(f"Mask has no usable CRS: {mask_path}")


def reproject_mask(mask_path: Path, reference) -> np.ndarray:
    destination = np.zeros((reference.height, reference.width), dtype="uint8")
    with rasterio.open(mask_path) as src:
        source = (src.read(1) > 0).astype("uint8")
        reproject(
            source=source,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=0,
            dst_transform=reference.transform,
            dst_crs=reference.crs,
            dst_nodata=0,
            resampling=Resampling.nearest,
        )
    return destination.astype(bool)


def valid_band(src, band: int) -> np.ndarray:
    arr = src.read(band, masked=True).astype("float32").filled(np.nan)
    arr[~np.isfinite(arr)] = np.nan
    return arr


def run_agb_mask_sensitivity() -> tuple[pd.DataFrame, pd.DataFrame]:
    first_path = AGB_DIR / "Zoige_AGB_GuardedConsensus_PLSR_90m_1988.tif"
    with rasterio.open(first_path) as reference:
        masks = {name: reproject_mask(path, reference) for name, path in MASKS.items()}

    annual_rows = []
    for year in YEARS:
        raster_path = AGB_DIR / f"Zoige_AGB_GuardedConsensus_PLSR_90m_{year}.tif"
        if not raster_path.exists():
            raise FileNotFoundError(raster_path)
        with rasterio.open(raster_path) as src:
            guarded = valid_band(src, 1)
            peak = valid_band(src, 3)
        for mask_name, spatial_mask in masks.items():
            for product, data in [("guarded", guarded), ("peak_only", peak)]:
                valid = spatial_mask & np.isfinite(data)
                values = data[valid]
                annual_rows.append(
                    {
                        "mask": mask_name,
                        "year": year,
                        "product": product,
                        "mask_pixel_count": int(spatial_mask.sum()),
                        "valid_pixel_count": int(valid.sum()),
                        "valid_coverage_fraction": float(valid.sum() / spatial_mask.sum()),
                        "mean_g_m2": float(np.mean(values)),
                        "p05_g_m2": float(np.percentile(values, 5)),
                        "p50_g_m2": float(np.percentile(values, 50)),
                        "p95_g_m2": float(np.percentile(values, 95)),
                    }
                )

    annual = pd.DataFrame(annual_rows)
    trend_rows = []
    for (mask_name, product), group in annual.groupby(["mask", "product"]):
        group = group.sort_values("year")
        slope, intercept, low, high = stats.theilslopes(
            group["mean_g_m2"], group["year"], alpha=0.95
        )
        tau, p_value = stats.kendalltau(group["year"], group["mean_g_m2"])
        trend_rows.append(
            {
                "mask": mask_name,
                "mask_area_km2_native": mask_area_km2(MASKS[mask_name]),
                "product": product,
                "n_years": int(len(group)),
                "sen_slope_g_m2_yr": float(slope),
                "sen_low95_g_m2_yr": float(low),
                "sen_high95_g_m2_yr": float(high),
                "kendall_tau": float(tau),
                "kendall_p_value": float(p_value),
                "minimum_coverage_fraction": float(group["valid_coverage_fraction"].min()),
                "mean_coverage_fraction": float(group["valid_coverage_fraction"].mean()),
                "direction": "increase" if slope > 0 else "decrease" if slope < 0 else "stable",
            }
        )
    return annual, pd.DataFrame(trend_rows)


def main() -> None:
    samples = pd.read_csv(SAMPLE_FILE)
    samples = samples.loc[samples["valid_features"].astype(bool)].copy()
    required = ["sample_id", "lon", "lat", "class", *BAND_NAMES]
    missing = [column for column in required if column not in samples.columns]
    if missing:
        raise ValueError(f"Missing sample columns: {missing}")

    fold_metrics, predictions = run_spatial_cv(samples)
    fold_metrics.to_csv(OUT_DIR / "boundary_spatial_block_metrics.csv", index=False)
    predictions.to_csv(OUT_DIR / "boundary_spatial_block_predictions.csv", index=False)

    mask_rows = []
    y_true = samples["class"].to_numpy(dtype="int16")
    for name, path in MASKS.items():
        y_pred = sample_mask_values(samples, path)
        metrics = classification_metrics(y_true, y_pred)
        metrics.update(
            {
                "mask": name,
                "area_km2": mask_area_km2(path),
                "diagnostic_role": "all-sample map inclusion; not independent validation",
            }
        )
        mask_rows.append(metrics)
    pd.DataFrame(mask_rows).to_csv(
        OUT_DIR / "boundary_mask_sample_inclusion_diagnostics.csv", index=False
    )

    annual, trends = run_agb_mask_sensitivity()
    annual.to_csv(OUT_DIR / "boundary_mask_agb_annual_sensitivity.csv", index=False)
    trends.to_csv(OUT_DIR / "boundary_mask_agb_trend_sensitivity.csv", index=False)

    metadata = {
        "spatial_block_size_m": BLOCK_SIZE_M,
        "spatial_cv_folds": N_SPLITS,
        "sample_count": int(len(samples)),
        "mask_names": list(MASKS),
        "years": [YEARS[0], YEARS[-1]],
        "notes": [
            "Spatial CV refits the RF in each fold and predicts only held-out 25 km blocks.",
            "Mask inclusion metrics use all samples and are diagnostics, not independent validation.",
            "AGB trend sensitivity uses the frozen 90 m products without refitting AGB models.",
        ],
    }
    (OUT_DIR / "boundary_validation_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    overall = fold_metrics.loc[fold_metrics["fold"].astype(str) == "all"].iloc[0]
    print(
        "Spatial OOF:",
        f"accuracy={overall['accuracy']:.3f}",
        f"balanced_accuracy={overall['balanced_accuracy']:.3f}",
        f"AUC={overall['auc']:.3f}",
        f"peat_F1={overall['peat_f1']:.3f}",
    )
    print(trends[["mask", "product", "sen_slope_g_m2_yr", "direction"]].to_string(index=False))


if __name__ == "__main__":
    main()
