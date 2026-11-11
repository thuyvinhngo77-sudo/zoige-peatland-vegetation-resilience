"""Produce and independently validate the frozen 2025 Peak-only AGB product."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import joblib
import matplotlib
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform as transform_coordinates
from rasterio.windows import Window


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._internal import peak_only_batch as peak_batch  # noqa: E402
from scripts._internal import annual_agb_helpers as pilot  # noqa: E402


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

YEAR = 2025
RUN_DIR = ROOT / "agb_2025_independent_validation"
OUTPUT_DIR = RUN_DIR / "outputs"
RASTER_DIR = RUN_DIR / "rasters"
FIGURE_DIR = RUN_DIR / "figures"
VRT_DIR = RUN_DIR / "virtual_inputs"
REPORT_PATH = RUN_DIR / "report_2025_independent_validation.md"

MODEL_PATH = (
    ROOT
    / "agb_ecological_weighted_validation"
    / "models"
    / "ecological_weighted_candidate.joblib"
)
TRAINING_FEATURES = (
    ROOT
    / "agb_ecological_weighted_validation"
    / "outputs"
    / "climate_augmented_sample_features.csv"
)
FIELD_PATH = (
    ROOT
    / "agb_field_data"
    / "field_2025_qtp"
    / "qtp_2025_dry_agb_zoige_aoi_training_points.csv"
)
MAIN95_PATH = (
    ROOT
    / "zoigepeatland"
    / "output_basic_boundary"
    / "Zoige_RF_basic_analysis_mask_main95_fill400.tif"
)

SUPPORTS = {
    "source_center_30m": 0,
    "source_center_3x3_90m": 1,
    "source_center_5x5_150m": 2,
}
PRIMARY_SUPPORT = "grid_product_90m"
PRIMARY_QC_FLAG = "pass_basic_checks"
PRIMARY_SCENARIO = "quality_controlled_primary"
ALL_RECORDS_SCENARIO = "all_records_sensitivity"
NODATA = pilot.NODATA

# These gates are fixed before viewing the 2025 predictions. Passing them is
# a numerical screen, not automatic production approval.
SCREENING_GATES = {
    "minimum_valid_records": 12,
    "minimum_pearson_r": 0.40,
    "minimum_predictive_r_squared": 0.0,
    "maximum_rmse_g_m2": 250.0,
    "maximum_absolute_bias_g_m2": 100.0,
}


def ensure_dirs() -> None:
    for path in [OUTPUT_DIR, RASTER_DIR, FIGURE_DIR, VRT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def check_inputs() -> dict[str, object]:
    image_paths = sorted(
        path
        for path in pilot.IMAGE_DIR.glob(
            "Zoige_Landsat_PeakNDVI_JulSep_2025_30m*.tif"
        )
        if " (1)" not in path.stem and "PreSurvey" not in path.stem
    )
    required = [MODEL_PATH, TRAINING_FEATURES, FIELD_PATH, MAIN95_PATH]
    missing = [str(path) for path in required if not path.exists()]
    return {
        "year": YEAR,
        "peak_tile_count": len(image_paths),
        "peak_tiles": [str(path) for path in image_paths],
        "required_peak_tile_count": 4,
        "missing_required_files": missing,
        "ready": len(image_paths) == 4 and not missing,
        "legacy_gs_input_accepted": False,
    }


def sample_raster_values(path: Path, points: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    with rasterio.open(path) as src:
        xs, ys = transform_coordinates(
            "EPSG:4326",
            src.crs,
            points["longitude"].astype(float).tolist(),
            points["latitude"].astype(float).tolist(),
        )
        values = list(src.sample(zip(xs, ys), masked=False))
        names = [description or f"band_{index}" for index, description in enumerate(src.descriptions, 1)]
        for value in values:
            record: dict[str, float] = {}
            for name, item in zip(names, value):
                number = float(item)
                record[name] = np.nan if number == src.nodata else number
            rows.append(record)
    return pd.DataFrame(rows)


def source_window_medians(
    source: rasterio.io.DatasetReader,
    row: int,
    col: int,
    radius: int,
) -> tuple[dict[str, float], int]:
    size = 2 * radius + 1
    data = source.read(
        list(range(1, min(source.count, len(pilot.SOURCE_BANDS)) + 1)),
        window=Window(col - radius, row - radius, size, size),
        boundless=True,
        fill_value=NODATA,
        masked=False,
    ).astype(np.float64)
    valid_index = pilot.SOURCE_BANDS.index("valid_mask")
    valid = data[valid_index] >= 0.5
    valid &= np.isfinite(data[valid_index])
    valid_count = int(valid.sum())
    result: dict[str, float] = {}
    for index, name in enumerate(pilot.SOURCE_BANDS[: data.shape[0]]):
        values = data[index][valid]
        values = values[np.isfinite(values) & (values != NODATA)]
        result[name] = float(np.median(values)) if values.size else np.nan
    for name in pilot.SOURCE_BANDS[data.shape[0] :]:
        result[name] = 0.0
    return result, valid_count


def source_feature_record(
    dynamic: dict[str, float],
    terrain: np.ndarray,
) -> dict[str, float]:
    arrays = {
        f"pk_{name}": float(dynamic[name])
        for name in pilot.DYNAMIC_FEATURES
        if name not in {"NIRv", "NDTI"}
    }
    arrays["pk_NIRv"] = float(dynamic["NIR"] * dynamic["NDVI"])
    denominator = dynamic["SWIR1"] + dynamic["SWIR2"]
    arrays["pk_NDTI"] = (
        float((dynamic["SWIR1"] - dynamic["SWIR2"]) / denominator)
        if np.isfinite(denominator) and abs(denominator) > 1e-8
        else np.nan
    )
    for index, name in enumerate(pilot.TERRAIN_FEATURES):
        arrays[name] = float(terrain[index])
    return arrays


def extract_centered_predictions(
    points: pd.DataFrame,
    model,
    features: list[str],
    lower: np.ndarray,
    upper: np.ndarray,
) -> pd.DataFrame:
    vrt_path = pilot.build_mosaic_vrt(YEAR)
    records: list[dict[str, object]] = []
    with rasterio.open(vrt_path) as source, rasterio.open(pilot.TERRAIN_STACK) as terrain_src:
        descriptions = list(terrain_src.descriptions)
        terrain_indexes = [descriptions.index(name) + 1 for name in pilot.TERRAIN_FEATURES]
        options = {
            "crs": source.crs,
            "transform": source.transform,
            "width": source.width,
            "height": source.height,
        }
        xs, ys = transform_coordinates(
            "EPSG:4326",
            source.crs,
            points["longitude"].astype(float).tolist(),
            points["latitude"].astype(float).tolist(),
        )
        with WarpedVRT(terrain_src, resampling=Resampling.bilinear, **options) as terrain_vrt:
            for (_, point), x, y in zip(points.iterrows(), xs, ys):
                source_row, source_col = source.index(x, y)
                terrain = terrain_vrt.read(
                    terrain_indexes,
                    window=Window(source_col, source_row, 1, 1),
                )[:, 0, 0]
                for support, radius in SUPPORTS.items():
                    dynamic, window_valid_count = source_window_medians(
                        source, source_row, source_col, radius
                    )
                    feature_values = source_feature_record(dynamic, terrain)
                    matrix = np.asarray([feature_values[name] for name in features], dtype=float)
                    is_valid = bool(window_valid_count > 0 and np.isfinite(matrix).all())
                    prediction = (
                        float(np.clip(model.predict(matrix.reshape(1, -1))[0], 0.0, 2000.0))
                        if is_valid
                        else np.nan
                    )
                    ood_fraction = (
                        float(np.mean((matrix < lower) | (matrix > upper)))
                        if is_valid
                        else np.nan
                    )
                    records.append(
                        {
                            "sample_id": point["sample_id"],
                            "support": support,
                            "window_size_pixels": 2 * radius + 1,
                            "support_scale_m": 30 * (2 * radius + 1),
                            "window_valid_pixel_count": window_valid_count,
                            "predicted_agb_g_m2": prediction,
                            "ood_feature_fraction": ood_fraction,
                        }
                    )
    return pd.DataFrame(records)


def metric_values(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[valid]
    predicted = predicted[valid]
    n = int(observed.size)
    if n == 0:
        return {
            "n": 0,
            **{key: np.nan for key in [
            "pearson_r", "spearman_r", "predictive_r_squared",
            "rmse_g_m2", "mae_g_m2", "bias_g_m2", "absolute_bias_g_m2",
            "regression_slope", "regression_intercept",
            ]},
        }
    residual = predicted - observed
    denominator = float(np.sum((observed - observed.mean()) ** 2))
    pearson = float(np.corrcoef(observed, predicted)[0, 1]) if n >= 3 and np.std(observed) > 0 and np.std(predicted) > 0 else np.nan
    spearman = float(pd.Series(observed).corr(pd.Series(predicted), method="spearman")) if n >= 3 else np.nan
    slope, intercept = (np.polyfit(observed, predicted, 1) if n >= 2 and np.std(observed) > 0 else (np.nan, np.nan))
    return {
        "n": n,
        "pearson_r": pearson,
        "spearman_r": spearman,
        "predictive_r_squared": float(1.0 - np.sum(residual**2) / denominator) if denominator > 0 else np.nan,
        "rmse_g_m2": float(np.sqrt(np.mean(residual**2))),
        "mae_g_m2": float(np.mean(np.abs(residual))),
        "bias_g_m2": float(np.mean(residual)),
        "absolute_bias_g_m2": float(abs(np.mean(residual))),
        "regression_slope": float(slope),
        "regression_intercept": float(intercept),
    }


def calculate_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    primary_qc = predictions["independent_qc_pass"].fillna(False).astype(bool)
    scenarios = {
        PRIMARY_SCENARIO: primary_qc,
        ALL_RECORDS_SCENARIO: np.ones(len(predictions), dtype=bool),
        "strict_main95_subgroup": primary_qc
        & predictions["inside_main95"].fillna(False).astype(bool),
    }
    for support in [PRIMARY_SUPPORT, *SUPPORTS]:
        subset_support = predictions["support"].eq(support)
        for scenario, scenario_mask in scenarios.items():
            subset = predictions.loc[subset_support & scenario_mask].copy()
            record_metrics = metric_values(
                subset["agb_mean_g_m2"].to_numpy(float),
                subset["predicted_agb_g_m2"].to_numpy(float),
            )
            rows.append({"support": support, "scenario": scenario, "weighting": "record_level", **record_metrics})
            site = subset.groupby("same_site_group", as_index=False).agg(
                observed=("agb_mean_g_m2", "mean"),
                predicted=("predicted_agb_g_m2", "mean"),
            )
            site_metrics = metric_values(site["observed"].to_numpy(float), site["predicted"].to_numpy(float))
            rows.append({"support": support, "scenario": scenario, "weighting": "site_equal_weight", **site_metrics})
    return pd.DataFrame(rows)


def cluster_bootstrap(
    predictions: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    source = predictions[
        predictions["support"].eq(PRIMARY_SUPPORT)
        & predictions["independent_qc_pass"].fillna(False).astype(bool)
        & np.isfinite(predictions["predicted_agb_g_m2"])
    ].copy()
    groups = source["same_site_group"].dropna().unique().tolist()
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    if len(groups) < 2:
        return pd.DataFrame()
    for replicate in range(n_bootstrap):
        selected = rng.choice(groups, size=len(groups), replace=True)
        pieces = []
        for draw, group in enumerate(selected):
            piece = source[source["same_site_group"].eq(group)].copy()
            piece["bootstrap_group"] = f"{draw}_{group}"
            pieces.append(piece)
        sample = pd.concat(pieces, ignore_index=True)
        metrics = metric_values(
            sample["agb_mean_g_m2"].to_numpy(float),
            sample["predicted_agb_g_m2"].to_numpy(float),
        )
        rows.append({"replicate": replicate, **metrics})
    bootstrap = pd.DataFrame(rows)
    interval_rows = []
    for name in ["pearson_r", "predictive_r_squared", "rmse_g_m2", "mae_g_m2", "bias_g_m2"]:
        values = bootstrap[name].dropna().to_numpy(float)
        interval_rows.append(
            {
                "metric": name,
                "estimate": metric_values(
                    source["agb_mean_g_m2"].to_numpy(float),
                    source["predicted_agb_g_m2"].to_numpy(float),
                )[name],
                "bootstrap_p025": float(np.quantile(values, 0.025)) if values.size else np.nan,
                "bootstrap_p975": float(np.quantile(values, 0.975)) if values.size else np.nan,
                "bootstrap_replicates": n_bootstrap,
                "cluster_variable": "same_site_group",
            }
        )
    return pd.DataFrame(interval_rows)


def validation_figure(predictions: pd.DataFrame, metrics: pd.DataFrame) -> None:
    supports = [PRIMARY_SUPPORT, *SUPPORTS]
    labels = {
        PRIMARY_SUPPORT: "90 m product grid (primary)",
        "source_center_30m": "30 m center pixel",
        "source_center_3x3_90m": "Centered 3x3 (about 90 m)",
        "source_center_5x5_150m": "Centered 5x5 (about 150 m)",
    }
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.2), constrained_layout=True)
    for axis, support in zip(axes.ravel(), supports):
        subset = predictions[predictions["support"].eq(support)].dropna(
            subset=["agb_mean_g_m2", "predicted_agb_g_m2"]
        )
        qc_pass = subset["independent_qc_pass"].fillna(False).astype(bool)
        axis.scatter(
            subset.loc[qc_pass, "agb_mean_g_m2"],
            subset.loc[qc_pass, "predicted_agb_g_m2"],
            s=42,
            color="#2A7F9E",
            edgecolor="white",
            linewidth=0.6,
            label="Quality-controlled field records",
        )
        axis.scatter(
            subset.loc[~qc_pass, "agb_mean_g_m2"],
            subset.loc[~qc_pass, "predicted_agb_g_m2"],
            s=54,
            marker="D",
            color="#C65D3B",
            edgecolor="white",
            linewidth=0.6,
            label="QC-review records (sensitivity only)",
        )
        limit = max(500.0, float(np.nanmax(subset[["agb_mean_g_m2", "predicted_agb_g_m2"]].to_numpy())) * 1.06) if len(subset) else 500.0
        axis.plot([0, limit], [0, limit], color="0.35", linestyle="--", linewidth=1)
        axis.set_xlim(0, limit)
        axis.set_ylim(0, limit)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(labels[support], fontsize=11)
        axis.set_xlabel("Observed dry AGB (g m$^{-2}$)")
        axis.set_ylabel("Predicted AGB (g m$^{-2}$)")
        axis.spines[["top", "right"]].set_visible(False)
        row = metrics[
            metrics["support"].eq(support)
            & metrics["scenario"].eq(PRIMARY_SCENARIO)
            & metrics["weighting"].eq("record_level")
        ].iloc[0]
        axis.text(
            0.04,
            0.96,
            f"n={int(row['n'])}\nr={row['pearson_r']:.2f}\nRMSE={row['rmse_g_m2']:.1f}",
            transform=axis.transAxes,
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9, "pad": 3},
        )
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("Independent 2025 validation of the frozen Peak-only PLSR", fontsize=14)
    fig.savefig(FIGURE_DIR / "field_validation_2025.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "field_validation_2025.pdf", bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> str:
    shown = frame.copy()
    for column in shown.columns:
        if pd.api.types.is_float_dtype(shown[column]):
            shown[column] = shown[column].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
        else:
            shown[column] = shown[column].astype(str)
    columns = list(shown.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in shown.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def build_report(
    annual_summary: pd.DataFrame,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    decision: dict[str, object],
) -> None:
    primary = metrics[
        metrics["support"].eq(PRIMARY_SUPPORT)
        & metrics["scenario"].eq(PRIMARY_SCENARIO)
        & metrics["weighting"].eq("record_level")
    ]
    selected = metrics[
        metrics["scenario"].eq(PRIMARY_SCENARIO)
        & metrics["weighting"].eq("record_level")
    ][["support", "n", "pearson_r", "predictive_r_squared", "rmse_g_m2", "mae_g_m2", "bias_g_m2"]]
    report = f"""# 2025年AGB独立验证

## 固定协议

- 2025年影像：Landsat 8/9 Collection 2 Level 2，7月1日至9月30日，逐景计算指数后执行`qualityMosaic('NDVI')`。
- 模型：冻结的`Baseline_Peak_PLSR`，20个特征、3个PLS成分；2025样点未参与拟合。
- 主产品：30 m输入经有效像元3×3中值聚合，输出90 m网格。
- 2025年缺少与1985-2024 TerraClimate生产栈完全一致的气候输入，因此本轮不伪造Guarded结果，主结果为Peak-only。
- 主验证仅使用原始数据中预先标记为`{PRIMARY_QC_FLAG}`的记录；全部候选记录仅用于附加敏感性分析。筛选依据为预先记录的质量标记，而不是模型预测结果或事后AGB阈值。

## 2025产品质量

{markdown_table(annual_summary.drop(columns=['output_path']))}

## 独立验证结果

{markdown_table(selected)}

主结论读取`grid_product_90m`。其余三个支持尺度仅用于样方-像元尺度敏感性，不替代主产品。

## 数值筛查

- 有效样点数：{int(primary.iloc[0]['n']) if not primary.empty else 0}；
- 预设数值门槛同时通过：{'是' if decision['meets_numeric_screen'] else '否'}；
- `production_approved`：否。本脚本不会仅凭一次年外验证自动冻结生产状态；
- 2025是否并入长期趋势：{'允许进入后续人工复核' if decision['eligible_for_manual_series_review'] else '暂不允许'}。

## 解释边界

2025数据首先承担年外独立验证作用。只有影像覆盖、训练域外比例、偏差和误差均经复核后，才把2025追加到1988-2024长期序列。样点集中在少数湿地，因此同时提供按`same_site_group`聚类bootstrap区间和样地等权指标。
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def run(n_bootstrap: int, seed: int) -> None:
    ensure_dirs()
    check = check_inputs()
    (OUTPUT_DIR / "input_check_2025.json").write_text(
        json.dumps(check, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not check["ready"]:
        raise FileNotFoundError(
            "2025 input is not ready. Download the four annual Peak tiles to "
            f"{pilot.IMAGE_DIR}. Current tile count: {check['peak_tile_count']}."
        )

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    features = list(bundle["features"])
    lower, upper = peak_batch.training_limits(features)
    aoi = gpd.read_file(pilot.AOI_PATH).to_crs("EPSG:32648")
    geometries = [geometry.__geo_interface__ for geometry in aoi.geometry if geometry is not None]

    # Reuse the frozen 2013-2024 production implementation while isolating
    # the new raster in a dedicated 2025 directory.
    peak_batch.RASTER_DIR = RASTER_DIR
    pilot.VRT_DIR = VRT_DIR
    annual_row = peak_batch.run_year(YEAR, model, features, lower, upper, geometries)
    annual_summary = pd.DataFrame([annual_row])
    annual_summary.to_csv(OUTPUT_DIR / "annual_product_qc_2025.csv", index=False, encoding="utf-8-sig")
    raster_path = Path(annual_row["output_path"])

    fields = pd.read_csv(FIELD_PATH)
    fields = fields.drop_duplicates(subset=["sample_id"]).copy()
    if "quality_flag" not in fields.columns:
        raise ValueError("The 2025 field table must contain a quality_flag column.")
    normalized_quality_flag = fields["quality_flag"].astype("string").str.strip().str.lower()
    fields["independent_qc_pass"] = normalized_quality_flag.eq(PRIMARY_QC_FLAG)
    if not fields["independent_qc_pass"].any():
        raise ValueError(
            f"No 2025 field records have quality_flag={PRIMARY_QC_FLAG!r}."
        )

    grid = sample_raster_values(raster_path, fields)
    main95 = sample_raster_values(MAIN95_PATH, fields).iloc[:, 0]
    fields["inside_main95"] = main95.fillna(0).astype(float) >= 0.5
    grid_predictions = fields.copy()
    grid_predictions["support"] = PRIMARY_SUPPORT
    grid_predictions["window_size_pixels"] = 3
    grid_predictions["support_scale_m"] = 90
    grid_predictions["window_valid_pixel_count"] = grid["valid_count"]
    grid_predictions["predicted_agb_g_m2"] = grid["AGB_full_aoi_g_m2"]
    grid_predictions["ood_feature_fraction"] = grid["ood_feature_fraction"]
    grid_predictions["peatland_probability"] = grid["peatland_probability"]

    centered = extract_centered_predictions(fields, model, features, lower, upper)
    centered = centered.merge(
        fields,
        on="sample_id",
        how="left",
        validate="many_to_one",
    )
    centered["peatland_probability"] = centered["sample_id"].map(
        grid_predictions.set_index("sample_id")["peatland_probability"]
    )
    centered["inside_main95"] = centered["sample_id"].map(
        grid_predictions.set_index("sample_id")["inside_main95"]
    )
    predictions = pd.concat([grid_predictions, centered], ignore_index=True, sort=False)
    predictions.to_csv(OUTPUT_DIR / "field_predictions_2025.csv", index=False, encoding="utf-8-sig")

    metrics = calculate_metrics(predictions)
    metrics.to_csv(OUTPUT_DIR / "metrics_2025.csv", index=False, encoding="utf-8-sig")
    intervals = cluster_bootstrap(predictions, n_bootstrap=n_bootstrap, seed=seed)
    intervals.to_csv(OUTPUT_DIR / "bootstrap_intervals_2025.csv", index=False, encoding="utf-8-sig")
    validation_figure(predictions, metrics)

    primary = metrics[
        metrics["support"].eq(PRIMARY_SUPPORT)
        & metrics["scenario"].eq(PRIMARY_SCENARIO)
        & metrics["weighting"].eq("record_level")
    ].iloc[0]
    gates = {
        "valid_records": bool(primary["n"] >= SCREENING_GATES["minimum_valid_records"]),
        "pearson_r": bool(primary["pearson_r"] >= SCREENING_GATES["minimum_pearson_r"]),
        "predictive_r_squared": bool(primary["predictive_r_squared"] >= SCREENING_GATES["minimum_predictive_r_squared"]),
        "rmse": bool(primary["rmse_g_m2"] <= SCREENING_GATES["maximum_rmse_g_m2"]),
        "absolute_bias": bool(primary["absolute_bias_g_m2"] <= SCREENING_GATES["maximum_absolute_bias_g_m2"]),
        "annual_ood": bool(annual_row["ood_fraction_gt_0_25"] <= 0.25),
    }
    decision = {
        "year": YEAR,
        "model": bundle["experiment"],
        "training_sample_count": bundle["training_sample_count"],
        "2025_field_records_used_for_model_fitting": 0,
        "2025_field_records_quality_controlled_primary": int(fields["independent_qc_pass"].sum()),
        "2025_field_records_all_candidates": int(len(fields)),
        "2025_field_records_sensitivity_only": int((~fields["independent_qc_pass"]).sum()),
        "primary_quality_flag": PRIMARY_QC_FLAG,
        "primary_scenario": PRIMARY_SCENARIO,
        "primary_support": PRIMARY_SUPPORT,
        "screening_thresholds": SCREENING_GATES,
        "screening_gates": gates,
        "meets_numeric_screen": bool(all(gates.values())),
        "production_approved": False,
        "eligible_for_manual_series_review": bool(all(gates.values())),
        "guarded_2025_generated": False,
        "guarded_2025_reason": "Frozen TerraClimate production stack ends in 2024.",
    }
    (OUTPUT_DIR / "decision_2025.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    build_report(annual_summary, predictions, metrics, decision)
    print(metrics.to_string(index=False))
    print(f"\nReport: {REPORT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-input", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()
    ensure_dirs()
    if args.check_input:
        print(json.dumps(check_inputs(), ensure_ascii=False, indent=2))
        return
    run(n_bootstrap=args.bootstrap, seed=args.seed)


if __name__ == "__main__":
    main()
