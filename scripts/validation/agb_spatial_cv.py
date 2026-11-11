"""Evaluate map-ready ecological stratification and quality weighting for AGB.

The experiment keeps the same 136 samples, PeakNDVI 3x3 features, 5 km
spatial blocks, and fold-wise source harmonization used by the current
baseline. Source labels are used only in the observation harmonization layer;
they are never predictors in a map-ready biomass model.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import rasterio
from scipy import stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

matplotlib.use("Agg")
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "agb_ecological_weighted_validation"
OUTPUT_DIR = RUN_DIR / "outputs"
FIGURE_DIR = RUN_DIR / "figures"
MODEL_DIR = RUN_DIR / "models"

SAMPLE_FEATURES = (
    ROOT
    / "agb_multitemporal_fusion_validation"
    / "outputs"
    / "multitemporal_sample_features.csv"
)
TIMING_FEATURES = (
    ROOT
    / "agb_timing_partial_validation"
    / "outputs"
    / "timing_extracted_sample_features.csv"
)
HOMOGENEITY_FLAGS = (
    ROOT
    / "agb_pixel_scale_homogeneity_optimization"
    / "outputs"
    / "sample_homogeneity_quality_flags.csv"
)
HOMOGENEITY_THRESHOLDS = (
    ROOT
    / "agb_pixel_scale_homogeneity_optimization"
    / "outputs"
    / "heterogeneity_thresholds.csv"
)
PEAT_PROBABILITY = (
    ROOT
    / "zoigepeatland"
    / "output_basic_boundary"
    / "Zoige_RF_basic_probability.tif"
)

SEED = 20260710
BOOTSTRAP_REPLICATES = 1000
TARGET = "agb_g_m2"
HARMONIZED_TARGET = "harmonized_agb_g_m2"
SOURCE = "source_dataset_group"
REFERENCE_SOURCE = "existing_master_160"
BLOCK = "cv_spatial_block_5km"

DYNAMIC_COLUMNS = [
    "Blue",
    "Green",
    "Red",
    "NIR",
    "SWIR1",
    "SWIR2",
    "NDVI",
    "EVI",
    "NDMI",
    "NDWI",
    "MNDWI",
    "FVC",
    "LSWI",
    "NIRv",
    "NDTI",
    "valid_count",
    "water_fraction",
]
TERRAIN = ["elevation", "slope", "relative_elevation"]
BASE_FEATURES = [f"pk_{name}" for name in DYNAMIC_COLUMNS] + TERRAIN
PEAT_FEATURES = ["peatland_probability", "peatland_candidate_domain"]
ECOLOGY_DERIVED = [
    "wetness_composite",
    "wetness_composite_sq",
    "peat_x_wetness",
    "peat_x_NDMI",
    "peat_x_MNDWI",
    "peat_x_NIRv",
    "peat_x_EVI",
    "peat_x_SWIR1",
    "peat_x_relative_elevation",
    "relative_elevation_x_wetness",
    "slope_x_wetness",
    "NIRv_sq",
    "EVI_sq",
    "SWIR1_sq",
]


@dataclass(frozen=True)
class Experiment:
    name: str
    feature_set: str
    model_name: str
    weight_scheme: str


EXPERIMENTS = [
    Experiment("Baseline_Peak_PLSR", "base", "PLSR", "none"),
    Experiment("Baseline_Peak_Ridge", "base", "Ridge", "none"),
    Experiment("Peat_PLSR", "peat", "PLSR", "none"),
    Experiment("Ecology_PLSR", "ecology", "PLSR", "none"),
    Experiment("Ecology_Ridge", "ecology", "Ridge", "none"),
    Experiment("Ecology_Huber", "ecology", "Huber", "none"),
    Experiment(
        "Ecology_Ridge_heterogeneity_weighted",
        "ecology",
        "Ridge",
        "heterogeneity",
    ),
    Experiment(
        "Ecology_Huber_heterogeneity_weighted",
        "ecology",
        "Huber",
        "heterogeneity",
    ),
    Experiment(
        "Ecology_Ridge_quality_weighted",
        "ecology",
        "Ridge",
        "quality",
    ),
    Experiment(
        "Ecology_Huber_quality_weighted",
        "ecology",
        "Huber",
        "quality",
    ),
    Experiment(
        "Ecology_ElasticNet_quality_weighted",
        "ecology",
        "ElasticNet",
        "quality",
    ),
]


def ensure_dirs() -> None:
    for path in (OUTPUT_DIR, FIGURE_DIR, MODEL_DIR):
        path.mkdir(parents=True, exist_ok=True)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    shown = frame.copy()
    for column in shown.columns:
        if pd.api.types.is_float_dtype(shown[column]):
            shown[column] = shown[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.3f}"
            )
        else:
            shown[column] = shown[column].map(
                lambda value: "" if pd.isna(value) else str(value)
            )
    columns = list(shown.columns)
    lines = ["| " + " | ".join(columns) + " |"]
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in shown.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def sample_probability(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    with rasterio.open(PEAT_PROBABILITY) as src:
        if str(src.crs).upper() != "EPSG:4326":
            raise ValueError(f"Expected EPSG:4326 peat probability, got {src.crs}")
        coordinates = list(zip(output["longitude"], output["latitude"]))
        sampled = list(src.sample(coordinates, masked=True))
        values = []
        available = []
        for item in sampled:
            value = item[0]
            is_available = not np.ma.is_masked(value) and np.isfinite(float(value))
            available.append(float(is_available))
            values.append(float(value) if is_available else 0.0)
    output["peatland_probability"] = np.clip(values, 0.0, 1.0)
    output["peatland_candidate_domain"] = available
    return output


def normalize_weights_within_source(
    frame: pd.DataFrame, raw: pd.Series
) -> pd.Series:
    normalized = pd.Series(index=frame.index, dtype=float)
    for _, indices in frame.groupby(SOURCE, sort=False).groups.items():
        values = raw.loc[indices].astype(float)
        mean = float(values.mean())
        normalized.loc[indices] = values / mean if mean > 0 else 1.0
    normalized = normalized.clip(0.50, 1.50)
    return normalized / normalized.mean()


def add_quality_weights(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    threshold = pd.read_csv(HOMOGENEITY_THRESHOLDS).iloc[0]
    q75 = float(threshold["heterogeneity_q75"])
    heterogeneity = output["local_heterogeneity_score"].fillna(q75)
    raw_heterogeneity = np.exp(-0.5 * np.square(heterogeneity / max(q75, 1e-6)))
    raw_heterogeneity = pd.Series(
        np.clip(raw_heterogeneity, 0.50, 1.0), index=output.index
    )

    records = pd.to_numeric(output["n_field_records"], errors="coerce").fillna(1)
    replicate = 0.75 + 0.25 * np.clip(
        np.log1p(records) / np.log1p(6.0), 0.0, 1.0
    )
    tier_map = {
        "primary_training": 1.00,
        "primary_training_candidate": 1.00,
        "development_validation": 0.90,
        "ecological_extension_training_candidate": 0.90,
    }
    tier = output["model_use_tier"].map(tier_map).fillna(0.90)

    output["weight_none"] = 1.0
    output["weight_heterogeneity"] = normalize_weights_within_source(
        output, raw_heterogeneity
    )
    output["weight_quality"] = normalize_weights_within_source(
        output, raw_heterogeneity * replicate * tier
    )
    output["weight_raw_heterogeneity"] = raw_heterogeneity
    output["weight_replicate_component"] = replicate
    output["weight_tier_component"] = tier
    return output


def add_ecology_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    wetness = output[["pk_NDMI", "pk_LSWI", "pk_MNDWI"]].mean(axis=1)
    output["wetness_composite"] = wetness
    output["wetness_composite_sq"] = np.square(wetness)
    probability = output["peatland_probability"]
    output["peat_x_wetness"] = probability * wetness
    output["peat_x_NDMI"] = probability * output["pk_NDMI"]
    output["peat_x_MNDWI"] = probability * output["pk_MNDWI"]
    output["peat_x_NIRv"] = probability * output["pk_NIRv"]
    output["peat_x_EVI"] = probability * output["pk_EVI"]
    output["peat_x_SWIR1"] = probability * output["pk_SWIR1"]
    output["peat_x_relative_elevation"] = (
        probability * output["relative_elevation"]
    )
    output["relative_elevation_x_wetness"] = (
        output["relative_elevation"] * wetness
    )
    output["slope_x_wetness"] = output["slope"] * wetness
    output["NIRv_sq"] = np.square(output["pk_NIRv"])
    output["EVI_sq"] = np.square(output["pk_EVI"])
    output["SWIR1_sq"] = np.square(output["pk_SWIR1"])
    return output


def load_samples() -> pd.DataFrame:
    frame = pd.read_csv(SAMPLE_FEATURES)
    peak_metadata = pd.read_csv(TIMING_FEATURES)
    peak_metadata = peak_metadata[
        peak_metadata["timing_variant"].eq("PeakNDVI_JulSep")
    ].drop_duplicates("same_site_group")
    metadata_columns = [
        "same_site_group",
        "model_use_tier",
        "recommended_split",
        "source_title",
        "n_field_records",
    ]
    frame = frame.merge(
        peak_metadata[metadata_columns],
        on="same_site_group",
        how="left",
        validate="one_to_one",
    )
    quality = pd.read_csv(HOMOGENEITY_FLAGS)
    frame = frame.merge(
        quality,
        on="same_site_group",
        how="left",
        validate="one_to_one",
    )
    frame = sample_probability(frame)
    frame = add_ecology_features(frame)
    frame = add_quality_weights(frame)

    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")
    frame = frame[
        frame[TARGET].between(0, 2000, inclusive="both")
        & frame[BASE_FEATURES].notna().all(axis=1)
    ].copy()
    frame["year"] = frame["year"].astype(int)
    for column in ("same_site_group", SOURCE, BLOCK):
        frame[column] = frame[column].astype(str)
    return frame.reset_index(drop=True)


def features_for(feature_set: str) -> list[str]:
    if feature_set == "base":
        return list(BASE_FEATURES)
    if feature_set == "peat":
        return list(BASE_FEATURES + PEAT_FEATURES)
    if feature_set == "ecology":
        return list(BASE_FEATURES + PEAT_FEATURES + ECOLOGY_DERIVED)
    raise ValueError(feature_set)


def source_dummies(frame: pd.DataFrame) -> list[str]:
    return [f"src_{source}" for source in sorted(frame[SOURCE].unique())]


def make_huber_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", RobustScaler()),
            (
                "regressor",
                HuberRegressor(alpha=0.02, epsilon=1.35, max_iter=3000),
            ),
        ]
    )


def make_model_and_grid(model_name: str, n_features: int):
    if model_name == "PLSR":
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
                ("regressor", PLSRegression(scale=False, max_iter=1500)),
            ]
        )
        components = [value for value in (2, 3, 4, 5, 6) if value < n_features]
        return model, {"regressor__n_components": components}, False
    if model_name == "Ridge":
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
                ("regressor", Ridge()),
            ]
        )
        return model, {"regressor__alpha": [0.3, 1.0, 3.0, 10.0, 30.0, 100.0]}, True
    if model_name == "Huber":
        return make_huber_pipeline(), None, True
    if model_name == "ElasticNet":
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", RobustScaler()),
                (
                    "regressor",
                    ElasticNet(max_iter=40000, random_state=SEED),
                ),
            ]
        )
        grid = {
            "regressor__alpha": [0.5, 2.0, 10.0, 30.0],
            "regressor__l1_ratio": [0.1, 0.5, 0.9],
        }
        return model, grid, True
    raise ValueError(model_name)


def fit_tuned_model(
    train: pd.DataFrame,
    features: list[str],
    target: np.ndarray,
    model_name: str,
    weight_scheme: str,
) -> tuple[object, dict[str, object]]:
    model, grid, supports_weight = make_model_and_grid(model_name, len(features))
    fit_params: dict[str, np.ndarray] = {}
    if supports_weight and weight_scheme != "none":
        fit_params["regressor__sample_weight"] = train[
            f"weight_{weight_scheme}"
        ].to_numpy(float)
    if not grid:
        model.fit(train[features], target, **fit_params)
        return model, {}

    groups = train[BLOCK].astype(str).to_numpy()
    inner = GroupKFold(n_splits=min(4, len(np.unique(groups))))
    search = GridSearchCV(
        model,
        grid,
        scoring="neg_root_mean_squared_error",
        cv=inner,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )
    search.fit(train[features], target, groups=groups, **fit_params)
    return search.best_estimator_, search.best_params_


def counterfactual_reference(
    frame: pd.DataFrame, dummy_names: list[str]
) -> pd.DataFrame:
    output = frame.copy()
    for name in dummy_names:
        output[name] = 0.0
    reference = f"src_{REFERENCE_SOURCE}"
    if reference in output.columns:
        output[reference] = 1.0
    return output


def harmonize_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    dummy_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    bridge_features = BASE_FEATURES + dummy_names
    bridge = make_huber_pipeline()
    bridge.fit(train[bridge_features], train[TARGET])

    def apply(frame: pd.DataFrame) -> pd.DataFrame:
        output = frame.copy()
        reference = counterfactual_reference(frame, dummy_names)
        current_prediction = bridge.predict(frame[bridge_features]).reshape(-1)
        reference_prediction = bridge.predict(reference[bridge_features]).reshape(-1)
        output["source_correction_g_m2"] = current_prediction - reference_prediction
        output[HARMONIZED_TARGET] = np.clip(
            output[TARGET] - output["source_correction_g_m2"], 0, 2000
        )
        return output

    return apply(train), apply(test), bridge


def metric_dict(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    y = observed[valid].astype(float)
    p = predicted[valid].astype(float)
    if len(y) < 2:
        return {
            "n": int(len(y)),
            "rmse_g_m2": np.nan,
            "mae_g_m2": np.nan,
            "bias_g_m2": np.nan,
            "predictive_r_squared": np.nan,
            "pearson_r": np.nan,
            "spearman_r": np.nan,
        }
    return {
        "n": int(len(y)),
        "rmse_g_m2": float(np.sqrt(mean_squared_error(y, p))),
        "mae_g_m2": float(mean_absolute_error(y, p)),
        "bias_g_m2": float(np.mean(p - y)),
        "predictive_r_squared": float(r2_score(y, p)),
        "pearson_r": float(stats.pearsonr(y, p).statistic)
        if y.std() > 0 and p.std() > 0
        else np.nan,
        "spearman_r": float(stats.spearmanr(y, p).statistic)
        if y.std() > 0 and p.std() > 0
        else np.nan,
    }


def spatial_predictions(
    frame: pd.DataFrame,
    experiment: Experiment,
    dummy_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = frame[BLOCK].to_numpy()
    outer = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    features = features_for(experiment.feature_set)
    outputs = []
    tuning_rows = []
    for fold, (train_index, test_index) in enumerate(outer.split(frame, groups=groups)):
        train = frame.iloc[train_index].copy()
        test = frame.iloc[test_index].copy()
        train, test, _ = harmonize_fold(train, test, dummy_names)
        model, parameters = fit_tuned_model(
            train,
            features,
            train[HARMONIZED_TARGET].to_numpy(float),
            experiment.model_name,
            experiment.weight_scheme,
        )
        prediction = np.clip(model.predict(test[features]).reshape(-1), 0, 2000)
        columns = [
            "same_site_group",
            "year",
            SOURCE,
            "validation_group",
            "vegetation_type",
            "peatland_relevance",
            "longitude",
            "latitude",
            BLOCK,
            TARGET,
            HARMONIZED_TARGET,
            "source_correction_g_m2",
            "peatland_probability",
            "peatland_candidate_domain",
            "wetness_composite",
            "local_heterogeneity_score",
            "n_field_records",
            "model_use_tier",
            "weight_heterogeneity",
            "weight_quality",
        ]
        part = test[columns].copy()
        part["prediction_g_m2"] = prediction
        part["experiment"] = experiment.name
        part["feature_set"] = experiment.feature_set
        part["model_name"] = experiment.model_name
        part["weight_scheme"] = experiment.weight_scheme
        part["feature_count"] = len(features)
        part["fold"] = fold
        outputs.append(part)
        tuning_rows.append(
            {
                "experiment": experiment.name,
                "fold": fold,
                "model_name": experiment.model_name,
                "weight_scheme": experiment.weight_scheme,
                "parameters": json.dumps(parameters, ensure_ascii=False, sort_keys=True),
            }
        )
    return pd.concat(outputs, ignore_index=True), pd.DataFrame(tuning_rows)


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for experiment, part in predictions.groupby("experiment", sort=False):
        harmonized = metric_dict(
            part[HARMONIZED_TARGET].to_numpy(float),
            part["prediction_g_m2"].to_numpy(float),
        )
        raw = metric_dict(
            part[TARGET].to_numpy(float), part["prediction_g_m2"].to_numpy(float)
        )
        first = part.iloc[0]
        row = {
            "experiment": experiment,
            "feature_set": first["feature_set"],
            "model_name": first["model_name"],
            "weight_scheme": first["weight_scheme"],
            "feature_count": int(first["feature_count"]),
            **harmonized,
        }
        row.update({f"raw_{key}": value for key, value in raw.items() if key != "n"})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["predictive_r_squared", "pearson_r", "rmse_g_m2"],
        ascending=[False, False, True],
    )


def bootstrap_comparison(
    predictions: pd.DataFrame, baseline_name: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = predictions[predictions["experiment"].eq(baseline_name)].copy()
    baseline = baseline.sort_values("same_site_group").reset_index(drop=True)
    blocks = baseline[BLOCK].astype(str).to_numpy()
    unique_blocks = np.unique(blocks)
    rng = np.random.default_rng(SEED)
    sampled_block_sets = [
        rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        for _ in range(BOOTSTRAP_REPLICATES)
    ]
    metric_names = [
        "pearson_r",
        "predictive_r_squared",
        "rmse_g_m2",
        "mae_g_m2",
        "bias_g_m2",
    ]
    interval_rows = []
    paired_rows = []

    for experiment, part in predictions.groupby("experiment", sort=False):
        part = part.sort_values("same_site_group").reset_index(drop=True)
        if not np.array_equal(
            part["same_site_group"].to_numpy(), baseline["same_site_group"].to_numpy()
        ):
            raise ValueError(f"Sample mismatch for bootstrap: {experiment}")
        values = {name: [] for name in metric_names}
        deltas = {name: [] for name in metric_names}
        for sampled_blocks in sampled_block_sets:
            indices = np.concatenate(
                [np.flatnonzero(blocks == block) for block in sampled_blocks]
            )
            candidate_metric = metric_dict(
                part[HARMONIZED_TARGET].to_numpy(float)[indices],
                part["prediction_g_m2"].to_numpy(float)[indices],
            )
            baseline_metric = metric_dict(
                baseline[HARMONIZED_TARGET].to_numpy(float)[indices],
                baseline["prediction_g_m2"].to_numpy(float)[indices],
            )
            for name in metric_names:
                if np.isfinite(candidate_metric[name]):
                    values[name].append(candidate_metric[name])
                if np.isfinite(candidate_metric[name]) and np.isfinite(
                    baseline_metric[name]
                ):
                    deltas[name].append(candidate_metric[name] - baseline_metric[name])

        for name in metric_names:
            array = np.asarray(values[name], dtype=float)
            interval_rows.append(
                {
                    "experiment": experiment,
                    "metric": name,
                    "bootstrap_n": len(array),
                    "mean": float(np.mean(array)),
                    "ci_low_2_5": float(np.quantile(array, 0.025)),
                    "ci_high_97_5": float(np.quantile(array, 0.975)),
                }
            )
        r_delta = np.asarray(deltas["pearson_r"], dtype=float)
        rmse_delta = np.asarray(deltas["rmse_g_m2"], dtype=float)
        r2_delta = np.asarray(deltas["predictive_r_squared"], dtype=float)
        paired_rows.append(
            {
                "experiment": experiment,
                "baseline": baseline_name,
                "delta_pearson_r_mean": float(np.mean(r_delta)),
                "delta_pearson_r_ci_low": float(np.quantile(r_delta, 0.025)),
                "delta_pearson_r_ci_high": float(np.quantile(r_delta, 0.975)),
                "probability_delta_r_gt_0": float(np.mean(r_delta > 0)),
                "delta_predictive_r2_mean": float(np.mean(r2_delta)),
                "delta_rmse_mean_g_m2": float(np.mean(rmse_delta)),
                "delta_rmse_ci_low": float(np.quantile(rmse_delta, 0.025)),
                "delta_rmse_ci_high": float(np.quantile(rmse_delta, 0.975)),
                "probability_delta_rmse_lt_0": float(np.mean(rmse_delta < 0)),
            }
        )
    return pd.DataFrame(interval_rows), pd.DataFrame(paired_rows)


def choose_recommended(
    summary: pd.DataFrame, paired: pd.DataFrame
) -> tuple[pd.Series, pd.Series]:
    baseline = summary[
        summary["experiment"].eq("Baseline_Peak_PLSR")
    ].iloc[0]
    numerical = summary.sort_values(
        ["predictive_r_squared", "pearson_r", "rmse_g_m2"],
        ascending=[False, False, True],
    ).iloc[0]
    paired_row = paired[paired["experiment"].eq(numerical["experiment"])].iloc[0]
    material = (
        numerical["experiment"] != baseline["experiment"]
        and (
            numerical["pearson_r"] - baseline["pearson_r"] >= 0.03
            or baseline["rmse_g_m2"] - numerical["rmse_g_m2"]
            >= 0.03 * baseline["rmse_g_m2"]
        )
        and paired_row["probability_delta_r_gt_0"] >= 0.80
        and paired_row["probability_delta_rmse_lt_0"] >= 0.80
    )
    recommended = numerical if material else baseline
    return recommended, numerical


def leave_year_predictions(
    frame: pd.DataFrame,
    experiment: Experiment,
    dummy_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = features_for(experiment.feature_set)
    outputs = []
    tuning_rows = []
    for year in sorted(frame["year"].unique()):
        train = frame[frame["year"].ne(year)].copy()
        test = frame[frame["year"].eq(year)].copy()
        train, test, _ = harmonize_fold(train, test, dummy_names)
        model, parameters = fit_tuned_model(
            train,
            features,
            train[HARMONIZED_TARGET].to_numpy(float),
            experiment.model_name,
            experiment.weight_scheme,
        )
        prediction = np.clip(model.predict(test[features]).reshape(-1), 0, 2000)
        part = test[
            [
                "same_site_group",
                "year",
                SOURCE,
                "validation_group",
                BLOCK,
                TARGET,
                HARMONIZED_TARGET,
                "source_correction_g_m2",
                "peatland_candidate_domain",
                "peatland_probability",
            ]
        ].copy()
        part["prediction_g_m2"] = prediction
        part["held_out_year"] = year
        outputs.append(part)
        tuning_rows.append(
            {
                "experiment": experiment.name,
                "validation_mode": "leave_year",
                "held_out_year": year,
                "parameters": json.dumps(parameters, ensure_ascii=False, sort_keys=True),
            }
        )
    return pd.concat(outputs, ignore_index=True), pd.DataFrame(tuning_rows)


def summarize_leave_year(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, part in predictions.groupby("held_out_year", sort=True):
        rows.append(
            {
                "held_out_year": year,
                **metric_dict(
                    part[HARMONIZED_TARGET].to_numpy(float),
                    part["prediction_g_m2"].to_numpy(float),
                ),
            }
        )
    rows.append(
        {
            "held_out_year": "all",
            **metric_dict(
                predictions[HARMONIZED_TARGET].to_numpy(float),
                predictions["prediction_g_m2"].to_numpy(float),
            ),
        }
    )
    return pd.DataFrame(rows)


def residual_summary(
    predictions: pd.DataFrame, recommended_name: str
) -> pd.DataFrame:
    selected = predictions[predictions["experiment"].eq(recommended_name)].copy()
    selected["residual_g_m2"] = (
        selected["prediction_g_m2"] - selected[HARMONIZED_TARGET]
    )
    selected["peat_domain_label"] = np.where(
        selected["peatland_candidate_domain"].ge(0.5),
        "peat_candidate",
        "outside_candidate",
    )
    rows = []
    for grouping, columns in {
        "source_year": [SOURCE, "year"],
        "peat_domain": ["peat_domain_label"],
        "peatland_relevance": ["peatland_relevance"],
        "heterogeneity_quartile": ["heterogeneity_quartile"],
    }.items():
        if grouping == "heterogeneity_quartile":
            selected["heterogeneity_quartile"] = pd.qcut(
                selected["local_heterogeneity_score"],
                4,
                labels=["Q1_low", "Q2", "Q3", "Q4_high"],
                duplicates="drop",
            )
        for keys, part in selected.groupby(columns, dropna=False, observed=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {"grouping": grouping}
            for column, value in zip(columns, keys):
                row[column] = value
            row.update(
                {
                    "n": len(part),
                    "observed_mean_g_m2": part[HARMONIZED_TARGET].mean(),
                    "predicted_mean_g_m2": part["prediction_g_m2"].mean(),
                    "bias_g_m2": part["residual_g_m2"].mean(),
                    "mae_g_m2": part["residual_g_m2"].abs().mean(),
                    "rmse_g_m2": np.sqrt(np.mean(np.square(part["residual_g_m2"]))),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def fit_final_bundle(
    frame: pd.DataFrame,
    experiment: Experiment,
    dummy_names: list[str],
) -> dict[str, object]:
    train, _, bridge = harmonize_fold(frame, frame.copy(), dummy_names)
    features = features_for(experiment.feature_set)
    model, parameters = fit_tuned_model(
        train,
        features,
        train[HARMONIZED_TARGET].to_numpy(float),
        experiment.model_name,
        experiment.weight_scheme,
    )
    bundle = {
        "model": model,
        "source_harmonization_bridge": bridge,
        "source_dummy_names": dummy_names,
        "reference_source": REFERENCE_SOURCE,
        "features": features,
        "base_features": BASE_FEATURES,
        "peat_features": PEAT_FEATURES,
        "ecology_derived_features": ECOLOGY_DERIVED,
        "experiment": experiment.name,
        "model_name": experiment.model_name,
        "weight_scheme": experiment.weight_scheme,
        "parameters": parameters,
        "peat_probability_path": str(PEAT_PROBABILITY),
        "probability_nodata_rule": "fill_with_zero_and_set_candidate_domain_to_zero",
        "training_sample_count": len(train),
        "seed": SEED,
    }
    joblib.dump(bundle, MODEL_DIR / "ecological_weighted_candidate.joblib")
    metadata = {key: value for key, value in bundle.items() if key not in {"model", "source_harmonization_bridge"}}
    (MODEL_DIR / "ecological_weighted_candidate_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return bundle


def make_figures(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
    recommended_name: str,
) -> None:
    ordered = summary.sort_values("pearson_r")
    colors = ["#D47A3A" if name == recommended_name else "#4B7F9E" for name in ordered["experiment"]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].barh(ordered["experiment"], ordered["pearson_r"], color=colors)
    axes[0].axvline(0.60, color="#B33A3A", linestyle="--", linewidth=1.2)
    axes[0].set_xlabel("Spatial-block Pearson r")
    axes[0].set_title("Correlation")
    axes[1].barh(ordered["experiment"], ordered["rmse_g_m2"], color=colors)
    axes[1].axvline(185.0, color="#B33A3A", linestyle="--", linewidth=1.2)
    axes[1].set_xlabel("RMSE (g m$^{-2}$)")
    axes[1].set_title("Absolute error")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "01_spatial_validation_comparison.png", dpi=300)
    plt.close(fig)

    names = ["Baseline_Peak_PLSR"]
    if recommended_name not in names:
        names.append(recommended_name)
    fig, axes = plt.subplots(1, len(names), figsize=(7 * len(names), 6), squeeze=False)
    for axis, name in zip(axes[0], names):
        part = predictions[predictions["experiment"].eq(name)]
        axis.scatter(
            part[HARMONIZED_TARGET],
            part["prediction_g_m2"],
            c=part["peatland_probability"],
            cmap="viridis",
            s=28,
            alpha=0.78,
            edgecolors="none",
        )
        low = min(part[HARMONIZED_TARGET].min(), part["prediction_g_m2"].min())
        high = max(part[HARMONIZED_TARGET].max(), part["prediction_g_m2"].max())
        axis.plot([low, high], [low, high], "--", color="#555555", linewidth=1)
        row = summary[summary["experiment"].eq(name)].iloc[0]
        axis.set_title(
            f"{name}\nr={row['pearson_r']:.3f}, R2={row['predictive_r_squared']:.3f}, "
            f"RMSE={row['rmse_g_m2']:.1f}"
        )
        axis.set_xlabel("Fold-harmonized observed AGB (g m$^{-2}$)")
        axis.set_ylabel("Out-of-fold predicted AGB (g m$^{-2}$)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "02_observed_vs_predicted.png", dpi=300)
    plt.close(fig)

    selected = predictions[predictions["experiment"].eq(recommended_name)].copy()
    selected["residual_g_m2"] = selected["prediction_g_m2"] - selected[HARMONIZED_TARGET]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    selected.boxplot(column="residual_g_m2", by="year", ax=axes[0], grid=False)
    axes[0].axhline(0, color="#555555", linestyle="--")
    axes[0].set_title("Residuals by year")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Prediction - harmonized observation (g m$^{-2}$)")
    selected.boxplot(
        column="residual_g_m2",
        by="peatland_candidate_domain",
        ax=axes[1],
        grid=False,
    )
    axes[1].axhline(0, color="#555555", linestyle="--")
    axes[1].set_title("Residuals outside/inside peat candidate domain")
    axes[1].set_xlabel("Peat candidate domain (0/1)")
    axes[1].set_ylabel("Prediction - harmonized observation (g m$^{-2}$)")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "03_residual_diagnostics.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for source, part in frame.groupby(SOURCE):
        axes[0].hist(part["weight_quality"], bins=12, alpha=0.55, label=source)
        axes[1].hist(part["peatland_probability"], bins=12, alpha=0.55, label=source)
    axes[0].set_title("Predeclared quality weights")
    axes[0].set_xlabel("Within-source normalized weight")
    axes[1].set_title("Map-ready peatland probability")
    axes[1].set_xlabel("Probability (outside candidate domain = 0)")
    for axis in axes:
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "04_weights_and_ecological_domain.png", dpi=300)
    plt.close(fig)


def write_report(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    recommended: pd.Series,
    numerical: pd.Series,
    leave_year: pd.DataFrame,
    residuals: pd.DataFrame,
) -> None:
    baseline = summary[summary["experiment"].eq("Baseline_Peak_PLSR")].iloc[0]
    selected_pair = paired[paired["experiment"].eq(recommended["experiment"])].iloc[0]
    stage_pass = (
        recommended["pearson_r"] >= 0.60
        and recommended["predictive_r_squared"] >= 0.35
        and recommended["rmse_g_m2"] <= 185.0
        and abs(recommended["bias_g_m2"]) <= 20.0
    )
    coverage = pd.DataFrame(
        [
            {
                "n_samples": len(frame),
                "peat_candidate_n": int(frame["peatland_candidate_domain"].sum()),
                "outside_candidate_n": int((1 - frame["peatland_candidate_domain"]).sum()),
                "peat_probability_mean_all": frame["peatland_probability"].mean(),
                "quality_weight_min": frame["weight_quality"].min(),
                "quality_weight_max": frame["weight_quality"].max(),
            }
        ]
    )
    shown_summary = summary[
        [
            "experiment",
            "feature_count",
            "weight_scheme",
            "n",
            "pearson_r",
            "predictive_r_squared",
            "rmse_g_m2",
            "mae_g_m2",
            "bias_g_m2",
        ]
    ]
    domain = residuals[residuals["grouping"].eq("peat_domain")][
        [
            "peat_domain_label",
            "n",
            "observed_mean_g_m2",
            "predicted_mean_g_m2",
            "bias_g_m2",
            "mae_g_m2",
            "rmse_g_m2",
        ]
    ]
    report = f"""# AGB生态分层与质量加权验证

## 设计

- 固定使用136个样点、PeakNDVI Jul-Sep影像、3x3中位数和5 km空间分块。
- 目标仍采用每个外层训练折内估计的来源协议校正；正式候选不使用source dummy。
- 新增可制图变量：泥炭地概率、泥炭候选域、湿度复合量及其与光谱和地形的交互项。
- 质量权重只使用重复数、预先定义的样点用途等级和像元异质性，并在每个来源内部归一化。

{markdown_table(coverage)}

泥炭地概率栅格在候选区外为NoData。本分析将区外明确编码为概率0和候选域0；这不是把未知值静默填零。

## 空间分块结果

{markdown_table(shown_summary)}

## 与当前基线比较

- 当前基线：`Baseline_Peak_PLSR`，r={baseline['pearson_r']:.3f}，R²={baseline['predictive_r_squared']:.3f}，RMSE={baseline['rmse_g_m2']:.1f} g/m²。
- 数值最优：`{numerical['experiment']}`，r={numerical['pearson_r']:.3f}，R²={numerical['predictive_r_squared']:.3f}，RMSE={numerical['rmse_g_m2']:.1f} g/m²。
- 保守推荐：`{recommended['experiment']}`，r={recommended['pearson_r']:.3f}，R²={recommended['predictive_r_squared']:.3f}，RMSE={recommended['rmse_g_m2']:.1f} g/m²，Bias={recommended['bias_g_m2']:.1f} g/m²。
- 推荐方案相对基线的空间块Bootstrap平均变化：Δr={selected_pair['delta_pearson_r_mean']:+.3f}，ΔRMSE={selected_pair['delta_rmse_mean_g_m2']:+.1f} g/m²；P(Δr>0)={selected_pair['probability_delta_r_gt_0']:.3f}，P(ΔRMSE<0)={selected_pair['probability_delta_rmse_lt_0']:.3f}。

若数值最优方案没有同时达到预设的实质改善幅度和Bootstrap支持，保守推荐会保留原基线，避免从多个候选中挑选偶然最高值。

## 阶段目标

目标为r≥0.60、预测R²≥0.35、RMSE≤185 g/m²、|Bias|≤20 g/m²。

**本轮是否同时达标：{'是' if stage_pass else '否'}。**

## 推荐模型留一年验证

{markdown_table(leave_year)}

## 泥炭生态域残差

{markdown_table(domain)}

## 解释

- 若生态交互项稳定优于基线，说明source dummy此前捕捉到的部分差异可以由真实、可制图的生态域替代。
- 若权重方案只降低RMSE却不提高r，说明它主要抑制噪声，未增加空间排序能力。
- 若所有方案提升有限，主要瓶颈仍是来源与年份完全混杂、2021-2022高值样点的协议或光谱饱和，而不是模型算法。
- 即使空间分块改善，留一年结果仍决定模型能否用于长时间序列绝对AGB制图。
"""
    (RUN_DIR / "report_ecological_weighted_validation.md").write_text(
        report, encoding="utf-8"
    )


def write_readme() -> None:
    text = """# AGB ecological weighted validation

This experiment compares the current Peak-only PLSR baseline with map-ready
peatland/ecological interactions and predeclared quality weights. It does not
use source labels as prediction features.

Run from the project root:

```powershell
python scripts/validation/agb_spatial_cv.py
```

Main outputs are in `outputs/`, figures in `figures/`, and the interpretation
is in `report_ecological_weighted_validation.md`.
"""
    (RUN_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    write_readme()
    frame = load_samples()
    frame.to_csv(
        OUTPUT_DIR / "ecological_weighted_sample_features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    dummy_names = source_dummies(frame)
    all_predictions = []
    all_tuning = []
    for experiment in EXPERIMENTS:
        print(f"Spatial validation: {experiment.name}", flush=True)
        prediction, tuning = spatial_predictions(frame, experiment, dummy_names)
        all_predictions.append(prediction)
        all_tuning.append(tuning)
    predictions = pd.concat(all_predictions, ignore_index=True)
    tuning = pd.concat(all_tuning, ignore_index=True)
    summary = summarize_predictions(predictions)
    intervals, paired = bootstrap_comparison(predictions, "Baseline_Peak_PLSR")
    recommended, numerical = choose_recommended(summary, paired)

    selected_experiment = next(
        item for item in EXPERIMENTS if item.name == recommended["experiment"]
    )
    print(f"Leave-year validation: {selected_experiment.name}", flush=True)
    year_predictions, year_tuning = leave_year_predictions(
        frame, selected_experiment, dummy_names
    )
    leave_year = summarize_leave_year(year_predictions)
    residuals = residual_summary(predictions, recommended["experiment"])
    fit_final_bundle(frame, selected_experiment, dummy_names)

    predictions.to_csv(
        OUTPUT_DIR / "ecological_spatial_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        OUTPUT_DIR / "ecological_validation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat([tuning, year_tuning], ignore_index=True).to_csv(
        OUTPUT_DIR / "nested_tuning_parameters.csv",
        index=False,
        encoding="utf-8-sig",
    )
    intervals.to_csv(
        OUTPUT_DIR / "bootstrap_metric_intervals.csv",
        index=False,
        encoding="utf-8-sig",
    )
    paired.to_csv(
        OUTPUT_DIR / "paired_improvement_vs_baseline.csv",
        index=False,
        encoding="utf-8-sig",
    )
    year_predictions.to_csv(
        OUTPUT_DIR / "recommended_leave_year_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    leave_year.to_csv(
        OUTPUT_DIR / "recommended_leave_year_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    residuals.to_csv(
        OUTPUT_DIR / "recommended_domain_residuals.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selected = pd.DataFrame(
        [
            {
                **recommended.to_dict(),
                "numerical_best_experiment": numerical["experiment"],
                "selection_rule": (
                    "numerical best requires delta_r>=0.03 or RMSE reduction>=3%, "
                    "plus paired block-bootstrap support>=0.80; otherwise retain baseline"
                ),
            }
        ]
    )
    selected.to_csv(
        OUTPUT_DIR / "recommended_ecological_model.csv",
        index=False,
        encoding="utf-8-sig",
    )

    make_figures(frame, predictions, summary, recommended["experiment"])
    write_report(
        frame,
        summary,
        paired,
        recommended,
        numerical,
        leave_year,
        residuals,
    )
    print(
        f"Recommended: {recommended['experiment']} | "
        f"r={recommended['pearson_r']:.3f}, "
        f"R2={recommended['predictive_r_squared']:.3f}, "
        f"RMSE={recommended['rmse_g_m2']:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
