#!/usr/bin/env python
"""Build targeted human and vegetation SHAP responses for the reduced model."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
from pyproj import Transformer
from rasterio.warp import Resampling
from scipy.stats import theilslopes

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "agb_driver_xgboost_shap_reduced_2001_2024"
OUTPUTS = OUT_ROOT / "outputs"
FIGURES = OUT_ROOT / "figures"
PANELS = FIGURES / "panels"
CACHE = OUT_ROOT / "cache"
MAIN_PATH = ROOT / "scripts" / "drivers" / "xgboost_shap_reduced.py"

TRAJECTORY_START = 1988
TRAJECTORY_END = 2024
MIN_TRAJECTORY_YEARS = 28
DELTA_BIC = 6.0


def load_main_module():
    spec = importlib.util.spec_from_file_location("section34_main", MAIN_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.10, 1.03, f"({label})", transform=ax.transAxes,
        ha="left", va="bottom", fontsize=9, fontweight="bold",
    )


def add_panel_subtitle(ax: plt.Axes, text: str, y: float = -0.20) -> None:
    ax.text(
        0.5, y, text, transform=ax.transAxes,
        ha="center", va="top", fontsize=8.5,
    )


def style_boxed_axis(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("0.25")
        spine.set_linewidth(0.8)
    ax.tick_params(direction="in", top=True, right=True)


def save(fig: plt.Figure, stem: str, panel: bool = False) -> None:
    directory = PANELS if panel else FIGURES
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.png", dpi=500, bbox_inches="tight", facecolor="white")
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(directory / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(
        directory / f"{stem}.tif",
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
    """Export a standalone panel crop from a composite figure."""
    PANELS.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = Bbox.union([axis.get_tightbbox(renderer) for axis in axes])
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    bbox = bbox.expanded(1 + pad_fraction, 1 + pad_fraction)
    for suffix, kwargs in (
        ("png", {"dpi": 500}),
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


def aggregate_early_peak(module, grid: dict, rebuild: bool = False) -> pd.DataFrame:
    cache_path = CACHE / "strict_peak_only_5km_1988_2000.pkl.gz"
    if cache_path.exists() and not rebuild:
        return pd.read_pickle(cache_path, compression="gzip")
    base = module.grid_frame(grid)[["row", "col"]]
    frames = []
    for year in range(TRAJECTORY_START, module.START_YEAR):
        path = module.AGB_DIR / f"Zoige_AGB_GuardedConsensus_PLSR_90m_{year}.tif"
        with rasterio.open(path) as src:
            peak = src.read(3).astype("float32")
            valid_count = src.read(8).astype("float32")
            if src.nodata is not None:
                peak[peak == src.nodata] = np.nan
                valid_count[valid_count == src.nodata] = np.nan
            strict = module.strict_mask_on_grid(src)
            valid = strict & np.isfinite(peak) & np.isfinite(valid_count) & (valid_count > 0)
            strict_area = module.warp_array(
                strict.astype("float32"), src.transform, src.crs, grid, Resampling.average
            )
            valid_area = module.warp_array(
                valid.astype("float32"), src.transform, src.crs, grid, Resampling.average
            )
            peak_mean = module.warp_array(
                np.where(valid, peak, np.nan), src.transform, src.crs, grid, Resampling.average
            )
        part = base.copy()
        part["year"] = year
        part["peak_agb"] = peak_mean.ravel()
        part["strict_area_fraction"] = strict_area.ravel()
        part["strict_valid_fraction"] = np.divide(
            valid_area,
            strict_area,
            out=np.full_like(valid_area, np.nan),
            where=strict_area > 0,
        ).ravel()
        frames.append(part)
        print(f"Peak-only trajectory aggregation: {year}", flush=True)
    result = pd.concat(frames, ignore_index=True)
    result.to_pickle(cache_path, compression="gzip")
    return result


def bic(rss: float, n: int, k: int) -> float:
    rss = max(float(rss), 1e-9)
    return n * math.log(rss / n) + k * math.log(n)


def fit_trajectory(group: pd.DataFrame) -> dict:
    data = group[["year", "peak_agb"]].dropna().sort_values("year")
    years = data["year"].to_numpy(dtype=float)
    values = data["peak_agb"].to_numpy(dtype=float)
    n = len(data)
    result = {
        "n_years": n,
        "trajectory_class": "Insufficient data",
        "sen_slope_g_m2_yr": np.nan,
        "break_year": np.nan,
        "step_magnitude_g_m2": np.nan,
        "bic_constant": np.nan,
        "bic_linear": np.nan,
        "bic_quadratic": np.nan,
        "bic_step": np.nan,
    }
    if n < MIN_TRAJECTORY_YEARS:
        return result
    centered = years - years.mean()
    constant = np.full(n, values.mean())
    linear_coef = np.polyfit(centered, values, 1)
    linear = np.polyval(linear_coef, centered)
    quadratic_coef = np.polyfit(centered, values, 2)
    quadratic = np.polyval(quadratic_coef, centered)
    b0 = bic(np.sum((values - constant) ** 2), n, 1)
    b1 = bic(np.sum((values - linear) ** 2), n, 2)
    b2 = bic(np.sum((values - quadratic) ** 2), n, 3)
    best_step = None
    for split in range(6, n - 5):
        prediction = np.r_[
            np.repeat(values[:split].mean(), split),
            np.repeat(values[split:].mean(), n - split),
        ]
        score = bic(np.sum((values - prediction) ** 2), n, 3)
        candidate = {
            "bic": score,
            "break_year": int(years[split]),
            "magnitude": float(values[split:].mean() - values[:split].mean()),
        }
        if best_step is None or score < best_step["bic"]:
            best_step = candidate
    slope = float(theilslopes(values, years).slope)
    direction = "increase" if slope >= 0 else "decrease"
    result.update({
        "sen_slope_g_m2_yr": slope,
        "break_year": best_step["break_year"],
        "step_magnitude_g_m2": best_step["magnitude"],
        "bic_constant": b0,
        "bic_linear": b1,
        "bic_quadratic": b2,
        "bic_step": best_step["bic"],
    })
    if b0 - b1 < DELTA_BIC:
        result["trajectory_class"] = "Stable or no clear trend"
        return result
    nonlinear = min(b2, best_step["bic"])
    if b1 - nonlinear >= DELTA_BIC:
        if best_step["bic"] <= b2:
            if 2012 <= best_step["break_year"] <= 2014:
                result["trajectory_class"] = "Sensor-transition step"
            else:
                step_direction = "increase" if best_step["magnitude"] >= 0 else "decrease"
                result["trajectory_class"] = f"Abrupt {step_direction}"
        else:
            endpoint_change = quadratic[-1] - quadratic[0]
            curve_direction = "increase" if endpoint_change >= 0 else "decrease"
            result["trajectory_class"] = f"Curved {curve_direction}"
    else:
        result["trajectory_class"] = f"Linear {direction}"
    return result


def build_trajectory_table(module, grid: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    early = aggregate_early_peak(module, grid)
    recent = pd.read_pickle(
        CACHE / "strict_peak_only_5km_2001_2024.pkl.gz", compression="gzip"
    )
    panel = pd.concat([early, recent], ignore_index=True)
    panel = panel[
        (panel["strict_area_fraction"] >= module.MIN_MASK_FRACTION)
        & (panel["strict_valid_fraction"] >= module.MIN_VALID_FRACTION)
    ].copy()
    panel["cell_id"] = panel["row"].astype(str) + "_" + panel["col"].astype(str)
    coordinates = module.grid_frame(grid)[["row", "col", "x", "y"]]
    rows = []
    for cell_id, group in panel.groupby("cell_id"):
        fitted = fit_trajectory(group)
        base = group.iloc[0]
        fitted.update({
            "cell_id": cell_id,
            "row": int(base["row"]),
            "col": int(base["col"]),
            "strict_area_fraction": float(group["strict_area_fraction"].median()),
        })
        rows.append(fitted)
    trajectories = pd.DataFrame(rows).merge(
        coordinates, on=["row", "col"], how="left", validate="one_to_one"
    )
    transformer = Transformer.from_crs(grid["crs"], "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(trajectories["x"].to_numpy(), trajectories["y"].to_numpy())
    trajectories["longitude"] = lon
    trajectories["latitude"] = lat
    trajectories["area_km2"] = trajectories["strict_area_fraction"] * 25.0
    trajectories["sensor_transition_break"] = trajectories["break_year"].between(2012, 2014)
    trajectories.to_csv(
        OUTPUTS / "peak_only_trajectory_diagnostics_1988_2024.csv",
        index=False,
        encoding="utf-8-sig",
    )
    panel.to_pickle(OUTPUTS / "peak_only_trajectory_panel_1988_2024.pkl.gz", compression="gzip")
    summary = trajectories.groupby("trajectory_class", as_index=False).agg(
        n_cells=("cell_id", "size"),
        area_km2=("area_km2", "sum"),
        median_sen_slope_g_m2_yr=("sen_slope_g_m2_yr", "median"),
    )
    summary["area_share_pct"] = 100 * summary["area_km2"] / summary["area_km2"].sum()
    summary.to_csv(
        OUTPUTS / "peak_only_trajectory_class_summary_1988_2024.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return trajectories, panel


CLASS_ORDER = [
    "Abrupt decrease",
    "Curved decrease",
    "Linear decrease",
    "Stable or no clear trend",
    "Sensor-transition step",
    "Linear increase",
    "Curved increase",
    "Abrupt increase",
]
CLASS_COLORS = {
    "Abrupt decrease": "#8C2D04",
    "Curved decrease": "#D95F0E",
    "Linear decrease": "#FEC44F",
    "Stable or no clear trend": "#BDBDBD",
    "Sensor-transition step": "#7B6BA8",
    "Linear increase": "#A1DAB4",
    "Curved increase": "#41B6C4",
    "Abrupt increase": "#225EA8",
    "Insufficient data": "#FFFFFF",
}


def format_geo_axes(ax: plt.Axes, label_right: bool = True) -> None:
    ax.tick_params(
        top=True, right=True, labeltop=True, labelright=label_right,
        direction="in", pad=2,
    )
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.1f}°E")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.1f}°N")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_aspect("equal", adjustable="box")
    style_boxed_axis(ax)


def plot_trajectory(trajectories: pd.DataFrame, panel: pd.DataFrame) -> None:
    valid = trajectories[trajectories["trajectory_class"] != "Insufficient data"].copy()
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.6))
    fig.subplots_adjust(left=0.10, right=0.96, top=0.96, bottom=0.10, hspace=0.50, wspace=0.34)
    ax = axes[0, 0]
    for category in CLASS_ORDER:
        part = valid[valid["trajectory_class"] == category]
        if part.empty:
            continue
        ax.scatter(
            part["longitude"], part["latitude"], s=15,
            marker="s", color=CLASS_COLORS[category], linewidths=0,
        )
    format_geo_axes(ax)
    add_panel_label(ax, "a")
    add_panel_subtitle(ax, "Trajectory classes")
    handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=CLASS_COLORS[x],
               markeredgecolor="none", markersize=6, label=x)
        for x in CLASS_ORDER if (valid["trajectory_class"] == x).any()
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True, framealpha=0.92,
              edgecolor="#CCCCCC", fontsize=7)

    ax = axes[0, 1]
    summary = valid.groupby("trajectory_class", as_index=False)["area_km2"].sum()
    summary["share"] = 100 * summary["area_km2"] / summary["area_km2"].sum()
    summary["trajectory_class"] = pd.Categorical(
        summary["trajectory_class"], CLASS_ORDER, ordered=True
    )
    summary = summary.sort_values("trajectory_class")
    ax.barh(
        summary["trajectory_class"].astype(str), summary["share"],
        color=[CLASS_COLORS[x] for x in summary["trajectory_class"].astype(str)],
    )
    for y, value in enumerate(summary["share"]):
        ax.text(value + 0.5, y, f"{value:.1f}%", va="center", fontsize=7)
    ax.set_xlim(0, float(summary["share"].max()) + 8)
    ax.set_xlabel("Area share (%)")
    add_panel_label(ax, "b")
    add_panel_subtitle(ax, "Trajectory-class area share", y=-0.24)
    style_boxed_axis(ax)

    ax = axes[1, 0]
    slope_limit = np.nanpercentile(np.abs(valid["sen_slope_g_m2_yr"]), 97.5)
    norm = TwoSlopeNorm(vmin=-slope_limit, vcenter=0, vmax=slope_limit)
    scatter = ax.scatter(
        valid["longitude"], valid["latitude"], s=15, marker="s",
        c=valid["sen_slope_g_m2_yr"], cmap="RdBu", norm=norm, linewidths=0,
    )
    format_geo_axes(ax, label_right=False)
    add_panel_label(ax, "c")
    add_panel_subtitle(ax, "Pixel-scale Sen slope")
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.040, pad=0.03)
    cbar.set_label("Sen slope (g m$^{-2}$ yr$^{-1}$)")

    ax = axes[1, 1]
    regional = panel.groupby("year").apply(
        lambda x: np.average(x["peak_agb"], weights=x["strict_area_fraction"]),
        include_groups=False,
    )
    years = regional.index.to_numpy(dtype=float)
    values = regional.to_numpy(dtype=float)
    slope, intercept = theilslopes(values, years)[:2]
    ax.plot(years, values, color="#2F6FB0", linewidth=1.2, marker="o", markersize=2.6)
    ax.plot(years, intercept + slope * years, color="#C44E52", linewidth=1.6)
    ax.text(
        0.03, 0.95, f"Sen slope = {slope:.2f} g m$^{{-2}}$ yr$^{{-1}}$",
        transform=ax.transAxes, va="top", fontsize=8,
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Area-weighted AGB (g m$^{-2}$)")
    add_panel_label(ax, "d")
    add_panel_subtitle(ax, "Regional AGB trajectory", y=-0.24)
    style_boxed_axis(ax)
    save_panel_crop(fig, [axes[0, 0]], "Fig_9a_trajectory_class_map")
    save_panel_crop(fig, [axes[0, 1]], "Fig_9b_trajectory_class_area_share")
    save_panel_crop(fig, [axes[1, 0], cbar.ax], "Fig_9c_pixel_sen_slope_map")
    save_panel_crop(fig, [axes[1, 1]], "Fig_9d_regional_AGB_trajectory")
    save(fig, "Fig_9_peak_only_trajectory_diagnostics")


def binned_response(shap_frame: pd.DataFrame, feature: str) -> pd.DataFrame:
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
    summary.insert(0, "feature", feature)
    return summary


TARGET_RESPONSES = [
    ("lhgi_z", "Grazing intensity anomaly (SD)", "primary"),
    ("hfp_z", "Human footprint anomaly (SD)", "primary"),
    ("lai_era_z", "LAI anomaly (SD)", "primary"),
    ("fvc_z", "FVC anomaly (SD)", "sensitivity"),
]


def draw_response_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    label: str,
    letter: str,
    show_ylabel: bool = True,
) -> None:
    ax.fill_between(
        data["feature_value"].to_numpy(), data["shap_p25"].to_numpy(),
        data["shap_p75"].to_numpy(), color="#A9C8E4", alpha=0.45, linewidth=0,
    )
    ax.plot(data["feature_value"], data["shap_median"], color="#2166AC", linewidth=1.8)
    ax.axhline(0, color="#666666", linewidth=0.8)
    ax.axvline(0, color="#BBBBBB", linewidth=0.7, linestyle="--")
    ax.set_xlabel(label)
    if show_ylabel:
        ax.set_ylabel(
            "Median held-out SHAP value (g m\u207b\u00b2)",
            fontfamily="DejaVu Sans",
        )
    else:
        ax.set_ylabel("")
    add_panel_label(ax, letter)
    style_boxed_axis(ax)


def plot_targeted_responses() -> None:
    shap_values = pd.read_pickle(OUTPUTS / "heldout_shap_values.pkl.gz", compression="gzip")
    primary = shap_values[shap_values["model"] == "Reduced full"].copy()
    sensitivity = shap_values[
        shap_values["model"] == "Reduced full + FVC sensitivity"
    ].copy()
    summaries = []
    for feature, label, source in TARGET_RESPONSES:
        frame = primary if source == "primary" else sensitivity
        part = binned_response(frame, feature)
        part["feature_label"] = label
        part["model_role"] = source
        summaries.append(part)
    result = pd.concat(summaries, ignore_index=True)
    result.to_csv(
        OUTPUTS / "heldout_shap_human_vegetation_responses.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.2))
    fig.subplots_adjust(left=0.11, right=0.98, top=0.96, bottom=0.11, hspace=0.36, wspace=0.24)
    for index, (feature, label, _) in enumerate(TARGET_RESPONSES):
        part = result[result["feature"] == feature].sort_values("feature_value")
        draw_response_panel(
            axes.ravel()[index],
            part,
            label,
            chr(ord("a") + index),
            show_ylabel=index % 2 == 0,
        )
    save(fig, "Fig_11_human_vegetation_SHAP_responses")

    panel_stems = (
        "Fig_11a_grazing_intensity_SHAP_response",
        "Fig_11b_human_footprint_SHAP_response",
        "Fig_11c_LAI_state_SHAP_response",
        "Fig_11d_FVC_state_SHAP_response",
    )
    for index, (feature, label, _) in enumerate(TARGET_RESPONSES):
        part = result[result["feature"] == feature].sort_values("feature_value")
        fig, ax = plt.subplots(figsize=(3.45, 3.05), constrained_layout=True)
        draw_response_panel(
            ax,
            part,
            label,
            chr(ord("a") + index),
            show_ylabel=index % 2 == 0,
        )
        save(fig, panel_stems[index], panel=True)


def export_existing_composite_panels() -> None:
    metrics = pd.read_csv(OUTPUTS / "blocked_validation_metrics.csv")
    importance = pd.read_csv(OUTPUTS / "heldout_shap_importance_full_exogenous.csv")
    grouped = pd.read_csv(OUTPUTS / "heldout_shap_group_importance.csv")
    overall = metrics[metrics["fold"] == "overall_oof"].copy()
    model_order = [
        "CRU core", "All climate", "Climate + vegetation",
        "Climate + human", "Full exogenous", "Full + FVC sensitivity",
    ]
    colors = {"spatial block": "#2F6FB0", "temporal block": "#D65F4A"}

    fig, ax = plt.subplots(figsize=(6.8, 3.4), constrained_layout=True)
    x = np.arange(len(model_order))
    width = 0.36
    for offset, validation in ((-width / 2, "spatial block"), (width / 2, "temporal block")):
        values = overall.set_index(["model", "validation"]).reindex(
            pd.MultiIndex.from_product([model_order, [validation]])
        )["r2"].to_numpy()
        ax.bar(x + offset, values, width, color=colors[validation], label=validation.capitalize())
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_ylabel("Out-of-fold predictive $R^2$")
    ax.set_xticks(x, model_order, rotation=25, ha="right")
    ax.legend(frameon=False, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "Fig_3_4_2a_model_validation", panel=True)

    fig, ax = plt.subplots(figsize=(4.8, 3.4), constrained_layout=True)
    ordered = grouped.sort_values("share_pct")
    palette = {
        "CRU climate": "#4C78A8", "TerraClimate": "#72B7B2",
        "ERA5-Land": "#59A14F", "Vegetation state": "#F2CF5B",
        "Human activity": "#B279A2", "Soil and terrain": "#9D9D9D",
    }
    ax.barh(ordered["factor_group"], ordered["share_pct"],
            color=[palette.get(x, "#777777") for x in ordered["factor_group"]])
    ax.set_xlabel("Grouped held-out SHAP contribution (%)")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "Fig_3_4_2c_grouped_shap", panel=True)

    fig, ax = plt.subplots(figsize=(5.0, 4.5), constrained_layout=True)
    top = importance.head(15).sort_values("mean_abs_shap")
    ax.barh(top["feature_label"], top["mean_abs_shap"],
            color=[palette.get(x, "#777777") for x in top["factor_group"]])
    ax.set_xlabel("Mean |held-out SHAP| (g m$^{-2}$)")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "Fig_3_4_2d_feature_shap", panel=True)


def refresh_main_shap_outputs(module) -> None:
    groups = module.json.loads((OUTPUTS / "feature_manifest.json").read_text(encoding="utf-8"))
    shap_values = pd.read_pickle(OUTPUTS / "heldout_shap_values.pkl.gz", compression="gzip")
    primary = shap_values[shap_values["model"] == "Full exogenous"].copy()
    importance, grouped = module.summarize_shap(primary, groups)
    dependence = module.binned_dependence(primary, importance, top_n=4)
    importance.to_csv(
        OUTPUTS / "heldout_shap_importance_full_exogenous.csv",
        index=False,
        encoding="utf-8-sig",
    )
    grouped.to_csv(
        OUTPUTS / "heldout_shap_group_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    dependence.to_csv(
        OUTPUTS / "heldout_shap_dependence_binned.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics = pd.read_csv(OUTPUTS / "blocked_validation_metrics.csv")
    module.plot_model_comparison(metrics, grouped, importance)
    module.plot_beeswarm_and_dependence(primary, importance, dependence)


def main() -> None:
    style()
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_targeted_responses()
    print("Reduced-model targeted SHAP response figures complete.", flush=True)


if __name__ == "__main__":
    main()
