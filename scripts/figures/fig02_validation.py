"""Build manuscript figures for Results section 3.1.

The main figure combines the strict peatland mask, spatially blocked boundary
validation, Peak-only AGB spatial validation, and validation-transfer tests.
The supplementary figure compares the quality-controlled 2025 primary set
with the complete-record sensitivity analysis.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
import rasterio
from scipy.stats import pearsonr


ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "agb_paper_validation_freeze"
INPUT_DIR = RUN_DIR / "outputs"
FIG_DIR = RUN_DIR / "figures_section_3_1"
PANEL_DIR = FIG_DIR / "separate_reference_style_panels"
SOURCE_DIR = RUN_DIR / "source_data_section_3_1"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PANEL_DIR.mkdir(parents=True, exist_ok=True)
SOURCE_DIR.mkdir(parents=True, exist_ok=True)

MASK_PATH = (
    ROOT
    / "zoigepeatland"
    / "output_basic_boundary"
    / "Zoige_RF_basic_analysis_mask_main95_fill400.tif"
)
BOUNDARY_METRICS = INPUT_DIR / "boundary_spatial_block_metrics.csv"
BOUNDARY_PREDICTIONS = INPUT_DIR / "boundary_spatial_block_predictions.csv"
MODEL_METRICS = (
    ROOT / "agb_dual_grid_2013_2024" / "outputs" / "model_spatial_block_validation.csv"
)
MODEL_PREDICTIONS = (
    ROOT
    / "paper_figures_group_meeting_2026_07_17"
    / "outputs"
    / "fig01_spatial_validation_predictions.csv"
)
EVIDENCE_SUMMARY = INPUT_DIR / "paper_validation_evidence_summary.csv"
VALIDATION_2025_METRICS = (
    ROOT / "agb_2025_independent_validation" / "outputs" / "metrics_2025.csv"
)
VALIDATION_2025_PREDICTIONS = (
    ROOT / "agb_2025_independent_validation" / "outputs" / "field_predictions_2025.csv"
)

BLUE = "#356FB3"
TEAL = "#2A9D8F"
ORANGE = "#D97732"
RED = "#C74E45"
NAVY = "#244C7A"
LIGHT_BLUE = "#CFE6EA"
GRAY = "#72777D"
LIGHT_GRAY = "#E8EAED"
BLACK = "#252525"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.2,
        "axes.titlesize": 8.1,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.7,
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    }
)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="top",
    )


def save_bundle(fig: plt.Figure, stem: str) -> None:
    for suffix, kwargs in (
        ("png", {"dpi": 400}),
        ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
        ("pdf", {}),
        ("svg", {}),
    ):
        fig.savefig(FIG_DIR / f"{stem}.{suffix}", bbox_inches="tight", **kwargs)


def save_panel_crop(fig: plt.Figure, ax: plt.Axes, stem: str) -> None:
    """Export one supplementary panel from its source composite."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = Bbox.union([ax.get_tightbbox(renderer)])
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted()).expanded(1.04, 1.04)
    for suffix, kwargs in (
        ("png", {"dpi": 400}),
        ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
        ("pdf", {}),
        ("svg", {}),
    ):
        fig.savefig(
            PANEL_DIR / f"{stem}.{suffix}",
            bbox_inches=bbox,
            facecolor="white",
            **kwargs,
        )


def rmse_from_role(role: str) -> float:
    match = re.search(r"RMSE=([0-9.]+)", str(role))
    if not match:
        raise ValueError(f"RMSE not found in evidence role: {role}")
    return float(match.group(1))


def p_label(value: float) -> str:
    return "p<0.001" if value < 0.001 else f"p={value:.3f}"


def add_fit_line(ax: plt.Axes, observed: np.ndarray, predicted: np.ndarray) -> None:
    slope, intercept = np.polyfit(observed, predicted, 1)
    x_values = np.array([float(np.nanmin(observed)), float(np.nanmax(observed))])
    ax.plot(x_values, intercept + slope * x_values, color=RED, lw=1.1, zorder=1)


def load_transfer_metrics() -> pd.DataFrame:
    spatial = pd.read_csv(MODEL_METRICS).set_index("experiment")
    evidence = pd.read_csv(EVIDENCE_SUMMARY)
    metrics_2025 = pd.read_csv(VALIDATION_2025_METRICS)

    rows: list[dict] = []
    for experiment, label, family in (
        ("Baseline_Peak_PLSR", "Peak-only | spatial", "Peak-only"),
        ("TC_CRU_consensus_core3_PLSR", "Guarded | spatial", "Guarded"),
    ):
        row = spatial.loc[experiment]
        rows.append(
            {
                "label": label,
                "family": family,
                "validation": "5-km spatial blocks",
                "n": int(row["n"]),
                "pearson_r": float(row["pearson_r"]),
                "predictive_r_squared": float(row["predictive_r_squared"]),
                "rmse_g_m2": float(row["rmse_g_m2"]),
            }
        )

    for evidence_name, label, family in (
        ("Baseline_Peak_PLSR_leave_year", "Peak-only | leave-year", "Peak-only"),
        ("Guarded_consensus_leave_year", "Guarded | leave-year", "Guarded"),
    ):
        row = evidence[evidence["evidence"].eq(evidence_name)].iloc[0]
        rows.append(
            {
                "label": label,
                "family": family,
                "validation": "leave-year",
                "n": np.nan,
                "pearson_r": float(row["primary_value"]),
                "predictive_r_squared": float(row["secondary_value"]),
                "rmse_g_m2": rmse_from_role(row["paper_role"]),
            }
        )

    row_2025 = metrics_2025[
        metrics_2025["support"].eq("grid_product_90m")
        & metrics_2025["scenario"].eq("quality_controlled_primary")
        & metrics_2025["weighting"].eq("record_level")
    ].iloc[0]
    rows.append(
        {
            "label": "Peak-only | 2025",
            "family": "2025 independent",
            "validation": "2025 independent",
            "n": int(row_2025["n"]),
            "pearson_r": float(row_2025["pearson_r"]),
            "predictive_r_squared": float(row_2025["predictive_r_squared"]),
            "rmse_g_m2": float(row_2025["rmse_g_m2"]),
        }
    )
    result = pd.DataFrame(rows)
    result.to_csv(SOURCE_DIR / "validation_transfer_metrics.csv", index=False)
    return result


def plot_boundary_map(ax: plt.Axes, predictions: pd.DataFrame) -> None:
    with rasterio.open(MASK_PATH) as src:
        factor = 5
        height = max(1, src.height // factor)
        width = max(1, src.width // factor)
        mask = src.read(1, out_shape=(height, width)) > 0
        bounds = src.bounds

    shown = np.ma.masked_where(~mask, mask.astype(float))
    ax.imshow(
        shown,
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        origin="upper",
        cmap=ListedColormap([LIGHT_BLUE]),
        interpolation="nearest",
        alpha=0.95,
        zorder=0,
    )
    ax.contour(
        np.linspace(bounds.left, bounds.right, mask.shape[1]),
        np.linspace(bounds.top, bounds.bottom, mask.shape[0]),
        mask.astype(float),
        levels=[0.5],
        colors=[NAVY],
        linewidths=0.55,
        zorder=1,
    )

    for class_value, color, label in (
        (0, GRAY, "Non-peat samples"),
        (1, TEAL, "Peat samples"),
    ):
        subset = predictions[predictions["class"].eq(class_value)]
        ax.scatter(
            subset["lon"],
            subset["lat"],
            s=7,
            c=color,
            alpha=0.55,
            edgecolors="white",
            linewidths=0.18,
            label=label,
            zorder=2,
        )
    errors = predictions[~predictions["correct"]]
    ax.scatter(
        errors["lon"],
        errors["lat"],
        s=17,
        facecolors="none",
        edgecolors=RED,
        linewidths=0.8,
        marker="o",
        label="OOF errors",
        zorder=3,
    )
    ax.set_xlim(bounds.left, bounds.right)
    ax.set_ylim(bounds.bottom, bounds.top)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title("Strict peatland analysis mask (4,673.4 km²)", loc="left")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.80,
        handletextpad=0.35,
        labelspacing=0.22,
        borderpad=0.25,
    )

    latitude = 33.25
    bar_km = 50
    bar_deg = bar_km / (111.32 * np.cos(np.deg2rad(latitude)))
    x0 = bounds.left + 0.17
    y0 = bounds.bottom + 0.16
    ax.plot([x0, x0 + bar_deg], [y0, y0], color=BLACK, lw=2.0, solid_capstyle="butt")
    ax.text(x0 + bar_deg / 2, y0 + 0.045, "50 km", ha="center", va="bottom", fontsize=6.2)
    panel_label(ax, "a")


def plot_confusion_matrix(ax: plt.Axes, metric: pd.Series, label: str = "b") -> None:
    matrix = np.array(
        [[metric["tn"], metric["fp"]], [metric["fn"], metric["tp"]]], dtype=float
    )
    row_percent = matrix / matrix.sum(axis=1, keepdims=True) * 100.0
    ax.imshow(row_percent, cmap="Blues", vmin=0, vmax=100)
    for row in range(2):
        for column in range(2):
            color = "white" if row_percent[row, column] > 55 else BLACK
            ax.text(
                column,
                row,
                f"{int(matrix[row, column])}\n({row_percent[row, column]:.1f}%)",
                ha="center",
                va="center",
                color=color,
                fontsize=7.5,
                fontweight="bold" if row == column else "normal",
            )
    ax.set_xticks([0, 1], ["Non-peat", "Peat"])
    ax.set_yticks([0, 1], ["Non-peat", "Peat"])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Observed class")
    ax.set_title("25-km spatial-block boundary validation", loc="left")
    ax.text(
        0.5,
        -0.25,
        (
            f"n={int(metric['n'])}   Accuracy={metric['accuracy']:.3f}   "
            f"AUC={metric['auc']:.3f}\nKappa={metric['kappa']:.3f}   "
            f"Peat F1={metric['peat_f1']:.3f}"
        ),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=6.7,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    panel_label(ax, label)


def plot_peak_scatter(
    ax: plt.Axes,
    predictions: pd.DataFrame,
    metric: pd.Series,
    label: str = "c",
) -> None:
    observed = predictions["harmonized_agb_g_m2"].to_numpy(float)
    predicted = predictions["prediction_g_m2"].to_numpy(float)
    upper = 1200.0
    ax.scatter(
        observed,
        predicted,
        color=BLUE,
        s=18,
        alpha=0.72,
        edgecolors="white",
        linewidths=0.28,
        zorder=2,
    )
    ax.plot([0, upper], [0, upper], ls="--", lw=0.9, color=GRAY, zorder=1)
    add_fit_line(ax, observed, predicted)
    p_value = float(pearsonr(observed, predicted).pvalue)
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Observed AGB (g m$^{-2}$)")
    ax.set_ylabel("Predicted AGB (g m$^{-2}$)")
    ax.set_title("Peak-only AGB: 5-km spatial-block OOF", loc="left")
    ax.text(
        0.04,
        0.96,
        (
            f"n={int(metric['n'])}\n"
            f"r={metric['pearson_r']:.3f}, {p_label(p_value)}\n"
            f"RMSE={metric['rmse_g_m2']:.1f} g m$^{{-2}}$\n"
            f"MAE={metric['mae_g_m2']:.1f} g m$^{{-2}}$"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.7,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2},
    )
    panel_label(ax, label)


def plot_transfer_summary(ax: plt.Axes, metrics: pd.DataFrame) -> None:
    colors = {
        "Peak-only": BLUE,
        "Guarded": TEAL,
        "2025 independent": RED,
    }
    y = np.arange(len(metrics))[::-1]
    ax.axvline(0, color=GRAY, lw=0.8, zorder=0)
    for yi, (_, row) in zip(y, metrics.iterrows()):
        color = colors[row["family"]]
        ax.plot(
            [row["predictive_r_squared"], row["pearson_r"]],
            [yi, yi],
            color=color,
            alpha=0.45,
            lw=1.0,
            zorder=1,
        )
        ax.scatter(row["pearson_r"], yi, s=31, marker="o", color=color, edgecolor="white", linewidth=0.45, zorder=3)
        ax.scatter(row["predictive_r_squared"], yi, s=29, marker="s", color=color, edgecolor="white", linewidth=0.45, zorder=3)
        ax.text(0.68, yi, f"RMSE {row['rmse_g_m2']:.1f}", va="center", ha="left", fontsize=6.3, color=BLACK)
    ax.set_yticks(y, metrics["label"])
    ax.set_xlim(-0.72, 0.98)
    ax.set_xticks([-0.5, 0.0, 0.5])
    ax.set_xlabel("Validation statistic")
    ax.set_title("Spatial and temporal transfer tests", loc="left")
    ax.grid(axis="x", color=LIGHT_GRAY, lw=0.6, zorder=0)
    ax.scatter([], [], marker="o", s=28, color=GRAY, label="Pearson r")
    ax.scatter([], [], marker="s", s=26, color=GRAY, label="Predictive R$^2$")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, -0.29), ncol=2, frameon=False, columnspacing=1.0)
    ax.text(
        0.99,
        -0.22,
        "RMSE in g m$^{-2}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.2,
        color=GRAY,
    )
    panel_label(ax, "d")


def build_main_figure() -> None:
    boundary_predictions = pd.read_csv(BOUNDARY_PREDICTIONS)
    boundary_metrics = pd.read_csv(BOUNDARY_METRICS)
    boundary_overall = boundary_metrics[
        boundary_metrics["validation"].eq("spatial_block_25km_oof")
    ].iloc[0]
    model_predictions = pd.read_csv(MODEL_PREDICTIONS)
    peak_predictions = model_predictions[
        model_predictions["experiment"].eq("Baseline_Peak_PLSR")
    ].copy()
    model_metrics = pd.read_csv(MODEL_METRICS).set_index("experiment")
    peak_metric = model_metrics.loc["Baseline_Peak_PLSR"]
    load_transfer_metrics()
    predictions_2025 = pd.read_csv(VALIDATION_2025_PREDICTIONS)
    predictions_2025 = predictions_2025[
        predictions_2025["support"].eq("grid_product_90m")
    ].copy()
    qc_pass_2025 = predictions_2025["independent_qc_pass"].astype(str).str.lower().eq("true")
    metrics_2025 = pd.read_csv(VALIDATION_2025_METRICS)
    primary_2025 = metrics_2025[
        metrics_2025["support"].eq("grid_product_90m")
        & metrics_2025["scenario"].eq("quality_controlled_primary")
        & metrics_2025["weighting"].eq("record_level")
    ].iloc[0]

    boundary_predictions.to_csv(SOURCE_DIR / "boundary_oof_predictions.csv", index=False)
    peak_predictions.to_csv(SOURCE_DIR / "peak_only_spatial_oof_predictions.csv", index=False)

    fig = plt.figure(figsize=(7.2, 6.15))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.06, 0.94],
        height_ratios=[0.94, 1.06],
        hspace=0.50,
        wspace=0.38,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])
    plot_boundary_map(ax_a, boundary_predictions)
    plot_confusion_matrix(ax_b, boundary_overall)
    plot_peak_scatter(ax_c, peak_predictions, peak_metric)
    scatter_2025(
        ax_d,
        predictions_2025.loc[qc_pass_2025],
        primary_2025,
        "2025 quality-controlled validation",
        False,
    )
    panel_label(ax_d, "d")
    fig.subplots_adjust(left=0.095, right=0.975, top=0.965, bottom=0.09)
    save_bundle(fig, "Fig_2_boundary_and_agb_validation")
    plt.close(fig)


def scatter_2025(
    ax: plt.Axes,
    data: pd.DataFrame,
    metric: pd.Series,
    title: str,
    show_review: bool,
) -> None:
    qc_pass = data["independent_qc_pass"].astype(str).str.lower().eq("true")
    ax.scatter(
        data.loc[qc_pass, "agb_mean_g_m2"],
        data.loc[qc_pass, "predicted_agb_g_m2"],
        s=33,
        color=BLUE,
        edgecolors="white",
        linewidths=0.5,
        label="Quality-controlled records",
        zorder=2,
    )
    if show_review and (~qc_pass).any():
        ax.scatter(
            data.loc[~qc_pass, "agb_mean_g_m2"],
            data.loc[~qc_pass, "predicted_agb_g_m2"],
            s=43,
            marker="D",
            color=ORANGE,
            edgecolors="white",
            linewidths=0.5,
            label="QC-review records",
            zorder=3,
        )
    upper = 1300 if show_review else 500
    ax.plot([0, upper], [0, upper], color=GRAY, ls="--", lw=0.9, zorder=0)
    observed = data["agb_mean_g_m2"].to_numpy(float)
    predicted = data["predicted_agb_g_m2"].to_numpy(float)
    add_fit_line(ax, observed, predicted)
    p_value = float(pearsonr(observed, predicted).pvalue)
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Observed dry AGB (g m$^{-2}$)")
    ax.set_ylabel("Predicted AGB (g m$^{-2}$)")
    ax.set_title(title, loc="left")
    ax.text(
        0.04,
        0.96,
        (
            f"n={int(metric['n'])}\n"
            f"r={metric['pearson_r']:.3f}, {p_label(p_value)}\n"
            f"RMSE={metric['rmse_g_m2']:.1f} g m$^{{-2}}$\n"
            f"MAE={metric['mae_g_m2']:.1f} g m$^{{-2}}$"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 2},
    )


def build_manuscript_figure_3() -> None:
    boundary_metrics = pd.read_csv(BOUNDARY_METRICS)
    boundary_overall = boundary_metrics[
        boundary_metrics["validation"].eq("spatial_block_25km_oof")
    ].iloc[0]
    model_predictions = pd.read_csv(MODEL_PREDICTIONS)
    peak_predictions = model_predictions[
        model_predictions["experiment"].eq("Baseline_Peak_PLSR")
    ].copy()
    model_metrics = pd.read_csv(MODEL_METRICS).set_index("experiment")
    peak_metric = model_metrics.loc["Baseline_Peak_PLSR"]

    predictions_2025 = pd.read_csv(VALIDATION_2025_PREDICTIONS)
    predictions_2025 = predictions_2025[
        predictions_2025["support"].eq("grid_product_90m")
    ].copy()
    qc_pass_2025 = predictions_2025["independent_qc_pass"].astype(str).str.lower().eq("true")
    metrics_2025 = pd.read_csv(VALIDATION_2025_METRICS)
    primary_2025 = metrics_2025[
        metrics_2025["support"].eq("grid_product_90m")
        & metrics_2025["scenario"].eq("quality_controlled_primary")
        & metrics_2025["weighting"].eq("record_level")
    ].iloc[0]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75))
    plot_confusion_matrix(axes[0], boundary_overall, label="a")
    axes[0].set_title("Peatland boundary: 25-km OOF", loc="left")
    plot_peak_scatter(axes[1], peak_predictions, peak_metric, label="b")
    axes[1].set_title("Peak-only AGB: 5-km OOF", loc="left")
    scatter_2025(
        axes[2],
        predictions_2025.loc[qc_pass_2025],
        primary_2025,
        "2025 quality-controlled validation",
        False,
    )
    panel_label(axes[2], "c")
    fig.subplots_adjust(left=0.075, right=0.99, top=0.92, bottom=0.23, wspace=0.47)
    save_bundle(fig, "Fig_3_validation_of_peatland_delineation_and_agb_estimation")
    plt.close(fig)


def build_2025_supplement() -> None:
    predictions = pd.read_csv(VALIDATION_2025_PREDICTIONS)
    predictions = predictions[predictions["support"].eq("grid_product_90m")].copy()
    metrics = pd.read_csv(VALIDATION_2025_METRICS)
    primary = metrics[
        metrics["support"].eq("grid_product_90m")
        & metrics["scenario"].eq("quality_controlled_primary")
        & metrics["weighting"].eq("record_level")
    ].iloc[0]
    sensitivity = metrics[
        metrics["support"].eq("grid_product_90m")
        & metrics["scenario"].eq("all_records_sensitivity")
        & metrics["weighting"].eq("record_level")
    ].iloc[0]
    qc_pass = predictions["independent_qc_pass"].astype(str).str.lower().eq("true")

    predictions.to_csv(SOURCE_DIR / "independent_2025_predictions.csv", index=False)
    metrics[
        metrics["support"].eq("grid_product_90m")
        & metrics["weighting"].eq("record_level")
        & metrics["scenario"].isin(["quality_controlled_primary", "all_records_sensitivity"])
    ].to_csv(SOURCE_DIR / "independent_2025_metrics.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25))
    scatter_2025(
        axes[0],
        predictions.loc[qc_pass],
        primary,
        "Quality-controlled 2025 records (primary)",
        False,
    )
    scatter_2025(
        axes[1],
        predictions,
        sensitivity,
        "All 2025 candidate records (sensitivity)",
        True,
    )
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    axes[0].legend(loc="lower right", frameon=False)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.94, bottom=0.16, wspace=0.30)
    save_panel_crop(fig, axes[0], "Fig_S1a_quality_controlled_2025_records")
    save_panel_crop(fig, axes[1], "Fig_S1b_all_2025_candidate_records")
    save_bundle(fig, "Fig_S1_independent_2025_validation_sensitivity")
    plt.close(fig)


def main() -> None:
    build_main_figure()
    build_manuscript_figure_3()
    build_2025_supplement()
    print(f"Figures written to: {FIG_DIR}")
    print(f"Source data written to: {SOURCE_DIR}")


if __name__ == "__main__":
    main()
