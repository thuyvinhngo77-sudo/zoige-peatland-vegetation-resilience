#!/usr/bin/env python
"""Create manuscript figures for Section 3.2 from frozen analysis outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.transforms import Bbox
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject
from rasterio.windows import Window, transform as window_transform
from scipy.stats import kendalltau, theilslopes


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "agb_peakonly_spatiotemporal_1988_2025"
OUT = RUN / "outputs"
RASTERS = RUN / "rasters"
FIGURES = RUN / "figures_section_3_2_final"
PANELS = FIGURES / "panels"

REFERENCE = (
    ROOT / "agb_dual_grid_2013_2024" / "rasters_90m"
    / "Zoige_AGB_GuardedConsensus_PLSR_90m_2024.tif"
)
STRICT_MASK = (
    ROOT / "zoigepeatland" / "output_basic_boundary"
    / "Zoige_RF_basic_analysis_mask_main95_fill400.tif"
)
MULTIYEAR = RASTERS / "PeakOnly_AGB_multiyear_stats_1988_2025_90m.tif"
PIXEL_TREND = RASTERS / "PeakOnly_AGB_pixel_trend_1988_2025_90m.tif"
ANNUAL = OUT / "annual_strict_main95_peakonly_stats_1988_2025.csv"
STATE = OUT / "annual_fvc_lai_state_1988_2025.csv"
REGIONAL_TRENDS = OUT / "regional_trend_sensitivity_1988_2025.csv"
TREND_AREA = OUT / "pixel_trend_class_area_1988_2025.csv"
PEAK_GUARDED = OUT / "annual_peak_guarded_comparison_1988_2024.csv"

REPRESENTATIVE_YEARS = (1989, 2000, 2013, 2025)

# Fixed classes retain cross-year comparability while resolving the narrow
# central distribution of peatland AGB more clearly than a full-range scale.
AGB_CLASS_BOUNDS = [0, 250, 275, 300, 325, 350, 375, 2001]
AGB_CLASS_LABELS = [
    "<250", "250–275", "275–300", "300–325",
    "325–350", "350–375", "≥375",
]
AGB_CLASS_COLORS = [
    "#fff7bc", "#d9f0a3", "#addd8e", "#78c679",
    "#41b6c4", "#1d91c0", "#225ea8",
]


mpl.use("Agg")
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7.5,
    "axes.linewidth": 0.7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "legend.fontsize": 6.5,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
})


def load_grid() -> tuple[dict, np.ndarray, tuple[int, int, int, int]]:
    with rasterio.open(REFERENCE) as ref:
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


def read_bbox(path: Path, band: int, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, r1, c0, c1 = bbox
    window = Window(c0, r0, c1 - c0, r1 - r0)
    with rasterio.open(path) as src:
        array = src.read(band, window=window).astype("float32")
        invalid = ~np.isfinite(array)
        if src.nodata is not None:
            invalid |= np.isclose(array, src.nodata)
        array[invalid] = np.nan
    return array


def map_extent(profile: dict, bbox: tuple[int, int, int, int]) -> list[float]:
    r0, r1, c0, c1 = bbox
    transform = profile["transform"]
    left = transform.c + c0 * transform.a
    right = transform.c + c1 * transform.a
    top = transform.f + r0 * transform.e
    bottom = transform.f + r1 * transform.e
    return [left / 1000, right / 1000, bottom / 1000, top / 1000]


def geographic_grid(
    profile: dict,
    bbox: tuple[int, int, int, int],
) -> tuple[object, object, int, int, list[float]]:
    r0, r1, c0, c1 = bbox
    height = r1 - r0
    width = c1 - c0
    src_transform = window_transform(Window(c0, r0, width, height), profile["transform"])
    left, bottom, right, top = array_bounds(height, width, src_transform)
    dst_transform, dst_width, dst_height = calculate_default_transform(
        profile["crs"], "EPSG:4326", width, height, left, bottom, right, top
    )
    dst_left, dst_bottom, dst_right, dst_top = array_bounds(
        dst_height, dst_width, dst_transform
    )
    extent = [dst_left, dst_right, dst_bottom, dst_top]
    return src_transform, dst_transform, dst_width, dst_height, extent


def reproject_local_array(
    array: np.ndarray,
    profile: dict,
    src_transform,
    dst_transform,
    dst_width: int,
    dst_height: int,
    *,
    is_mask: bool = False,
) -> np.ndarray:
    if is_mask:
        destination = np.zeros((dst_height, dst_width), dtype="uint8")
        reproject(
            source=array.astype("uint8"),
            destination=destination,
            src_transform=src_transform,
            src_crs=profile["crs"],
            dst_transform=dst_transform,
            dst_crs="EPSG:4326",
            src_nodata=0,
            dst_nodata=0,
            resampling=Resampling.nearest,
        )
        return destination > 0

    destination = np.full((dst_height, dst_width), np.nan, dtype="float32")
    reproject(
        source=array.astype("float32"),
        destination=destination,
        src_transform=src_transform,
        src_crs=profile["crs"],
        dst_transform=dst_transform,
        dst_crs="EPSG:4326",
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=Resampling.nearest,
    )
    return destination


def square_geographic_extent(extent: list[float]) -> list[float]:
    left, right, bottom, top = extent
    centre_lon = (left + right) / 2
    centre_lat = (bottom + top) / 2
    lon_span = right - left
    lat_span = top - bottom
    cos_lat = np.cos(np.deg2rad(centre_lat))
    physical_span = max(lon_span * cos_lat, lat_span) * 1.07
    half_lon = physical_span / (2 * cos_lat)
    half_lat = physical_span / 2
    return [
        centre_lon - half_lon,
        centre_lon + half_lon,
        centre_lat - half_lat,
        centre_lat + half_lat,
    ]


def _longitude_label(value: float, _position: int) -> str:
    return f"{value:.1f}\N{DEGREE SIGN}E"


def _latitude_label(value: float, _position: int) -> str:
    return f"{value:.1f}\N{DEGREE SIGN}N"


def style_geographic_map_axis(ax: plt.Axes, extent: list[float]) -> None:
    left, right, bottom, top = square_geographic_extent(extent)
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_box_aspect(1)

    lon_ticks = np.linspace(left, right, 4)[1:-1]
    lat_ticks = np.linspace(bottom, top, 4)[1:-1]
    ax.xaxis.set_major_locator(FixedLocator(lon_ticks))
    ax.yaxis.set_major_locator(FixedLocator(lat_ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(_longitude_label))
    ax.yaxis.set_major_formatter(FuncFormatter(_latitude_label))
    ax.tick_params(
        direction="in", top=True, right=True,
        labeltop=True, labelright=True, pad=2,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("0.25")
        spine.set_linewidth(0.7)


def style_map_axis(ax: plt.Axes, show_y: bool = True) -> None:
    ax.set_xlabel("Easting (km)")
    if show_y:
        ax.set_ylabel("Northing (km)")
    else:
        ax.tick_params(labelleft=False)
    ax.tick_params(direction="in", top=True, right=True)
    ax.set_aspect("equal", adjustable="box")


def add_boundary(ax: plt.Axes, local_mask: np.ndarray, extent: list[float]) -> None:
    ax.contour(
        local_mask.astype(float), levels=[0.5], colors="0.20", linewidths=0.35,
        extent=extent, origin="upper",
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08, 1.03, f"({label})", transform=ax.transAxes, fontsize=8.5,
        fontweight="bold", va="bottom", ha="left",
    )


def style_boxed_axis(ax: plt.Axes) -> None:
    """Apply the common manuscript frame used by every non-map panel."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("0.25")
        spine.set_linewidth(0.7)
    ax.tick_params(direction="in", top=True, right=True)


def add_panel_subtitle(ax: plt.Axes, text: str, y: float = -0.22) -> None:
    ax.text(
        0.5, y, text, transform=ax.transAxes,
        ha="center", va="top", fontsize=8,
    )


def agb_class_style() -> tuple[ListedColormap, BoundaryNorm, list[float]]:
    cmap = ListedColormap(AGB_CLASS_COLORS)
    norm = BoundaryNorm(AGB_CLASS_BOUNDS, cmap.N, clip=True)
    centers = [
        (left + right) / 2
        for left, right in zip(AGB_CLASS_BOUNDS[:-1], AGB_CLASS_BOUNDS[1:])
    ]
    return cmap, norm, centers


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (
        ("png", {"dpi": 450}),
        ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
        ("pdf", {}),
        ("svg", {}),
    ):
        fig.savefig(FIGURES / f"{stem}.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def save_panel_crop(
    fig: plt.Figure,
    axes: list[plt.Axes],
    stem: str,
    pad_fraction: float = 0.04,
) -> None:
    """Export one logical panel from a composite without duplicating plot code."""
    PANELS.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = Bbox.union([axis.get_tightbbox(renderer) for axis in axes])
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    bbox = bbox.expanded(1 + pad_fraction, 1 + pad_fraction)
    for suffix, kwargs in (
        ("png", {"dpi": 450}),
        ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
        ("pdf", {}),
        ("svg", {}),
    ):
        fig.savefig(
            PANELS / f"{stem}.{suffix}",
            bbox_inches=bbox,
            facecolor="white",
            **kwargs,
        )


def annual_raster(year: int) -> tuple[Path, int]:
    if year <= 2024:
        return (
            ROOT / "agb_dual_grid_2013_2024" / "rasters_90m"
            / f"Zoige_AGB_GuardedConsensus_PLSR_90m_{year}.tif",
            3,
        )
    return (
        ROOT / "agb_2025_independent_validation" / "rasters"
        / "Zoige_AGB_PeakOnly_PLSR_90m_2025.tif",
        1,
    )


def figure_4_mean_agb(profile: dict, strict: np.ndarray, bbox) -> None:
    local_mask = strict[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    mean_agb = read_bbox(MULTIYEAR, 1, bbox)
    temporal_cv = read_bbox(MULTIYEAR, 3, bbox)
    mean_agb[~local_mask] = np.nan
    temporal_cv[~local_mask] = np.nan

    src_transform, dst_transform, dst_width, dst_height, extent = geographic_grid(
        profile, bbox
    )
    mean_agb = reproject_local_array(
        mean_agb, profile, src_transform, dst_transform, dst_width, dst_height
    )
    temporal_cv = reproject_local_array(
        temporal_cv, profile, src_transform, dst_transform, dst_width, dst_height
    )
    local_mask = reproject_local_array(
        local_mask, profile, src_transform, dst_transform, dst_width, dst_height,
        is_mask=True,
    )
    mean_agb[~local_mask] = np.nan
    temporal_cv[~local_mask] = np.nan

    fig = plt.figure(figsize=(7.2, 4.35))
    axes = [
        fig.add_axes([0.075, 0.28, 0.36, 0.60]),
        fig.add_axes([0.565, 0.28, 0.36, 0.60]),
    ]
    cmap, norm, centers = agb_class_style()
    mean_im = axes[0].imshow(
        mean_agb, extent=extent, origin="upper", cmap=cmap, norm=norm,
        interpolation="nearest", aspect="auto",
    )
    axes[0].contour(
        local_mask.astype(float), levels=[0.5], colors="0.68", linewidths=0.18,
        extent=extent, origin="upper",
    )
    style_geographic_map_axis(axes[0], extent)
    axes[0].text(
        0.5, -0.115, "Mean AGB", transform=axes[0].transAxes,
        ha="center", va="top", fontsize=8,
    )
    add_panel_label(axes[0], "a")

    cv_im = axes[1].imshow(
        temporal_cv, extent=extent, origin="upper", cmap="YlOrRd",
        vmin=0.08, vmax=0.20, interpolation="nearest", aspect="auto",
    )
    axes[1].contour(
        local_mask.astype(float), levels=[0.5], colors="0.68", linewidths=0.18,
        extent=extent, origin="upper",
    )
    style_geographic_map_axis(axes[1], extent)
    axes[1].text(
        0.5, -0.115, "Interannual variability", transform=axes[1].transAxes,
        ha="center", va="top", fontsize=8,
    )
    add_panel_label(axes[1], "b")

    mean_cax = fig.add_axes([0.055, 0.075, 0.40, 0.023])
    cbar = fig.colorbar(
        mean_im, cax=mean_cax, orientation="horizontal",
        boundaries=AGB_CLASS_BOUNDS, spacing="uniform",
    )
    cbar.set_label("AGB (g m$^{-2}$)")
    cbar.set_ticks(centers)
    cbar.set_ticklabels(AGB_CLASS_LABELS)
    cbar.ax.tick_params(labelsize=5.2, pad=1.2)

    cv_cax = fig.add_axes([0.545, 0.075, 0.40, 0.023])
    cbar = fig.colorbar(cv_im, cax=cv_cax, orientation="horizontal", extend="both")
    cbar.set_label("Temporal coefficient of variation")
    cbar.set_ticks([0.08, 0.12, 0.16, 0.20])
    save_panel_crop(fig, [axes[0], mean_cax], "Fig_4a_mean_AGB_1988_2025")
    save_panel_crop(fig, [axes[1], cv_cax], "Fig_4b_AGB_interannual_variability_1988_2025")
    save_figure(fig, "Fig_4_AGB_spatial_pattern_and_variability_1988_2025")


def figure_s3_full_series_trends(profile: dict, strict: np.ndarray, bbox) -> None:
    local_mask = strict[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    slope = read_bbox(PIXEL_TREND, 1, bbox)
    classes = read_bbox(PIXEL_TREND, 4, bbox)
    slope[~local_mask] = np.nan
    classes[~local_mask] = np.nan

    src_transform, dst_transform, dst_width, dst_height, extent = geographic_grid(
        profile, bbox
    )
    slope = reproject_local_array(
        slope, profile, src_transform, dst_transform, dst_width, dst_height
    )
    classes = reproject_local_array(
        classes, profile, src_transform, dst_transform, dst_width, dst_height
    )
    local_mask = reproject_local_array(
        local_mask, profile, src_transform, dst_transform, dst_width, dst_height,
        is_mask=True,
    )
    slope[~local_mask] = np.nan
    classes[~local_mask] = np.nan

    fig = plt.figure(figsize=(7.2, 4.35))
    axes = [
        fig.add_axes([0.075, 0.28, 0.36, 0.60]),
        fig.add_axes([0.565, 0.28, 0.36, 0.60]),
    ]
    norm = TwoSlopeNorm(vmin=-4, vcenter=0, vmax=4)
    im = axes[0].imshow(
        slope, extent=extent, origin="upper", cmap="RdBu", norm=norm,
        interpolation="nearest", aspect="auto",
    )
    axes[0].contour(
        local_mask.astype(float), levels=[0.5], colors="0.68", linewidths=0.18,
        extent=extent, origin="upper",
    )
    style_geographic_map_axis(axes[0], extent)
    axes[0].text(
        0.5, -0.115, "Full-series Theil–Sen slope",
        transform=axes[0].transAxes, ha="center", va="top", fontsize=8,
    )
    add_panel_label(axes[0], "a")

    class_colors = ["#9b1d20", "#e88973", "#dedede", "#80b8d6", "#2166ac"]
    cmap = ListedColormap(class_colors)
    class_norm = BoundaryNorm([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], cmap.N)
    axes[1].imshow(
        classes, extent=extent, origin="upper", cmap=cmap, norm=class_norm,
        interpolation="nearest", aspect="auto",
    )
    axes[1].contour(
        local_mask.astype(float), levels=[0.5], colors="0.68", linewidths=0.18,
        extent=extent, origin="upper",
    )
    style_geographic_map_axis(axes[1], extent)
    axes[1].text(
        0.5, -0.115, "Approximate Mann–Kendall class",
        transform=axes[1].transAxes, ha="center", va="top", fontsize=8,
    )
    add_panel_label(axes[1], "b")

    cax = fig.add_axes([0.085, 0.105, 0.35, 0.025])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("AGB trend (g m$^{-2}$ yr$^{-1}$)")
    cbar.set_ticks([-4, -2, 0, 2, 4])

    area = pd.read_csv(TREND_AREA).set_index("trend_class_code")
    class_labels = [
        ("Sig. decrease", -2), ("Non-sig. decrease", -1),
        ("Stable", 0), ("Non-sig. increase", 1), ("Sig. increase", 2),
    ]
    handles = []
    for color, (label, code) in zip(class_colors, class_labels):
        fraction = float(area.loc[code, "fraction_of_trend_valid_area"])
        handles.append(Patch(facecolor=color, edgecolor="none", label=f"{label} ({fraction:.1%})"))
    legend_ax = fig.add_axes([0.525, 0.035, 0.46, 0.15])
    legend_ax.axis("off")
    legend_ax.legend(
        handles=handles, loc="center", ncol=2, frameon=False,
        columnspacing=1.0, handlelength=1.6,
    )
    save_figure(fig, "Fig_S3_full_series_AGB_trend_maps_1988_2025")


def _sen_line(years: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, float, float]:
    valid = np.isfinite(years) & np.isfinite(values)
    result = theilslopes(values[valid], years[valid], 0.95)
    tau, pvalue = kendalltau(years[valid], values[valid])
    fitted = result.intercept + result.slope * years
    return fitted, float(result.slope), float(pvalue)


def add_landsat_transitions(ax: plt.Axes, annotate: bool = False) -> None:
    transitions = [
        (1998.5, "L7 joined"),
        (2002.5, "L7 removed"),
        (2012.5, "L8"),
        (2020.5, "L9 joined"),
    ]
    for year, label in transitions:
        ax.axvline(year, color="0.62", ls=":", lw=0.7, zorder=0)
        if annotate:
            ax.text(
                year + 0.18, 0.03, label, transform=ax.get_xaxis_transform(),
                rotation=90, va="bottom", ha="left", fontsize=5.5, color="0.35",
            )


def figure_5_state_trajectories() -> None:
    annual = pd.read_csv(ANNUAL)
    state = pd.read_csv(STATE)
    trends = pd.read_csv(REGIONAL_TRENDS)
    x = annual["year"].to_numpy(dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.75), sharex=True)
    fig.subplots_adjust(left=0.11, right=0.985, top=0.965, bottom=0.095, hspace=0.42)

    ax = axes[0]
    ax.fill_between(
        x, annual["agb_p05_g_m2"], annual["agb_p95_g_m2"],
        color="#9ecae1", alpha=0.35, linewidth=0, label="P05–P95",
    )
    ax.plot(x, annual["agb_p50_g_m2"], color="#6baed6", lw=1.0, label="Median")
    ax.plot(
        x, annual["agb_mean_g_m2"], color="#225ea8", marker="o", ms=2.4,
        lw=1.2, label="Mean",
    )
    trend = trends.loc[trends["scenario"] == "1988-2025 all"].iloc[0]
    fitted = trend["sen_intercept"] + trend["sen_slope_g_m2_yr"] * x
    ax.plot(
        x, fitted, color="#252525", ls="--", lw=1.0,
        label="Full-series Sen fit",
    )
    add_landsat_transitions(ax, annotate=True)
    ax.text(
        0.012, 0.96,
        f"Unharmonized Sen slope = {trend['sen_slope_g_m2_yr']:.2f} "
        "g m$^{-2}$ yr$^{-1}$",
        transform=ax.transAxes, va="top", ha="left", fontsize=6.8,
    )
    ax.set_ylabel("AGB (g m$^{-2}$)")
    add_panel_subtitle(ax, "Regional AGB")
    ax.legend(loc="upper center", bbox_to_anchor=(0.67, 1.01), ncol=4, frameon=False)
    add_panel_label(ax, "a")

    ax = axes[1]
    fvc_year = state["year"].to_numpy(dtype=float)
    fvc = state["fvc_mean"].to_numpy(dtype=float)
    fvc_fit, fvc_slope, fvc_p = _sen_line(fvc_year, fvc)
    ax.plot(fvc_year, fvc, color="#1b9e77", marker="o", ms=2.4, lw=1.2)
    ax.plot(fvc_year, fvc_fit, color="#252525", ls="--", lw=1.0)
    add_landsat_transitions(ax)
    ax.text(
        0.012, 0.95, f"Sen slope = {fvc_slope:.4f} yr$^{{-1}}$\nMK p = {fvc_p:.1e}",
        transform=ax.transAxes, va="top", ha="left", fontsize=6.8,
    )
    ax.set_ylabel("FVC")
    add_panel_subtitle(ax, "Regional vegetation cover")
    add_panel_label(ax, "b")

    ax = axes[2]
    era_colors = {"AVHRR": "#756bb1", "VIIRS S-NPP": "#31a354", "VIIRS NOAA-20": "#e6550d"}
    for era, group in state.groupby("lai_sensor_era", sort=False):
        ax.plot(
            group["year"], group["lai_era_z"], color=era_colors[era],
            marker="o", ms=2.5, lw=1.2, label=era,
        )
    ax.axhline(0, color="0.55", lw=0.8)
    for year in (2013.5, 2022.5):
        ax.axvline(year, color="0.60", ls=":", lw=0.8)
    ax.set_ylabel("LAI (within-era z)")
    ax.set_xlabel("Year")
    add_panel_subtitle(ax, "Regional LAI state", y=-0.29)
    ax.legend(loc="lower left", ncol=3, frameon=False)
    add_panel_label(ax, "c")

    for ax in axes:
        ax.grid(axis="y", color="0.90", lw=0.55)
        style_boxed_axis(ax)
    panel_names = (
        "Fig_5a_regional_AGB_trajectory_1988_2025",
        "Fig_5b_regional_FVC_trajectory_1988_2025",
        "Fig_5c_regional_LAI_state_1988_2025",
    )
    for axis, stem in zip(axes, panel_names):
        save_panel_crop(fig, [axis], stem)
    save_figure(fig, "Fig_5_regional_AGB_FVC_LAI_trajectories_1988_2025")


def figure_6_sensor_diagnostics() -> None:
    annual = pd.read_csv(ANNUAL)
    comparison = pd.read_csv(PEAK_GUARDED)
    sensor_colors = {
        "Landsat 5 TM": "#636363",
        "Landsat 5 TM + Landsat 7 ETM+": "#e6ab02",
        "Landsat 7 ETM+ (SLC-off)": "#d95f02",
        "Landsat 8 OLI": "#1f78b4",
        "Landsat 8 OLI + Landsat 9 OLI-2": "#1b9e77",
    }

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.85))
    fig.subplots_adjust(left=0.12, right=0.985, top=0.95, bottom=0.13, hspace=0.62, wspace=0.32)

    ax = axes[0, 0]
    ax.plot(annual["year"], annual["agb_mean_g_m2"], color="0.72", lw=0.9, zorder=1)
    for policy, group in annual.groupby("sensor_policy", sort=False):
        ax.scatter(
            group["year"], group["agb_mean_g_m2"], s=18,
            color=sensor_colors.get(policy, "0.4"), edgecolor="white",
            linewidth=0.35, label=policy, zorder=2,
        )
    add_landsat_transitions(ax)
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean AGB (g m$^{-2}$)")
    add_panel_subtitle(ax, "Sensor composition of the annual series", y=-0.27)
    ax.set_ylim(top=421)
    ax.legend(
        loc="upper left", ncol=2, frameon=False, fontsize=5.0,
        handletextpad=0.3, borderaxespad=0.2,
    )
    add_panel_label(ax, "a")

    stages = [
        ("L5, 1988–1998", 1988, 1998),
        ("L5, 2003–2011", 2003, 2011),
        ("L8, 2013–2020", 2013, 2020),
        ("L8+L9, 2021–2025", 2021, 2025),
    ]
    rows = []
    for label, start, end in stages:
        group = annual.loc[annual["year"].between(start, end)]
        result = theilslopes(group["agb_mean_g_m2"], group["year"], 0.95)
        _, pvalue = kendalltau(group["year"], group["agb_mean_g_m2"])
        rows.append({
            "label": label,
            "slope": float(result.slope),
            "low": float(result.low_slope),
            "high": float(result.high_slope),
            "p": float(pvalue),
        })
    stage_data = pd.DataFrame(rows)

    ax = axes[0, 1]
    ypos = np.arange(len(stage_data))
    errors = np.vstack([
        stage_data["slope"] - stage_data["low"],
        stage_data["high"] - stage_data["slope"],
    ])
    ax.errorbar(
        stage_data["slope"], ypos, xerr=errors, fmt="o", color="#225ea8",
        capsize=2.5, ms=4, lw=1.0,
    )
    ax.axvline(0, color="0.45", lw=0.8)
    ax.set_yticks(ypos, stage_data["label"])
    ax.invert_yaxis()
    ax.set_xlabel("Within-stage Sen slope (g m$^{-2}$ yr$^{-1}$)")
    add_panel_subtitle(ax, "Sensor-stage trend sensitivity", y=-0.27)
    xmax = float(np.nanmax(stage_data["high"]))
    for y, row in zip(ypos, stage_data.itertuples()):
        ax.text(
            0.985, y, f"p={row.p:.2f}",
            transform=ax.get_yaxis_transform(), ha="right", va="center",
            fontsize=5.7,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.25},
        )
    ax.set_xlim(float(np.nanmin(stage_data["low"])) - 1, xmax + 4.5)
    add_panel_label(ax, "b")

    lookup = annual.set_index("year")["agb_mean_g_m2"]
    jump_labels = ["1998→1999\nL7 joined", "2012→2013\nL8", "2021→2022\nL9 joined"]
    jump_values = [
        float(lookup.loc[1999] - lookup.loc[1998]),
        float(lookup.loc[2013] - lookup.loc[2012]),
        float(lookup.loc[2022] - lookup.loc[2021]),
    ]
    ax = axes[1, 0]
    bars = ax.bar(jump_labels, jump_values, color=["#e6ab02", "#1f78b4", "#1b9e77"], width=0.62)
    ax.axhline(0, color="0.4", lw=0.8)
    for bar, value in zip(bars, jump_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, value + 2,
            f"{value:+.1f}", ha="center", va="bottom", fontsize=6.2,
        )
    ax.set_ylabel("One-year AGB change (g m$^{-2}$)")
    add_panel_subtitle(ax, "Changes at sensor transitions", y=-0.23)
    add_panel_label(ax, "c")

    ax = axes[1, 1]
    peak = (
        comparison["agb_mean_g_m2"] - comparison["agb_mean_g_m2"].mean()
    ) / comparison["agb_mean_g_m2"].std(ddof=0)
    guarded = (
        comparison["guarded_agb_mean_g_m2"]
        - comparison["guarded_agb_mean_g_m2"].mean()
    ) / comparison["guarded_agb_mean_g_m2"].std(ddof=0)
    model_r = float(np.corrcoef(peak, guarded)[0, 1])
    ax.plot(comparison["year"], peak, color="#225ea8", lw=1.2, label="Peak-only")
    ax.plot(comparison["year"], guarded, color="#d95f0e", lw=1.2, label="Guarded-consensus")
    ax.axhline(0, color="0.65", lw=0.8)
    ax.text(0.02, 0.95, f"r = {model_r:.3f}", transform=ax.transAxes, va="top")
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual AGB anomaly (z)")
    add_panel_subtitle(ax, "Model-structure sensitivity, 1988–2024", y=-0.23)
    ax.legend(loc="lower right", frameon=False)
    add_panel_label(ax, "d")

    for ax in axes.flat:
        ax.grid(axis="y", color="0.92", lw=0.55)
        style_boxed_axis(ax)
    panel_names = (
        "Fig_6a_annual_sensor_composition",
        "Fig_6b_sensor_stage_trend_sensitivity",
        "Fig_6c_sensor_transition_changes",
        "Fig_6d_peakonly_guarded_sensitivity",
    )
    for axis, stem in zip(axes.flat, panel_names):
        save_panel_crop(fig, [axis], stem)
    save_figure(fig, "Fig_6_sensor_stage_and_model_sensitivity")


def figure_s2_representative_maps(profile: dict, strict: np.ndarray, bbox) -> None:
    local_mask = strict[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    src_transform, dst_transform, dst_width, dst_height, extent = geographic_grid(
        profile, bbox
    )
    local_mask = reproject_local_array(
        local_mask, profile, src_transform, dst_transform, dst_width, dst_height,
        is_mask=True,
    )
    cmap, norm, centers = agb_class_style()
    fig = plt.figure(figsize=(7.2, 7.2))
    axes = [
        fig.add_axes([0.08, 0.60, 0.34, 0.34]),
        fig.add_axes([0.58, 0.60, 0.34, 0.34]),
        fig.add_axes([0.08, 0.18, 0.34, 0.34]),
        fig.add_axes([0.58, 0.18, 0.34, 0.34]),
    ]
    last = None
    for index, (ax, year) in enumerate(zip(axes, REPRESENTATIVE_YEARS)):
        path, band = annual_raster(year)
        values = read_bbox(path, band, bbox)
        values = reproject_local_array(
            values, profile, src_transform, dst_transform, dst_width, dst_height
        )
        values[~local_mask] = np.nan
        last = ax.imshow(
            values, extent=extent, origin="upper", cmap=cmap, norm=norm,
            interpolation="nearest", aspect="auto",
        )
        ax.contour(
            local_mask.astype(float), levels=[0.5], colors="0.68", linewidths=0.18,
            extent=extent, origin="upper",
        )
        style_geographic_map_axis(ax, extent)
        ax.text(
            0.5, -0.090, str(year), transform=ax.transAxes,
            ha="center", va="top", fontsize=8,
        )
        add_panel_label(ax, chr(ord("a") + index))
    cax = fig.add_axes([0.26, 0.055, 0.48, 0.020])
    cbar = fig.colorbar(
        last, cax=cax, orientation="horizontal",
        boundaries=AGB_CLASS_BOUNDS, spacing="uniform",
    )
    cbar.set_label("AGB (g m$^{-2}$)")
    cbar.set_ticks(centers)
    cbar.set_ticklabels(AGB_CLASS_LABELS)
    save_figure(fig, "Fig_S2_representative_AGB_maps_1989_2000_2013_2025")


def figure_s4_trend_sensitivity() -> None:
    trends = pd.read_csv(REGIONAL_TRENDS)
    comparison = pd.read_csv(PEAK_GUARDED)
    order = [
        "1988-2024 frozen", "1988-2025 all",
        "1988-2025 coverage >= 0.80", "1988-2025 coverage >= 0.90",
    ]
    data = trends.set_index("scenario").loc[order].reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.45))
    fig.subplots_adjust(left=0.13, right=0.985, top=0.92, bottom=0.25, wspace=0.30)
    ax = axes[0]
    ypos = np.arange(len(data))
    errors = np.vstack([
        data["sen_slope_g_m2_yr"] - data["sen_ci95_low"],
        data["sen_ci95_high"] - data["sen_slope_g_m2_yr"],
    ])
    ax.errorbar(
        data["sen_slope_g_m2_yr"], ypos, xerr=errors, fmt="o",
        color="#225ea8", capsize=2.5, ms=4,
    )
    ax.axvline(0, color="0.55", lw=0.8)
    ax.set_yticks(ypos, ["1988–2024\nfrozen", "1988–2025\nall", "Coverage\n≥0.80", "Coverage\n≥0.90"])
    ax.invert_yaxis()
    ax.set_xlabel("Sen slope (g m$^{-2}$ yr$^{-1}$)")
    add_panel_subtitle(ax, "Regional trend sensitivity", y=-0.28)
    add_panel_label(ax, "a")

    ax = axes[1]
    peak = (comparison["agb_mean_g_m2"] - comparison["agb_mean_g_m2"].mean()) / comparison["agb_mean_g_m2"].std(ddof=0)
    guarded = (comparison["guarded_agb_mean_g_m2"] - comparison["guarded_agb_mean_g_m2"].mean()) / comparison["guarded_agb_mean_g_m2"].std(ddof=0)
    ax.plot(comparison["year"], peak, color="#225ea8", lw=1.2, label="Peak-only")
    ax.plot(comparison["year"], guarded, color="#d95f0e", lw=1.2, label="Guarded-consensus")
    ax.axhline(0, color="0.65", lw=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual AGB anomaly (z)")
    add_panel_subtitle(ax, "Model-structure sensitivity", y=-0.28)
    ax.legend(loc="upper left", frameon=False)
    add_panel_label(ax, "b")

    for ax in axes:
        ax.grid(axis="x", color="0.92", lw=0.55)
        style_boxed_axis(ax)
    save_figure(fig, "Fig_S4_regional_trend_and_model_sensitivity")


def write_legends() -> None:
    text = """# Section 3.2 figure legends

## 中文

**图4. 1988–2025年若尔盖泥炭地AGB空间格局与年际变异。** (a) 多年平均AGB；(b) 年际变异系数。AGB为Peak-only模型在90 m网格上的年度估算结果。

**图5. 1988–2025年若尔盖泥炭地植被状态年际变化。** (a) 区域AGB均值、中位数、P05–P95范围及完整序列Sen拟合；(b) 区域平均FVC；(c) 在AVHRR、VIIRS S-NPP和VIIRS NOAA-20时期内分别标准化的LAI。竖线表示Landsat或LAI传感器阶段转换。

**图6. AGB时间序列的传感器阶段与模型结构敏感性。** (a) 各年度Landsat传感器组成；(b) 单一传感器阶段内的Sen斜率及95%置信区间；(c) 主要传感器转换前后的单年AGB变化；(d) Peak-only与Guarded-consensus年度AGB标准化异常。

## English

**Fig. 4. Spatial pattern and interannual variability of peatland aboveground biomass in Zoige from 1988 to 2025.** (a) Multi-year mean AGB. (b) Temporal coefficient of variation. Annual AGB was estimated with the Peak-only model on a 90-m output grid.

**Fig. 5. Interannual trajectories of peatland vegetation state from 1988 to 2025.** (a) Regional mean, median, P05–P95 range and full-series Sen fit of AGB. (b) Regional mean FVC. (c) LAI standardized separately within the AVHRR, VIIRS S-NPP and VIIRS NOAA-20 eras. Vertical lines denote Landsat or LAI sensor transitions.

**Fig. 6. Sensitivity of the AGB time series to sensor stage and model structure.** (a) Landsat sensor composition by year. (b) Within-stage Sen slopes with 95% confidence intervals. (c) One-year AGB changes across major sensor transitions. (d) Standardized annual AGB anomalies from the Peak-only and Guarded-consensus products.

"""
    (FIGURES / "figure_legends_section_3_2.md").write_text(text, encoding="utf-8")


def write_qa() -> None:
    from PIL import Image

    records = []
    for path in sorted(FIGURES.glob("*.png")):
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
            records.append({
                "file": path.name,
                "width_px": image.width,
                "height_px": image.height,
                "nonwhite_fraction": float(np.mean(np.any(rgb < 250, axis=2))),
                "pixel_std": float(rgb.std()),
                "nonblank": bool(rgb.std() > 5 and np.mean(np.any(rgb < 250, axis=2)) > 0.02),
            })
    pd.DataFrame(records).to_csv(FIGURES / "figure_qa.csv", index=False)
    metadata = {
        "core_conclusion": "Strict peatland AGB showed distinct spatial structure and interannual variability; the apparent full-period increase coincided with sensor transitions and was not significant within individual sensor stages.",
        "backend": "Python/matplotlib",
        "main_figures": ["Fig. 4", "Fig. 5", "Fig. 6"],
        "supplementary_figures": [],
        "source_tables": [str(ANNUAL), str(STATE), str(REGIONAL_TRENDS), str(TREND_AREA)],
        "source_rasters": [str(MULTIYEAR), str(PIXEL_TREND)],
    }
    (FIGURES / "figure_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    profile, strict, bbox = load_grid()
    figure_4_mean_agb(profile, strict, bbox)
    figure_5_state_trajectories()
    figure_6_sensor_diagnostics()
    figure_s2_representative_maps(profile, strict, bbox)
    figure_s3_full_series_trends(profile, strict, bbox)
    figure_s4_trend_sensitivity()
    write_legends()
    write_qa()
    print(f"Figures written to {FIGURES}")


if __name__ == "__main__":
    main()
