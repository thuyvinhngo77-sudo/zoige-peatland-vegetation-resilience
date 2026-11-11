"""Build the manuscript study-area map for the Zoige peatland analysis."""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import LightSource, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds
from rasterio.warp import reproject
from shapely.geometry import box, mapping


ROOT = Path(__file__).resolve().parents[2]
WORK_DIR = Path(__file__).resolve().parent
EXTERNAL_DIR = WORK_DIR / "external_data"
FIGURE_DIR = WORK_DIR / "figures"
PANEL_DIR = FIGURE_DIR / "panels"
SOURCE_DIR = WORK_DIR / "source_data"

AOI_PATH = ROOT / "zoigepeatland" / "data" / "Zoige.shp"
XUE_PRIOR_PATH = ROOT / "zoigepeatland" / "data" / "zoigepeatland1km.shp"
MAIN95_PATH = (
    ROOT
    / "zoigepeatland"
    / "output_basic_boundary"
    / "Zoige_RF_basic_analysis_mask_main95_fill400.tif"
)
AREA_PATH = (
    ROOT
    / "zoigepeatland"
    / "output_basic_boundary"
    / "Zoige_RF_basic_postprocess_area_statistics.csv"
)
CALIBRATION_SAMPLES_PATH = (
    ROOT
    / "agb_final_candidate_peak_source_corrected"
    / "outputs"
    / "peakndvi_sample_inventory.csv"
)
ALL_FIELD_SAMPLES_PATH = (
    ROOT
    / "agb_field_data"
    / "master_samples"
    / "zoige_agb_master_samples_2014_2025.csv"
)

COUNTRIES_PATH = ROOT / "paper_study" / "ne_110m_admin_0_countries.geojson"
PROVINCES_PATH = (
    EXTERNAL_DIR
    / "chn_adm1"
    / "geoBoundaries-CHN-ADM1.shp"
)
QTP_PATH = EXTERNAL_DIR / "qtp2021" / "TPboundaryPolygon.shp"
RIVERS_PATH = (
    EXTERNAL_DIR
    / "rivers"
    / "ne_10m_rivers_lake_centerlines.shp"
)

AGB_DATA_ROOT = Path(
    os.environ.get("AGB_COLLECT_DATA_ROOT", str(ROOT / "external_data"))
)


COLORS = {
    "ink": "#202726",
    "muted": "#68716f",
    "grid": "#9ca5a2",
    "aoi": "#3f4745",
    "river": "#4a8db7",
    "main_fill": "#1c8d7b",
    "main_edge": "#086255",
    "prior": "#c06149",
    "calibration": "#f3c243",
    "validation": "#7d4ba0",
    "qtp_fill": "#d9c7a0",
    "qtp_edge": "#987d4f",
    "study": "#c43c39",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.linewidth": 0.75,
            "axes.edgecolor": COLORS["ink"],
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def locate_dem() -> Path:
    candidates = list(AGB_DATA_ROOT.rglob("Zoige_Terrain_Hydro_30m.tif"))
    if not candidates:
        raise FileNotFoundError(
            "Zoige_Terrain_Hydro_30m.tif was not found. Set "
            "AGB_COLLECT_DATA_ROOT to the external-data directory."
        )
    return candidates[0]


def repair_geometries(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    result = frame.copy()
    result["geometry"] = result.geometry.make_valid()
    return result.loc[~result.geometry.is_empty & result.geometry.notna()].copy()


def read_preview(
    path: Path,
    band: int,
    max_side: int,
    resampling: Resampling,
) -> tuple[np.ndarray, rasterio.coords.BoundingBox, rasterio.Affine, object]:
    with rasterio.open(path) as src:
        scale = min(1.0, max_side / max(src.width, src.height))
        width = max(1, int(round(src.width * scale)))
        height = max(1, int(round(src.height * scale)))
        data = src.read(
            band,
            out_shape=(height, width),
            resampling=resampling,
        ).astype("float32")
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
        transform = from_bounds(*src.bounds, width=width, height=height)
        return data, src.bounds, transform, src.crs


def build_hillshade(
    dem_path: Path,
    aoi: gpd.GeoDataFrame,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    elevation, bounds, transform, crs = read_preview(
        dem_path,
        band=1,
        max_side=1500,
        resampling=Resampling.average,
    )
    aoi_in_crs = aoi.to_crs(crs)
    inside = geometry_mask(
        [mapping(geom) for geom in aoi_in_crs.geometry],
        out_shape=elevation.shape,
        transform=transform,
        invert=True,
    )
    valid = np.isfinite(elevation) & inside
    fill_value = float(np.nanmedian(elevation[valid]))
    filled = np.where(valid, elevation, fill_value)
    shade = LightSource(azdeg=315, altdeg=42).hillshade(
        filled,
        vert_exag=0.85,
        dx=30,
        dy=30,
    )
    shade[~valid] = np.nan
    elevation[~valid] = np.nan
    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
    return elevation, shade, extent


def main95_mean_elevation(dem_path: Path) -> float:
    with rasterio.open(MAIN95_PATH) as mask_src:
        mask = mask_src.read(1) > 0
        destination = np.full(mask.shape, np.nan, dtype="float32")
        with rasterio.open(dem_path) as dem_src:
            reproject(
                source=rasterio.band(dem_src, 1),
                destination=destination,
                src_transform=dem_src.transform,
                src_crs=dem_src.crs,
                dst_transform=mask_src.transform,
                dst_crs=mask_src.crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
    valid = mask & np.isfinite(destination)
    return float(np.nanmean(destination[valid]))


def main95_area_km2() -> float:
    with AREA_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["mask"] == "analysis_main95_fill400":
                return float(row["clean_area_km2"])
    raise KeyError("analysis_main95_fill400 was not found in the area table")


def format_lon(value: float, _: int | None = None) -> str:
    return f"{value:g}°E"


def format_lat(value: float, _: int | None = None) -> str:
    return f"{value:g}°N"


def style_geo_axes(
    ax: plt.Axes,
    x_ticks: list[float],
    y_ticks: list[float],
    grid: bool = True,
) -> None:
    ax.xaxis.set_major_locator(FixedLocator(x_ticks))
    ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        labeltop=True,
        labelright=True,
        pad=2,
        labelsize=6.3,
    )
    if grid:
        ax.grid(
            color=COLORS["grid"],
            linewidth=0.38,
            linestyle=(0, (2.2, 2.2)),
            alpha=0.48,
            zorder=0,
        )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.75)
        spine.set_color(COLORS["ink"])


def add_scale_bar(ax: plt.Axes, length_km: float = 50) -> None:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    latitude = ymin + 0.055 * (ymax - ymin)
    start_x = xmin + 0.065 * (xmax - xmin)
    delta_lon = length_km / (111.32 * math.cos(math.radians(latitude)))
    end_x = start_x + delta_lon
    ax.plot(
        [start_x, end_x],
        [latitude, latitude],
        color=COLORS["ink"],
        linewidth=2.2,
        solid_capstyle="butt",
        zorder=20,
    )
    tick_height = 0.012 * (ymax - ymin)
    ax.plot(
        [start_x, start_x],
        [latitude - tick_height, latitude + tick_height],
        color=COLORS["ink"],
        linewidth=0.8,
        zorder=20,
    )
    ax.plot(
        [end_x, end_x],
        [latitude - tick_height, latitude + tick_height],
        color=COLORS["ink"],
        linewidth=0.8,
        zorder=20,
    )
    ax.text(
        (start_x + end_x) / 2,
        latitude + 0.020 * (ymax - ymin),
        f"{length_km:g} km",
        ha="center",
        va="bottom",
        fontsize=6.2,
        color=COLORS["ink"],
        zorder=20,
    )


def add_north_arrow(ax: plt.Axes) -> None:
    ax.annotate(
        "N",
        xy=(0.935, 0.915),
        xytext=(0.935, 0.82),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=7.5,
        fontweight="bold",
        color=COLORS["ink"],
        arrowprops={
            "arrowstyle": "-|>",
            "facecolor": COLORS["ink"],
            "edgecolor": COLORS["ink"],
            "linewidth": 0.8,
        },
        zorder=25,
    )


def load_field_samples() -> tuple[pd.DataFrame, pd.DataFrame]:
    calibration = pd.read_csv(CALIBRATION_SAMPLES_PATH)
    independent_all = pd.read_csv(ALL_FIELD_SAMPLES_PATH)
    independent = independent_all.loc[
        independent_all["use_for_independent_test"]
        .astype(str)
        .str.lower()
        .eq("yes")
    ].copy()
    return calibration, independent


def load_layers() -> dict[str, object]:
    aoi = repair_geometries(gpd.read_file(AOI_PATH).to_crs(4326))
    xue = repair_geometries(gpd.read_file(XUE_PRIOR_PATH).to_crs(4326))
    qtp = repair_geometries(gpd.read_file(QTP_PATH).to_crs(4326))
    provinces = repair_geometries(gpd.read_file(PROVINCES_PATH).to_crs(4326))
    countries = repair_geometries(gpd.read_file(COUNTRIES_PATH).to_crs(4326))
    rivers = repair_geometries(
        gpd.read_file(RIVERS_PATH, bbox=tuple(aoi.total_bounds)).to_crs(4326)
    )
    if len(rivers):
        rivers = repair_geometries(gpd.clip(rivers, aoi))
    calibration, independent = load_field_samples()
    return {
        "aoi": aoi,
        "xue": xue,
        "qtp": qtp,
        "provinces": provinces,
        "countries": countries,
        "rivers": rivers,
        "calibration": calibration,
        "independent": independent,
    }


def draw_local_map(
    ax: plt.Axes,
    layers: dict[str, object],
    elevation: np.ndarray,
    hillshade: np.ndarray,
    terrain_extent: tuple[float, float, float, float],
    panel_label: str = "(a)",
) -> None:
    aoi = layers["aoi"]
    xue = layers["xue"]
    rivers = layers["rivers"]
    calibration = layers["calibration"]
    independent = layers["independent"]

    ax.set_facecolor("#f5f6f4")
    q_low, q_high = np.nanpercentile(elevation, [2, 98])
    ax.imshow(
        elevation,
        extent=terrain_extent,
        origin="upper",
        cmap="gist_earth",
        vmin=q_low,
        vmax=q_high,
        alpha=0.44,
        interpolation="bilinear",
        zorder=1,
    )
    ax.imshow(
        hillshade,
        extent=terrain_extent,
        origin="upper",
        cmap="Greys",
        vmin=0.05,
        vmax=0.95,
        alpha=0.48,
        interpolation="bilinear",
        zorder=2,
    )

    xue.plot(
        ax=ax,
        facecolor=COLORS["prior"],
        edgecolor="none",
        alpha=0.16,
        linewidth=0.0,
        zorder=3,
    )
    xue.boundary.plot(
        ax=ax,
        color=COLORS["prior"],
        linewidth=0.28,
        linestyle="-",
        alpha=0.48,
        zorder=7,
    )

    main_mask, mask_bounds, _, _ = read_preview(
        MAIN95_PATH,
        band=1,
        max_side=1300,
        resampling=Resampling.nearest,
    )
    binary = np.where(main_mask > 0, 1.0, np.nan)
    main_cmap = ListedColormap([COLORS["main_fill"]])
    ax.imshow(
        binary,
        extent=(mask_bounds.left, mask_bounds.right, mask_bounds.bottom, mask_bounds.top),
        origin="upper",
        cmap=main_cmap,
        vmin=0,
        vmax=1,
        alpha=0.50,
        interpolation="nearest",
        zorder=5,
    )
    contour_binary = np.nan_to_num(binary, nan=0.0)
    ax.contour(
        contour_binary,
        levels=[0.5],
        extent=(mask_bounds.left, mask_bounds.right, mask_bounds.bottom, mask_bounds.top),
        origin="upper",
        colors=[COLORS["main_edge"]],
        linewidths=0.55,
        zorder=8,
    )

    if len(rivers):
        rivers.plot(
            ax=ax,
            color=COLORS["river"],
            linewidth=np.clip(1.25 - 0.08 * rivers["scalerank"].to_numpy(), 0.35, 1.0),
            alpha=0.92,
            zorder=9,
        )

    ax.scatter(
        calibration["longitude"],
        calibration["latitude"],
        s=9.0,
        marker="o",
        facecolor=COLORS["calibration"],
        edgecolor="white",
        linewidth=0.35,
        alpha=0.88,
        zorder=12,
    )
    ax.scatter(
        independent["longitude"],
        independent["latitude"],
        s=20.0,
        marker="D",
        facecolor=COLORS["validation"],
        edgecolor="white",
        linewidth=0.45,
        alpha=0.95,
        zorder=13,
    )
    aoi.boundary.plot(ax=ax, color=COLORS["aoi"], linewidth=0.85, zorder=15)

    xmin, ymin, xmax, ymax = aoi.total_bounds
    dx = xmax - xmin
    dy = ymax - ymin
    ax.set_xlim(xmin - 0.035 * dx, xmax + 0.035 * dx)
    ax.set_ylim(ymin - 0.035 * dy, ymax + 0.035 * dy)
    center_lat = (ymin + ymax) / 2
    ax.set_aspect(1 / math.cos(math.radians(center_lat)))
    style_geo_axes(ax, [101, 102, 103], [32, 33, 34])
    add_scale_bar(ax, 50)
    add_north_arrow(ax)
    ax.text(
        0.018,
        0.985,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        fontweight="bold",
        color=COLORS["ink"],
        zorder=30,
    )


def draw_locator(ax: plt.Axes, layers: dict[str, object]) -> None:
    countries = layers["countries"]
    provinces = layers["provinces"]
    qtp = layers["qtp"]
    aoi = layers["aoi"]

    map_box = box(70, 15, 138, 56)
    nearby = countries.loc[countries.geometry.intersects(map_box)].copy()
    nearby.plot(ax=ax, facecolor="#f5f5f2", edgecolor="#b6bcb9", linewidth=0.38, zorder=1)
    provinces.plot(ax=ax, facecolor="#eceeeb", edgecolor="#a7afac", linewidth=0.34, zorder=2)
    qtp.plot(
        ax=ax,
        facecolor=COLORS["qtp_fill"],
        edgecolor=COLORS["qtp_edge"],
        linewidth=0.70,
        alpha=0.72,
        zorder=3,
    )

    xmin, ymin, xmax, ymax = aoi.total_bounds
    ax.add_patch(
        Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            facecolor="none",
            edgecolor=COLORS["study"],
            linewidth=1.25,
            zorder=8,
        )
    )
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2
    ax.plot(center_x, center_y, marker="o", ms=2.8, color=COLORS["study"], zorder=9)
    ax.annotate(
        "Zoige",
        xy=(center_x, center_y),
        xytext=(107.5, 38.2),
        fontsize=6.4,
        color=COLORS["study"],
        fontweight="bold",
        arrowprops={
            "arrowstyle": "-",
            "color": COLORS["study"],
            "linewidth": 0.65,
        },
        zorder=10,
    )

    ax.set_xlim(72, 136)
    ax.set_ylim(17, 55)
    ax.set_aspect(1 / math.cos(math.radians(36)))
    style_geo_axes(ax, [75, 90, 105, 120, 135], [20, 30, 40, 50], grid=True)
    ax.text(
        0.025,
        0.97,
        "(b)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        fontweight="bold",
        color=COLORS["ink"],
        zorder=20,
    )


def legend_handles(calibration_n: int, validation_n: int) -> list[object]:
    return [
        Patch(
            facecolor=COLORS["main_fill"],
            edgecolor=COLORS["main_edge"],
            alpha=0.72,
            label="Validated peatland domain",
        ),
        Patch(
            facecolor=COLORS["prior"],
            edgecolor=COLORS["prior"],
            linewidth=0.5,
            alpha=0.28,
            label="Xue et al. 1-km peatland prior",
        ),
        Line2D([0], [0], color=COLORS["river"], linewidth=1.2, label="Major rivers"),
        Line2D([0], [0], color=COLORS["aoi"], linewidth=0.9, label="Zoige analysis extent"),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=COLORS["calibration"],
            markeredgecolor="white",
            markeredgewidth=0.35,
            markersize=4.4,
            label=f"AGB calibration (n = {calibration_n})",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            markerfacecolor=COLORS["validation"],
            markeredgecolor="white",
            markeredgewidth=0.35,
            markersize=4.4,
            label=f"2025 validation (n = {validation_n})",
        ),
        Patch(
            facecolor=COLORS["qtp_fill"],
            edgecolor=COLORS["qtp_edge"],
            alpha=0.72,
            label="Tibetan Plateau (2021 boundary)",
        ),
    ]


def draw_information_panel(
    ax: plt.Axes,
    calibration_n: int,
    validation_n: int,
    area_km2: float,
    mean_elevation_m: float,
) -> None:
    ax.set_axis_off()
    ax.legend(
        handles=legend_handles(calibration_n, validation_n),
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        fontsize=6.25,
        handlelength=2.2,
        handletextpad=0.65,
        labelspacing=0.55,
        borderaxespad=0.0,
        frameon=False,
    )
    summary = (
        f"Validated peatland area   {area_km2:,.1f} km²\n"
        f"Mean elevation             {mean_elevation_m:,.0f} m a.s.l.\n"
        "Climate                    cold, seasonally humid alpine"
    )
    ax.text(
        0.0,
        0.08,
        summary,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.35,
        color=COLORS["ink"],
        linespacing=1.55,
        bbox={
            "boxstyle": "square,pad=0.55",
            "facecolor": "#f6f7f5",
            "edgecolor": "#c7ccca",
            "linewidth": 0.55,
        },
    )


def save_figure(fig: plt.Figure, stem: Path, dpi: int = 600) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=350, bbox_inches="tight", facecolor="white")
    fig.savefig(
        stem.with_suffix(".tif"),
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )


def export_source_summary(
    area_km2: float,
    mean_elevation_m: float,
    calibration: pd.DataFrame,
    independent: pd.DataFrame,
) -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "validated_peatland_area_km2": area_km2,
                "mean_elevation_m": mean_elevation_m,
                "agb_calibration_n": len(calibration),
                "agb_independent_2025_n": len(independent),
                "calibration_years": ",".join(
                    str(year) for year in sorted(calibration["year"].dropna().unique())
                ),
                "independent_years": ",".join(
                    str(int(year)) for year in sorted(independent["year"].dropna().unique())
                ),
            }
        ]
    ).to_csv(SOURCE_DIR / "Fig_1_study_area_summary.csv", index=False)
    calibration[["longitude", "latitude", "year", "agb_g_m2"]].to_csv(
        SOURCE_DIR / "Fig_1a_agb_calibration_locations.csv", index=False
    )
    independent[["longitude", "latitude", "year", "agb_g_m2"]].to_csv(
        SOURCE_DIR / "Fig_1a_agb_2025_validation_locations.csv", index=False
    )


def make_combined_figure(
    layers: dict[str, object],
    elevation: np.ndarray,
    hillshade: np.ndarray,
    terrain_extent: tuple[float, float, float, float],
    area_km2: float,
    mean_elevation_m: float,
) -> None:
    fig = plt.figure(figsize=(7.20, 5.25), facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.58, 1.0),
        height_ratios=(1.02, 0.98),
        left=0.055,
        right=0.985,
        bottom=0.085,
        top=0.985,
        wspace=0.16,
        hspace=0.17,
    )
    local_ax = fig.add_subplot(grid[:, 0])
    locator_ax = fig.add_subplot(grid[0, 1])
    info_ax = fig.add_subplot(grid[1, 1])

    draw_local_map(local_ax, layers, elevation, hillshade, terrain_extent)
    draw_locator(locator_ax, layers)
    draw_information_panel(
        info_ax,
        calibration_n=len(layers["calibration"]),
        validation_n=len(layers["independent"]),
        area_km2=area_km2,
        mean_elevation_m=mean_elevation_m,
    )

    fig.text(
        0.337,
        0.038,
        "Validated peatland domain and AGB observations",
        ha="center",
        va="center",
        fontsize=6.8,
        color=COLORS["ink"],
    )
    fig.text(
        0.807,
        0.502,
        "Location on the Tibetan Plateau",
        ha="center",
        va="center",
        fontsize=6.8,
        color=COLORS["ink"],
    )
    save_figure(fig, FIGURE_DIR / "Fig_1_study_area_map")
    plt.close(fig)


def make_standalone_panels(
    layers: dict[str, object],
    elevation: np.ndarray,
    hillshade: np.ndarray,
    terrain_extent: tuple[float, float, float, float],
    area_km2: float,
    mean_elevation_m: float,
) -> None:
    fig_a, ax_a = plt.subplots(figsize=(5.2, 5.35), facecolor="white")
    draw_local_map(ax_a, layers, elevation, hillshade, terrain_extent)
    fig_a.subplots_adjust(left=0.10, right=0.97, bottom=0.08, top=0.97)
    save_figure(fig_a, PANEL_DIR / "Fig_1a_local_study_area")
    plt.close(fig_a)

    fig_b, ax_b = plt.subplots(figsize=(5.3, 3.45), facecolor="white")
    draw_locator(ax_b, layers)
    ax_b.legend(
        handles=[
            Patch(
                facecolor=COLORS["qtp_fill"],
                edgecolor=COLORS["qtp_edge"],
                label="Tibetan Plateau",
            ),
            Line2D(
                [0],
                [0],
                color=COLORS["study"],
                linewidth=1.2,
                label="Zoige study area",
            ),
        ],
        loc="lower left",
        fontsize=6.4,
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#c7ccca",
    )
    fig_b.subplots_adjust(left=0.08, right=0.97, bottom=0.12, top=0.96)
    save_figure(fig_b, PANEL_DIR / "Fig_1b_location_map")
    plt.close(fig_b)

    fig_info, ax_info = plt.subplots(figsize=(4.25, 3.25), facecolor="white")
    draw_information_panel(
        ax_info,
        calibration_n=len(layers["calibration"]),
        validation_n=len(layers["independent"]),
        area_km2=area_km2,
        mean_elevation_m=mean_elevation_m,
    )
    fig_info.subplots_adjust(left=0.04, right=0.98, bottom=0.05, top=0.98)
    save_figure(fig_info, PANEL_DIR / "Fig_1_legend_information")
    plt.close(fig_info)


def main() -> None:
    setup_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PANEL_DIR.mkdir(parents=True, exist_ok=True)

    layers = load_layers()
    dem_path = locate_dem()
    elevation, hillshade, terrain_extent = build_hillshade(dem_path, layers["aoi"])
    area_km2 = main95_area_km2()
    mean_elevation_m = main95_mean_elevation(dem_path)

    make_combined_figure(
        layers,
        elevation,
        hillshade,
        terrain_extent,
        area_km2,
        mean_elevation_m,
    )
    make_standalone_panels(
        layers,
        elevation,
        hillshade,
        terrain_extent,
        area_km2,
        mean_elevation_m,
    )
    export_source_summary(
        area_km2,
        mean_elevation_m,
        layers["calibration"],
        layers["independent"],
    )

    print(f"Validated peatland area: {area_km2:.1f} km2")
    print(f"Mean elevation: {mean_elevation_m:.0f} m")
    print(f"Calibration samples: {len(layers['calibration'])}")
    print(f"2025 validation samples: {len(layers['independent'])}")
    print(FIGURE_DIR / "Fig_1_study_area_map.png")


if __name__ == "__main__":
    main()
