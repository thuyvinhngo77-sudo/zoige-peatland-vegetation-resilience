#!/usr/bin/env python
"""Build and plot four-core-event AGB/FVC drought-response maps.

The script preserves the 90 m analysis grid, uses annual Peak-only AGB,
calculates response ratios separately for each event, and then takes the
pixelwise median across events. Pixels need at least three valid events to be
shown in the across-event maps.
"""

from __future__ import annotations

import json
import math
import warnings
from contextlib import ExitStack
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter, MultipleLocator
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from rasterio.windows import Window


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "drought_response_maps_four_core_20260921"
RASTERS = RUN / "rasters"
FIGURES = RUN / "figures"
SOURCE_DATA = RUN / "source_data"

AGB_DIR = ROOT / "agb_dual_grid_2013_2024" / "rasters_90m"
FVC_DIR = (
    ROOT / "agb_resilience_section_3_3_1988_2025" / "rasters" / "fvc_90m"
)
STRICT_MASK = (
    ROOT
    / "zoigepeatland"
    / "output_basic_boundary"
    / "Zoige_RF_basic_analysis_mask_main95_fill400.tif"
)

NODATA = -9999.0
BLOCK = 512
MIN_VALID_EVENTS = 3
RATIO_MIN, RATIO_CENTER, RATIO_MAX = 0.85, 1.0, 1.15

EVENTS = [
    {
        "event_id": "1991",
        "pre": [1988, 1989, 1990],
        "event": [1991],
        "post": [1992, 1993],
    },
    {
        "event_id": "1997",
        "pre": [1995, 1996],
        "event": [1997],
        "post": [1998, 1999],
    },
    {
        "event_id": "2002",
        "pre": [1999, 2001],
        "event": [2002],
        "post": [2003, 2004, 2005],
    },
    {
        "event_id": "2015_2016",
        "pre": [2012, 2013, 2014],
        "event": [2015, 2016],
        "post": [2017, 2018, 2019],
    },
]

METRICS = ("resistance", "recovery", "resilience")


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.linewidth": 0.7,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.8,
        "ytick.major.size": 2.8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


def ensure_dirs() -> None:
    for path in (RASTERS, FIGURES, SOURCE_DATA):
        path.mkdir(parents=True, exist_ok=True)


def annual_path_band(variable: str, year: int) -> tuple[Path, int]:
    if variable == "AGB":
        return (
            AGB_DIR / f"Zoige_AGB_GuardedConsensus_PLSR_90m_{year}.tif",
            3,
        )
    if variable == "FVC":
        return FVC_DIR / f"Zoige_FVC_PeakNDVI_3x3Median_90m_{year}.tif", 1
    raise ValueError(variable)


def all_required_years() -> list[int]:
    return sorted(
        {
            year
            for event in EVENTS
            for phase in ("pre", "event", "post")
            for year in event[phase]
        }
    )


def iter_windows(width: int, height: int):
    for row in range(0, height, BLOCK):
        for col in range(0, width, BLOCK):
            yield Window(
                col,
                row,
                min(BLOCK, width - col),
                min(BLOCK, height - row),
            )


def read_float(
    source: rasterio.io.DatasetReader, band: int, window: Window
) -> np.ndarray:
    values = source.read(band, window=window).astype("float32")
    invalid = ~np.isfinite(values)
    if source.nodata is not None:
        invalid |= np.isclose(values, source.nodata)
    values[invalid] = np.nan
    return values


def reference_profile_and_mask() -> tuple[dict, np.ndarray]:
    reference_path, _ = annual_path_band("AGB", 2019)
    with rasterio.open(reference_path) as reference:
        profile = reference.profile.copy()
        strict = np.zeros(reference.shape, dtype="uint8")
        with rasterio.open(STRICT_MASK) as mask_source:
            reproject(
                source=rasterio.band(mask_source, 1),
                destination=strict,
                src_transform=mask_source.transform,
                src_crs=mask_source.crs,
                dst_transform=reference.transform,
                dst_crs=reference.crs,
                src_nodata=mask_source.nodata,
                dst_nodata=0,
                resampling=Resampling.nearest,
            )
    return profile, strict > 0


def output_profile(reference: dict, count: int) -> dict:
    profile = reference.copy()
    profile.update(
        count=count,
        dtype="float32",
        nodata=NODATA,
        compress="deflate",
        predictor=3,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    return profile


def required_observations(years: list[int], phase: str) -> int:
    if len(years) == 1:
        return 1
    if phase == "event":
        return len(years)
    return min(2, len(years))


def period_median(
    annual: dict[int, np.ndarray], years: list[int], phase: str
) -> np.ndarray:
    values = np.stack([annual[year] for year in years])
    count = np.isfinite(values).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = np.nanmedian(values, axis=0).astype("float32")
    result[count < required_observations(years, phase)] = np.nan
    return result


def calculate_event_metrics(
    annual: dict[int, np.ndarray], event: dict, minimum: float
) -> dict[str, np.ndarray]:
    pre = period_median(annual, event["pre"], "pre")
    during = period_median(annual, event["event"], "event")
    post = period_median(annual, event["post"], "post")
    valid = (
        np.isfinite(pre)
        & np.isfinite(during)
        & np.isfinite(post)
        & (pre > minimum)
        & (during > minimum)
    )
    outputs = {
        metric: np.full(pre.shape, np.nan, dtype="float32") for metric in METRICS
    }
    outputs["resistance"][valid] = during[valid] / pre[valid]
    outputs["recovery"][valid] = post[valid] / during[valid]
    outputs["resilience"][valid] = post[valid] / pre[valid]
    return outputs


def band_descriptions() -> list[str]:
    names = []
    for event in EVENTS:
        for metric in METRICS:
            names.append(f"{event['event_id']}_{metric}")
    names.extend(f"median_{metric}_four_core_events" for metric in METRICS)
    names.extend(["valid_core_event_count", "decline_frequency_resistance_lt_0p90"])
    return names


def build_response_raster(
    variable: str, reference: dict, strict_mask: np.ndarray
) -> Path:
    output = RASTERS / f"Zoige_{variable}_four_core_event_response_90m.tif"
    years = all_required_years()
    descriptions = band_descriptions()
    minimum = 20.0 if variable == "AGB" else 0.05

    input_specs = {year: annual_path_band(variable, year) for year in years}
    missing = [str(path) for path, _ in input_specs.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing annual inputs:\n" + "\n".join(missing))

    with ExitStack() as context:
        sources = {
            year: (context.enter_context(rasterio.open(path)), band)
            for year, (path, band) in input_specs.items()
        }
        destination = context.enter_context(
            rasterio.open(output, "w", **output_profile(reference, len(descriptions)))
        )
        for index, description in enumerate(descriptions, start=1):
            destination.set_band_description(index, description)
        destination.update_tags(
            variable=variable,
            core_events="1991;1997;2002;2015-2016",
            aggregation="pixelwise median of event-specific ratios",
            minimum_valid_events=MIN_VALID_EVENTS,
            spatial_grid_m=90,
            peatland_mask="strict main95 fill400",
            agb_input=(
                "annual Peak-only AGB, band 3 of dual-product raster"
                if variable == "AGB"
                else "not applicable"
            ),
            fvc_input=(
                "annual PeakNDVI-derived FVC valid-only 3x3 median"
                if variable == "FVC"
                else "not applicable"
            ),
        )

        for window in iter_windows(reference["width"], reference["height"]):
            annual = {
                year: read_float(source, band, window)
                for year, (source, band) in sources.items()
            }
            r0, c0 = int(window.row_off), int(window.col_off)
            r1 = r0 + int(window.height)
            c1 = c0 + int(window.width)
            local_mask = strict_mask[r0:r1, c0:c1]
            for values in annual.values():
                values[~local_mask] = np.nan

            event_results = [
                calculate_event_metrics(annual, event, minimum) for event in EVENTS
            ]
            arrays = []
            for result in event_results:
                arrays.extend(result[metric] for metric in METRICS)

            resistance_stack = np.stack(
                [result["resistance"] for result in event_results]
            )
            valid_count = np.isfinite(resistance_stack).sum(axis=0).astype("float32")
            valid_across = valid_count >= MIN_VALID_EVENTS

            for metric in METRICS:
                metric_stack = np.stack([result[metric] for result in event_results])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    median = np.nanmedian(metric_stack, axis=0).astype("float32")
                median[~valid_across] = np.nan
                arrays.append(median)

            decline_count = np.sum(resistance_stack < 0.90, axis=0).astype("float32")
            decline_frequency = np.full(valid_count.shape, np.nan, dtype="float32")
            decline_frequency[valid_across] = (
                100.0 * decline_count[valid_across] / valid_count[valid_across]
            )
            valid_count[~local_mask] = np.nan
            arrays.extend([valid_count, decline_frequency])

            for band_index, array in enumerate(arrays, start=1):
                destination.write(
                    np.where(np.isfinite(array), array, NODATA).astype("float32"),
                    band_index,
                    window=window,
                )
    return output


def summarize_raster(path: Path, strict_pixels: int) -> pd.DataFrame:
    rows = []
    with rasterio.open(path) as source:
        for band in range(1, source.count + 1):
            values = source.read(band).astype("float32")
            valid = np.isfinite(values)
            if source.nodata is not None:
                valid &= ~np.isclose(values, source.nodata)
            data = values[valid]
            rows.append(
                {
                    "file": path.name,
                    "band": band,
                    "description": source.descriptions[band - 1],
                    "valid_pixels": int(data.size),
                    "valid_fraction_of_strict_mask": data.size / strict_pixels,
                    "p05": float(np.quantile(data, 0.05)) if data.size else np.nan,
                    "median": float(np.median(data)) if data.size else np.nan,
                    "p95": float(np.quantile(data, 0.95)) if data.size else np.nan,
                }
            )
    return pd.DataFrame(rows)


def geographic_template(
    reference_path: Path, strict_mask: np.ndarray
) -> tuple[dict, np.ndarray, tuple[int, int, int, int], list[float]]:
    with rasterio.open(reference_path) as source:
        transform, width, height = calculate_default_transform(
            source.crs,
            "EPSG:4326",
            source.width,
            source.height,
            *source.bounds,
        )
        boundary = np.zeros((height, width), dtype="uint8")
        reproject(
            source=strict_mask.astype("uint8"),
            destination=boundary,
            src_transform=source.transform,
            src_crs=source.crs,
            dst_transform=transform,
            dst_crs="EPSG:4326",
            src_nodata=0,
            dst_nodata=0,
            resampling=Resampling.nearest,
        )
    rows, cols = np.where(boundary > 0)
    pad = 4
    r0 = max(0, int(rows.min()) - pad)
    r1 = min(height, int(rows.max()) + 1 + pad)
    c0 = max(0, int(cols.min()) - pad)
    c1 = min(width, int(cols.max()) + 1 + pad)
    left, top = transform * (c0, r0)
    right, bottom = transform * (c1, r1)
    meta = {"transform": transform, "width": width, "height": height}
    return meta, boundary, (r0, r1, c0, c1), [left, right, bottom, top]


def geographic_band(
    path: Path,
    band: int,
    template: dict,
    boundary: np.ndarray,
    crop: tuple[int, int, int, int],
    extent: list[float],
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    destination = np.full(
        (template["height"], template["width"]), np.nan, dtype="float32"
    )
    with rasterio.open(path) as source:
        reproject(
            source=rasterio.band(source, band),
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=source.nodata,
            dst_transform=template["transform"],
            dst_crs="EPSG:4326",
            dst_nodata=np.nan,
            resampling=Resampling.nearest,
        )
    destination[boundary == 0] = np.nan
    r0, r1, c0, c1 = crop
    return destination[r0:r1, c0:c1], boundary[r0:r1, c0:c1] > 0, extent


def map_axis(
    axis: plt.Axes,
    extent: list[float],
    show_right: bool,
) -> None:
    axis.set_xlim(extent[0], extent[1])
    axis.set_ylim(extent[2], extent[3])
    mean_latitude = 0.5 * (extent[2] + extent[3])
    axis.set_aspect(1 / np.cos(np.deg2rad(mean_latitude)), adjustable="box")
    axis.tick_params(
        direction="in",
        top=True,
        right=True,
        labeltop=True,
        labelbottom=False,
        labelleft=True,
        labelright=show_right,
        pad=2,
    )
    axis.xaxis.set_major_locator(MultipleLocator(1.0))
    axis.yaxis.set_major_locator(MultipleLocator(1.0))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f}°E"))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f}°N"))
    for spine in axis.spines.values():
        spine.set_color("0.25")
        spine.set_linewidth(0.7)


def export_figure(figure: plt.Figure, stem: str) -> None:
    target = FIGURES / stem
    figure.savefig(target.with_suffix(".png"), dpi=450, bbox_inches="tight")
    figure.savefig(
        target.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    figure.savefig(target.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(target.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def draw_figure(
    paths: dict[str, Path], reference: dict, strict_mask: np.ndarray
) -> None:
    template, boundary, crop, extent = geographic_template(paths["AGB"], strict_mask)
    norm = TwoSlopeNorm(vmin=RATIO_MIN, vcenter=RATIO_CENTER, vmax=RATIO_MAX)
    cmap = LinearSegmentedColormap.from_list(
        "muted_response",
        ["#8f1738", "#dc735f", "#f7f6f2", "#77adc5", "#0b4f78"],
    )
    cmap.set_bad("white")

    median_bands = {"Resistance": 13, "Recovery": 14, "Resilience": 15}
    figure, axes = plt.subplots(2, 3, figsize=(7.2, 5.45))
    image = None
    for row, variable in enumerate(("AGB", "FVC")):
        for col, (metric, band) in enumerate(median_bands.items()):
            array, valid_boundary, panel_extent = geographic_band(
                paths[variable], band, template, boundary, crop, extent
            )
            axis = axes[row, col]
            image = axis.imshow(
                array,
                extent=panel_extent,
                origin="upper",
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
            )
            axis.contour(
                valid_boundary.astype(float),
                levels=[0.5],
                colors="0.76",
                linewidths=0.14,
                extent=panel_extent,
                origin="upper",
            )
            map_axis(axis, panel_extent, show_right=col == 2)
            axis.text(
                -0.11,
                1.04,
                f"({chr(97 + row * 3 + col)})",
                transform=axis.transAxes,
                fontsize=8.5,
                fontweight="bold",
                ha="left",
                va="bottom",
            )
            axis.text(
                0.5,
                -0.105,
                f"Median {variable} {metric.lower()}",
                transform=axis.transAxes,
                ha="center",
                va="top",
                fontsize=7.5,
            )

    color_axis = figure.add_axes([0.20, 0.065, 0.60, 0.025])
    colorbar = figure.colorbar(
        image, cax=color_axis, orientation="horizontal", extend="both"
    )
    colorbar.set_ticks([0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15])
    colorbar.ax.tick_params(labelsize=6, length=2)
    colorbar.set_label(
        "Median response ratio across four core drought episodes (1 = no net change)",
        fontsize=6.6,
        labelpad=2,
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.96,
        bottom=0.17,
        top=0.95,
        wspace=0.27,
        hspace=0.34,
    )
    export_figure(figure, "Fig_Sx_four_core_event_spatial_response_AGB_FVC")


def write_legend_and_metadata(paths: dict[str, Path]) -> None:
    legend = """**Fig. Sx | Spatial patterns of vegetation resistance, recovery and resilience across four core drought episodes in Zoige peatlands.** Pixel-level median resistance, recovery and resilience across the 1991, 1997, 2002 and 2015-2016 climate-defined drought episodes are shown for (a-c) AGB and (d-f) FVC. Response ratios were calculated separately for each event and then summarized as the pixelwise median across valid events. Only pixels with valid observations for at least three core episodes are displayed. A ratio of 1 indicates no net change. Recovery above 1 indicates an increase relative to the event-period state and does not necessarily indicate a return to the pre-event baseline. AGB and FVC were analysed on the 90 m grid; LAI was excluded because of its coarser spatial support.
"""
    (RUN / "figure_legend.md").write_text(legend, encoding="utf-8")
    metadata = {
        "core_events": EVENTS,
        "metrics": {
            "resistance": "event/pre",
            "recovery": "post/event",
            "resilience": "post/pre",
        },
        "minimum_valid_events": MIN_VALID_EVENTS,
        "agb_source": "annual 90 m Peak-only AGB (band 3 of dual-product raster)",
        "fvc_source": "annual 90 m PeakNDVI-derived FVC",
        "mask": "strict main95 peatland mask with fill400",
        "outputs": {key: str(value) for key, value in paths.items()},
        "figure_scale": [RATIO_MIN, RATIO_MAX],
        "software": {
            "python": "3.x",
            "matplotlib": mpl.__version__,
            "rasterio": rasterio.__version__,
        },
    }
    (RUN / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    ensure_dirs()
    reference, strict_mask = reference_profile_and_mask()
    paths = {
        variable: build_response_raster(variable, reference, strict_mask)
        for variable in ("AGB", "FVC")
    }
    summaries = pd.concat(
        [summarize_raster(path, int(strict_mask.sum())) for path in paths.values()],
        ignore_index=True,
    )
    summaries.to_csv(SOURCE_DATA / "four_core_event_map_summary.csv", index=False)
    draw_figure(paths, reference, strict_mask)
    write_legend_and_metadata(paths)
    print(f"Figure and rasters written to {RUN}")


if __name__ == "__main__":
    main()
