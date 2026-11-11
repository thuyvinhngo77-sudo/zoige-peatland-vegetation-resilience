#!/usr/bin/env python
"""Prepare QA-filtered LAI/FAPAR and FVC on the common 5 km grid."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from netCDF4 import Dataset
from pyproj import Transformer
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "agb_lai_resilience_breakpoint_1988_2024"
OUTPUTS = OUT_ROOT / "outputs"
RASTERS = OUT_ROOT / "rasters_5km"
CACHE = OUT_ROOT / "cache"

BASE_PANEL = ROOT / "agb_driver_xgboost_shap_v2" / "cache" / "driver_panel_5km.pkl.gz"
GRID_JSON = ROOT / "agb_driver_xgboost_shap_v2" / "outputs" / "analysis_grid.json"
NOAA_ROOT = ROOT / "climate_data" / "input" / "noaa_lai_fapar_daily_789"
FVC_ROOT = Path(
    os.environ.get(
        "LANDSAT_COMPOSITE_ROOT",
        str(ROOT / "data" / "landsat_composites"),
    )
)

FVC_BAND = 13
EXPECTED_DAYS = 92
MIN_LAI_VALID_DAYS = 3
KNOWN_FVC_CACHE_REBUILD_YEARS = {2006, 2014, 2015, 2017, 2020, 2021, 2022}


def ensure_dirs() -> None:
    for path in (OUTPUTS, RASTERS / "lai_fapar", RASTERS / "fvc", CACHE):
        path.mkdir(parents=True, exist_ok=True)


def load_grid() -> dict:
    spec = json.loads(GRID_JSON.read_text(encoding="utf-8"))
    left, bottom, right, top = spec["bounds"]
    return {
        "crs": spec["crs"],
        "transform": from_origin(left, top, spec["grid_m"], spec["grid_m"]),
        "width": spec["width"],
        "height": spec["height"],
        "bounds": (left, bottom, right, top),
        "grid_m": spec["grid_m"],
    }


def grid_lonlat_bounds(grid: dict) -> tuple[float, float, float, float]:
    left, bottom, right, top = grid["bounds"]
    transformer = Transformer.from_crs(grid["crs"], "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(
        [left, left, right, right], [bottom, top, bottom, top]
    )
    return min(lon) - 0.15, min(lat) - 0.15, max(lon) + 0.15, max(lat) + 0.15


def warp_array(
    values: np.ndarray,
    src_transform,
    src_crs,
    grid: dict,
    resampling=Resampling.average,
) -> np.ndarray:
    dst = np.full((grid["height"], grid["width"]), np.nan, dtype="float32")
    reproject(
        source=values.astype("float32"),
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=np.nan,
        dst_transform=grid["transform"],
        dst_crs=grid["crs"],
        dst_nodata=np.nan,
        resampling=resampling,
    )
    return dst


def write_multiband(path: Path, arrays: list[np.ndarray], names: list[str], grid: dict) -> None:
    profile = {
        "driver": "GTiff",
        "width": grid["width"],
        "height": grid["height"],
        "count": len(arrays),
        "dtype": "float32",
        "crs": grid["crs"],
        "transform": grid["transform"],
        "nodata": -9999.0,
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
    }
    with rasterio.open(path, "w", **profile) as dst:
        for index, (array, name) in enumerate(zip(arrays, names), 1):
            dst.write(np.where(np.isfinite(array), array, -9999).astype("float32"), index)
            dst.set_band_description(index, name)


def read_multiband(path: Path) -> tuple[list[np.ndarray], list[str]]:
    with rasterio.open(path) as src:
        arrays = []
        for index in range(1, src.count + 1):
            arr = src.read(index).astype("float32")
            arr[arr == src.nodata] = np.nan
            arrays.append(arr)
        return arrays, list(src.descriptions)


def sensor_metadata(path: Path) -> tuple[str, str]:
    name = path.name
    if "AVHRR" in name:
        family = "AVHRR"
    elif "NPP15C1" in name:
        family = "VIIRS_S-NPP"
    elif "JP115C1" in name:
        family = "VIIRS_NOAA-20"
    else:
        family = "unknown"
    match = re.search(r"NOAA-(\d+)", name)
    platform = f"NOAA-{match.group(1)}" if match else ("S-NPP" if "S-NPP" in name else "unknown")
    return family, platform


def quick_readability(files: list[Path]) -> tuple[int, list[str]]:
    readable, bad = 0, []
    for path in files:
        try:
            with Dataset(path, "r") as ds:
                _ = ds.variables["LAI"].shape
                _ = ds.variables["QA"].shape
            readable += 1
        except Exception:
            bad.append(path.name)
    return readable, bad


def build_lai_year(year: int, grid: dict, rebuild: bool) -> tuple[dict, dict]:
    out_path = RASTERS / "lai_fapar" / f"Zoige_NOAA_LAI_FAPAR_JulSep_{year}_5km.tif"
    files = sorted((NOAA_ROOT / str(year)).glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No NOAA LAI/FAPAR files for {year}")
    family, platform = sensor_metadata(files[0])
    readable_files = 0
    unreadable_files = []
    if out_path.exists() and not rebuild:
        arrays, _ = read_multiband(out_path)
        lai, fapar, valid_days, valid_fraction = arrays
        readable_files, unreadable_files = quick_readability(files)
    else:
        lon_min, lat_min, lon_max, lat_max = grid_lonlat_bounds(grid)
        lai_daily, fapar_daily = [], []
        src_transform = None
        flip_y = False
        for path in files:
            try:
                with xr.open_dataset(path, decode_cf=False, mask_and_scale=False) as ds:
                    lon = ds["longitude"].values
                    lat = ds["latitude"].values
                    lon_idx = np.where((lon >= lon_min) & (lon <= lon_max))[0]
                    lat_idx = np.where((lat >= lat_min) & (lat <= lat_max))[0]
                    if not len(lon_idx) or not len(lat_idx):
                        raise ValueError(f"NOAA grid does not cover target for {path.name}")
                    lon_slice = slice(lon_idx.min(), lon_idx.max() + 1)
                    lat_slice = slice(lat_idx.min(), lat_idx.max() + 1)
                    raw_lai = ds["LAI"].isel(time=0, longitude=lon_slice, latitude=lat_slice).values
                    raw_fapar = ds["FAPAR"].isel(time=0, longitude=lon_slice, latitude=lat_slice).values
                    qa = ds["QA"].isel(time=0, longitude=lon_slice, latitude=lat_slice).values
                    # QA bits 0-1: retrieval quality; bits 2-5 value 24 denotes
                    # water. The separate domain-test bit is retained in the
                    # source archive but is not required after Quality_OK,
                    # because doing so removes most cloudy high-plateau cells.
                    valid = ((qa & 3) == 0) & ((qa & 60) != 24)
                    valid &= (raw_lai >= 0) & (raw_lai <= 7000)
                    valid &= (raw_fapar >= 0) & (raw_fapar <= 1000)
                    lai_daily.append(np.where(valid, raw_lai * 0.001, np.nan).astype("float32"))
                    fapar_daily.append(np.where(valid, raw_fapar * 0.001, np.nan).astype("float32"))
                    if src_transform is None:
                        lons = lon[lon_slice]
                        lats = lat[lat_slice]
                        dx = float(np.median(np.diff(lons)))
                        dy = float(abs(np.median(np.diff(lats))))
                        flip_y = bool(lats[0] < lats[-1])
                        north = lats[-1] if flip_y else lats[0]
                        src_transform = from_origin(lons[0] - dx / 2, north + dy / 2, dx, dy)
                    readable_files += 1
            except Exception:
                unreadable_files.append(path.name)
                continue
        if not lai_daily:
            raise ValueError(f"All NOAA LAI/FAPAR files are unreadable for {year}")
        lai_stack = np.stack(lai_daily)
        fapar_stack = np.stack(fapar_daily)
        native_valid_days = np.sum(np.isfinite(lai_stack), axis=0).astype("float32")
        with np.errstate(all="ignore"):
            native_lai = np.nanmedian(lai_stack, axis=0).astype("float32")
            native_fapar = np.nanmedian(fapar_stack, axis=0).astype("float32")
        native_lai[native_valid_days < MIN_LAI_VALID_DAYS] = np.nan
        native_fapar[native_valid_days < MIN_LAI_VALID_DAYS] = np.nan
        if flip_y:
            native_lai = np.flipud(native_lai)
            native_fapar = np.flipud(native_fapar)
            native_valid_days = np.flipud(native_valid_days)
        lai = warp_array(native_lai, src_transform, "EPSG:4326", grid)
        fapar = warp_array(native_fapar, src_transform, "EPSG:4326", grid)
        valid_days = warp_array(native_valid_days, src_transform, "EPSG:4326", grid)
        valid_fraction = valid_days / max(readable_files, 1)
        write_multiband(
            out_path,
            [lai, fapar, valid_days, valid_fraction],
            ["LAI_JulSep_median", "FAPAR_JulSep_median", "valid_day_count", "valid_day_fraction"],
            grid,
        )
    completeness = {
        "year": year,
        "file_count": len(files),
        "expected_file_count": EXPECTED_DAYS,
        "file_completeness": len(files) / EXPECTED_DAYS,
        "readable_file_count": readable_files,
        "unreadable_file_count": len(unreadable_files),
        "readable_completeness": readable_files / EXPECTED_DAYS,
        "sensor_family": family,
        "platform": platform,
        "annual_status": (
            "complete" if readable_files >= 90 else
            "usable_with_daily_gaps" if readable_files >= 70 else
            "incomplete"
        ),
        "unreadable_files": ";".join(unreadable_files),
    }
    arrays = {
        "lai": lai,
        "fapar": fapar,
        "lai_valid_days": valid_days,
        "lai_valid_fraction": valid_fraction,
    }
    print(
        f"LAI/FAPAR prepared: {year} "
        f"({readable_files}/{len(files)} readable, {family})",
        flush=True,
    )
    return arrays, completeness


def warp_fvc_tile(path: Path, grid: dict) -> np.ndarray:
    dst = np.full((grid["height"], grid["width"]), np.nan, dtype="float32")
    with rasterio.open(path) as src:
        reproject(
            source=rasterio.band(src, FVC_BAND),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=np.nan,
            dst_transform=grid["transform"],
            dst_crs=grid["crs"],
            dst_nodata=np.nan,
            resampling=Resampling.average,
        )
    return dst


def build_fvc_year(year: int, grid: dict, rebuild: bool) -> tuple[np.ndarray, dict]:
    out_path = RASTERS / "fvc" / f"Zoige_Landsat_FVC_PeakNDVI_JulSep_{year}_5km.tif"
    candidates = sorted(FVC_ROOT.glob(f"*PeakNDVI_JulSep_{year}_30m*.tif"))
    # Keep one file per Earth Engine tile coordinate. Prefer the original name
    # over Windows duplicate copies such as " (1).tif".
    tile_map = {}
    for path in candidates:
        match = re.search(r"-(\d{10})-(\d{10})", path.name)
        key = match.group(0) if match else path.name
        if key not in tile_map or " (" in tile_map[key].name:
            tile_map[key] = path
    tiles = sorted(tile_map.values())
    if not tiles:
        raise FileNotFoundError(f"No Landsat FVC tiles for {year}")
    force_rebuild = rebuild or year in KNOWN_FVC_CACHE_REBUILD_YEARS
    if out_path.exists() and not force_rebuild:
        arrays, _ = read_multiband(out_path)
        fvc = arrays[0]
    else:
        warped = [warp_fvc_tile(path, grid) for path in tiles]
        with np.errstate(all="ignore"):
            fvc = np.nanmean(np.stack(warped), axis=0).astype("float32")
        fvc[(fvc < 0) | (fvc > 1)] = np.nan
        write_multiband(out_path, [fvc], ["FVC_PeakNDVI_JulSep"], grid)
    tile_status = "complete" if len(tiles) == 4 else "incomplete"
    info = {"year": year, "fvc_tile_count": len(tiles), "fvc_tile_status": tile_status}
    print(f"FVC prepared: {year} ({len(tiles)} tiles)", flush=True)
    return fvc, info


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def regional_summaries(panel: pd.DataFrame, completeness: pd.DataFrame) -> pd.DataFrame:
    records = []
    metadata = completeness.set_index("year")
    for year, group in panel.groupby("year"):
        full = group[group["full_coverage"] >= 0.5]
        peat = group[(group["peat_area_fraction"] >= 0.1) & (group["peat_valid_fraction"] >= 0.5)]
        for region, frame in (("full_aoi", full), ("peatland", peat)):
            if region == "peatland":
                weights = frame["peat_area_fraction"]
                agb = weighted_mean(frame["peat_agb"], weights)
            else:
                weights = frame["full_coverage"].clip(lower=0)
                agb = weighted_mean(frame["full_agb"], weights)
            record = {
                "year": int(year),
                "region": region,
                "agb_g_m2": agb,
                "fvc": weighted_mean(frame["fvc"], weights),
                "lai": weighted_mean(frame["lai"], weights),
                "fapar": weighted_mean(frame["fapar"], weights),
                "lai_valid_fraction": weighted_mean(frame["lai_valid_fraction"], weights),
                "n_grid_cells": len(frame),
                "status": "observed",
            }
            if year in metadata.index:
                record.update(metadata.loc[year, ["file_count", "sensor_family", "platform", "annual_status"]].to_dict())
            records.append(record)
    return pd.DataFrame(records)


def temporary_2011_series(summary: pd.DataFrame, lai_2011: dict, fvc_2011: np.ndarray, base: pd.DataFrame) -> pd.DataFrame:
    temp = summary.copy()
    grid_frame = base[base["year"] == 2010][
        ["row", "col", "full_coverage", "peat_area_fraction", "peat_valid_fraction"]
    ].copy()
    grid_frame["lai"] = lai_2011["lai"].ravel()
    grid_frame["fapar"] = lai_2011["fapar"].ravel()
    grid_frame["lai_valid_fraction"] = lai_2011["lai_valid_fraction"].ravel()
    # The partial 2011 FVC raster is deliberately ignored in the regional
    # temporary series. FVC and AGB use a transparent 2010-2012 interpolation.
    rows = []
    for region in ["full_aoi", "peatland"]:
        before = summary[(summary["region"] == region) & (summary["year"] == 2010)].iloc[0]
        after = summary[(summary["region"] == region) & (summary["year"] == 2012)].iloc[0]
        if region == "peatland":
            frame = grid_frame[(grid_frame["peat_area_fraction"] >= 0.1) & (grid_frame["peat_valid_fraction"] >= 0.5)]
            weights = frame["peat_area_fraction"]
        else:
            frame = grid_frame[grid_frame["full_coverage"] >= 0.5]
            weights = frame["full_coverage"]
        rows.append({
            "year": 2011,
            "region": region,
            "agb_g_m2": float((before["agb_g_m2"] + after["agb_g_m2"]) / 2),
            "fvc": float((before["fvc"] + after["fvc"]) / 2),
            "lai": weighted_mean(frame["lai"], weights),
            "fapar": weighted_mean(frame["fapar"], weights),
            "lai_valid_fraction": weighted_mean(frame["lai_valid_fraction"], weights),
            "n_grid_cells": len(frame),
            "status": "interpolated_visualization_only",
            "file_count": 92,
            "sensor_family": "AVHRR",
            "platform": "NOAA-19",
            "annual_status": "LAI_observed_AGB_FVC_interpolated",
        })
    return pd.concat([temp, pd.DataFrame(rows)], ignore_index=True).sort_values(["region", "year"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-lai", action="store_true")
    parser.add_argument("--rebuild-fvc", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    ensure_dirs()
    grid = load_grid()
    base = pd.read_pickle(BASE_PANEL, compression="gzip")
    agb_years = sorted(base["year"].unique())
    years = sorted(set(agb_years) | {2011})
    lai_by_year, fvc_by_year, completeness_rows, fvc_rows = {}, {}, [], []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(build_lai_year, year, grid, args.rebuild_lai): year
            for year in years
        }
        for future in as_completed(futures):
            year = futures[future]
            lai_by_year[year], info = future.result()
            completeness_rows.append(info)
    with ProcessPoolExecutor(max_workers=max(1, min(args.workers, 3))) as executor:
        futures = {
            executor.submit(build_fvc_year, year, grid, args.rebuild_fvc): year
            for year in years
        }
        for future in as_completed(futures):
            year = futures[future]
            fvc_by_year[year], fvc_info = future.result()
            fvc_rows.append(fvc_info)

    completeness = pd.DataFrame(completeness_rows).merge(pd.DataFrame(fvc_rows), on="year", how="outer")
    completeness.to_csv(OUTPUTS / "annual_input_completeness.csv", index=False, encoding="utf-8-sig")

    panel = base.copy()
    state_rows = []
    rr, cc = np.indices((grid["height"], grid["width"]))
    for year in agb_years:
        state_rows.append(pd.DataFrame({
            "year": year,
            "row": rr.ravel(),
            "col": cc.ravel(),
            "lai": lai_by_year[year]["lai"].ravel(),
            "fapar": lai_by_year[year]["fapar"].ravel(),
            "lai_valid_days": lai_by_year[year]["lai_valid_days"].ravel(),
            "lai_valid_fraction": lai_by_year[year]["lai_valid_fraction"].ravel(),
            "fvc": fvc_by_year[year].ravel(),
        }))
    state = pd.concat(state_rows, ignore_index=True)
    panel = panel.merge(state, on=["year", "row", "col"], how="left", validate="one_to_one")
    panel.to_pickle(CACHE / "annual_state_panel_5km.pkl.gz", compression="gzip")
    summary = regional_summaries(panel, completeness)
    summary.to_csv(OUTPUTS / "annual_regional_state_summary.csv", index=False, encoding="utf-8-sig")
    temporary = temporary_2011_series(summary, lai_by_year[2011], fvc_by_year[2011], base)
    temp_dir = OUT_ROOT / "temporary_group_meeting"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temporary.to_csv(temp_dir / "annual_regional_state_with_2011_visualization_only.csv", index=False, encoding="utf-8-sig")
    print(f"Annual state panel complete: {len(panel):,} rows", flush=True)


if __name__ == "__main__":
    main()
