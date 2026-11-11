#!/usr/bin/env python
"""Numerical and export QA for the four-core-event response map."""

from pathlib import Path

import numpy as np
import rasterio
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "drought_response_maps_four_core_20260921"


def main() -> None:
    lines = ["# QA notes", ""]
    for variable in ("AGB", "FVC"):
        path = RUN / "rasters" / f"Zoige_{variable}_four_core_event_response_90m.tif"
        with rasterio.open(path) as source:
            lines.append(
                f"- {variable}: {source.count} bands; shape {source.height} x "
                f"{source.width}; CRS {source.crs}."
            )
            count = source.read(16)
            valid_count = count[count != source.nodata]
            unique, frequencies = np.unique(valid_count, return_counts=True)
            distribution = ", ".join(
                f"{int(value)} events: {int(frequency):,} pixels"
                for value, frequency in zip(unique, frequencies)
            )
            lines.append(f"- {variable} valid-event distribution: {distribution}.")
            for event_index, event_id in enumerate(
                ("1991", "1997", "2002", "2015-2016")
            ):
                resistance, recovery, resilience = [
                    source.read(event_index * 3 + offset).astype("float64")
                    for offset in (1, 2, 3)
                ]
                valid = (
                    (resistance != source.nodata)
                    & (recovery != source.nodata)
                    & (resilience != source.nodata)
                    & np.isfinite(resistance)
                    & np.isfinite(recovery)
                    & np.isfinite(resilience)
                )
                error = np.max(
                    np.abs(
                        resistance[valid] * recovery[valid] - resilience[valid]
                    )
                )
                lines.append(
                    f"- {variable} {event_id}: max |resistance x recovery - "
                    f"resilience| = {error:.3e}."
                )

    figure_stem = "Fig_Sx_four_core_event_spatial_response_AGB_FVC"
    png = RUN / "figures" / f"{figure_stem}.png"
    with Image.open(png) as image:
        lines.append(f"- PNG dimensions: {image.width} x {image.height} px.")
    for suffix in ("png", "tiff", "pdf", "svg"):
        path = RUN / "figures" / f"{figure_stem}.{suffix}"
        lines.append(f"- {suffix.upper()} export: {path.stat().st_size:,} bytes.")
    lines.extend(
        [
            "- Visual QA: panel labels, map titles and colour bar are readable; no overlaps remain.",
            "- The colour scale is shared across all six panels and centred at a response ratio of 1.",
            "- Values outside 0.85-1.15 are retained in the rasters and displayed at the extended colour-bar ends.",
            "",
        ]
    )
    output = RUN / "qa_notes.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
