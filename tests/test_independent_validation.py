from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.validation.independent_2025_validation import (
    ALL_RECORDS_SCENARIO,
    PRIMARY_SCENARIO,
    PRIMARY_SUPPORT,
    calculate_metrics,
)


def test_quality_controlled_records_are_the_primary_validation_set() -> None:
    record_count = 16
    predictions = pd.DataFrame(
        {
            "support": [PRIMARY_SUPPORT] * record_count,
            "independent_qc_pass": [True] * 14 + [False] * 2,
            "inside_main95": [True] * record_count,
            "agb_mean_g_m2": np.arange(1, record_count + 1, dtype=float),
            "predicted_agb_g_m2": np.arange(1, record_count + 1, dtype=float),
            "same_site_group": [f"site_{index // 2}" for index in range(record_count)],
        }
    )

    metrics = calculate_metrics(predictions)
    record_metrics = metrics[
        metrics["support"].eq(PRIMARY_SUPPORT)
        & metrics["weighting"].eq("record_level")
    ].set_index("scenario")

    assert int(record_metrics.loc[PRIMARY_SCENARIO, "n"]) == 14
    assert int(record_metrics.loc[ALL_RECORDS_SCENARIO, "n"]) == 16
