from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "drought_reanalysis_20260916"
OUT = RUN / "figures_manuscript"
DATA = OUT / "source_data"
TABLES = RUN / "outputs"

CORE_EVENTS = ["1991", "1997", "2002", "2015-2016"]
EXPANDED_EVENTS = ["1994", "2000", "2022"]
VARIABLES = ["AGB", "FVC", "LAI"]

COLORS = {
    "AGB": "#2F6FAE",
    "FVC": "#2A927B",
    "LAI": "#7B65A7",
    "core": "#B95756",
    "expanded": "#C28E3B",
    "other": "#B7BEC2",
    "ink": "#252525",
    "grid": "#D9D9D9",
}


def configure_style():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def full_frame(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color(COLORS["ink"])


def panel_label(ax, letter):
    ax.text(
        0.0,
        1.045,
        f"({letter})",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontweight="bold",
        fontsize=10,
        clip_on=False,
    )


def bottom_title(ax, text, y=-0.24):
    ax.text(
        0.5,
        y,
        text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        clip_on=False,
    )


def save_figure(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def load_data():
    climate = pd.read_csv(TABLES / "annual_climate_fixed_support.csv")
    registry = pd.read_csv(TABLES / "event_registry.csv", dtype={"id": str})
    metrics = pd.read_csv(
        TABLES / "event_metric_summary.csv", dtype={"event_id": str}
    )
    trajectories = pd.read_csv(
        TABLES / "annual_response_trajectories.csv", dtype={"event_id": str}
    )
    recovery = pd.read_csv(
        TABLES / "conditional_recovery_curves.csv", dtype={"event_id": str}
    )
    return climate, registry, metrics, trajectories, recovery


def event_year_sets(registry):
    core = set()
    expanded = set()
    for row in registry.itertuples():
        target = core if row.tier == "core" else expanded
        target.update(range(int(row.start), int(row.end) + 1))
    return core, expanded


def metric_panel(ax, data, metric, letter, subtitle, limited):
    events = CORE_EVENTS
    positions = np.arange(len(events), dtype=float)
    offsets = {"AGB": -0.18, "FVC": 0.0, "LAI": 0.18}
    markers = {"AGB": "o", "FVC": "s", "LAI": "^"}

    for variable in VARIABLES:
        q = data[(data["variable"] == variable) & (data["metric"] == metric)].copy()
        q = q.set_index("event_id").reindex(events)
        x = positions + offsets[variable]
        y = q["median"].to_numpy(float)
        lo = q["ci_low"].to_numpy(float)
        hi = q["ci_high"].to_numpy(float)
        err = np.vstack([y - lo, hi - y])
        for i, event in enumerate(events):
            is_limited = (event, variable) in limited
            ax.errorbar(
                x[i],
                y[i],
                yerr=err[:, i : i + 1],
                fmt=markers[variable],
                ms=5.6,
                color=COLORS[variable],
                markerfacecolor="white" if is_limited else COLORS[variable],
                markeredgecolor=COLORS[variable],
                markeredgewidth=1.1,
                elinewidth=1.1,
                capsize=2.3,
                zorder=3,
            )

    ax.axhline(1.0, color="#666666", linestyle="--", linewidth=0.9, zorder=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(["1991", "1997", "2002", "2015–16"])
    ax.set_ylabel("Response ratio")
    ax.set_ylim(0.78, 1.13)
    ax.set_yticks(np.arange(0.8, 1.11, 0.1))
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.5, alpha=0.55)
    full_frame(ax)
    panel_label(ax, letter)
    bottom_title(ax, subtitle)


def make_figure7(climate, registry, metrics):
    fixed = metrics[
        (metrics["scenario"] == "fixed_window")
        & (metrics["event_id"].isin(CORE_EVENTS))
    ].copy()
    core_years, expanded_years = event_year_sets(registry)

    fig = plt.figure(figsize=(7.2, 6.45), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.16, 1.0],
        hspace=0.52,
        wspace=0.25,
        left=0.09,
        right=0.91,
        top=0.91,
        bottom=0.18,
    )
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    years = climate["year"].astype(int).to_numpy()
    colors = [
        COLORS["core"]
        if y in core_years
        else COLORS["expanded"]
        if y in expanded_years
        else COLORS["other"]
        for y in years
    ]
    ax_a.bar(years, climate["pre_jas_z"], width=0.82, color=colors, edgecolor="none")
    ax_a.axhline(0, color=COLORS["ink"], linewidth=0.75)
    ax_a.axhline(-1, color=COLORS["core"], linestyle="--", linewidth=1.0)
    ax_a.set_ylabel("JAS precipitation anomaly (z)")
    ax_a.set_xlim(1987.3, 2024.7)
    ax_a.set_ylim(-3.05, 2.15)
    ax_a.set_xticks(np.arange(1988, 2025, 4))

    ax_ai = ax_a.twinx()
    ax_ai.plot(
        years,
        climate["ai"],
        color="#50575A",
        linewidth=0.7,
        alpha=0.55,
        zorder=2,
    )
    ai_point_colors = [
        COLORS["core"]
        if y in core_years
        else COLORS["expanded"]
        if y in expanded_years
        else "#666D70"
        for y in years
    ]
    ax_ai.scatter(
        years,
        climate["ai"],
        s=13,
        c=ai_point_colors,
        edgecolor="white",
        linewidth=0.25,
        zorder=3,
    )
    ai_median = float(climate["ai"].median())
    ax_ai.axhline(ai_median, color="#555555", linestyle=":", linewidth=0.9)
    ax_ai.set_ylabel("Annual aridity index")
    ax_ai.set_ylim(0.88, 1.78)
    ax_ai.tick_params(direction="in", right=True)
    full_frame(ax_a)
    ax_ai.spines["top"].set_visible(False)
    ax_ai.spines["bottom"].set_visible(False)
    ax_ai.spines["left"].set_visible(False)
    ax_ai.spines["right"].set_linewidth(0.8)
    panel_label(ax_a, "a")
    bottom_title(ax_a, "Climate-defined dry episodes", y=-0.20)

    legend_handles = [
        Patch(facecolor=COLORS["core"], label="Core drought"),
        Patch(facecolor=COLORS["expanded"], label="Expanded dry-year sensitivity"),
        Patch(facecolor=COLORS["other"], label="Other years"),
    ]
    ax_a.legend(
        handles=legend_handles,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        handlelength=1.8,
        columnspacing=1.6,
    )

    resistance_limited = {("2015-2016", "LAI")}
    postpre_limited = {
        ("1997", "AGB"),
        ("1997", "FVC"),
        ("2002", "AGB"),
        ("2002", "FVC"),
        ("2015-2016", "LAI"),
    }
    metric_panel(
        ax_b,
        fixed,
        "resistance",
        "b",
        "State during core droughts",
        resistance_limited,
    )
    metric_panel(
        ax_c,
        fixed,
        "resilience",
        "c",
        "Post-event state relative to baseline",
        postpre_limited,
    )
    ax_c.set_ylabel("")
    ax_c.tick_params(labelleft=True)

    variable_handles = [
        Line2D(
            [0],
            [0],
            marker={"AGB": "o", "FVC": "s", "LAI": "^"}[v],
            color=COLORS[v],
            markerfacecolor=COLORS[v],
            linewidth=0,
            markersize=6,
            label=v,
        )
        for v in VARIABLES
    ]
    variable_handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="#666666",
            markerfacecolor="white",
            linewidth=0,
            markersize=6,
            label="Limited comparable window",
        )
    )
    fig.legend(
        handles=variable_handles,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        columnspacing=1.5,
        handletextpad=0.45,
    )

    DATA.mkdir(parents=True, exist_ok=True)
    climate.assign(
        event_class=[
            "core" if y in core_years else "expanded" if y in expanded_years else "other"
            for y in years
        ]
    ).to_csv(DATA / "Fig7a_climate_selection.csv", index=False)
    fixed.to_csv(DATA / "Fig7bc_core_event_ratios.csv", index=False)
    save_figure(fig, "Fig7_core_drought_responses")


def make_figure8(trajectories, recovery):
    traj = trajectories[
        (trajectories["scenario"] == "fixed_window")
        & (trajectories["event_id"].isin(CORE_EVENTS))
    ].copy()
    rec = recovery[
        (recovery["scenario"] == "fixed_window")
        & (recovery["event_id"] == "1991")
        & (recovery["variable"] == "FVC")
        & (~recovery["sustained_two_years"])
    ].copy()

    fig = plt.figure(figsize=(7.2, 7.25), constrained_layout=False)
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.0, 1.0, 0.86],
        hspace=0.58,
        wspace=0.22,
        left=0.13,
        right=0.98,
        top=0.92,
        bottom=0.12,
    )
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]
    letters = ["a", "b", "c", "d"]

    for ax, event, letter in zip(axes, CORE_EVENTS, letters):
        q_event = traj[traj["event_id"] == event]
        event_years = sorted(q_event.loc[q_event["is_event"], "year"].unique())
        if event_years:
            ax.axvspan(
                min(event_years) - 0.45,
                max(event_years) + 0.45,
                color=COLORS["core"],
                alpha=0.10,
                linewidth=0,
                zorder=0,
            )
        for variable in VARIABLES:
            q = q_event[q_event["variable"] == variable].sort_values("year")
            x = q["year"].to_numpy(float)
            y = q["ratio"].to_numpy(float)
            lo = q["ci_low"].to_numpy(float)
            hi = q["ci_high"].to_numpy(float)
            ax.fill_between(x, lo, hi, color=COLORS[variable], alpha=0.12, linewidth=0)
            ax.plot(
                x,
                y,
                color=COLORS[variable],
                marker="o",
                markersize=3.7,
                linewidth=1.35,
                label=variable,
            )
        ax.axhline(1.0, color="#666666", linestyle="--", linewidth=0.85)
        xmin = int(q_event["year"].min())
        xmax = int(q_event["year"].max())
        ax.set_xlim(xmin - 0.35, xmax + 0.35)
        ax.set_xticks(np.arange(xmin, xmax + 1, 2 if xmax - xmin >= 6 else 1))
        ax.set_ylim(0.78, 1.22)
        ax.set_yticks([0.8, 0.9, 1.0, 1.1, 1.2])
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.5, alpha=0.5)
        full_frame(ax)
        panel_label(ax, letter)
        bottom_title(ax, event.replace("-", "–"), y=-0.22)

    axes[0].set_ylabel("State / pre-event baseline")
    axes[2].set_ylabel("State / pre-event baseline")
    axes[1].set_ylabel("")
    axes[3].set_ylabel("")

    indicator_handles = [
        Line2D([0], [0], color=COLORS[v], marker="o", lw=1.5, ms=4.5, label=v)
        for v in VARIABLES
    ]
    fig.legend(
        handles=indicator_handles,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.985),
        columnspacing=2.0,
    )

    ax_e = fig.add_subplot(gs[2, :])
    threshold_colors = {0.90: "#8CCDBB", 0.95: "#4AA58C", 1.00: "#176B59"}
    threshold_labels = {
        0.90: "90% of baseline",
        0.95: "95% of baseline",
        1.00: "100% of baseline",
    }
    for target in [0.90, 0.95, 1.00]:
        q = rec[np.isclose(rec["target_fraction"], target)].sort_values("post_year")
        ax_e.plot(
            q["post_year"],
            q["attained_fraction"],
            color=threshold_colors[target],
            marker="o",
            markersize=4.8,
            linewidth=1.6,
            label=threshold_labels[target],
        )
        if len(q):
            row = q.iloc[-1]
            ax_e.annotate(
                f"{row['attained_fraction'] * 100:.1f}%",
                (row["post_year"], row["attained_fraction"]),
                xytext=(7, 0),
                textcoords="offset points",
                va="center",
                color=threshold_colors[target],
                fontsize=8,
            )
    ax_e.set_xlim(0.85, 2.28)
    ax_e.set_xticks([1, 2])
    ax_e.set_ylim(0, 1.04)
    ax_e.set_yticks(np.arange(0, 1.01, 0.2))
    ax_e.set_xlabel("Years after the 1991 episode")
    ax_e.set_ylabel("Attained fraction of declined FVC area")
    ax_e.grid(axis="y", color=COLORS["grid"], linewidth=0.5, alpha=0.55)
    full_frame(ax_e)
    panel_label(ax_e, "e")
    bottom_title(ax_e, "Conditional FVC attainment after the 1991 drought", y=-0.31)
    ax_e.legend(loc="upper left", ncol=3, columnspacing=1.2, handlelength=2.0)
    ax_e.text(
        0.995,
        0.04,
        "Initial cohort: >10% event-period decline (52.4% of valid peatland area)",
        transform=ax_e.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#555555",
    )

    DATA.mkdir(parents=True, exist_ok=True)
    traj.to_csv(DATA / "Fig8ad_core_event_trajectories.csv", index=False)
    rec.to_csv(DATA / "Fig8e_1991_FVC_conditional_recovery.csv", index=False)
    save_figure(fig, "Fig8_core_event_trajectories_and_recovery")


def write_caption_file():
    caption = """# Revised manuscript figure captions

**Fig. 7 | Climate-defined dry episodes and vegetation responses in Zoige peatlands.** (a) Standardized July–September precipitation anomalies and annual aridity index during 1988–2024. Red bars and points denote four core drought episodes selected using precipitation anomaly <= -1 and aridity above its period median; gold denotes three expanded dry-year sensitivity episodes. (b) Event-period resistance and (c) post-event state relative to the pre-event baseline for the four core episodes. Points are peatland-area-weighted medians and error bars are 95% intervals from 2,000 resamples of 25 km spatial blocks. Filled symbols denote estimates supported by the predefined comparable-window checks; open symbols denote estimates with limited comparable baseline or follow-up observations. AGB and FVC were evaluated on the fixed 5 km grid, whereas LAI retained its approximately 5 km observational support.

**Fig. 8 | Annual vegetation trajectories and conditional recovery during core drought episodes.** (a–d) Annual AGB, FVC and LAI states relative to their event-specific pre-event baselines for the 1991, 1997, 2002 and 2015–2016 episodes. Shading around lines denotes 95% spatial-block bootstrap intervals, red bands denote climate-defined event years and the horizontal line marks the pre-event baseline. (e) Cumulative fraction of the 1991 FVC cohort that declined by more than 10% during the event and subsequently attained 90%, 95% or 100% of its pre-event baseline. Attainment was evaluated only within the observed two-year follow-up and does not imply sustained recovery or causal attribution.
"""
    (OUT / "figure_captions.md").write_text(caption, encoding="utf-8")


def main():
    configure_style()
    OUT.mkdir(parents=True, exist_ok=True)
    climate, registry, metrics, trajectories, recovery = load_data()
    make_figure7(climate, registry, metrics)
    make_figure8(trajectories, recovery)
    write_caption_file()
    print(f"Wrote revised manuscript figures to {OUT}")


if __name__ == "__main__":
    main()
