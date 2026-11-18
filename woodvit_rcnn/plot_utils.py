"""Plots of the evaluation. Figures are drawn on the Agg canvas (no display needed); the output
path is always an argument."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

CM_TO_INCH = 1 / 2.54
OSA_LABELS = {"C_OSA": "C-OSA", "M_OSA": "M-OSA", "U_OSA": "U-OSA"}


def new_figure(width_cm: float = 12.0, height_cm: float = 6.5) -> Figure:
    fig = Figure(figsize=(width_cm * CM_TO_INCH, height_cm * CM_TO_INCH), dpi=150)
    FigureCanvasAgg(fig)
    return fig


def save_figure(fig: Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    return path


def plot_osa_series(series: pd.DataFrame, column_prefix: str, xlabel: str, path: str | Path) -> Path:
    """C-, M- and U-OSA over a threshold series (table of ``osa.osa_series``)."""
    columns = [c for c in series.columns if c.startswith(column_prefix)]
    thresholds = [float(c[len(column_prefix):]) for c in columns]
    fig = new_figure()
    ax = fig.add_subplot(1, 1, 1)
    for key, label in OSA_LABELS.items():
        values = series.loc[series["metric"] == key, columns].to_numpy(dtype=float).ravel()
        ax.plot(thresholds, values, marker="o", markersize=3, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Share of annotated area")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(thresholds)
    ax.set_xticklabels([f"{t:.2f}" for t in thresholds], rotation=45, ha="right", fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left")
    return save_figure(fig, path)
