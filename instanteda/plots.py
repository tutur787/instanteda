import os

import numpy as np
import polars as pl
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import Normalize

from instanteda.constants import CMAP, DPI, FIGSIZE, FONT_SIZE, HATCHES
from instanteda.utils import (
    _cat_columns,
    _datetime_columns,
    _get_colors,
    _num_columns,
    _show_fig,
    _truncate,
)


def _plot_categorical(pldf, save_dir):
    cat_cols = _cat_columns(pldf)
    figs = {}
    for col in cat_cols:
        vc = pldf[col].value_counts().sort("count", descending=True)
        if vc.height > 10:
            vc = vc.head(10)

        labels = vc[col].to_list()
        values = vc["count"].to_numpy()
        n = len(labels)
        colors = _get_colors(n)

        fig, ax = plt.subplots(figsize=FIGSIZE)
        bars = ax.bar(
            range(n), values,
            color=colors, edgecolor="black", linewidth=0.8,
        )
        for i, bar in enumerate(bars):
            bar.set_hatch(HATCHES[i % len(HATCHES)])

        ax.set_xticks(range(n))
        ax.set_xticklabels([_truncate(v) for v in labels], rotation=45, ha="right")
        ax.set_ylabel("Count")
        ax.set_xlabel(col)
        fig.tight_layout()

        path = os.path.join(save_dir, f"cat_{col}.png")
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
        figs[col] = fig
        _show_fig(fig)
    return figs


def _plot_numerical(pldf, save_dir):
    num_cols = _num_columns(pldf)
    cmap_obj = matplotlib.colormaps[CMAP]
    figs = {}
    for col in num_cols:
        data = pldf[col].drop_nulls().to_numpy()
        if len(data) == 0:
            continue

        fig, ax = plt.subplots(figsize=FIGSIZE)
        n_bins = min(30, max(10, int(np.sqrt(len(data)))))
        _, bins, patches = ax.hist(
            data, bins=n_bins, edgecolor="black", linewidth=0.8,
        )

        norm = Normalize(vmin=bins.min(), vmax=bins.max())
        for patch, left in zip(patches, bins[:-1]):
            patch.set_facecolor(cmap_obj(norm(left)))
            patch.set_hatch("///")

        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        fig.tight_layout()

        path = os.path.join(save_dir, f"num_{col}.png")
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
        figs[col] = fig
        _show_fig(fig)
    return figs


def _plot_datetime(pldf, save_dir):
    """Line plot of record count over time for each datetime column."""
    dt_cols = _datetime_columns(pldf)
    cmap_obj = matplotlib.colormaps[CMAP]
    figs = {}
    for col in dt_cols:
        series = pldf[col].drop_nulls().sort()
        if series.len() < 2:
            continue

        dates = series.to_pandas()
        date_nums = mdates.date2num(dates)
        n_bins = min(50, max(10, len(dates) // 20))
        counts, edges = np.histogram(date_nums, bins=n_bins)
        centers = mdates.num2date((edges[:-1] + edges[1:]) / 2)

        fig, ax = plt.subplots(figsize=FIGSIZE)
        color = cmap_obj(0.3)
        ax.plot(centers, counts, color=color, linewidth=1.5)
        ax.fill_between(centers, counts, alpha=0.3, color=color, hatch="///")

        ax.set_xlabel(col)
        ax.set_ylabel("Record count")
        ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(mdates.AutoDateLocator()))
        fig.autofmt_xdate()
        fig.tight_layout()

        path = os.path.join(save_dir, f"dt_{col}.png")
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
        figs[col] = fig
        _show_fig(fig)
    return figs


def _plot_correlations(pldf, save_dir):
    num_cols = _num_columns(pldf)
    if len(num_cols) < 2:
        return None

    n = len(num_cols)
    corr_vals = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            r = pldf.select(pl.corr(num_cols[i], num_cols[j])).item()
            r = 0.0 if r is None else float(r)
            corr_vals[i, j] = r
            corr_vals[j, i] = r

    fig, ax = plt.subplots(figsize=FIGSIZE)
    im = ax.imshow(corr_vals, cmap=CMAP, vmin=-1, vmax=1, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(num_cols, rotation=45, ha="right")
    ax.set_yticklabels(num_cols)

    thresh = 0.5
    for i in range(n):
        for j in range(n):
            val = corr_vals[i, j]
            color = "white" if abs(val) > thresh else "black"
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center", color=color,
                fontsize=max(FONT_SIZE - 4, 6),
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=FONT_SIZE - 2)
    fig.tight_layout()

    path = os.path.join(save_dir, "correlations.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    _show_fig(fig)
    return fig
