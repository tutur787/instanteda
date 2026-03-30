import os

import numpy as np
import pandas as pd
import polars as pl
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from instanteda.constants import CMAP, DPI, FIGSIZE, FONT_SIZE, HATCHES
from instanteda.utils import (
    _get_colors,
    _is_continuous,
    _is_datetime_dtype,
    _num_columns,
    _show_fig,
    _truncate,
)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def _plot_col_by_target(pldf, col, target, target_continuous, save_dir):
    """Dispatch to the right plot for one feature column vs. the target.

    datetime col  (any target)                ->  line over time
    datetime target x continuous col          ->  line of col over time
    datetime target x categorical col         ->  skip
    continuous  target x continuous  col       ->  scatter
    continuous  target x categorical col      ->  box plot
    categorical target x continuous  col      ->  overlaid histograms
    categorical target x categorical col      ->  grouped bar chart
    """
    col_is_dt = _is_datetime_dtype(pldf.schema[col])
    target_is_dt = _is_datetime_dtype(pldf.schema[target])

    if col_is_dt and target_is_dt:
        return None
    if col_is_dt:
        if target_continuous:
            return _line_dt_cont(pldf, col, target, save_dir)
        return _line_dt_cat(pldf, col, target, save_dir)
    if target_is_dt:
        if _is_continuous(pldf[col]):
            return _line_dt_cont(pldf, target, col, save_dir)
        return None

    col_continuous = _is_continuous(pldf[col])

    if target_continuous and col_continuous:
        return _scatter_cont_cont(pldf, col, target, save_dir)
    if target_continuous and not col_continuous:
        return _box_cont_target_cat_col(pldf, col, target, save_dir)
    if not target_continuous and col_continuous:
        return _hist_cat_target_cont_col(pldf, col, target, save_dir)
    return _bar_cat_target_cat_col(pldf, col, target, save_dir)


# ---------------------------------------------------------------------------
# datetime column x continuous value
# ---------------------------------------------------------------------------

def _line_dt_cont(pldf, dt_col, val_col, save_dir):
    """Scatter of a continuous value over a datetime axis."""
    sub = pldf.select([dt_col, val_col]).drop_nulls().sort(dt_col)
    if sub.height < 2:
        return None

    dates = sub[dt_col].to_pandas()
    values = sub[val_col].to_numpy().astype(float)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    cmap_obj = matplotlib.colormaps[CMAP]
    ax.scatter(
        dates, values, c=values, cmap=CMAP, s=8, alpha=0.6,
        edgecolors="black", linewidths=0.2,
    )

    window = max(sub.height // 30, 2)
    rolling = pd.Series(values, index=dates).rolling(window, center=True).mean()
    ax.plot(rolling.index, rolling.values, color=cmap_obj(0.85), linewidth=1.8)

    ax.set_xlabel(dt_col)
    ax.set_ylabel(val_col)
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(mdates.AutoDateLocator()))
    fig.autofmt_xdate()
    fig.tight_layout()

    path = os.path.join(save_dir, f"target_{val_col}_by_{dt_col}_line.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    _show_fig(fig)
    return fig


# ---------------------------------------------------------------------------
# datetime column x categorical value
# ---------------------------------------------------------------------------

def _line_dt_cat(pldf, dt_col, cat_col, save_dir):
    """Multi-line record count over time, one line per category."""
    sub = pldf.select([dt_col, cat_col]).drop_nulls().sort(dt_col)
    groups = sorted(sub[cat_col].unique().to_list(), key=str)
    if len(groups) < 2:
        return None

    all_date_nums = mdates.date2num(sub[dt_col].to_pandas())
    n_bins = min(50, max(10, len(all_date_nums) // 20))
    _, edges = np.histogram(all_date_nums, bins=n_bins)
    centers = mdates.num2date((edges[:-1] + edges[1:]) / 2)

    n_groups = len(groups)
    colors = _get_colors(n_groups)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for i, g in enumerate(groups):
        g_dates = mdates.date2num(
            sub.filter(pl.col(cat_col) == g)[dt_col].to_pandas()
        )
        counts, _ = np.histogram(g_dates, bins=edges)
        ax.plot(centers, counts, color=colors[i], linewidth=1.5, label=_truncate(g))

    ax.set_xlabel(dt_col)
    ax.set_ylabel("Record count")
    ax.legend(title=cat_col, fontsize=FONT_SIZE - 4, title_fontsize=FONT_SIZE - 3)
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(mdates.AutoDateLocator()))
    fig.autofmt_xdate()
    fig.tight_layout()

    path = os.path.join(save_dir, f"target_{cat_col}_by_{dt_col}_line.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    _show_fig(fig)
    return fig


# ---------------------------------------------------------------------------
# continuous target x continuous variable
# ---------------------------------------------------------------------------

def _scatter_cont_cont(pldf, col, target, save_dir):
    sub = pldf.select([col, target]).drop_nulls()
    if sub.height == 0:
        return None

    x = sub[col].to_numpy()
    y = sub[target].to_numpy()

    fig, ax = plt.subplots(figsize=FIGSIZE)
    sc = ax.scatter(
        x, y, c=y, cmap=CMAP, s=12, alpha=0.7,
        edgecolors="black", linewidths=0.3,
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(target, fontsize=FONT_SIZE - 2)
    cbar.ax.tick_params(labelsize=FONT_SIZE - 2)
    ax.set_xlabel(col)
    ax.set_ylabel(target)
    fig.tight_layout()

    path = os.path.join(save_dir, f"target_{col}_scatter.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    _show_fig(fig)
    return fig


# ---------------------------------------------------------------------------
# continuous target x categorical variable
# ---------------------------------------------------------------------------

def _box_cont_target_cat_col(pldf, col, target, save_dir):
    sub = pldf.select([col, target]).drop_nulls()
    vc = sub[col].value_counts().sort("count", descending=True)
    groups = vc[col].to_list()
    if len(groups) > 10:
        groups = groups[:10]
        sub = sub.filter(pl.col(col).is_in(groups))
    groups = sorted(groups, key=str)

    n = len(groups)
    colors = _get_colors(n)
    data_by_group = [
        sub.filter(pl.col(col) == g)[target].to_numpy() for g in groups
    ]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bp = ax.boxplot(
        data_by_group, patch_artist=True, widths=0.6,
        medianprops=dict(color="black", linewidth=1.2),
        flierprops=dict(marker="o", markersize=3, linestyle="none"),
    )
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(colors[i])
        box.set_edgecolor("black")
        box.set_linewidth(0.8)
        box.set_hatch(HATCHES[i % len(HATCHES)])

    ax.set_xticklabels([_truncate(g) for g in groups], rotation=45, ha="right")
    ax.set_xlabel(col)
    ax.set_ylabel(target)
    fig.tight_layout()

    path = os.path.join(save_dir, f"target_{col}_box.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    _show_fig(fig)
    return fig


# ---------------------------------------------------------------------------
# categorical target x continuous variable
# ---------------------------------------------------------------------------

def _hist_cat_target_cont_col(pldf, col, target, save_dir):
    sub = pldf.select([col, target]).drop_nulls()
    vc = sub[target].value_counts().sort("count", descending=True)
    groups = vc[target].to_list()[:10]
    groups = sorted(groups, key=str)
    n_groups = len(groups)
    if n_groups < 2:
        return None
    sub = sub.filter(pl.col(target).is_in(groups))

    colors = _get_colors(n_groups)
    all_vals = sub[col].to_numpy()
    n_bins = min(30, max(10, int(np.sqrt(len(all_vals)))))
    bin_edges = np.linspace(all_vals.min(), all_vals.max(), n_bins + 1)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for i, g in enumerate(groups):
        vals = sub.filter(pl.col(target) == g)[col].to_numpy()
        ax.hist(
            vals, bins=bin_edges, color=colors[i], edgecolor="black",
            linewidth=0.6, alpha=0.55, hatch=HATCHES[i % len(HATCHES)],
            label=_truncate(g),
        )

    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    ax.legend(title=target, fontsize=FONT_SIZE - 4, title_fontsize=FONT_SIZE - 3)
    fig.tight_layout()

    path = os.path.join(save_dir, f"target_{col}_hist.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    _show_fig(fig)
    return fig


# ---------------------------------------------------------------------------
# categorical target x categorical variable
# ---------------------------------------------------------------------------

def _bar_cat_target_cat_col(pldf, col, target, save_dir):
    sub = pldf.select([col, target]).drop_nulls()
    vc = sub[target].value_counts().sort("count", descending=True)
    groups = vc[target].to_list()[:10]
    groups = sorted(groups, key=str)
    if len(groups) < 2:
        return None

    ct = (
        sub.group_by([col, target])
        .len()
        .pivot(on=target, index=col, values="len")
        .fill_null(0)
    )

    ct_pd = ct.to_pandas().set_index(col)
    ct_pd.columns = [str(c) for c in ct_pd.columns]

    if len(ct_pd) > 10:
        top = ct_pd.sum(axis=1).nlargest(10).index
        ct_pd = ct_pd.loc[top]

    ct_norm = ct_pd.div(ct_pd.sum(axis=1), axis=0)
    categories = ct_norm.index.tolist()
    groups = [g for g in [str(g) for g in groups] if g in ct_norm.columns]
    n_groups = len(groups)
    if n_groups < 2:
        return None

    n_cats = len(categories)
    x = np.arange(n_cats)
    width = 0.8 / n_groups
    colors = _get_colors(n_groups)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for i, g in enumerate(groups):
        offset = (i - n_groups / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, ct_norm[g].values, width=width,
            color=colors[i], edgecolor="black", linewidth=0.8,
            label=_truncate(g),
        )
        for bar in bars:
            bar.set_hatch(HATCHES[i % len(HATCHES)])

    ax.set_xticks(x)
    ax.set_xticklabels([_truncate(c) for c in categories], rotation=45, ha="right")
    ax.set_ylabel("Proportion")
    ax.set_xlabel(col)
    ax.legend(title=target, fontsize=FONT_SIZE - 4, title_fontsize=FONT_SIZE - 3)
    fig.tight_layout()

    path = os.path.join(save_dir, f"target_{col}_bar.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    _show_fig(fig)
    return fig


# ---------------------------------------------------------------------------
# Target summary table
# ---------------------------------------------------------------------------

def _table_target_summary(pldf, target, target_continuous):
    """Descriptive statistics grouped by target (computed in polars)."""
    if target_continuous:
        num_cols = [c for c in _num_columns(pldf) if c != target]
        if not num_cols:
            s = pldf[target].drop_nulls()
            return pd.DataFrame(
                {"count": [s.len()], "mean": [s.mean()], "std": [s.std()],
                 "min": [s.min()], "max": [s.max()]},
                index=[target],
            )

        corrs = {}
        for c in num_cols:
            r = pldf.select(pl.corr(c, target)).item()
            corrs[c] = 0.0 if r is None else float(r)
        result = pd.DataFrame({"Correlation": corrs})
        return result.sort_values("Correlation", key=abs, ascending=False)

    num_cols = [c for c in pldf.columns
                if c != target and _is_continuous(pldf[c])]

    if not num_cols:
        vc = pldf[target].value_counts().sort("count", descending=True)
        pdf = vc.to_pandas()
        pdf["Percent (%)"] = pdf["count"] / pldf.height * 100
        return pdf.rename(columns={"count": "Count"}).set_index(target)

    parts = []
    for c in num_cols:
        stats = (
            pldf.group_by(target)
            .agg([
                pl.col(c).mean().alias("Mean"),
                pl.col(c).std().alias("Std"),
                pl.col(c).median().alias("Median"),
                pl.col(c).min().cast(pl.Float64).alias("Min"),
                pl.col(c).max().cast(pl.Float64).alias("Max"),
            ])
            .with_columns(pl.lit(c).alias("Feature"))
        )
        parts.append(stats)

    result = pl.concat(parts).sort(target)
    pdf = result.to_pandas()
    pdf[target] = pdf[target].astype(str)
    return pdf.set_index([target, "Feature"])
