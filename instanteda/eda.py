import os

import numpy as np
import pandas as pd
import polars as pl
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

FIGSIZE = (3.54, 3.54)
FONT_FAMILY = "DejaVu Sans"
FONT_SIZE = 14
CMAP = "cividis"
DPI = 300
HATCHES = ["///", "\\\\\\", "xxx", "+++", "...", "ooo", "**", "---", "|||", "OOO"]

_CATEGORICAL_RATIO = 0.05
_CAT_DTYPES = {pl.String, pl.Utf8, pl.Categorical}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _in_notebook():
    """Return True when running inside IPython / Jupyter."""
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


def _show_fig(fig):
    """Display *fig* inline (if in a notebook) then free it."""
    if _in_notebook():
        from IPython.display import display
        display(fig)
    plt.close(fig)


def _show_df(df, title=None):
    """Display a DataFrame inline (if in a notebook)."""
    if not _in_notebook():
        return
    from IPython.display import display, Markdown
    if title:
        display(Markdown(f"**{title}**"))
    display(df)


def _to_polars(df):
    """Accept pandas or polars input; return a polars DataFrame."""
    if isinstance(df, pl.DataFrame):
        return df
    if isinstance(df, pl.LazyFrame):
        return df.collect()
    if isinstance(df, pd.DataFrame):
        return pl.from_pandas(df)
    raise TypeError(f"Expected pandas or polars DataFrame, got {type(df)}")


def _setup_style():
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE - 2,
            "ytick.labelsize": FONT_SIZE - 2,
            "legend.fontsize": FONT_SIZE - 2,
            "figure.figsize": FIGSIZE,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "axes.linewidth": 1.0,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )


def _get_colors(n):
    cmap = matplotlib.colormaps[CMAP]
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def _cat_columns(pldf):
    return [c for c in pldf.columns if pldf.schema[c] in _CAT_DTYPES]


def _num_columns(pldf):
    return [c for c in pldf.columns if pldf.schema[c].is_numeric()]


def _is_continuous(series):
    """Decide whether a polars Series should be treated as continuous.

    Returns False (i.e. categorical) when the series is non-numeric, binary,
    or integer-valued with nunique/nrows < 5 %.
    """
    if not series.dtype.is_numeric():
        return False
    non_null = series.drop_nulls()
    if len(non_null) == 0:
        return False
    n_unique = non_null.n_unique()
    if n_unique <= 2:
        return False
    ratio = n_unique / len(non_null)
    if ratio < _CATEGORICAL_RATIO:
        if series.dtype.is_integer():
            return False
        try:
            rounded = non_null.cast(pl.Float64).round(0)
            if (non_null.cast(pl.Float64) == rounded).all():
                return False
        except Exception:
            return False
    return True


# ---------------------------------------------------------------------------
# Table rendering (pandas/matplotlib — operates on small result sets)
# ---------------------------------------------------------------------------

def _save_table_as_png(df, path):
    """Render a pandas DataFrame as a publication-quality three-line table PNG."""
    df_display = df.copy()
    for col in df_display.columns:
        if pd.api.types.is_float_dtype(df_display[col]):
            df_display[col] = df_display[col].map(
                lambda x: f"{x:.2f}" if pd.notnull(x) else ""
            )

    nrows, ncols = df_display.shape
    col_width = 1.3
    row_height = 0.45
    idx_width = max(len(str(i)) for i in df_display.index) * 0.12 + 0.6
    fig_width = max(idx_width + col_width * ncols, FIGSIZE[0])
    fig_height = max(row_height * (nrows + 1.5), 1.5)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=df_display.astype(str).values,
        colLabels=[str(c) for c in df_display.columns],
        rowLabels=[str(i) for i in df_display.index],
        cellLoc="center",
        rowLoc="right",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(FONT_SIZE - 2)
    table.scale(1, 1.6)

    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0)
        cell.set_edgecolor("black")
        cell.set_facecolor("none")

        if row == 0:
            cell.set_text_props(fontweight="bold", fontfamily=FONT_FAMILY)
            cell.visible_edges = "BT"
            cell.set_linewidth(1.2)
        elif row == nrows:
            cell.visible_edges = "B"
            cell.set_linewidth(1.2)
        else:
            cell.visible_edges = ""

        if col == -1:
            cell.set_text_props(fontstyle="italic", fontfamily=FONT_FAMILY)

    fig.savefig(
        path, dpi=DPI, bbox_inches="tight", pad_inches=0.05,
        facecolor="white", edgecolor="none",
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Table helpers (polars computation → small pandas result)
# ---------------------------------------------------------------------------

def _table_missing(pldf):
    n = pldf.height
    nulls = (
        pldf.null_count()
        .unpivot(variable_name="Column", value_name="Count")
        .with_columns((pl.col("Count") / n * 100).alias("Percent (%)"))
    )
    has_missing = nulls.filter(pl.col("Count") > 0)
    result = has_missing if has_missing.height > 0 else nulls
    return result.to_pandas().set_index("Column")


def _table_duplicates(pldf):
    n_total = pldf.height
    n_unique = pldf.unique().height
    n_dup = n_total - n_unique
    pct = n_dup / n_total * 100 if n_total > 0 else 0.0
    return pd.DataFrame(
        {"Value": [n_dup, n_total, f"{pct:.1f}"]},
        index=["Duplicate rows", "Total rows", "Percentage (%)"],
    )


def _table_types(pldf):
    cols = pldf.columns
    return pd.DataFrame(
        {
            "Type": [str(pldf.schema[c]) for c in cols],
            "Non-null": [pldf.height - pldf[c].null_count() for c in cols],
            "Unique": [pldf[c].n_unique() for c in cols],
        },
        index=cols,
    )


def _table_describe(pldf):
    num_cols = _num_columns(pldf)
    if not num_cols:
        return pd.DataFrame()

    stat_names = [
        "count", "mean", "std", "min", "25 pctl", "50 pctl", "75 pctl", "max",
    ]
    exprs = []
    for c in num_cols:
        exprs.extend([
            pl.col(c).count().cast(pl.Float64).alias(f"{c}__count"),
            pl.col(c).mean().alias(f"{c}__mean"),
            pl.col(c).std().alias(f"{c}__std"),
            pl.col(c).min().cast(pl.Float64).alias(f"{c}__min"),
            pl.col(c).quantile(0.25).alias(f"{c}__25 pctl"),
            pl.col(c).quantile(0.50).alias(f"{c}__50 pctl"),
            pl.col(c).quantile(0.75).alias(f"{c}__75 pctl"),
            pl.col(c).max().cast(pl.Float64).alias(f"{c}__max"),
        ])

    row = pldf.select(exprs)
    data = {c: [row[f"{c}__{s}"][0] for s in stat_names] for c in num_cols}
    return pd.DataFrame(data, index=stat_names).T


# ---------------------------------------------------------------------------
# Plot helpers (polars for data extraction → matplotlib for rendering)
# ---------------------------------------------------------------------------

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
        ax.set_xticklabels([str(v) for v in labels], rotation=45, ha="right")
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


# ---------------------------------------------------------------------------
# Target-aware helpers
# ---------------------------------------------------------------------------

def _plot_col_by_target(pldf, col, target, target_continuous, save_dir):
    """Dispatch to the right plot for one feature column vs. the target.

    continuous  target × continuous  col  →  scatter
    continuous  target × categorical col  →  box plot
    categorical target × continuous  col  →  overlaid histograms
    categorical target × categorical col  →  grouped bar chart
    """
    col_continuous = _is_continuous(pldf[col])

    if target_continuous and col_continuous:
        return _scatter_cont_cont(pldf, col, target, save_dir)
    if target_continuous and not col_continuous:
        return _box_cont_target_cat_col(pldf, col, target, save_dir)
    if not target_continuous and col_continuous:
        return _hist_cat_target_cont_col(pldf, col, target, save_dir)
    return _bar_cat_target_cat_col(pldf, col, target, save_dir)


# -- continuous target × continuous variable --------------------------------

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


# -- continuous target × categorical variable ------------------------------

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

    ax.set_xticklabels([str(g) for g in groups], rotation=45, ha="right")
    ax.set_xlabel(col)
    ax.set_ylabel(target)
    fig.tight_layout()

    path = os.path.join(save_dir, f"target_{col}_box.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    _show_fig(fig)
    return fig


# -- categorical target × continuous variable ------------------------------

def _hist_cat_target_cont_col(pldf, col, target, save_dir):
    sub = pldf.select([col, target]).drop_nulls()
    groups = sorted(sub[target].unique().to_list(), key=str)
    n_groups = len(groups)
    if n_groups < 2:
        return None

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
            label=str(g),
        )

    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    ax.legend(title=target, fontsize=FONT_SIZE - 4, title_fontsize=FONT_SIZE - 3)
    fig.tight_layout()

    path = os.path.join(save_dir, f"target_{col}_hist.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    _show_fig(fig)
    return fig


# -- categorical target × categorical variable -----------------------------

def _bar_cat_target_cat_col(pldf, col, target, save_dir):
    sub = pldf.select([col, target]).drop_nulls()
    groups = sorted(sub[target].unique().to_list(), key=str)
    n_groups = len(groups)
    if n_groups < 2:
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
            label=g,
        )
        for bar in bars:
            bar.set_hatch(HATCHES[i % len(HATCHES)])

    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in categories], rotation=45, ha="right")
    ax.set_ylabel("Proportion")
    ax.set_xlabel(col)
    ax.legend(title=target, fontsize=FONT_SIZE - 4, title_fontsize=FONT_SIZE - 3)
    fig.tight_layout()

    path = os.path.join(save_dir, f"target_{col}_bar.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    _show_fig(fig)
    return fig


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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def eda(df, target=None, save_dir="eda"):
    """Run exploratory data analysis and save publication-ready PNGs.

    Accepts pandas or polars DataFrames (or LazyFrames).  Heavy computation
    (null counts, duplicates, descriptive statistics, correlations, group-by
    aggregations) runs in polars for speed on large files.  Small result
    tables are converted to pandas for rendering and display.

    Parameters
    ----------
    df : pandas.DataFrame | polars.DataFrame | polars.LazyFrame
        Input data.
    target : str, optional
        Target column name.  When provided, each feature is plotted against
        the target using the appropriate chart type (scatter, box plot,
        overlaid histogram, or grouped bar chart) depending on whether the
        target and feature are continuous or categorical.
    save_dir : str
        Directory where PNGs are written.

    Returns
    -------
    dict
        Mapping of result names to pandas DataFrames (tables) or
        matplotlib Figures (plots).
    """
    pldf = _to_polars(df)
    os.makedirs(save_dir, exist_ok=True)
    _setup_style()

    tables = {
        "missing": _table_missing(pldf),
        "duplicates": _table_duplicates(pldf),
        "types": _table_types(pldf),
        "describe": _table_describe(pldf),
    }
    for name, tbl in tables.items():
        _save_table_as_png(tbl, os.path.join(save_dir, f"{name}.png"))
        _show_df(tbl, title=name.replace("_", " ").title())

    cat_figs = _plot_categorical(pldf, save_dir)
    num_figs = _plot_numerical(pldf, save_dir)
    corr_fig = _plot_correlations(pldf, save_dir)

    results = dict(tables)
    if cat_figs:
        results["categorical_distributions"] = cat_figs
    if num_figs:
        results["numerical_distributions"] = num_figs
    if corr_fig is not None:
        results["correlations"] = corr_fig

    if target is not None and target in pldf.columns:
        target_continuous = _is_continuous(pldf[target])

        target_tbl = _table_target_summary(pldf, target, target_continuous)
        _save_table_as_png(
            target_tbl, os.path.join(save_dir, "target_summary.png"),
        )
        _show_df(target_tbl, title="Target Summary")
        results["target_summary"] = target_tbl

        by_target_figs = {}
        feature_cols = [c for c in pldf.columns if c != target]
        for col in feature_cols:
            fig = _plot_col_by_target(
                pldf, col, target, target_continuous, save_dir,
            )
            if fig is not None:
                by_target_figs[col] = fig
        if by_target_figs:
            results["by_target"] = by_target_figs

    return results
