import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

from instanteda.constants import DPI, FIGSIZE, FONT_FAMILY, FONT_SIZE
from instanteda.utils import _num_columns


def _save_table_as_png(df, path, title=None):
    """Render a pandas DataFrame as a publication-quality three-line table PNG."""
    if df.empty:
        return
    df_display = df.copy()
    for col in df_display.columns:
        if pd.api.types.is_float_dtype(df_display[col]):
            df_display[col] = df_display[col].map(
                lambda x: (
                    ""
                    if pd.isnull(x)
                    else f"{x:.2e}" if abs(x) >= 1e6 or (0 < abs(x) < 1e-3)
                    else f"{x:.2f}"
                )
            )

    nrows, ncols = df_display.shape
    col_width = 1.3
    row_height = 0.45
    idx_width = max(len(str(i)) for i in df_display.index) * 0.12 + 0.6
    fig_width = max(idx_width + col_width * ncols, FIGSIZE[0])
    fig_height = max(row_height * (nrows + 1.5), 1.5)
    if title:
        fig_height += 0.45

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    if title:
        ax.set_title(
            title, fontsize=FONT_SIZE, fontweight="bold",
            fontfamily=FONT_FAMILY, pad=12,
        )

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


def _friendly_dtype(dtype):
    """Return a short, readable dtype name for publication tables."""
    s = str(dtype)
    if s.startswith("Datetime"):
        return "datetime"
    if s.startswith("Duration"):
        return "duration"
    return s.lower()


def _table_types(pldf):
    cols = pldf.columns
    return pd.DataFrame(
        {
            "Type": [_friendly_dtype(pldf.schema[c]) for c in cols],
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
            pl.col(c).count().cast(pl.Float64).alias(f"{c}__count").round(2),
            pl.col(c).mean().alias(f"{c}__mean").round(2),
            pl.col(c).std().alias(f"{c}__std").round(2),
            pl.col(c).min().cast(pl.Float64).alias(f"{c}__min").round(2),
            pl.col(c).quantile(0.25).alias(f"{c}__25 pctl").round(2),
            pl.col(c).quantile(0.50).alias(f"{c}__50 pctl").round(2),
            pl.col(c).quantile(0.75).alias(f"{c}__75 pctl").round(2),
            pl.col(c).max().cast(pl.Float64).alias(f"{c}__max").round(2),
        ])

    row = pldf.select(exprs)
    data = {c: [row[f"{c}__{s}"][0] for s in stat_names] for c in num_cols}
    return pd.DataFrame(data, index=stat_names).T
