import os
import warnings

from instanteda.utils import _is_continuous, _setup_style, _show_df, _to_polars
from instanteda.tables import (
    _save_table_as_png,
    _table_describe,
    _table_duplicates,
    _table_missing,
    _table_types,
)
from instanteda.plots import (
    _plot_categorical,
    _plot_correlations,
    _plot_datetime,
    _plot_numerical,
)
from instanteda.target import _plot_col_by_target, _table_target_summary


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
    _TABLE_TITLES = {
        "missing": "Missing Values",
        "duplicates": "Duplicate Rows",
        "types": "Column Types",
        "describe": "Descriptive Statistics",
    }
    for name, tbl in tables.items():
        title = _TABLE_TITLES.get(name, name.replace("_", " ").title())
        _save_table_as_png(tbl, os.path.join(save_dir, f"{name}.png"), title=title)
        _show_df(tbl, title=title)

    cat_figs = _plot_categorical(pldf, save_dir)
    num_figs = _plot_numerical(pldf, save_dir)
    dt_figs = _plot_datetime(pldf, save_dir)
    corr_fig = _plot_correlations(pldf, save_dir)

    results = dict(tables)
    if cat_figs:
        results["categorical_distributions"] = cat_figs
    if num_figs:
        results["numerical_distributions"] = num_figs
    if dt_figs:
        results["datetime_distributions"] = dt_figs
    if corr_fig is not None:
        results["correlations"] = corr_fig

    if target is not None and target in pldf.columns:
        target_continuous = _is_continuous(pldf[target])

        if not target_continuous:
            n_unique = pldf[target].n_unique()
            if n_unique > 10:
                warnings.warn(
                    f"Target '{target}' has {n_unique} unique values — "
                    "only the top 10 most frequent are shown in per-feature plots.",
                    UserWarning,
                    stacklevel=2,
                )

        target_tbl = _table_target_summary(pldf, target, target_continuous)
        _save_table_as_png(
            target_tbl, os.path.join(save_dir, "target_summary.png"),
            title="Target Summary",
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
