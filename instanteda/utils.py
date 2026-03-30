import warnings

import pandas as pd
import polars as pl
import matplotlib
import matplotlib.pyplot as plt

from instanteda.constants import (
    CMAP,
    DPI,
    FIGSIZE,
    FONT_FAMILY,
    FONT_SIZE,
    _CAT_DTYPES,
    _CATEGORICAL_RATIO,
    _MAX_LABEL_LEN,
)


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
    """Accept pandas or polars input; return a polars DataFrame.

    Object-dtype columns are inspected before conversion: if >= 80 % of
    non-null values parse as dates they become Datetime; otherwise they
    are cast to str (preserving nulls).  This runs before ``pl.from_pandas``
    so that date strings are never misclassified as categorical.
    """
    if isinstance(df, pl.DataFrame):
        return df
    if isinstance(df, pl.LazyFrame):
        return df.collect()
    if isinstance(df, pd.DataFrame):
        prepared = df.copy()
        for col in prepared.columns:
            if prepared[col].dtype != object:
                continue
            non_null_count = prepared[col].notna().sum()
            if non_null_count == 0:
                prepared[col] = prepared[col].astype(str)
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    parsed = pd.to_datetime(prepared[col], errors="coerce")
                if parsed.notna().sum() >= non_null_count * 0.8:
                    prepared[col] = parsed
                    continue
            except Exception:
                pass
            nulls = prepared[col].isna()
            prepared[col] = prepared[col].astype(str)
            prepared.loc[nulls, col] = None
        return pl.from_pandas(prepared)
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


def _truncate(label, maxlen=_MAX_LABEL_LEN):
    s = str(label)
    return s if len(s) <= maxlen else s[: maxlen - 1] + "\u2026"


def _is_datetime_dtype(dtype):
    """True for Date and Datetime variants, not Time or Duration."""
    if not dtype.is_temporal():
        return False
    s = str(dtype)
    return s.startswith("Date")


def _cat_columns(pldf):
    return [
        c for c in pldf.columns
        if pldf.schema[c] in _CAT_DTYPES and not _is_datetime_dtype(pldf.schema[c])
    ]


def _num_columns(pldf):
    return [c for c in pldf.columns if pldf.schema[c].is_numeric()]


def _datetime_columns(pldf):
    return [c for c in pldf.columns if _is_datetime_dtype(pldf.schema[c])]


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
