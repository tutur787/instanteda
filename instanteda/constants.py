import polars as pl

FIGSIZE = (3.54, 3.54)
FONT_FAMILY = "DejaVu Sans"
FONT_SIZE = 12
CMAP = "cividis"
DPI = 600
HATCHES = ["///", "\\\\\\", "xxx", "+++", "...", "ooo", "***", "---", "|||", "OOO"]

_CATEGORICAL_RATIO = 0.05
_CAT_DTYPES = {pl.String, pl.Utf8, pl.Categorical}
_MAX_LABEL_LEN = 15
