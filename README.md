# instanteda

**Instantly make paper-ready EDA plots and tables.**

One function call. Any DataFrame. Publication-quality PNGs -- sized, typeset, and styled for direct inclusion in scientific manuscripts.

```python
from instanteda import eda

results = eda(df, target="label", save_dir="figures")
```

## Why instanteda

- You have a dataset and need to explore it before modeling
- You want figures you can drop straight into a LaTeX document or journal submission
- You don't want to spend an hour configuring matplotlib for every project

instanteda does the entire exploratory data analysis in a single call and writes every output to disk as a 600 DPI PNG.

## Features

**Publication-ready formatting**
- 3.54 x 3.54 in (90 mm) single-column figures, 600 DPI
- DejaVu Sans, 12 pt
- Cividis colormap with hatch textures for grayscale and colorblind accessibility
- Three-line tables with titled headers, scientific notation for large numbers

**Smart type detection**
- Numeric columns with low unique-to-row ratio and integer values are automatically treated as categorical (catches 0/1 flags, ordinal labels, encoded classes)
- Date strings in mixed formats are auto-parsed to datetime
- Datetime columns get line plots, not bar charts

**Target-aware analysis** -- when a target column is provided, every feature is plotted against it using the right chart:

| Target | Feature | Plot |
|---|---|---|
| Continuous | Continuous | Scatter |
| Continuous | Categorical | Box plot |
| Categorical | Continuous | Overlaid histograms |
| Categorical | Categorical | Grouped bar chart |
| Any | Datetime | Line over time |
| Datetime | Continuous | Line over time |

**Fast on large files** -- heavy computation (null counts, duplicates, descriptive stats, correlations, group-by aggregations) runs in Polars. Small result tables are converted to pandas for rendering.

**Jupyter support** -- tables and plots display inline in notebooks. PNGs are always saved to disk regardless.

## Install

```bash
pip install instanteda
```

From source:

```bash
git clone https://github.com/tutur787/instanteda.git
cd instanteda
pip install .
```

For development:

```bash
pip install -e ".[dev]"
```

## Quickstart

```python
from instanteda import eda
import pandas as pd

df = pd.read_csv("data.csv")

# Basic EDA
results = eda(df)

# With a target variable
results = eda(df, target="species")

# Custom output directory
results = eda(df, target="price", save_dir="figures")
```

Polars DataFrames and LazyFrames work too:

```python
import polars as pl

df = pl.read_csv("data.csv")
results = eda(df, target="label")
```

## Output

All PNGs are written to `save_dir` (default `eda/`):

```
eda/
  # Tables
  missing.png              Null counts per column
  duplicates.png           Duplicate row summary
  types.png                Column types, non-null counts, unique counts
  describe.png             Descriptive statistics (count, mean, std, quartiles)

  # Distributions
  cat_{column}.png         Bar chart per categorical column
  num_{column}.png         Histogram per numerical column
  dt_{column}.png          Record-count line plot per datetime column
  correlations.png         Correlation heatmap

  # Target analysis (when target is provided)
  target_summary.png                   Grouped stats or correlations
  target_{col}_scatter.png             Continuous x continuous
  target_{col}_box.png                 Continuous target x categorical feature
  target_{col}_hist.png                Categorical target x continuous feature
  target_{col}_bar.png                 Categorical x categorical
  target_{col}_by_{dt_col}_line.png    Value or counts over time
```

## Project structure

```
instanteda/
├── __init__.py      Public API — re-exports eda()
├── constants.py     Shared constants (figure size, DPI, colormap, hatches)
├── utils.py         DataFrame conversion, style setup, column classification
├── tables.py        Table computation and PNG rendering
├── plots.py         Univariate distribution plots (categorical, numerical, datetime, correlations)
├── target.py        Target-aware plots and summary tables
└── eda.py           Main entry point — orchestrates tables, plots, and target analysis
```

## API

```python
eda(df, target=None, save_dir="eda")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `df` | `pandas.DataFrame`, `polars.DataFrame`, `polars.LazyFrame` | -- | Input data |
| `target` | `str` or `None` | `None` | Target column for grouped analysis |
| `save_dir` | `str` | `"eda"` | Output directory (created if needed) |

Returns a `dict` mapping result names to `pandas.DataFrame` (tables) or `matplotlib.Figure` (plots).

## Requirements

- Python >= 3.9
- numpy >= 1.24
- pandas >= 2.0
- polars >= 1.0
- matplotlib >= 3.7

## License

MIT
