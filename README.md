# instanteda

**Instantly make paper-ready EDA plots and tables.**

One function call turns any DataFrame into publication-quality PNG figures and three-line tables -- sized, typeset, and styled for direct inclusion in scientific manuscripts.

## Features

- **Single-column journal figures** -- 3.54 x 3.54 in (90 mm), 300 DPI, DejaVu Sans 14 pt
- **Cividis colormap with hatch textures** -- accessible in grayscale and for colorblind readers
- **Three-line publication tables** -- rendered as PNGs with proper typographic rules
- **Smart target analysis** -- automatically picks the right chart for every feature x target combination:

| Target | Feature | Plot |
|---|---|---|
| Continuous | Continuous | Scatter |
| Continuous | Categorical | Box plot |
| Categorical | Continuous | Overlaid histograms |
| Categorical | Categorical | Grouped bar chart |

- **Automatic type detection** -- numeric columns that look like class labels (integer-valued, low unique-to-row ratio) are treated as categorical
- **Polars engine** -- heavy computation (null counts, duplicates, descriptive stats, correlations, group-by aggregations) runs in Polars for speed on large files
- **Jupyter inline display** -- tables and plots render inline when running in a notebook; PNGs are always saved to disk

## Install

```bash
pip install instanteda
```

Or install from source:

```bash
git clone https://github.com/tutur787/instanteda.git
cd instanteda
pip install .
```

## Quickstart

```python
from instanteda import eda
import pandas as pd

df = pd.read_csv("data.csv")

# Basic EDA -- saves all outputs to ./eda/
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

## What gets generated

```
eda/
  missing.png            # Null counts per column
  duplicates.png         # Duplicate row summary
  types.png              # Column types, non-null counts, unique counts
  describe.png           # Descriptive statistics (count, mean, std, quartiles)
  cat_{column}.png       # Bar chart per categorical column
  num_{column}.png       # Histogram per numerical column
  correlations.png       # Correlation heatmap

  # When target is provided:
  target_summary.png     # Stats grouped by target (or correlations if continuous)
  target_{col}_scatter.png   # Continuous x continuous
  target_{col}_box.png       # Continuous target x categorical feature
  target_{col}_hist.png      # Categorical target x continuous feature
  target_{col}_bar.png       # Categorical x categorical
```

## API

```python
eda(df, target=None, save_dir="eda")
```

| Parameter | Type | Description |
|---|---|---|
| `df` | `pandas.DataFrame`, `polars.DataFrame`, or `polars.LazyFrame` | Input data |
| `target` | `str` or `None` | Target column for grouped analysis |
| `save_dir` | `str` | Directory for output PNGs (created if needed) |

**Returns** a `dict` mapping result names to `pandas.DataFrame` (tables) or `matplotlib.Figure` (plots).

## Requirements

- Python >= 3.9
- numpy >= 1.24
- pandas >= 2.0
- polars >= 1.0
- matplotlib >= 3.7

## License

MIT
