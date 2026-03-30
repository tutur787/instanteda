"""Stress tests for instanteda — edge cases that should not crash."""

import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import polars as pl
import pytest

from instanteda.eda import eda
from instanteda.utils import (
    _to_polars,
    _cat_columns,
    _num_columns,
    _datetime_columns,
    _is_continuous,
)
from instanteda.tables import (
    _table_missing,
    _table_duplicates,
    _table_types,
    _table_describe,
)
from instanteda.target import _table_target_summary
from instanteda.plots import _plot_correlations


@pytest.fixture()
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


# ── 1. Completely empty DataFrame ──────────────────────────────────────────


class TestEmptyDataFrame:
    """Zero rows *and* zero columns."""

    def test_pandas_empty(self, tmp_dir):
        df = pd.DataFrame()
        result = eda(df, save_dir=tmp_dir)
        assert isinstance(result, dict)
        assert "missing" in result
        assert "duplicates" in result

    def test_polars_empty(self, tmp_dir):
        df = pl.DataFrame()
        result = eda(df, save_dir=tmp_dir)
        assert isinstance(result, dict)

    def test_zero_rows_with_columns(self, tmp_dir):
        df = pd.DataFrame({"a": pd.Series(dtype="float64"),
                           "b": pd.Series(dtype="str")})
        result = eda(df, save_dir=tmp_dir)
        assert isinstance(result, dict)

    def test_zero_rows_with_target(self, tmp_dir):
        df = pd.DataFrame({"x": pd.Series(dtype="float64"),
                           "y": pd.Series(dtype="float64")})
        result = eda(df, target="y", save_dir=tmp_dir)
        assert isinstance(result, dict)


# ── 2. Only categorical columns ───────────────────────────────────────────


class TestOnlyCategorical:
    """No numerical columns at all."""

    def test_all_string_columns(self, tmp_dir):
        df = pd.DataFrame({
            "color": ["red", "blue", "green", "red", "blue"],
            "size": ["S", "M", "L", "M", "S"],
            "shape": ["circle", "square", "circle", "triangle", "square"],
        })
        result = eda(df, save_dir=tmp_dir)
        assert result["describe"].empty
        assert "correlations" not in result
        assert "categorical_distributions" in result

    def test_cat_only_with_target(self, tmp_dir):
        df = pd.DataFrame({
            "animal": ["cat", "dog", "cat", "dog", "bird"],
            "color": ["black", "white", "white", "black", "green"],
        })
        result = eda(df, target="animal", save_dir=tmp_dir)
        assert "target_summary" in result

    def test_polars_categorical_dtype(self, tmp_dir):
        df = pl.DataFrame({
            "a": pl.Series(["x", "y", "z", "x"], dtype=pl.Categorical),
            "b": pl.Series(["p", "q", "p", "q"], dtype=pl.Categorical),
        })
        result = eda(df, save_dir=tmp_dir)
        assert isinstance(result, dict)


# ── 3. Target column doesn't exist ────────────────────────────────────────


class TestMissingTarget:
    """Target name not among the DataFrame columns."""

    def test_nonexistent_target_ignored(self, tmp_dir):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = eda(df, target="DOES_NOT_EXIST", save_dir=tmp_dir)
        assert "target_summary" not in result

    def test_empty_string_target(self, tmp_dir):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = eda(df, target="", save_dir=tmp_dir)
        assert "target_summary" not in result

    def test_none_target(self, tmp_dir):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = eda(df, target=None, save_dir=tmp_dir)
        assert "target_summary" not in result


# ── 4. Single-row DataFrame ───────────────────────────────────────────────


class TestSingleRow:
    """Only one observation — many statistics become degenerate."""

    def test_one_row_numerical(self, tmp_dir):
        df = pd.DataFrame({"x": [42.0], "y": [7.0]})
        result = eda(df, save_dir=tmp_dir)
        assert isinstance(result, dict)

    def test_one_row_with_target(self, tmp_dir):
        df = pd.DataFrame({"feat": [1.0], "label": ["A"]})
        result = eda(df, target="label", save_dir=tmp_dir)
        assert isinstance(result, dict)

    def test_one_row_categorical(self, tmp_dir):
        df = pd.DataFrame({"a": ["hello"], "b": ["world"]})
        result = eda(df, save_dir=tmp_dir)
        assert isinstance(result, dict)

    def test_one_row_mixed(self, tmp_dir):
        df = pd.DataFrame({"num": [3.14], "cat": ["pi"], "dt": pd.to_datetime(["2024-01-01"])})
        result = eda(df, save_dir=tmp_dir)
        assert isinstance(result, dict)


# ── 5. All-null DataFrame ─────────────────────────────────────────────────


class TestAllNull:
    """Every value is NaN/None."""

    def test_all_nan_numeric(self, tmp_dir):
        df = pd.DataFrame({"a": [np.nan, np.nan], "b": [np.nan, np.nan]})
        result = eda(df, save_dir=tmp_dir)
        assert isinstance(result, dict)

    def test_all_none_string(self, tmp_dir):
        df = pd.DataFrame({"x": [None, None], "y": [None, None]})
        result = eda(df, save_dir=tmp_dir)
        assert isinstance(result, dict)

    def test_all_null_with_target(self, tmp_dir):
        df = pd.DataFrame({"feat": [np.nan, np.nan], "target": [np.nan, np.nan]})
        result = eda(df, target="target", save_dir=tmp_dir)
        assert isinstance(result, dict)


# ── 6. Single-column DataFrame ────────────────────────────────────────────


class TestSingleColumn:
    """Only one column — correlations and target-vs-feature are impossible."""

    def test_one_numeric_column(self, tmp_dir):
        df = pd.DataFrame({"val": range(100)})
        result = eda(df, save_dir=tmp_dir)
        assert "correlations" not in result
        assert "numerical_distributions" in result

    def test_one_cat_column(self, tmp_dir):
        df = pd.DataFrame({"label": ["a", "b", "a", "c", "b"]})
        result = eda(df, save_dir=tmp_dir)
        assert "categorical_distributions" in result

    def test_single_col_as_target(self, tmp_dir):
        df = pd.DataFrame({"only": [1, 2, 3, 4, 5]})
        result = eda(df, target="only", save_dir=tmp_dir)
        assert isinstance(result, dict)


# ── 7. Large cardinality categorical ──────────────────────────────────────


class TestHighCardinality:
    """Categorical column with many unique values — plots should cap at 10."""

    def test_many_categories(self, tmp_dir):
        df = pd.DataFrame({"id": [f"user_{i}" for i in range(500)]})
        result = eda(df, save_dir=tmp_dir)
        assert isinstance(result, dict)

    def test_high_cardinality_as_target(self, tmp_dir):
        n = 200
        df = pd.DataFrame({
            "feature": np.random.randn(n),
            "group": [f"g{i % 50}" for i in range(n)],
        })
        result = eda(df, target="group", save_dir=tmp_dir)
        assert isinstance(result, dict)


# ── 8. Constant columns ──────────────────────────────────────────────────


class TestConstantColumns:
    """Every row has the same value — zero variance."""

    def test_constant_numeric(self, tmp_dir):
        df = pd.DataFrame({"x": [5.0] * 20, "y": [5.0] * 20})
        result = eda(df, save_dir=tmp_dir)
        assert isinstance(result, dict)

    def test_constant_target(self, tmp_dir):
        df = pd.DataFrame({"feat": range(10), "target": ["A"] * 10})
        result = eda(df, target="target", save_dir=tmp_dir)
        assert isinstance(result, dict)

    def test_constant_numeric_target(self, tmp_dir):
        df = pd.DataFrame({"feat": range(10), "target": [0.0] * 10})
        result = eda(df, target="target", save_dir=tmp_dir)
        assert isinstance(result, dict)


# ── 9. Duplicate rows ────────────────────────────────────────────────────


class TestDuplicateRows:
    """100% duplicate rows."""

    def test_all_duplicates(self, tmp_dir):
        row = {"a": 1, "b": "x", "c": 3.14}
        df = pd.DataFrame([row] * 50)
        result = eda(df, save_dir=tmp_dir)
        dup_table = result["duplicates"]
        assert int(dup_table.loc["Duplicate rows", "Value"]) == 49


# ── 10. Mixed types & tricky dtypes ──────────────────────────────────────


class TestMixedAndTrickyTypes:
    def test_boolean_column(self, tmp_dir):
        df = pd.DataFrame({
            "flag": [True, False, True, False, True],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        result = eda(df, save_dir=tmp_dir)
        assert isinstance(result, dict)

    def test_integer_and_float_mix(self, tmp_dir):
        df = pd.DataFrame({
            "int_col": [1, 2, 3, 4, 5],
            "float_col": [1.1, 2.2, 3.3, 4.4, 5.5],
        })
        result = eda(df, save_dir=tmp_dir)
        assert isinstance(result, dict)

    def test_datetime_only(self, tmp_dir):
        df = pd.DataFrame({
            "date1": pd.date_range("2020-01-01", periods=30),
            "date2": pd.date_range("2021-06-01", periods=30),
        })
        result = eda(df, save_dir=tmp_dir)
        assert isinstance(result, dict)
        assert result["describe"].empty

    def test_datetime_with_target(self, tmp_dir):
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=50),
            "value": np.random.randn(50),
        })
        result = eda(df, target="value", save_dir=tmp_dir)
        assert isinstance(result, dict)


# ── 11. Polars LazyFrame input ────────────────────────────────────────────


class TestPolarsLazyFrame:
    def test_lazy_frame(self, tmp_dir):
        lf = pl.LazyFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        result = eda(lf, save_dir=tmp_dir)
        assert isinstance(result, dict)


# ── 12. Invalid input types ──────────────────────────────────────────────


class TestInvalidInput:
    def test_dict_raises(self, tmp_dir):
        with pytest.raises(TypeError, match="Expected pandas or polars"):
            eda({"a": [1, 2]}, save_dir=tmp_dir)

    def test_list_raises(self, tmp_dir):
        with pytest.raises(TypeError, match="Expected pandas or polars"):
            eda([[1, 2], [3, 4]], save_dir=tmp_dir)


# ── 13. Columns with spaces / special characters ─────────────────────────


class TestSpecialColumnNames:
    def test_spaces_in_names(self, tmp_dir):
        df = pd.DataFrame({
            "my feature": [1, 2, 3, 4, 5],
            "target col": [10, 20, 30, 40, 50],
        })
        result = eda(df, target="target col", save_dir=tmp_dir)
        assert isinstance(result, dict)

    def test_unicode_column_names(self, tmp_dir):
        df = pd.DataFrame({"température": [20, 22, 19], "catégorie": ["a", "b", "a"]})
        result = eda(df, save_dir=tmp_dir)
        assert isinstance(result, dict)


# ── 14. Very wide DataFrame (many columns, few rows) ─────────────────────


class TestWideDataFrame:
    def test_100_columns_5_rows(self, tmp_dir):
        data = {f"col_{i}": np.random.randn(5) for i in range(100)}
        df = pd.DataFrame(data)
        result = eda(df, save_dir=tmp_dir)
        assert isinstance(result, dict)
        assert "correlations" in result


# ── 15. Binary integer columns (should be treated as categorical) ────────


class TestBinaryColumns:
    def test_binary_int(self, tmp_dir):
        df = pd.DataFrame({
            "flag": [0, 1, 0, 1, 0, 1, 0, 1],
            "value": np.random.randn(8),
        })
        pldf = _to_polars(df)
        assert not _is_continuous(pldf["flag"])

    def test_binary_target(self, tmp_dir):
        df = pd.DataFrame({
            "feat": np.random.randn(50),
            "label": [0, 1] * 25,
        })
        result = eda(df, target="label", save_dir=tmp_dir)
        assert isinstance(result, dict)


# ── 16. Heavy nulls mixed in ─────────────────────────────────────────────


class TestHeavyNulls:
    def test_mostly_null_numeric(self, tmp_dir):
        vals = [np.nan] * 95 + list(range(5))
        df = pd.DataFrame({"sparse": vals, "dense": range(100)})
        result = eda(df, save_dir=tmp_dir)
        assert isinstance(result, dict)

    def test_nulls_in_target(self, tmp_dir):
        df = pd.DataFrame({
            "feat": range(20),
            "target": [None] * 10 + ["A"] * 5 + ["B"] * 5,
        })
        result = eda(df, target="target", save_dir=tmp_dir)
        assert isinstance(result, dict)


# ── 17. Utility function edge cases ──────────────────────────────────────


class TestUtilityEdges:
    def test_to_polars_rejects_string(self):
        with pytest.raises(TypeError):
            _to_polars("not a dataframe")

    def test_is_continuous_empty_series(self):
        s = pl.Series("x", [], dtype=pl.Float64)
        assert not _is_continuous(s)

    def test_is_continuous_all_null(self):
        s = pl.Series("x", [None, None, None], dtype=pl.Float64)
        assert not _is_continuous(s)

    def test_cat_columns_on_numeric_df(self):
        pldf = pl.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        assert _cat_columns(pldf) == []

    def test_num_columns_on_string_df(self):
        pldf = pl.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
        assert _num_columns(pldf) == []

    def test_datetime_columns_on_non_temporal(self):
        pldf = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        assert _datetime_columns(pldf) == []


# ── 18. Output directory creation ─────────────────────────────────────────


class TestOutputDir:
    def test_nested_dir_created(self, tmp_dir):
        nested = os.path.join(tmp_dir, "a", "b", "c")
        df = pd.DataFrame({"x": [1, 2, 3]})
        eda(df, save_dir=nested)
        assert os.path.isdir(nested)
        assert any(f.endswith(".png") for f in os.listdir(nested))
