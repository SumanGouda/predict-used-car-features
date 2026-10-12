# test_data_cleaning.py
import json

import numpy as np
import pandas as pd
import pytest
from utils.preprocessing import handle_missing_values


def make_df(n, n_missing, kind="num"):
    if kind == "num":
        vals = [float(i) for i in range(n - n_missing)] + [np.nan] * n_missing
    else:
        vals = ["a"] * (n - n_missing - 1) + ["b"] + [None] * n_missing
    return pd.DataFrame({"x": vals})


# ---------- early exits ----------

def test_column_not_found():
    df = make_df(10, 0)
    out, meta = handle_missing_values(df, "nope")
    assert meta["action"] == "column_not_found"
    assert out is df


def test_empty_dataframe():
    out, meta = handle_missing_values(pd.DataFrame({"x": []}), "x")
    assert meta["action"] == "empty_dataframe"
    assert meta["missing_count"] == 0
    assert meta["missing_percentage"] is None


def test_no_missing_values():
    df = make_df(10, 0)
    out, meta = handle_missing_values(df, "x")
    assert meta["action"] == "no_missing_values"
    assert meta["missing_percentage"] == 0.0
    assert out["x"].equals(df["x"])


# ---------- each strategy ----------

def test_fill_numeric_uses_median():
    df = make_df(100, 5)
    expected = df["x"].median()
    out, meta = handle_missing_values(df, "x")
    assert meta["action"] == "fill"
    assert meta["fill_strategy"] == "median"
    assert meta["fill_value"] == expected
    assert out["x"].isnull().sum() == 0
    assert "x_was_missing" not in out.columns


def test_fill_categorical_uses_mode():
    out, meta = handle_missing_values(make_df(100, 5, "cat"), "x")
    assert meta["action"] == "fill"
    assert meta["fill_strategy"] == "mode"
    assert meta["fill_value"] == "a"
    assert out["x"].isnull().sum() == 0


def test_fill_and_indicator_numeric_uses_mean():
    df = make_df(100, 20)
    expected = df["x"].mean()
    out, meta = handle_missing_values(df, "x")
    assert meta["action"] == "fill_and_indicator"
    assert meta["fill_strategy"] == "mean"
    assert meta["fill_value"] == pytest.approx(expected)
    assert meta["indicator_column"] == "x_was_missing"
    assert out["x_was_missing"].sum() == 20
    assert out["x"].isnull().sum() == 0


def test_fill_and_indicator_categorical_uses_mode():
    out, meta = handle_missing_values(make_df(100, 20, "cat"), "x")
    assert meta["fill_strategy"] == "mode"
    assert meta["fill_value"] == "a"
    assert out["x_was_missing"].sum() == 20


def test_drop_rows_resets_index():
    out, meta = handle_missing_values(make_df(100, 40), "x")
    assert meta["action"] == "drop_rows"
    assert meta["rows_dropped"] == 40
    assert len(out) == 60
    assert list(out.index) == list(range(60))


def test_drop_column():
    df = make_df(100, 60)
    df["other"] = 1
    out, meta = handle_missing_values(df, "x")
    assert meta["action"] == "drop_column"
    assert "x" not in out.columns
    assert "other" in out.columns


# ---------- boundaries (thresholds use strict >) ----------

@pytest.mark.parametrize(
    "n_missing, action",
    [
        (10, "fill"),                # exactly 10% -> not > 10
        (11, "fill_and_indicator"),
        (30, "fill_and_indicator"),  # exactly 30% -> not > 30
        (31, "drop_rows"),
        (50, "drop_rows"),           # exactly 50% -> not > 50
        (51, "drop_column"),
    ],
)
def test_threshold_boundaries(n_missing, action):
    _, meta = handle_missing_values(make_df(100, n_missing), "x")
    assert meta["action"] == action


# ---------- metadata ----------

def test_metadata_is_json_serializable():
    for n_missing in (0, 5, 20, 40, 60):
        _, meta = handle_missing_values(make_df(100, n_missing), "x")
        json.dumps(meta)  # fails if numpy types leak through


def test_missing_stats_recorded():
    _, meta = handle_missing_values(make_df(200, 5), "x")
    assert meta["missing_count"] == 5
    assert meta["missing_percentage"] == 2.5
