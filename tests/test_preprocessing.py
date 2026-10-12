import json

import numpy as np
import pandas as pd
import pytest
from functools import partial

from config.settings import OWNERSHIP_MAP
from utils.cleaners import clean_car_name, clean_ownership, clean_price
from utils.preprocessing import (
    _apply_frequency_encoding,
    _apply_target_encoding,
    _encode_column_ohe,
    cleaning_pipeline,
    clean_using_functions,
    clean_using_regex,
    handle_missing_values,
)


def make_df(n, n_missing, kind="num"):
    if kind == "num":
        vals = [float(i) for i in range(n - n_missing)] + [np.nan] * n_missing
    else:
        vals = ["a"] * (n - n_missing - 1) + ["b"] + [None] * n_missing
    return pd.DataFrame({"x": vals})


# =====================================================
# handle_missing_values
# =====================================================

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


def test_metadata_is_json_serializable():
    for n_missing in (0, 5, 20, 40, 60):
        _, meta = handle_missing_values(make_df(100, n_missing), "x")
        json.dumps(meta)


def test_missing_stats_recorded():
    _, meta = handle_missing_values(make_df(200, 5), "x")
    assert meta["missing_count"] == 5
    assert meta["missing_percentage"] == 2.5


# =====================================================
# clean_using_regex
# =====================================================

def test_regex_tuple_rule_extracts_number():
    df = pd.DataFrame({"Mileage": ["19.5 kmpl", "21 kmpl", None]})
    out = clean_using_regex(df, {"Mileage": (r"(\d+\.?\d*)", float)})
    assert out["Mileage"].iloc[:2].tolist() == [19.5, 21.0]
    assert pd.isna(out["Mileage"].iloc[2])
    assert out["Mileage"].dtype == float


def test_regex_strips_commas():
    df = pd.DataFrame({"Kms Driven": ["1,20,000 kms", "45,000 kms"]})
    out = clean_using_regex(df, {"Kms Driven": (r"([\d,]+)", float)})
    assert out["Kms Driven"].tolist() == [120000.0, 45000.0]


def test_regex_dict_rule_maps_values():
    df = pd.DataFrame({"Transmission": ["Automatic", "Manual", "CVT"]})
    out = clean_using_regex(df, {"Transmission": {"Automatic": 1, "Manual": 0}})
    assert out["Transmission"].iloc[:2].tolist() == [1, 0]
    assert pd.isna(out["Transmission"].iloc[2])   # unmapped -> NaN


def test_regex_leaves_other_columns_and_input_untouched():
    df = pd.DataFrame({"Mileage": ["19 kmpl"], "other": ["keep me"]})
    out = clean_using_regex(df, {"Mileage": (r"(\d+)", float)})
    assert out["other"].tolist() == ["keep me"]
    assert df["Mileage"].tolist() == ["19 kmpl"]   # original not mutated


# =====================================================
# clean_using_functions
# =====================================================

def test_functions_single_column_rule():
    df = pd.DataFrame({"Price": ["5 Lakh", "1 Crore"]})
    out = clean_using_functions(df, {"Price": clean_price})
    assert out["Price"].tolist() == [500000.0, 10000000.0]


def test_functions_partial_with_map():
    df = pd.DataFrame({"Ownership": ["First Owner", "Third Owner"]})
    rule = {"Ownership": partial(clean_ownership, map_dict=OWNERSHIP_MAP)}
    out = clean_using_functions(df, rule)
    assert out["Ownership"].tolist() == [1, 3]


def test_functions_tuple_rule_expands_and_drops_source():
    df = pd.DataFrame({"car_name": ["Maruti Swift VXI", "Honda"], "keep": [1, 2]})
    out = clean_using_functions(df, {"car_name": (clean_car_name, ["brand", "model"])})
    assert "car_name" not in out.columns
    assert out["brand"].tolist() == ["Maruti", "Honda"]
    assert out["model"].tolist() == ["Swift VXI", "Unknown"]
    assert out["keep"].tolist() == [1, 2]


def test_functions_does_not_mutate_input():
    df = pd.DataFrame({"Price": ["5 Lakh"]})
    clean_using_functions(df, {"Price": clean_price})
    assert df["Price"].tolist() == ["5 Lakh"]


# =====================================================
# _encode_column_ohe
# =====================================================

def test_ohe_drops_first_sorted_category():
    df = pd.DataFrame({"Fuel": ["Petrol", "Diesel", "CNG", "Petrol"]})
    dummies, meta = _encode_column_ohe(df, "Fuel", drop_first=True)
    assert list(dummies.columns) == ["Fuel_Diesel", "Fuel_Petrol"]
    assert meta["all_categories"] == ["CNG", "Diesel", "Petrol"]
    assert meta["dropped_baseline_category"] == "CNG"
    assert meta["encoded_columns"] == list(dummies.columns)
    assert dummies["Fuel_Petrol"].tolist() == [1, 0, 0, 1]


def test_ohe_keep_all_categories():
    df = pd.DataFrame({"Fuel": ["Petrol", "Diesel", "CNG"]})
    dummies, meta = _encode_column_ohe(df, "Fuel", drop_first=False)
    assert dummies.shape[1] == 3
    assert meta["dropped_baseline_category"] is None


def test_ohe_nan_row_is_all_zeros():
    df = pd.DataFrame({"Fuel": ["Petrol", "Diesel", None]})
    dummies, _ = _encode_column_ohe(df, "Fuel", drop_first=False)
    assert dummies.iloc[2].sum() == 0


# =====================================================
# _apply_frequency_encoding
# =====================================================

def test_frequency_encoding_replaces_column_with_counts():
    df = pd.DataFrame({"Engine Type": ["V6", "I4", "I4", "I4", "V6"]})
    out, meta = _apply_frequency_encoding(df, "Engine Type")
    assert out["Engine Type"].tolist() == [2, 3, 3, 3, 2]
    m = meta["Engine Type"]
    assert m["frequency_map"] == {"I4": 3, "V6": 2}
    assert m["encoded_column"] == "Engine Type"
    assert m["normalize"] is False


def test_frequency_encoding_normalized():
    df = pd.DataFrame({"c": ["a", "a", "a", "b", "b"]})
    out, meta = _apply_frequency_encoding(df, "c", normalize=True)
    assert out["c"].tolist() == pytest.approx([0.6, 0.6, 0.6, 0.4, 0.4])
    assert meta["c"]["unseen_default_value"] == 0.0


def test_frequency_encoding_keep_original_adds_new_column():
    df = pd.DataFrame({"c": ["a", "a", "b"]})
    out, meta = _apply_frequency_encoding(df, "c", drop_original=False)
    assert out["c"].tolist() == ["a", "a", "b"]
    assert out["c_freq"].tolist() == [2, 2, 1]
    assert meta["c"]["encoded_column"] == "c_freq"


def test_frequency_encoding_leaves_nan_as_nan_and_input_untouched():
    df = pd.DataFrame({"c": ["a", "a", None]})
    out, _ = _apply_frequency_encoding(df, "c")
    assert out["c"].iloc[:2].tolist() == [2, 2]
    assert pd.isna(out["c"].iloc[2])
    assert df["c"].iloc[:2].tolist() == ["a", "a"]
    assert df["c"].isna().tolist() == [False, False, True]


# =====================================================
# _apply_target_encoding
# =====================================================

def test_target_encoding_uses_group_means():
    df = pd.DataFrame({"brand": ["A", "A", "B"], "Price": [100.0, 200.0, 300.0]})
    out, meta = _apply_target_encoding(df, "brand", target_column="Price")
    assert out["brand"].tolist() == [150.0, 150.0, 300.0]
    m = meta["brand"]
    assert m["encoding_map"] == {"A": 150.0, "B": 300.0}
    assert m["unseen_default_value"] == pytest.approx(200.0)   # global mean
    assert m["target_column"] == "Price"


def test_target_encoding_keep_original():
    df = pd.DataFrame({"brand": ["A", "B"], "Price": [100.0, 300.0]})
    out, meta = _apply_target_encoding(df, "brand", drop_original=False)
    assert out["brand"].tolist() == ["A", "B"]
    assert out["brand_target_encoded"].tolist() == [100.0, 300.0]


def test_target_encoding_missing_target_raises():
    with pytest.raises(ValueError, match="missing"):
        _apply_target_encoding(pd.DataFrame({"brand": ["A"]}), "brand")


def test_target_encoding_non_numeric_target_raises():
    df = pd.DataFrame({"brand": ["A"], "Price": ["cheap"]})
    with pytest.raises(ValueError, match="numeric"):
        _apply_target_encoding(df, "brand")


# =====================================================
# cleaning_pipeline (integration)
# =====================================================

@pytest.fixture
def raw_cars():
    return pd.DataFrame({
        "car_name": ["Maruti Swift VXI", "Hyundai i20 Asta", "Maruti Baleno Zeta", "Honda City ZX"],
        "Price": ["5.2 Lakh", "6 Lakh", "7.5 Lakh", "9 Lakh"],
        "Mileage": ["19 kmpl", "20.5 kmpl", "21 kmpl", "18 kmpl"],
        "Kms Driven": ["12,000 kms", "45,000 kms", "30,000 kms", "60,000 kms"],
        "Fuel": ["Petrol", "Diesel", "Petrol", "CNG"],
        "Drive Type": ["FWD", "AWD", "FWD", "RWD"],
        "Turbo Charger": ["No", "Yes", "No", "Twin"],
        "Engine Type": ["V6", "4 Cylinder", "4 Cylinder", "4 Cylinder"],
        "Ownership": ["First Owner", "Second Owner", "First Owner", "Third Owner"],
        "Transmission": ["Manual", "Automatic", "Manual", "Manual"],
        "Emission Norm Compliance": ["BS VI", "BS IV", "BS VI", "BS VI 2.0"],
    })


def test_pipeline_end_to_end(raw_cars, tmp_path):
    out = cleaning_pipeline(raw_cars, tmp_path / "meta" / "encoding_metadata.json")

    # source columns replaced
    assert "car_name" not in out.columns
    assert {"brand", "model"} <= set(out.columns)
    for gone in ("Fuel", "Drive Type", "Turbo Charger"):
        assert gone not in out.columns

    # one-hot columns (first sorted category dropped)
    assert {"Fuel_Diesel", "Fuel_Petrol",
            "Drive Type_FWD", "Drive Type_RWD",
            "Turbo Charger_Twin", "Turbo Charger_Yes"} <= set(out.columns)

    # value cleaning
    assert out["Price"].tolist() == pytest.approx([520000, 600000, 750000, 900000])
    assert out["Kms Driven"].tolist() == [12000.0, 45000.0, 30000.0, 60000.0]
    assert out["Ownership"].tolist() == [1, 2, 1, 3]
    assert out["Transmission"].tolist() == [0, 1, 0, 0]
    assert out["Emission Norm Compliance"].tolist() == [6, 4, 6, 7]
    assert out["Engine Type"].tolist() == [1, 3, 3, 3]   # frequency encoded

    assert not out.isnull().any().any()


def test_pipeline_writes_metadata_files(raw_cars, tmp_path):
    meta_path = tmp_path / "meta" / "encoding_metadata.json"
    cleaning_pipeline(raw_cars, meta_path)

    assert meta_path.exists()
    assert (meta_path.parent / "fill_missing.json").exists()

    enc = json.loads(meta_path.read_text(encoding="utf-8"))
    assert set(enc) == {"Fuel", "Drive Type", "Turbo Charger", "Engine Type"}
    assert enc["Fuel"]["encoding_type"] == "one_hot"
    assert enc["Engine Type"]["encoding_type"] == "frequency"


def test_pipeline_does_not_mutate_input(raw_cars, tmp_path):
    cols_before = list(raw_cars.columns)
    cleaning_pipeline(raw_cars, tmp_path / "m.json")
    assert list(raw_cars.columns) == cols_before
    assert raw_cars["Price"].iloc[0] == "5.2 Lakh"


def test_pipeline_without_metadata_path(raw_cars):
    out = cleaning_pipeline(raw_cars, None)   # should skip file writing, not crash
    assert len(out) == 4
    