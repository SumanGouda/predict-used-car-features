import numpy as np
import pandas as pd
import pytest

from config.settings import EMISSION_NORM_ORDINAL_MAP, OWNERSHIP_MAP
from utils.cleaners import (
    clean_car_name,
    clean_drive_type,
    clean_emission_norm,
    clean_ownership,
    clean_price,
    clean_turbo_charger,
    engine_shape,
)

EMPTY_VALUES = [None, np.nan, pd.NA, "", "  ", "nan", "None", "NONE"]


# ---------- clean_drive_type ----------

@pytest.mark.parametrize("raw, expected", [
    ("FWD", "FWD"),
    ("2WD", "FWD"),
    ("4x2", "FWD"),
    ("Front Wheel Drive", "FWD"),
    ("Two Wheel Drive", "FWD"),
    ("RWD", "RWD"),
    ("Rear-Wheel Drive", "RWD"),
    ("RWD (with MTT)", "RWD"),
    ("AWD", "AWD"),
    ("4WD", "AWD"),
    ("4x4", "AWD"),
    ("All Wheel Drive", "AWD"),
    ("Permanent All Wheel Drive Quattro", "AWD"),
    ("  fwd  ", "FWD"),          # whitespace + case
])
def test_drive_type_variants(raw, expected):
    assert clean_drive_type(raw) == expected


@pytest.mark.parametrize("raw", EMPTY_VALUES + ["hovercraft"])
def test_drive_type_empty_or_unknown_returns_none(raw):
    assert clean_drive_type(raw) is None


# ---------- clean_turbo_charger ----------

@pytest.mark.parametrize("raw, expected", [
    ("No", "No"),
    ("N/A", "No"),
    ("Naturally Aspirated", "No"),
    ("0", "No"),
    ("false", "No"),
    ("Yes", "Yes"),
    ("Turbo", "Yes"),
    ("Single", "Yes"),
    ("1", "Yes"),
    (1, "Yes"),                   # non-string input
    ("Twin", "Twin"),
    ("Twin Turbo", "Twin"),
    ("Dual", "Twin"),
    ("Bi-Turbo", "Twin"),
])
def test_turbo_variants(raw, expected):
    assert clean_turbo_charger(raw) == expected


@pytest.mark.parametrize("raw", EMPTY_VALUES + ["maybe"])
def test_turbo_empty_or_unknown_returns_none(raw):
    assert clean_turbo_charger(raw) is None


# ---------- clean_emission_norm ----------
# Expected values are keys of the real map, so the test follows the map.

@pytest.mark.parametrize("raw, canonical", [
    ("BS I", "BS I"),
    ("BS II", "BS II"),
    ("BS III", "BS III"),
    ("BSIII", "BS III"),
    ("BS IV", "BS IV"),
    ("BS VI", "BS VI"),
    ("BSVI", "BS VI"),
    ("BS VI 2.0", "BS VI 2.0"),   # must not fall into plain "BS VI"
    ("BS6.0", "BS VI 2.0"),
    ("Bharat Stage II", "BS II"),
    ("Bharat Stage III", "BS III"),
    ("Bharat Stage IV", "BS IV"),
    ("Bharat Stage VI", "BS VI"),
    ("Euro IV", "Euro IV"),
    ("Euro V", "Euro V"),
    ("Euro VI", "Euro VI"),
    ("EU 6", "Euro VI"),
    ("ZEV", "ZEV"),
    ("something else", "Unknown"),
])
def test_emission_norm_variants(raw, canonical):
    result = clean_emission_norm(raw, EMISSION_NORM_ORDINAL_MAP)
    assert result == EMISSION_NORM_ORDINAL_MAP[canonical]


@pytest.mark.parametrize("raw", EMPTY_VALUES)
def test_emission_norm_empty_uses_unknown_rank(raw):
    assert clean_emission_norm(raw, EMISSION_NORM_ORDINAL_MAP) == EMISSION_NORM_ORDINAL_MAP["Unknown"]


def test_emission_norm_uses_the_map_passed_in():
    custom = {k: 100 + i for i, k in enumerate(EMISSION_NORM_ORDINAL_MAP)}
    assert clean_emission_norm("BS IV", custom) == custom["BS IV"]
    assert clean_emission_norm(None, custom) == custom["Unknown"]


# ---------- clean_price ----------

@pytest.mark.parametrize("raw, expected", [
    ("₹5.2 Lakh", 520000.0),
    ("₹ 8.75 Lakh", 875000.0),
    ("5.2Lakh", 520000.0),
    ("1.5 Crore", 15000000.0),
    ("50 Thousand", 50000.0),
    ("5.25 Lakh", 525000.0),
])
def test_price_suffixes(raw, expected):
    assert clean_price(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", EMPTY_VALUES + ["12345", "free"])
def test_price_empty_or_no_suffix_returns_nan(raw):
    assert np.isnan(clean_price(raw))


# ---------- clean_ownership ----------

@pytest.mark.parametrize("label, rank", list(OWNERSHIP_MAP.items()))
def test_ownership_labels(label, rank):
    assert clean_ownership(label, OWNERSHIP_MAP) == rank


@pytest.mark.parametrize("raw", ["first owner", "SECOND OWNER", "  third owner "])
def test_ownership_case_and_whitespace(raw):
    assert clean_ownership(raw, OWNERSHIP_MAP) > 0


@pytest.mark.parametrize("raw", EMPTY_VALUES + ["Tenth Owner"])
def test_ownership_empty_or_unknown_returns_zero(raw):
    assert clean_ownership(raw, OWNERSHIP_MAP) == 0


# ---------- clean_car_name ----------

@pytest.mark.parametrize("raw, expected", [
    ("Maruti Swift VXI", ("Maruti", "Swift VXI")),   # splits on first space only
    ("  Hyundai i20 ", ("Hyundai", "i20")),
    ("Maruti", ("Maruti", "Unknown")),               # no model
])
def test_car_name_split(raw, expected):
    assert clean_car_name(raw) == expected


@pytest.mark.parametrize("raw", EMPTY_VALUES)
def test_car_name_empty(raw):
    assert clean_car_name(raw) == ("Unknown", "Unknown")


# ---------- engine_shape ----------

@pytest.mark.parametrize("raw, expected", [
    ("V6", "V6"),
    ("V-8", "V8"),
    ("V 12", "V12"),
    ("W12", "W12"),
    ("Flat 6", "Flat6"),
    ("Boxer-4", "Flat4"),
    ("I4", "I4"),
    ("4 Cylinder", "I4"),
    ("6-cyl", "I6"),
    ("V-Type", "V"),
    ("In-line", "Inline"),
    ("Inline", "Inline"),
    ("V6 Turbo", "V6"),
    ("Rotary", "Unknown"),
])
def test_engine_shape(raw, expected):
    assert engine_shape(raw) == expected


@pytest.mark.parametrize("raw", EMPTY_VALUES)
def test_engine_shape_empty(raw):
    assert engine_shape(raw) == "Unknown"
    