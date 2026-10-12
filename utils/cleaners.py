import re
from typing import Any

import numpy as np
import pandas as pd

# This module must never import from `config` or `utils.data_cleaning`,
# otherwise the circular import comes back.


def clean_drive_type(value: Any) -> str:
    """Standardizes drive type variants into canonical labels without encoding."""
    if pd.isna(value) or str(value).strip().lower() in ["nan", "none", ""]:
        return None

    cleaned_val = (
        str(value).strip().upper().replace(" ", "").replace("-", "")
    )
    fwd_variants = {
        "FWD",
        "2WD",
        "4X2",
        "FRONTWHEELDRIVE",
        "TWOWHEELDRIVE",
        "TWOWHHEELDRIVE",
        "TWOWHHHEELDRIVE",
    }
    rwd_variants = {
        "RWD",
        "RWD(WITHMTT)",
        "REARWHEELDRIVEWITHESP",
        "REARWHEELDRIVE",
    }
    awd_variants = {
        "AWD",
        "4WD",
        "4X4",
        "ALLWHEELDRIVE",
        "FOURWHEELDRIVE",
        "FOURWHELLDRIVE",
        "PERMANENTALLWHEELDRIVEQUATTRO",
    }
    if cleaned_val in fwd_variants:
        return "FWD"
    if cleaned_val in rwd_variants:
        return "RWD"
    if cleaned_val in awd_variants:
        return "AWD"

    return None


def clean_turbo_charger(value: Any) -> str:
    """Standardizes turbo charger variants into canonical labels without encoding."""
    if pd.isna(value) or str(value).strip().lower() in ["nan", "none", ""]:
        return None
    cleaned_val = str(value).strip().upper().replace(" ", "")
    no_variants = {
        "NO",
        "N",
        "NA",
        "N/A",
        "NONE",
        "FALSE",
        "0",
        "NOTAVAILABLE",
        "NOTURBO",
        "NATURALLYASPIRATED",
    }
    yes_variants = {
        "YES",
        "Y",
        "TRUE",
        "1",
        "TURBO",
        "TURBOCHARGED",
        "SINGLETURBO",
        "SINGLE",
    }
    twin_variants = {
        "TWIN",
        "TWINTURBO",
        "DUALTURBO",
        "DUAL",
        "BITURBO",
        "BI-TURBO",
        "TWIN-TURBO",
    }
    if cleaned_val in no_variants:
        return "No"
    if cleaned_val in yes_variants:
        return "Yes"
    if cleaned_val in twin_variants:
        return "Twin"

    return None


def clean_emission_norm(value: Any, map_dict) -> int:
    """Normalizes emission norm variants and maps them to an ordinal strictness rank."""
    if pd.isna(value) or str(value).strip().lower() in ["nan", "none", ""]:
        return map_dict["Unknown"]

    cleaned_val = str(value).strip().upper()

    if "ZEV" in cleaned_val:
        canonical = "ZEV"
    elif "6.0" in cleaned_val or "VI 2.0" in cleaned_val:
        canonical = "BS VI 2.0"
    elif "BS III" in cleaned_val or "BSIII" in cleaned_val or "BHARAT STAGE III" in cleaned_val:
        canonical = "BS III"
    elif "BS IV" in cleaned_val or "BSIV" in cleaned_val or "BHARAT STAGE IV" in cleaned_val:
        canonical = "BS IV"
    elif "BS VI" in cleaned_val or "BSVI" in cleaned_val or "BHARAT STAGE VI" in cleaned_val:
        canonical = "BS VI"
    elif "BS II" in cleaned_val or "BHARAT STAGE II" in cleaned_val:
        canonical = "BS II"
    elif "BS I" in cleaned_val or "BHARAT STAGE I" in cleaned_val:
        canonical = "BS I"
    elif "EURO VI" in cleaned_val or "EU 6" in cleaned_val:
        canonical = "Euro VI"
    elif "EURO V" in cleaned_val:
        canonical = "Euro V"
    elif "EURO IV" in cleaned_val:
        canonical = "Euro IV"
    else:
        canonical = "Unknown"

    return map_dict[canonical]


def clean_price(value: Any) -> float:
    """Converts price strings with Lakh/Crore/Thousand suffixes into a numeric value, rounded to 1 decimal place."""
    if pd.isna(value) or str(value).strip().lower() in ["nan", "none", ""]:
        return np.nan

    cleaned_val = str(value).replace("₹", "").strip()

    if "Lakh" in cleaned_val:
        return round(float(cleaned_val.replace("Lakh", "").strip()) * 100000, 1)
    if "Crore" in cleaned_val:
        return round(float(cleaned_val.replace("Crore", "").strip()) * 10000000, 1)
    if "Thousand" in cleaned_val:
        return round(float(cleaned_val.replace("Thousand", "").strip()) * 1000, 1)

    return np.nan


def clean_ownership(value: Any, map_dict: dict) -> int:
    """Maps ownership label variants to an ordinal rank via map_dict."""
    if pd.isna(value) or str(value).strip().lower() in ["nan", "none", ""]:
        return 0

    cleaned_val = str(value).strip().title()
    return map_dict.get(cleaned_val, 0)


def clean_car_name(value: Any) -> tuple[str, str]:
    """Splits a raw car name string into (brand, model) without encoding."""
    if pd.isna(value) or str(value).strip().lower() in ["nan", "none", ""]:
        return "Unknown", "Unknown"

    parts = str(value).strip().split(" ", 1)
    brand = parts[0] if parts[0] else "Unknown"
    model = parts[1] if len(parts) > 1 and parts[1] else "Unknown"

    return brand, model


def engine_shape(value: Any) -> str:
    if pd.isna(value) or str(value).strip().lower() in ["nan", "none", ""]:
        return "Unknown"
    val_upper = str(value).strip().upper()
    v_match = re.search(r"\bV[-\s]?(\d{1,2})\b", val_upper)
    if v_match:
        return f"V{v_match.group(1)}"
    w_match = re.search(r"\bW[-\s]?(\d{1,2})\b", val_upper)
    if w_match:
        return f"W{w_match.group(1)}"
    flat_match = re.search(r"\b(?:FLAT|BOXER)[-\s]?(\d{1,2})\b", val_upper)
    if flat_match:
        return f"Flat{flat_match.group(1)}"
    i_match = re.search(r"\bI[-\s]?(\d{1,2})\b", val_upper)
    if i_match:
        return f"I{i_match.group(1)}"
    cyl_match = re.search(r"\b(\d{1,2})[-\s]?CYL(?:INDER)?S?\b", val_upper)
    if cyl_match:
        return f"I{cyl_match.group(1)}"
    if re.search(r"\bV[-\s]?TYPE\b", val_upper):
        return "V"
    if "IN-LINE" in val_upper or "INLINE" in val_upper or "IN LINE" in val_upper:
        return "Inline"
    return "Unknown"
