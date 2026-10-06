import json
import re
from pathlib import Path 
from altair import value
import pandas as pd
import numpy as np
from typing import Any

def handle_missing_values(df: pd.DataFrame, col: str,) -> tuple[pd.DataFrame, dict]:
    """
    Handles missing values and returns metadata describing
    the operation performed.
    """

    metadata = {
        "column": col,
        "missing_count": None,
        "missing_percentage": None,
        "action": None,
        "fill_strategy": None,
        "fill_value": None,
        "indicator_column": None,
    }

    if col not in df.columns:
        metadata["action"] = "column_not_found"
        print(f"Column '{col}' not found in DataFrame.")
        return df, metadata

    missing_count = df[col].isnull().sum()
    total_rows = len(df)

    metadata["missing_count"] = int(missing_count)

    if total_rows == 0:
        metadata["action"] = "empty_dataframe"
        print("DataFrame is empty.")
        return df, metadata

    missing_pct = (missing_count / total_rows) * 100
    metadata["missing_percentage"] = round(missing_pct, 2)

    print(f"{col}: {missing_pct:.2f}% missing", end=" → ")

    # > 50% missing → drop column
    if missing_pct > 50:
        df = df.drop(columns=[col])

        metadata["action"] = "drop_column"

        print("Dropped column (>50% missing)")

    # 30–50% missing → drop rows
    elif missing_pct > 30:
        rows_before = len(df)

        df = df.dropna(subset=[col]).reset_index(drop=True)

        metadata["action"] = "drop_rows"
        metadata["rows_dropped"] = rows_before - len(df)

        print("Dropped rows (30-50% missing)")

    # 10–30% missing → fill + indicator
    elif missing_pct > 10:
        indicator_col = f"{col}_was_missing"

        df[indicator_col] = df[col].isnull().astype(int)

        if (
            df[col].dtype == "object"
            or isinstance(df[col].dtype, pd.CategoricalDtype)
        ):
            mode_vals = df[col].mode()
            fill_val = mode_vals[0] if not mode_vals.empty else "Unknown"

            metadata["fill_strategy"] = "mode"

        else:
            fill_val = df[col].mean()

            metadata["fill_strategy"] = "mean"

        df[col] = df[col].fillna(fill_val)

        metadata["action"] = "fill_and_indicator"
        metadata["fill_value"] = (
            fill_val.item()
            if hasattr(fill_val, "item")
            else fill_val
        )
        metadata["indicator_column"] = indicator_col

        print(
            f"Filled + added indicator column '{indicator_col}'"
        )

    # < 10% missing → fill only
    elif missing_pct > 0:
        if (
            df[col].dtype == "object"
            or isinstance(df[col].dtype, pd.CategoricalDtype)
        ):
            mode_vals = df[col].mode()
            fill_val = mode_vals[0] if not mode_vals.empty else "Unknown"

            metadata["fill_strategy"] = "mode"

        else:
            fill_val = df[col].median()

            metadata["fill_strategy"] = "median"

        df[col] = df[col].fillna(fill_val)

        metadata["action"] = "fill"

        metadata["fill_value"] = (
            fill_val.item()
            if hasattr(fill_val, "item")
            else fill_val
        )

        print(f"Filled with {metadata['fill_strategy']}")

    else:
        metadata["action"] = "no_missing_values"

        print("No missing values")

    return df, metadata

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

def cleaning_pipeline(
    df: pd.DataFrame, regex_clean_dict: dict, func_clean_dict: dict, 
    ohe_features: list, metadata_json_path: Path,
) -> pd.DataFrame:

    """
        Cleans dataframe columns dynamically using regex extraction,

        dictionary mapping, custom cleaning functions, One-Hot Encoding, boolean
        conversion, and missing value handling at the end based on configuration
        settings.

        Any remaining object/category columns are left untouched (e.g. for
        XGBoost native categorical handling) and reported.
    """
    df = df.copy()
    missing_value_metadata_registry = {}
    ohe_metadata_registry = {} 

    for col in list(df.columns):
        if col in regex_clean_dict:
            rule = regex_clean_dict[col] 
            if isinstance(rule, tuple):
                pattern, dtype = rule
                extracted = df[col].astype(str).str.extract(pattern, expand=False)
                df[col] = pd.to_numeric(extracted, errors="coerce").astype(dtype)
            elif isinstance(rule, dict):
                df[col] = df[col].map(rule)
 
    for col in list(df.columns):
        if col in func_clean_dict:
            rule = func_clean_dict[col] 
            if isinstance(rule, tuple):
                clean_func, new_cols = rule
                df[new_cols] = df[col].apply(clean_func).apply(pd.Series)
                df = df.drop(columns=[col]) 
            else:
                df[col] = df[col].apply(rule)
 
    for col in ohe_features:
        if col not in df.columns:
            continue 
        dummies, meta = _encode_column_ohe(df, column=col, drop_first=True)
        ohe_metadata_registry[col] = meta
        df = pd.concat([df.drop(columns=[col]), dummies], axis=1,)

    bool_cols = df.select_dtypes(include=["bool"]).columns
    if not bool_cols.empty:
        df[bool_cols] = df[bool_cols].astype(int)

    for col in list(df.columns):
        if col not in df.columns:
            continue
        df, metadata = handle_missing_values(df, col)
        missing_value_metadata_registry[col] = metadata

    remaining_categorical = df.select_dtypes(
        include=["object", "category"]
    ).columns
    if not remaining_categorical.empty:
        print(
            "Columns left as string/object dtype "
            f"(not encoded): {list(remaining_categorical)}"
        )

    if metadata_json_path:
        metadata_json_path.parent.mkdir(parents=True, exist_ok=True,)
        if ohe_metadata_registry:
            with open(metadata_json_path, "w", encoding="utf-8",) as f:
                json.dump(ohe_metadata_registry, f, indent=4,)
            print("OHE metadata successfully saved to: " f"{metadata_json_path}")

        if missing_value_metadata_registry:
            fill_missing_path = (metadata_json_path.parent / "fill_missing.json")
            with open(fill_missing_path, "w", encoding="utf-8") as f:
                json.dump(missing_value_metadata_registry, f, indent=4, default=str,) 
            print("Missing-value metadata successfully saved to: " f"{fill_missing_path}")

    return df

def _encode_column_ohe(
    df: pd.DataFrame, column: str, drop_first: bool = True
) -> tuple[pd.DataFrame, dict]:
    """Generates one-hot encoded dummies and tracks category mappings and dropped reference level."""
    categories = sorted(df[column].dropna().unique().tolist())
    dummies = pd.get_dummies(
        df[column], prefix=column, drop_first=drop_first, dtype=int
    )
    dropped_feature = categories[0] if (drop_first and categories) else None
    metadata = {
        "original_column": column,
        "all_categories": categories,
        "encoded_columns": list(dummies.columns),
        "dropped_baseline_category": dropped_feature,
    }
    return dummies, metadata

def clean_engine_type(value: Any) -> str:
    if pd.isna(value) or str(value).strip().lower() in ["nan", "none", ""]:
        return "Unknown", "Unknown"
    shape = _extract_engine_shape(value)
    volume = _extract_engine_volume(value)
    return shape, volume

def _extract_engine_shape(value: Any) -> str:
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

def _extract_engine_volume(value: Any) -> str:
    if pd.isna(value):
        return "Unknown"
    val_str = str(value).strip()
    if val_str.lower() in ["nan", "none", "", "unknown", "null"]:
        return "Unknown"
    
    liter_match = re.search(r'\b(\d+\.\d+|\d+)\s*[-_]?\s*(?:L|LTR|LITER|LITRE)S?\b',  val_str, re.IGNORECASE)
    if liter_match:
        vol = float(liter_match.group(1))
        if vol > 10.0:
            return f"{vol / 1000:.1f}L"
        return f"{vol:.1f}L"

    cc_match = re.search(r'\b(\d{3,4})\s*CC\b', val_str, re.IGNORECASE)
    if cc_match:
        cc_val = float(cc_match.group(1))
        return f"{cc_val / 1000:.1f}L"

    standalone_match = re.search(
        r'\b(\d\.\d)\s*(?=[A-Z]|Turbo|TSI|CRDi|Kappa|MPI|VVT|GDI|Multijet|T-GDi|T-CRDi|Revotron|Revotorq|Bi-fuel|Dual|Naturally|Ingenium)', val_str, re.IGNORECASE
    )
    if standalone_match:
        return f"{float(standalone_match.group(1)):.1f}L"
    
    return "Unknown"
