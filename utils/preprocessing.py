import json
from pathlib import Path 
import pandas as pd 
from config.settings import FEATURE_REGEX_PATTERNS, FUNC_CLEAN_DICT, ENCODING_MAP


def cleaning_pipeline(df: pd.DataFrame, metadata_output_path: Path,) -> pd.DataFrame:
    """
        Cleans dataframe columns dynamically using regex extraction,

        dictionary mapping, custom cleaning functions, One-Hot Encoding, boolean
        conversion, and missing value handling at the end based on configuration
        settings.

        Any remaining object/category columns are left untouched (e.g. for
        XGBoost native categorical handling) and reported.
    """
    df = df.copy()
    fill_missing_meta_registry = {}
    encoding_meta_registry = {} 

    df = clean_using_regex(df, FEATURE_REGEX_PATTERNS)
    df = clean_using_functions(df, FUNC_CLEAN_DICT)
 
    for encoding_type, config in ENCODING_MAP.items(): 
        items = config.items() if isinstance(config, dict) else [(col, None) for col in config] 
        for col, label_map in items:
            if col not in df.columns:
                continue 
            if encoding_type == "ohe_encoding":
                dummies, meta = _encode_column_ohe(df, column=col, drop_first=True)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                encoding_meta_registry[col] = {
                    "encoding_type": "one_hot",
                    "metadata": meta,
                }
            elif encoding_type == "freq_encoding":
                df, meta = _apply_frequency_encoding(df, column_name=col, normalize=False, drop_original=True)
                encoding_meta_registry[col] = {
                    "encoding_type": "frequency",
                    "metadata": meta[col],
                }
            elif encoding_type == "target_encoding":
                df, meta = _apply_target_encoding(df, column_name=col, target_column="Price", drop_original=True)
                encoding_meta_registry[col] = {
                    "encoding_type": "target",
                    "metadata": meta[col],
                }
            elif encoding_type == "label_encoding":
                df, meta = _apply_label_encoding(df, column_name=col, label_map=label_map, unseen_default_value=-1, drop_original=True)
                encoding_meta_registry[col] = {
                    "encoding_type": "label",
                    "metadata": meta[col],
                }
    
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if not bool_cols.empty:
        df[bool_cols] = df[bool_cols].astype(int)

    for col in list(df.columns):
        df, metadata = handle_missing_values(df, col)
        fill_missing_meta_registry[col] = metadata

    remaining_categorical = df.select_dtypes(include=["object", "str", "category"]).columns
    if not remaining_categorical.empty:
        print("Columns left as string/object dtype " f"(not encoded): {list(remaining_categorical)}")

    if metadata_output_path:
        metadata_output_path.parent.mkdir(parents=True, exist_ok=True)

        if encoding_meta_registry:
            with open(metadata_output_path, "w", encoding="utf-8") as f:
                json.dump(encoding_meta_registry, f, indent=4, default=str)
            print(f"Encoding metadata successfully saved to: {metadata_output_path}")
        if fill_missing_meta_registry:
            fill_missing_path = metadata_output_path.parent / "fill_missing.json"
            with open(fill_missing_path, "w", encoding="utf-8") as f:
                json.dump(fill_missing_meta_registry, f, indent=4, default=str)
            print(f"Missing-value metadata successfully saved to: {fill_missing_path}")

    return df


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

        if pd.api.types.is_numeric_dtype(df[col]):
            fill_val = df[col].mean()
            metadata["fill_strategy"] = "mean"
        else:
            mode_vals = df[col].mode()
            fill_val = mode_vals[0] if not mode_vals.empty else "Unknown"
            metadata["fill_strategy"] = "mode"

        df[col] = df[col].fillna(fill_val)
        metadata["action"] = "fill_and_indicator"
        metadata["fill_value"] = (
            fill_val.item() if hasattr(fill_val, "item") else fill_val
        )
        metadata["indicator_column"] = indicator_col
        print(f"Filled + added indicator column '{indicator_col}'")

    # < 10% missing → fill only
    elif missing_pct > 0:
        if pd.api.types.is_numeric_dtype(df[col]):
            fill_val = df[col].median()
            metadata["fill_strategy"] = "median"
        else:
            mode_vals = df[col].mode()
            fill_val = mode_vals[0] if not mode_vals.empty else "Unknown"
            metadata["fill_strategy"] = "mode"

        df[col] = df[col].fillna(fill_val)
        metadata["action"] = "fill"
        metadata["fill_value"] = (
            fill_val.item() if hasattr(fill_val, "item") else fill_val
        )
        print(f"Filled with {metadata['fill_strategy']}")
    
    else:
        metadata["action"] = "no_missing_values"
        print("No missing values")

    return df, metadata


def clean_using_regex(df: pd.DataFrame, regex_clean_dict: dict) -> pd.DataFrame:
    df = df.copy()  
    for col in list(df.columns):
        if col in regex_clean_dict:
            rule = regex_clean_dict[col] 
            if isinstance(rule, tuple):
                pattern, dtype = rule 
                extracted = df[col].astype(str).str.extract(pattern, expand=False) 
                
                # Safely strip commas if present
                if extracted.dtype == object or extracted.str.contains(",", na=False).any():
                    extracted = extracted.str.replace(",", "", regex=False)
                    
                df[col] = pd.to_numeric(extracted, errors="coerce").astype(dtype)
            elif isinstance(rule, dict):
                df[col] = df[col].map(rule) 
    return df


def clean_using_functions(df: pd.DataFrame, func_clean_dict: dict) -> pd.DataFrame:
    """Applies custom cleaning functions to DataFrame columns, supporting both 
    single-column transforms and multi-column expansions (tuples like car_name -> brand, model)."""
    df = df.copy() 
    for col in list(df.columns):
        if col in func_clean_dict:
            rule = func_clean_dict[col] 
            if isinstance(rule, tuple):
                clean_func, new_cols = rule
                df[new_cols] = df[col].apply(clean_func).apply(pd.Series)
                df = df.drop(columns=[col]) 
            else:
                df[col] = df[col].apply(rule)
    return df


def _encode_column_ohe(df: pd.DataFrame, column: str, drop_first: bool = True) -> tuple[pd.DataFrame, dict]:
    """Generates one-hot encoded dummies and tracks category mappings and dropped reference level."""
    categories = sorted(df[column].dropna().unique().tolist())
    dummies = pd.get_dummies(df[column], prefix=column, drop_first=drop_first, dtype=int)
    dropped_feature = categories[0] if (drop_first and categories) else None
    metadata = {
        "original_column"           : column,
        "all_categories"            : categories,
        "encoded_columns"           : list(dummies.columns),
        "dropped_baseline_category" : dropped_feature,
    }
    return dummies, metadata


def _apply_frequency_encoding(df: pd.DataFrame, column_name: str | list[str], normalize: bool = False, drop_original: bool = True,) -> tuple[pd.DataFrame, dict]:
    """Applies frequency encoding to specified categorical feature column(s) and tracks metadata and frequency mappings."""
    df = df.copy()
    cols = [column_name] if isinstance(column_name, str) else column_name
    metadata = {}

    for col in cols:
        freq_series = df[col].value_counts(normalize=normalize)
        freq_map = {
            (k.item() if hasattr(k, "item") else k): (v.item() if hasattr(v, "item") else v) for k, v in freq_series.to_dict().items()
        }
        default_val        = 0.0 if normalize else 0
        target_col_name    = col if drop_original else f"{col}_freq"

        encoded_series  = df[col].map(freq_map) 
        unseen_mask     = df[col].notna() & encoded_series.isna()
        encoded_series.loc[unseen_mask] = default_val

        if drop_original and target_col_name != col:
            df = df.drop(columns=[col])
        df[target_col_name] = encoded_series
        metadata[col] = {
            "original_column"       : col,
            "encoded_column"        : target_col_name,
            "normalize"             : normalize,
            "frequency_map"         : freq_map,
            "unseen_default_value"  : default_val,
        }
    return df, metadata


def _apply_target_encoding(df: pd.DataFrame, column_name: str | list[str], target_column: str = "Price", drop_original: bool = True,) -> tuple[pd.DataFrame, dict]:
    """Applies target encoding to specified categorical feature column(s) using a target column and tracks metadata and mappings."""
    df = df.copy()
    
    if target_column not in df.columns:
        raise ValueError(f"Target feature '{target_column}' is missing from the DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[target_column]):
        raise ValueError(f"Target feature '{target_column}' must be numeric to calculate target encoding means.")
    
    metadata    = {} 
    global_mean = float(df[target_column].mean())
    cols        = [column_name] if isinstance(column_name, str) else column_name

    for col in cols: 
        means_series = df.groupby(col)[target_column].mean().dropna()
        encoding_map = {
            (k.item() if hasattr(k, "item") else k): (v.item() if hasattr(v, "item") else v) for k, v in means_series.to_dict().items()
        } 
        target_col_name = col if drop_original else f"{col}_target_encoded"

        encoded_series  = df[col].map(encoding_map)
        unseen_mask     = df[col].notna() & encoded_series.isna()
        encoded_series.loc[unseen_mask] = global_mean
        
        if drop_original and target_col_name != col:
            df = df.drop(columns=[col])
            
        df[target_col_name] = encoded_series
        
        metadata[col] = {
            "original_column"       : col,
            "encoded_column"        : target_col_name,
            "target_column"         : target_column,
            "encoding_map"          : encoding_map,
            "unseen_default_value"  : global_mean,
        }
    return df, metadata


def _apply_label_encoding(df: pd.DataFrame, column_name: str, label_map: dict, unseen_default_value: int = -1, drop_original: bool = True) -> tuple[pd.DataFrame, dict]:
    """Applies static label encoding to a specified column using a label map and tracks metadata."""
    df = df.copy()
    metadata = {}

    target_col_name = column_name if drop_original else f"{column_name}_label_encoded"
    encoded_series = df[column_name].map(label_map)
    unseen_mask = df[column_name].notna() & encoded_series.isna()
    encoded_series.loc[unseen_mask] = unseen_default_value

    if drop_original and target_col_name != column_name:
        df = df.drop(columns=[column_name])
    df[target_col_name] = encoded_series
    
    metadata[column_name] = {
        "original_column": column_name,
        "encoded_column": target_col_name,
        "label_map": label_map,
        "unseen_default_value": unseen_default_value,
    }
    return df, metadata