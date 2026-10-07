import sys
import pandas as pd
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from config.settings import (
    FEATURE_REGEX_PATTERNS,
    FUNC_CLEAN_DICT,
    ENCODING_MAP,
    ENC_METADATA_FILE_POWER,
    OUTPUT_FILE_POWER,
    PROCESSED_FILE_POWER,
)
from utils.data_cleaning import cleaning_pipeline
from utils.eda import eda
from utils.pre_preprocessing import apply_preprocessing

def process(
    csv_file: str | Path, clean_dict: dict, func_clean_dict: dict, 
    ENCODING_MAP: dict, metadata_json: str | Path, output_path: str | Path,
) -> pd.DataFrame:
    """
        Cleans raw CSV data, generates EDA diagnostic charts, applies post-EDA
        preprocessing, and exports processed data to CSV.
    """
    csv_file        = Path(csv_file)
    output_path     = Path(output_path)
    metadata_json   = Path(metadata_json) 
    if any(not p.exists() for p in (csv_file, metadata_json, output_path)):
        raise FileNotFoundError("One or more required files/paths do not exist. Run fetch_data.py first.")
    df = pd.read_csv(csv_file)
    df = cleaning_pipeline(df, clean_dict, func_clean_dict, ENCODING_MAP, metadata_json)
 
    eda_output_dir = metadata_json.parent / "eda_reports"
    eda(df, eda_output_dir) 
    df = apply_preprocessing(df) 

    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=False)
    df.to_csv(output_path, index=False)
    print(f"Processed dataset successfully saved to: {output_path.resolve()}")

    return df

if __name__ == "__main__":
    process(
        csv_file=OUTPUT_FILE_POWER,
        clean_dict=FEATURE_REGEX_PATTERNS,
        func_clean_dict=FUNC_CLEAN_DICT,
        ENCODING_MAP=ENCODING_MAP,
        metadata_json=ENC_METADATA_FILE_POWER,
        output_path=PROCESSED_FILE_POWER,
    )
