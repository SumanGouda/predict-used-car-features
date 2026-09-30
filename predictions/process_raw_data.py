import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.settings import (
    FEATURE_REGEX_PATTERNS,
    FUNC_CLEAN_DICT,
    OHE_FEATURES,
    OHE_METADATA_FILE_MILEAGE,
    OUTPUT_FILE_MILEAGE,
    PROCESSED_FILE_MILEAGE,
)
from utils.data_cleaning import cleaning_pipeline
from utils.eda import eda
from utils.pre_preprocessing import apply_preprocessing


def process(
    csv_file: str | Path,
    clean_dict: dict,
    func_clean_dict: dict,
    ohe_features: list,
    metadata_json: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """Cleans raw CSV data, generates EDA diagnostic charts, applies post-EDA
    preprocessing, and exports processed data to CSV.
    """
    csv_file = Path(csv_file)
    metadata_json = Path(metadata_json)
    output_path = Path(output_path)

    if not csv_file.exists():
        raise FileNotFoundError(f"Raw data file not found at: {csv_file}. Run fetch_data.py first.")
 
    df = pd.read_csv(csv_file)
    df = cleaning_pipeline(df, clean_dict, func_clean_dict, ohe_features, metadata_json)
 
    eda_output_dir = metadata_json.parent / "eda_reports"
    eda(df, eda_output_dir) 
    df = apply_preprocessing(df) 

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Processed dataset successfully saved to: {output_path.resolve()}")

    return df


if __name__ == "__main__":
    process(
        csv_file=OUTPUT_FILE_MILEAGE,
        clean_dict=FEATURE_REGEX_PATTERNS,
        func_clean_dict=FUNC_CLEAN_DICT,
        ohe_features=OHE_FEATURES,
        metadata_json=OHE_METADATA_FILE_MILEAGE,
        output_path=PROCESSED_FILE_MILEAGE,
    )
