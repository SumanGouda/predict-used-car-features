import sys
import pandas as pd
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from config.settings import (
    ENC_METADATA_FILE_PRICE,
    OUTPUT_FILE_PRICE,
    PROCESSED_FILE_PRICE,
)
from utils.preprocessing import cleaning_pipeline
from utils.eda import eda
from utils.manual_processing import apply_preprocessing

def process(csv_file: str | Path, metadata_json: str | Path, output_path: str | Path,) -> pd.DataFrame:
    """
        Cleans raw CSV data, generates EDA diagnostic charts, applies post-EDA
        preprocessing, and exports processed data to CSV.
    """
    csv_file        = Path(csv_file)
    output_path     = Path(output_path)
    metadata_json   = Path(metadata_json)

    if not csv_file.exists():
        raise FileNotFoundError(f"{csv_file} not found. Run fetch_data.py first.")

    df = pd.read_csv(csv_file)
    df = cleaning_pipeline(df, metadata_json)

    eda_output_dir = metadata_json.parent / "eda_reports"
    eda(df, eda_output_dir)
    df = apply_preprocessing(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Processed dataset successfully saved to: {output_path.resolve()}")

    return df

if __name__ == "__main__":
    process(
        csv_file=OUTPUT_FILE_PRICE,
        metadata_json=ENC_METADATA_FILE_PRICE,
        output_path=PROCESSED_FILE_PRICE,
    )
