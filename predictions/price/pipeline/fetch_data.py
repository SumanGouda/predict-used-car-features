import sys
import sqlite3
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.model_helper import get_and_validate_features
from utils.cleaning import apply_cleaning_pipeline
from config.settings import (
    DB_FILE,
    FEATURE_REGEX_PATTERNS,
    FEATURES_FILE_PRICE,
    FUNC_CLEAN_DICT,
    OHE_FEATURES,
    OHE_METADATA_FILE_PRICE,
    OUTPUT_FILE_PRICE,
    PROCESS_RAW_DATA,
    PROCESSED_FILE_PRICE
)

def main(features_txt_path: str, db_path: str, output_csv_path: str) -> pd.DataFrame:
    """Reads target features from features.txt, extracts them from every city table
    in the SQLite database, combines everything into a single dataset, and exports to CSV.
    """
    db_file = Path(db_path)
    output_file = Path(output_csv_path)
    features_file = Path(features_txt_path)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    target_columns = get_and_validate_features(features_file, db_file)

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        )
        tables = [row[0] for row in cursor.fetchall()]

        if not tables:
            print("No tables found in the database.")
            return pd.DataFrame()

        all_data = []
        for table in tables:
            df_table = _extract_table(conn, cursor, table, target_columns)
            if not df_table.empty:
                all_data.append(df_table)

    if not all_data:
        print("No data retrieved from any table.")
        return pd.DataFrame()

    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.dropna(how="all", subset=target_columns, inplace=True)
    combined_df.drop_duplicates(subset=target_columns, inplace=True)
    combined_df.reset_index(drop=True, inplace=True)

    combined_df.to_csv(output_file, index=False)
    print(f"Loaded {len(target_columns)} features from '{features_file.name}'.")
    print(f"Successfully extracted {len(combined_df)} records across {len(tables)} city tables.")
    print(f"Warehouse dataset saved to: {output_file}")

    return combined_df


def _extract_table(
    conn: sqlite3.Connection, cursor: sqlite3.Cursor, table: str, target_columns: list
) -> pd.DataFrame:
    """Selects the target columns from a single table, aliasing missing columns to NULL."""
    cursor.execute(f"PRAGMA table_info('{table}');")
    db_cols_map = {col[1].strip().lower(): col[1] for col in cursor.fetchall()}

    select_clauses = [
        f'"{db_cols_map[col.strip().lower()]}" AS "{col}"'
        if col.strip().lower() in db_cols_map
        else f'NULL AS "{col}"'
        for col in target_columns
    ]

    query = f"SELECT {', '.join(select_clauses)} FROM \"{table}\""

    try:
        return pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"Error querying table '{table}': {e}")
        return pd.DataFrame()
    

def process(
    csv_file: str,
    regex_clean_dict: dict,
    func_clean_dict: dict,
    ohe_features: list,
    metadata_json: str,
    output_path: str,
):
    csv_file = Path(csv_file)
    metadata_json = Path(metadata_json)
    output_path = Path(output_path)

    df = pd.read_csv(csv_file)
    df = apply_cleaning_pipeline(df, regex_clean_dict, func_clean_dict, ohe_features, metadata_json)

    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    if PROCESS_RAW_DATA:
        main(FEATURES_FILE_PRICE, DB_FILE, OUTPUT_FILE_PRICE)
        process(
            csv_file=OUTPUT_FILE_PRICE,
            regex_clean_dict=FEATURE_REGEX_PATTERNS,
            func_clean_dict=FUNC_CLEAN_DICT,
            ohe_features=OHE_FEATURES,
            metadata_json=OHE_METADATA_FILE_PRICE,
            output_path=PROCESSED_FILE_PRICE,
        )

                                