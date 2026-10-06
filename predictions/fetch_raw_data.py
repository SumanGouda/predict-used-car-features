import sqlite3
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.settings import (
    DB_FILE,
    FEATURES_FILE_POWER,
    OUTPUT_FILE_POWER,
)
from utils.model_helper import get_and_validate_features


def fetch_raw_data(features_txt_path: str | Path, db_path: str | Path, output_csv_path: str | Path) -> pd.DataFrame:
    """Reads target features from features.txt, extracts them from every city table
    in the SQLite database, combines everything into a single dataset, and exports to raw CSV.
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
    print(f"Warehouse raw dataset saved to: {output_file}")

    return combined_df


def _extract_table(
    conn: sqlite3.Connection, cursor: sqlite3.Cursor, table: str, target_columns: list
) -> pd.DataFrame:
    """Selects target columns from a single table, aliasing missing columns to NULL."""
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


if __name__ == "__main__": 
    print("Fetching fresh raw data from SQLite database...")
    fetch_raw_data(FEATURES_FILE_POWER, DB_FILE, OUTPUT_FILE_POWER) 
        