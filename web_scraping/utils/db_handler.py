import pandas as pd
import sqlite3
from pathlib import Path

def analyze_mixed_features(df):
    """Prints missing value percentages, detects mixed data types across columns,

    and displays basic statistical summaries safely for numeric and categorical
    columns.
    """
    if df.empty:
        print("DataFrame is empty. Nothing to analyze.")
        return

    print("=" * 60)
    print("                DATA ANALYSIS REPORT               ")
    print("=" * 60)

    # 1. Missing Value Breakdown
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df)) * 100
    missing_df = pd.DataFrame(
        {"Missing Values": null_counts, "Percentage (%)": null_pct.round(2)}
    )
    print("\n--- Missing Values Summary ---")
    missing_rows = missing_df[missing_df["Missing Values"] > 0]
    if not missing_rows.empty:
        print(missing_rows)
    else:
        print("No missing values found across any columns.")

    # 2. Mixed Type Column Detection
    print("\n--- Column Data Types & Mixed Type Detection ---")
    for col in df.columns:
        inferred_types = set(df[col].dropna().apply(type).tolist())
        type_names = [t.__name__ for t in inferred_types]

        if len(inferred_types) > 1:
            print(f"[!] WARNING: Column '{col}' contains MIXED types: {type_names}")
        else:
            print(
                f"Column '{col}': {type_names[0] if type_names else 'Empty/All Null'}"
            )

    # 3. Safe Statistical Summaries
    print("\n--- Summary Statistics ---")
    # Check if there are any numeric columns in the DataFrame
    numeric_cols = df.select_dtypes(include=["number"]).columns

    if len(numeric_cols) > 0:
        desc = df[numeric_cols].describe().T
        # Safely select columns that exist in the describe output
        cols_to_show = [c for c in ["count", "mean", "min", "50%", "max"] if c in desc.columns]
        print(desc[cols_to_show])
    else:
        print("No numeric columns present in DataFrame. Displaying object summary:")
        print(df.describe(include=["object"]).T)

    print("=" * 60)
    
def save_to_city_table(df, city_name, db_path):
    """Saves the cleaned DataFrame into an SQLite database table named after

    the specified city name. Creates the table automatically if it doesn't
    exist.
    """
    if df.empty:
        print(f"Skipping empty DataFrame for city: {city_name}")
        return

    db_path = Path(db_path)
    # Ensure database directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize table name (e.g., 'New Delhi' -> 'New_Delhi')
    table_name = city_name.strip().replace(" ", "_")

    conn = sqlite3.connect(db_path)
    try: 
        df.to_sql(name=table_name, con=conn, if_exists="append", index=False)
        print(
            f"Successfully inserted {len(df)} records into SQLite table '{table_name}' in {db_path.name}"
        )
    except Exception as e:
        print(f"Error saving to database table '{table_name}': {e}")
    finally:
        conn.close()
