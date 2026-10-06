import sys
from pathlib import Path
import pandas as pd

# Setup Project Paths
PROJECT_ROOT = Path(__file__).resolve().parents[0]
BASE_DIR = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import analysis and database utilities
from utils.db_handler import analyze_mixed_features, save_to_city_table


def main():
    city = None

    raw_data_path = PROJECT_ROOT / "data" / "raw" / f"{city}_car_data.json"
    db_path = BASE_DIR / "database" / "car_database.db"

    if not raw_data_path.exists():
        print(f"Error: File not found at {raw_data_path}")
        return
 
    df = pd.read_json(raw_data_path, lines=True)
 
    df.dropna(how="all", inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    if df.empty:
        print("DataFrame is empty. Aborting insertion.")
        return
 
    analyze_mixed_features(df)
 
    while True:
        command = (
            input(
                "\nEnter 'OK' to save DataFrame to database, or 'SKIP' to cancel: "
            )
            .strip()
            .lower()
        )

        if command == "ok":
            save_to_city_table(df, city, db_path)
            break
        elif command == "skip":
            print("Skipping saving to database.")
            break
        else:
            print("Invalid input. Please enter 'OK' or 'SKIP'.")


if __name__ == "__main__":
    main()
