from pathlib import Path
from functools import partial
from utils.data_cleaning import (
    clean_drive_type, clean_turbo_charger, clean_price, 
    clean_car_name, clean_emission_norm, clean_ownership,
)

# ==========================================
# System & Path Configurations
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_FILE = PROJECT_ROOT / "database" / "car_database.db" 

# ==========================================
# Data Processing Paths
# ==========================================
# Mileage Paths
FEATURES_FILE_MILEAGE = (
    PROJECT_ROOT / "predictions" / "data" / "mileage" / "features.txt"
)
OUTPUT_FILE_MILEAGE = (
    PROJECT_ROOT / "predictions" / "data" / "mileage" / "raw" / "data.csv"
)
PROCESSED_FILE_MILEAGE = (
    PROJECT_ROOT / "predictions" / "data" / "mileage" / "processed" / "data.csv"
)
OHE_METADATA_FILE_MILEAGE = (
    PROJECT_ROOT / "predictions" / "data" / "mileage" / "processed" / "metadata" / "ohe_metadata.json"
)

# Power Paths
FEATURES_FILE_POWER = (
    PROJECT_ROOT / "predictions" / "data" / "power" / "features.txt"
)
OUTPUT_FILE_POWER = (
    PROJECT_ROOT / "predictions" / "data" / "power" / "raw" / "data.csv"
)
PROCESSED_FILE_POWER = (
    PROJECT_ROOT / "predictions" / "data" / "power" / "processed" / "data.csv"
)
OHE_METADATA_FILE_POWER = (
    PROJECT_ROOT / "predictions" / "data" / "power" / "processed" / "metadata" / "ohe_metadata.json"
)

# Price Paths
FEATURES_FILE_PRICE = (
    PROJECT_ROOT / "predictions" / "data" / "price" / "features.txt"
)
OUTPUT_FILE_PRICE = (
    PROJECT_ROOT / "predictions" / "data" / "price" / "raw" / "data.csv"
)
PROCESSED_FILE_PRICE = (
    PROJECT_ROOT / "predictions" / "data" / "data" / "processed" / "data.csv"
)
OHE_METADATA_FILE_PRICE = (
    PROJECT_ROOT / "predictions" / "model" / "price" / "ohe_metadata.json"
)

# ==========================================
# Preprocessing Rules & Maps
# ==========================================
FEATURE_REGEX_PATTERNS = {
    "Mileage": (r"(\d+\.?\d*)", float),
    "Engine": (r"(\d+)", float),
    "Kerb Weight": (r"(\d+)", float),
    "Power": (r"(\d+\.?\d*)", float),
    "Registration Year": (r"(\d{4})", float),
    "Kms Driven": (r"([\d,]+)", float),
    "Seats": (r"(\d+)", float),
    "Ground Clearance Unladen": (r"(\d+\.?\d*)", float),
    "Transmission Type": {"Automatic": 1, "Manual": 0},
    "Transmission": {"Automatic": 1, "Manual": 0},
}

EMISSION_NORM_ORDINAL_MAP = {
    "BS I": 1,
    "BS II": 2,
    "BS III": 3,
    "Euro IV": 4,
    "BS IV": 4,
    "Euro V": 5,
    "BS VI": 6,
    "Euro VI": 6,
    "BS VI 2.0": 7,
    "ZEV": 8,
    "Unknown": 0,
}

OWNERSHIP_MAP = {
    "First Owner": 1,
    "Second Owner": 2,
    "Third Owner": 3,
    "Fourth Owner": 4,
    "Fifth Owner": 5,
}

OHE_FEATURES = ["Fuel", "Transmission", "Drive Type", "Turbo Charger"]

FUNC_CLEAN_DICT = {
    "Drive Type": clean_drive_type,
    "Turbo Charger": clean_turbo_charger,
    "Price": clean_price,
    "Emission Norm Compliance": partial(clean_emission_norm, map_dict=EMISSION_NORM_ORDINAL_MAP),
    "car_name": (clean_car_name, ["brand", "model"]),
    "Ownership": partial(clean_ownership, map_dict=OWNERSHIP_MAP),
}

# ==========================================
# Model Training Settings: Mileage
# ==========================================
EXPERIMENT_NAME_MILEAGE = "Mileage_Prediction"
from sklearn.ensemble import RandomForestRegressor
MODEL_MILEAGE = RandomForestRegressor(random_state=42)
PARAM_GRID_MILEAGE = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
}
TARGET_COLUMN_MILEAGE = "Mileage"
DATASET_MILEAGE = PROCESSED_FILE_MILEAGE 

# ==========================================
# Model Training Settings: Power
# ==========================================
EXPERIMENT_NAME_POWER = "Power_Prediction"
from sklearn.ensemble import RandomForestRegressor
MODEL_POWER = RandomForestRegressor(random_state=42)
PARAM_GRID_POWER = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
}
TARGET_COLUMN_POWER = "Power"
DATASET_POWER = PROCESSED_FILE_POWER

# ==========================================
# Model Training Settings: Price
# ==========================================

