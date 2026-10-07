import os
from config.settings import (
    DATASET_MILEAGE,
    EXPERIMENT_NAME_MILEAGE,
    MODEL_MILEAGE,
    PARAM_GRID_MILEAGE,
    TARGET_COLUMN_MILEAGE,
)
from predictions.core.train_model_pipeline import run_pipeline

if __name__ == "__main__":
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    run_pipeline(
        MODEL_MILEAGE,
        DATASET_MILEAGE,
        PARAM_GRID_MILEAGE,
        TARGET_COLUMN_MILEAGE,
        EXPERIMENT_NAME_MILEAGE,
    )
    