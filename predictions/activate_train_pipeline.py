import os
from config.settings import (
    DATASET_PRICE,
    EXPERIMENT_NAME_PRICE,
    MODEL_PRICE,
    PARAM_GRID_PRICE,
    TARGET_COLUMN_PRICE,
)
from predictions.core.train_model_pipeline import run_pipeline

if __name__ == "__main__":
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    run_pipeline(
        MODEL_PRICE,
        DATASET_PRICE,
        PARAM_GRID_PRICE,
        TARGET_COLUMN_PRICE,
        EXPERIMENT_NAME_PRICE,
    )
    