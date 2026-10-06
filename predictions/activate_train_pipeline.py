import os
from config.settings import (
    DATASET_POWER,
    EXPERIMENT_NAME_POWER,
    MODEL_POWER,
    PARAM_GRID_POWER,
    TARGET_COLUMN_POWER,
)
from predictions.core.train_model_pipeline import run_pipeline

if __name__ == "__main__":
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    run_pipeline(
        MODEL_POWER,
        DATASET_POWER,
        PARAM_GRID_POWER,
        TARGET_COLUMN_POWER,
        EXPERIMENT_NAME_POWER,
    )
    