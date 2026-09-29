from predictions.core.train_model_pipeline import run_pipeline
from config.settings import (
    POWER_MODEL, POWER_PARAM_GRID, POWER_TARGET_COLUMN, 
    POWER_DATASET, POWER_EXPERIMENT_NAME,
)
import os

if __name__ == "__main__": 
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true" 
    run_pipeline(
    POWER_MODEL, POWER_DATASET, POWER_PARAM_GRID, POWER_TARGET_COLUMN, POWER_EXPERIMENT_NAME
)
