import matplotlib.pyplot as plt
from pathlib import Path
import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split

# Import custom module utilities
from predictions.core.evaluation import evaluate_model
from predictions.core.train_model import train_model
from utils.visualization import generate_performance_plots
from config.settings import MILEAGE_MODEL, MILEAGE_PARAM_GRID, TARGET_COLUMN, DATASET

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

def run_pipeline(model, dataset, grid_params, target):
    project_root = Path(__file__).resolve().parents[3]  
    mlruns_dir = project_root / "mlruns" 

    mlflow.set_tracking_uri(mlruns_dir.as_uri())

    mlflow.set_experiment("Car_Feature_Prediction")

    with mlflow.start_run(run_name=f"Predict_{target}") as run:
        print(f"Active MLflow Run ID: {run.info.run_id}")
        mlflow.set_tags(
            {
                "target_feature": target,
                "model_type": type(model).__name__,
                "pipeline_stage": "training",
            }
        )
        df = pd.read_csv(dataset)
        target_col = target
        train_df, val_df = train_test_split(
            df, test_size=0.2, random_state=42, shuffle=True
        )

        print("Training model...")
        train_results = train_model(train_df, val_df, model, target_col, grid_params,)
        best_model = train_results["model"]
        best_params = train_results["best_params"]
        if best_params:
            mlflow.log_params(best_params)

        print("Evaluating model...")
        train_eval = evaluate_model(best_model, train_df, target_col, task_type="regression")
        val_eval = evaluate_model(best_model, val_df, target_col, task_type="regression")

        for metric_name, val in train_eval["metrics"].items():
            mlflow.log_metric(f"train_{metric_name}", val)

        for metric_name, val in val_eval["metrics"].items():
            mlflow.log_metric(f"val_{metric_name}", val)

        print("Generating performance plots...")
        chunks_data = {
            "Train": {
                "y_true": train_eval["y_true"],
                "y_pred": train_eval["predictions"],
            },
            "Validation": {
                "y_true": val_eval["y_true"],
                "y_pred": val_eval["predictions"],
            },
        }
        fig = generate_performance_plots(chunks_data)
        mlflow.log_figure(fig, artifact_file=f"plots/{target_col}_performance.png")
        plt.close(fig)

        print("Logging model to MLflow...")
        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
        )
        print("Pipeline execution completed successfully.")

if __name__ == "__main__":
    run_pipeline(MILEAGE_MODEL, DATASET, MILEAGE_PARAM_GRID, TARGET_COLUMN)
