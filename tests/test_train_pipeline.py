import os
import mlflow
from xgboost import XGBRegressor
from predictions.core.train_model_pipeline import run_pipeline


def test_run_pipeline_end_to_end(sample_dataset, temp_mlruns_dir):
    """Tests if run_pipeline executes end-to-end, logs metrics, and saves MLflow artifacts."""
 
    mlflow.set_tracking_uri(temp_mlruns_dir)

    test_model = XGBRegressor(n_estimators=5, max_depth=2, random_state=42)
    test_grid = {"n_estimators": [2, 5]}
    target_col = "Mileage" 

    run_pipeline(
        model=test_model,
        dataset=sample_dataset,
        grid_params=test_grid,
        target=target_col,
    ) 
    experiment = mlflow.get_experiment_by_name("Car_Feature_Prediction")
    assert experiment is not None, "Experiment was not created in MLflow."

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) > 0, "No MLflow run was recorded."

    latest_run = runs.iloc[0]
    assert latest_run["tags.target_feature"] == target_col
    assert "metrics.val_rmse" in latest_run
    assert "metrics.train_rmse" in latest_run
