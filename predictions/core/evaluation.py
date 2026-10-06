from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
import numpy as np


def evaluate_model(
    model,
    test_df,
    target_column: str,
    task_type: str = "regression",
):
    """Evaluates a trained model on a target dataset split.

    Returns metrics dict and prediction arrays compatible with MLflow logging
    and downstream plotting utilities.
    """
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column].to_numpy().ravel()
    predictions = model.predict(X_test).ravel()

    if task_type == "classification":
        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(
                precision_score(
                    y_test, predictions, average="weighted", zero_division=0
                )
            ),
            "recall": float(
                recall_score(
                    y_test, predictions, average="weighted", zero_division=0
                )
            ),
            "f1_score": float(
                f1_score(
                    y_test, predictions, average="weighted", zero_division=0
                )
            ),
        }

    elif task_type == "regression":
        mse = mean_squared_error(y_test, predictions)
        metrics = {
            "mae": float(mean_absolute_error(y_test, predictions)),
            "mse": float(mse),
            "rmse": float(np.sqrt(mse)),
            "r2": float(r2_score(y_test, predictions)),
        }

    else:
        raise ValueError("task_type must be either 'classification' or 'regression'")

    return {
        "metrics": metrics,
        "y_true": y_test,
        "predictions": predictions,
    }
