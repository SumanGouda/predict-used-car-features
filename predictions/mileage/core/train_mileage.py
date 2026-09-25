"""
train_mileage.py
----------------
Utility module containing model training, hyperparameter optimization,
and metric calculation routines for predicting car mileage.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm 
from sklearn.model_selection import ParameterGrid, cross_val_score

from config.settings import (
    MILEAGE_CV_FOLDS,
    MILEAGE_MODEL_TYPE,
    MILEAGE_PARAM_GRID,
    MILEAGE_RANDOM_STATE,
)
from utils.model_registry import instantiate_model
from utils.model_helper import generate_learning_curve_data, compute_regression_metrics


def train_mileage_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_type: str = MILEAGE_MODEL_TYPE,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    cv_folds: int = MILEAGE_CV_FOLDS,
    random_state: int = MILEAGE_RANDOM_STATE,
    metadata_json_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Trains a model dynamically instantiated from utils.model_registry using GridSearchCV,
    evaluates metrics, and computes learning curve performance.
    """
    model_name = f"car_mileage_{model_type}"

    # Import model
    try:
        base_model = instantiate_model(model_type, random_state=random_state)
    except ValueError as err:
        raise KeyError(
            f"Model type '{model_type}' was not found in utils/model_registry.py. "
            f"Developer action required: Please add a getter function and map '{model_type}' "
            f"inside the REGISTRY dictionary in utils/model_registry.py."
        ) from err
 
    if param_grid is None:
        param_grid = MILEAGE_PARAM_GRID

    # Find optimum parameter using gridsearch
    grid = list(ParameterGrid(param_grid))
    best_score = -float("inf")
    best_params = None
    cv_results_list = []

    print(f"Running search across {len(grid)} hyperparameter combinations...")
    for params in tqdm(grid, desc="GridSearchCV Progress"):
        estimator = instantiate_model(
            model_type, **params, random_state=random_state
        )
        scores = cross_val_score(
            estimator, X_train, y_train, cv=cv_folds, scoring="r2", n_jobs=-1
        )
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))

        cv_results_list.append(
            {
                "params": params,
                "mean_test_score": mean_score,
                "std_test_score": std_score,
            }
        )

        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    # Refit the best model on the full training set
    best_estimator = instantiate_model(
        model_type, **best_params, random_state=random_state
    )
    best_estimator.fit(X_train, y_train)

    y_train_pred = best_estimator.predict(X_train)
    y_val_pred = best_estimator.predict(X_val)

    train_metrics = compute_regression_metrics(y_train, y_train_pred)
    val_metrics = compute_regression_metrics(y_val, y_val_pred)

    # Compute Learning Curve Data
    learning_curve_data = generate_learning_curve_data(
        estimator=best_estimator, X=X_train, y=y_train, cv=cv_folds, scoring="r2"
    )

    # Format cross-validation search results DataFrame
    cv_results_df = pd.DataFrame(cv_results_list)
    cv_results_df["rank_test_score"] = (
        cv_results_df["mean_test_score"]
        .rank(ascending=False, method="min")
        .astype(int)
    )
    cv_results_df = cv_results_df[
        ["params", "mean_test_score", "std_test_score", "rank_test_score"]
    ].sort_values(by="rank_test_score")
 
    return {
        "best_estimator": best_estimator,
        "model_name": model_name,
        "best_params": best_params,
        "param_grid": param_grid,
        "cv_best_score": best_score,
        "cv_results_df": cv_results_df,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "learning_curve": learning_curve_data,
        "feature_names": list(X_train.columns),
        "metadata_json_path": metadata_json_path,
    }
