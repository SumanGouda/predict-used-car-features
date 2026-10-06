"""
utils/model_registry.py
------------------------
Centralized model registry utility providing lazy dynamic imports
and model instantiation wrappers for ML training pipelines.
"""

from typing import Any, Callable, Dict
from sklearn.base import BaseEstimator


def _get_xgboost_regressor() -> type[BaseEstimator]:
    from xgboost import XGBRegressor

    return XGBRegressor


def _get_random_forest_regressor() -> type[BaseEstimator]:
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor


def _get_catboost_regressor() -> type[BaseEstimator]:
    from catboost import CatBoostRegressor

    return CatBoostRegressor


def _get_gradient_boosting_regressor() -> type[BaseEstimator]:
    from sklearn.ensemble import GradientBoostingRegressor

    return GradientBoostingRegressor


# Central registry mapping model string keys to lazy-loaded estimator classes
REGISTRY: Dict[str, Callable[[], type[BaseEstimator]]] = {
    "xgboost": _get_xgboost_regressor,
    "random_forest": _get_random_forest_regressor,
    "catboost": _get_catboost_regressor,
    "gradient_boosting": _get_gradient_boosting_regressor,
}


def get_model_class(model_type: str) -> type[BaseEstimator]:
    """
    Retrieves the uninstantiated model class from the registry based on model_type.

    Parameters:
    -----------
    model_type : str
        String key representing the target model architecture (e.g., 'xgboost').

    Returns:
    --------
    type[BaseEstimator]
        Uninstantiated sklearn-compatible estimator class.
    """
    key = model_type.lower().strip()
    if key not in REGISTRY:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Supported keys: {list(REGISTRY.keys())}"
        )
    return REGISTRY[key]()


def instantiate_model(model_type: str, **kwargs: Any) -> BaseEstimator:
    """
    Instantiates a model instance dynamically from the registry with supplied kwargs.

    Parameters:
    -----------
    model_type : str
        String key representing the model architecture.
    **kwargs : Any
        Keyword arguments passed directly to the model constructor.

    Returns:
    --------
    BaseEstimator
        An initialized instance of the requested estimator.
    """
    model_cls = get_model_class(model_type)
    return model_cls(**kwargs)
