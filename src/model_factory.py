import logging
from typing import Any, Dict

import lightgbm as lgb
import numpy as np
import torch
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline


def build_model_pipeline(model):
    """Build a simple pipeline with just the model"""
    return Pipeline([("model", model)])


def _detect_gpu_capabilities() -> Dict[str, Any]:
    """
    Detect available GPU capabilities and return optimal device configurations.
    
    Returns:
        Dict containing device configurations for LightGBM and XGBoost
    """
    config = {
        "pytorch_device": "cpu",
        "lgbm_device": None,
        "xgb_device": None,
        "xgb_tree_method": "hist",
        "gpu_available": False
    }
    
    # Check PyTorch GPU availability first
    if torch.cuda.is_available():
        config["pytorch_device"] = "cuda"
        config["gpu_available"] = True
        logging.debug("NVIDIA CUDA GPU detected.")
    elif torch.backends.mps.is_available():
        config["pytorch_device"] = "mps"
        config["gpu_available"] = True
        logging.debug("Apple Metal (MPS) GPU detected.")
    else:
        logging.debug("No PyTorch-compatible GPU found.")
    
    
    # Create small test data
    X_test = np.random.random((100, 10))
    y_test = np.random.random(100)

    # check GPU support
    if config["gpu_available"] and config["pytorch_device"]=="cuda":
        try:
            
            test_model = lgb.LGBMRegressor(
                device='gpu', 
                verbose=-1, 
                n_estimators=1,
                num_leaves=10
            )
            test_model.fit(X_test, y_test)
            
            config["lgbm_device"] = "cuda" if config["pytorch_device"] == "cuda" else "gpu"
            logging.info("LightGBM GPU support confirmed.")
        except Exception as e:
            logging.warning("LightGBM GPU support not available")
            config["lgbm_device"] = None
        
        if config["pytorch_device"] == "cuda":
            logging.info("XGBoost GPU support confirmed.")
            config["xgb_device"] = "cuda"
        else:
            logging.info("No GPU available, using CPU for XGBoost")
        
    else:
        logging.info("GPU acceleration disabled - using CPU for all models.")
    
    return config

def get_model_configs(
    random_state: int, 
    metric: str = "mape",
    n_jobs: int = 1,
) -> Dict[str, Dict[str, Any]]:
    """Returns a dictionary of models and their configurations"""
    
    # Detect GPU capabilities
    gpu_config = _detect_gpu_capabilities()

    configs = {
        "LightGBM": {
            "model": lgb.LGBMRegressor(
                random_state=random_state,
                objective="regression",
                metric=metric,
                n_jobs=n_jobs,
                verbose=-1,
                device=gpu_config["lgbm_device"],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
            ),
        },
        "XGBoost": {
            "model": xgb.XGBRegressor(
                random_state=random_state,
                tree_method=gpu_config["xgb_tree_method"], 
                objective="reg:squarederror",
                n_jobs=n_jobs,
                device=gpu_config["xgb_device"],
                eval_metric=metric,
            ),
        },
        "XGBoostQuantile": {
            "model": xgb.XGBRegressor(
                random_state=random_state,
                tree_method=gpu_config["xgb_tree_method"], 
                objective="reg:quantileerror",
                quantile_alpha=0.5,
                n_jobs=n_jobs,
                device=gpu_config["xgb_device"],
                eval_metric=metric,
            ),
        },
        "RandomForest": {
            "model": RandomForestRegressor(random_state=random_state, n_jobs=n_jobs),
        },
    }
    return configs


def get_model_params(
    model_name: str, trial, model_prefix: str = "model__"
) -> Dict[str, Any]:
    """Get hyperparameter search space for a given model"""
    params = {}

    if model_name == "RandomForest":
        params[f"{model_prefix}n_estimators"] = trial.suggest_int(
            f"{model_prefix}n_estimators", 200, 1800, step=100
        )
        params[f"{model_prefix}max_depth"] = trial.suggest_int(
            f"{model_prefix}max_depth", 3, 100
        )
        params[f"{model_prefix}min_samples_split"] = trial.suggest_int(
            f"{model_prefix}min_samples_split", 5, 30
        )
        params[f"{model_prefix}min_samples_leaf"] = trial.suggest_int(
            f"{model_prefix}min_samples_leaf", 2, 15
        )
        params[f"{model_prefix}max_features"] = trial.suggest_categorical(
            f"{model_prefix}max_features", [0.5, 0.7, 1.0]
        )

    elif model_name in ["XGBoost", "XGBoostQuantile"]:
        params[f"{model_prefix}max_depth"] = trial.suggest_int(
            f"{model_prefix}max_depth",
            3,
            100,
        )
        params[f"{model_prefix}learning_rate"] = trial.suggest_float(
            f"{model_prefix}learning_rate", 0.01, 0.3, log=True
        )
        params[f"{model_prefix}n_estimators"] = trial.suggest_int(
            f"{model_prefix}n_estimators", 400, 5000, step=100
        )
        params[f"{model_prefix}min_child_weight"] = trial.suggest_int(
            f"{model_prefix}min_child_weight",
            5,
            50,
        )
        params[f"{model_prefix}subsample"] = trial.suggest_float(
            f"{model_prefix}subsample", 0.6, 1.0
        )
        params[f"{model_prefix}colsample_bytree"] = trial.suggest_float(
            f"{model_prefix}colsample_bytree", 0.6, 1.0
        )
        params[f"{model_prefix}gamma"] = trial.suggest_float(
            f"{model_prefix}gamma",
            0,
            5.0,
        )
        params[f"{model_prefix}reg_lambda"] = trial.suggest_float(
            f"{model_prefix}reg_lambda", 1e-8, 200.0, log=True
        )
        params[f"{model_prefix}reg_alpha"] = trial.suggest_float(
            f"{model_prefix}reg_alpha", 1e-8, 200.0, log=True
        )

        if model_name == "XGBoostQuantile":
            params[f"{model_prefix}quantile_alpha"] = trial.suggest_float(
                f"{model_prefix}quantile_alpha", 0.4, 0.8, step=0.05
            )

    elif model_name == "LightGBM":
        params[f"{model_prefix}objective"] = trial.suggest_categorical(
            f"{model_prefix}objective",
            ["regression_l2", "huber", "regression_l1"],
        )

        params[f"{model_prefix}learning_rate"] = trial.suggest_float(
            f"{model_prefix}learning_rate", 1e-2, 0.1, log=True
        )
        params[f"{model_prefix}n_estimators"] = trial.suggest_int(
            f"{model_prefix}n_estimators", 800, 8000, step=100
        )
        params[f"{model_prefix}num_leaves"] = trial.suggest_int(
            f"{model_prefix}num_leaves", 20, 500
        )
        params[f"{model_prefix}max_depth"] = trial.suggest_int(
            f"{model_prefix}max_depth", 3, 200
        )
        params[f"{model_prefix}min_child_samples"] = trial.suggest_int(
            f"{model_prefix}min_child_samples", 5, 100
        )
        params[f"{model_prefix}subsample"] = trial.suggest_float(
            f"{model_prefix}subsample", 0.1, 1.0
        )
        params[f"{model_prefix}colsample_bytree"] = trial.suggest_float(
            f"{model_prefix}colsample_bytree", 0.1, 1.0
        )
        params[f"{model_prefix}reg_alpha"] = trial.suggest_float(
            f"{model_prefix}reg_alpha", 1e-8, 10.0, log=True
        )
        params[f"{model_prefix}reg_lambda"] = trial.suggest_float(
            f"{model_prefix}reg_lambda", 1e-8, 200.0, log=True
        )

    return params
