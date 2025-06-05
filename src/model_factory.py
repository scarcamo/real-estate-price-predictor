import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from typing import Dict, Any


def build_model_pipeline(model):
    """Build a simple pipeline with just the model"""
    return Pipeline([("model", model)])


def get_model_configs(random_state: int, metric: str = "mape") -> Dict[str, Dict[str, Any]]:
    """Returns a dictionary of models and their configurations"""
    configs = {
        "LightGBM": {
            "model": lgb.LGBMRegressor(
                random_state=random_state,
                objective="regression",
                metric=metric,
                n_jobs=-1,
                verbose=-1,
            ),
        },
        "XGBoost": {
            "model": xgb.XGBRegressor(
                random_state=random_state,
                tree_method="hist",
                objective="reg:squarederror",
                n_jobs=-1,
            ),
        },
        "XGBoostQuantile": {
            "model": xgb.XGBRegressor(
                random_state=random_state,
                tree_method="hist",
                objective="reg:quantileerror",
                quantile_alpha=0.5,
                n_jobs=-1,
            ),
        },
        "RandomForest": {
            "model": RandomForestRegressor(random_state=random_state, n_jobs=-1),
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
            f"{model_prefix}n_estimators", 400, 1500, step=100
        )
        params[f"{model_prefix}max_depth"] = trial.suggest_int(
            f"{model_prefix}max_depth", 3, 30
        )
        params[f"{model_prefix}min_samples_split"] = trial.suggest_int(
            f"{model_prefix}min_samples_split", 2, 50
        )
        params[f"{model_prefix}min_samples_leaf"] = trial.suggest_int(
            f"{model_prefix}min_samples_leaf", 1, 30
        )
        params[f"{model_prefix}max_features"] = trial.suggest_categorical(
            f"{model_prefix}max_features", ["sqrt", "log2", 0.5, 0.7, 1.0]
        )
        params[f"{model_prefix}bootstrap"] = trial.suggest_categorical(
            f"{model_prefix}bootstrap", [True, False]
        )

    elif model_name in ["XGBoost", "XGBoostQuantile"]:
        params[f"{model_prefix}max_depth"] = trial.suggest_int(
            f"{model_prefix}max_depth",
            3,
            30,
        )
        params[f"{model_prefix}learning_rate"] = trial.suggest_float(
            f"{model_prefix}learning_rate", 0.01, 0.3, log=True
        )
        params[f"{model_prefix}n_estimators"] = trial.suggest_int(
            f"{model_prefix}n_estimators", 100, 2000, step=100
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
            f"{model_prefix}reg_lambda", 1e-2, 200.0, log=True
        )
        params[f"{model_prefix}reg_alpha"] = trial.suggest_float(
            f"{model_prefix}reg_alpha", 1e-2, 200.0, log=True
        )

    elif model_name == "LightGBM":
        params[f"{model_prefix}objective"] = trial.suggest_categorical(
            f"{model_prefix}objective",
            ["regression_l1", "regression_l2", "huber"],
        )

        params[f"{model_prefix}learning_rate"] = trial.suggest_float(
            f"{model_prefix}learning_rate", 0.01, 0.3, log=True
        )
        params[f"{model_prefix}n_estimators"] = trial.suggest_int(
            f"{model_prefix}n_estimators", 700, 2500, step=100
        )
        params[f"{model_prefix}num_leaves"] = trial.suggest_int(
            f"{model_prefix}num_leaves", 20, 200
        )
        params[f"{model_prefix}max_depth"] = trial.suggest_int(
            f"{model_prefix}max_depth", 3, 40
        )
        params[f"{model_prefix}min_child_samples"] = trial.suggest_int(
            f"{model_prefix}min_child_samples", 5, 100
        )
        params[f"{model_prefix}subsample"] = trial.suggest_float(
            f"{model_prefix}subsample", 0.6, 1.0
        )
        params[f"{model_prefix}colsample_bytree"] = trial.suggest_float(
            f"{model_prefix}colsample_bytree", 0.5, 1.0
        )
        params[f"{model_prefix}reg_alpha"] = trial.suggest_float(
            f"{model_prefix}reg_alpha", 1e-2, 200.0, log=True
        )
        params[f"{model_prefix}reg_lambda"] = trial.suggest_float(
            f"{model_prefix}reg_lambda", 1e-2, 200.0, log=True
        )

    return params
