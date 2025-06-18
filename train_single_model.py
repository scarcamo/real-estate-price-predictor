#!/usr/bin/env python3
import argparse
import json
import logging
import os
import time
from typing import Any, Dict

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold

from src.config import load_config
from src.data_manager import DataManager
from src.model_factory import build_model_pipeline, get_model_configs
from src.trainer import evaluate_model

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

logging.getLogger("lightgbm").setLevel(logging.ERROR)
logging.getLogger("xgboost").setLevel(logging.ERROR)


def get_default_hyperparameters(model_name: str) -> Dict[str, Any]:
    """Get reasonable default hyperparameters for each model"""

    # updated for fast training
    defaults = {
        "LightGBM": {
            "objective": "huber",
            "min_child_samples": 37,
            "max_depth": 25,
            "num_leaves": 50,
            "reg_alpha": 0.0006591423461978351,
            "n_estimators": 500,
            "subsample": 0.9135201316522913,
            "learning_rate": 0.014490114072153929,
            "reg_lambda": 0.8582832434053244,
            "colsample_bytree": 0.5759062281855559,
        },
        "XGBoost": {
            "n_estimators": 1000,
            "max_depth": 10,
            "learning_rate": 0.1,
            "min_child_weight": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 1.0,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
        },
        "XGBoostQuantile": {
            "n_estimators": 1000,
            "max_depth": 10,
            "learning_rate": 0.1,
            "min_child_weight": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 1.0,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
        },
        "RandomForest": {
            "n_estimators": 500,
            "max_depth": 15,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "max_features": "sqrt",
            "bootstrap": True,
        },
    }
    return defaults.get(model_name, {})


def load_hyperparameters_from_mlflow(run_id: str) -> Dict[str, Any]:
    """Load hyperparameters from an MLflow run"""
    try:
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)

        # Extract hyperparameters from run params
        params = {}
        model_prefix = ""

        for key, value in run.data.params.items():
            if key.startswith(model_prefix):
                # Remove  prefix
                param_name = key.replace(model_prefix, "")

                # Try to convert to appropriate type
                try:
                    # Try int first
                    if value.isdigit() or (
                        value.startswith("-") and value[1:].isdigit()
                    ):
                        params[param_name] = int(value)
                    # Try float
                    elif "." in value:
                        params[param_name] = float(value)
                    # Try boolean
                    elif value.lower() in ["true", "false"]:
                        params[param_name] = value.lower() == "true"
                    # Keep as string
                    else:
                        params[param_name] = value
                except ValueError:
                    params[param_name] = value

        logger.info(f"Loaded {len(params)} hyperparameters from MLflow run {run_id}")
        logger.info(f"Parameters: {params}")
        return params

    except Exception as e:
        logger.error(f"Failed to load hyperparameters from MLflow run {run_id}: {e}")
        raise


def train_single_model(
    model_name: str,
    hyperparameters: Dict[str, Any],
    feature_set_path: str,
    config: Dict[str, Any],
    scoring_metric: str = "mape",
) -> None:
    """Train a single model with given hyperparameters"""

    logger.info("🚀 Starting single model training")
    logger.info(f"Model: {model_name}")
    logger.info(f"Feature Set: {feature_set_path}")
    logger.info(f"Scoring Metric: {scoring_metric}")
    logger.info(f"Hyperparameters: {hyperparameters}")

    start_time = time.time()

    # Setup experiment name
    feature_set_name = os.path.basename(feature_set_path).replace(".json", "")
    experiment_name = f"SingleModel_{model_name}_{feature_set_name}_{scoring_metric}"
    mlflow.set_experiment(experiment_name)

    # Initialize DataManager and get transformed data
    logger.info("🔄 Loading and transforming data using centralized DataManager")
    data_manager = DataManager(config)
    data_manager.load_data()

    # Get transformed data using the centralized method
    X_train_transformed, X_test_transformed, feature_info = (
        data_manager.get_transformed_data_from_feature_set(feature_set_path)
    )

    logger.info(f"✅ Data transformation completed")
    logger.info(f"📊 Training data shape: {X_train_transformed.shape}")
    logger.info(f"📊 Test data shape: {X_test_transformed.shape}")
    logger.info(f"📋 Feature info: {feature_info}")
    logger.info("🚀 Using OPTIMIZED approach: Data transformed once, then split for CV")

    # Get model configuration
    model_configs = get_model_configs(config["RANDOM_STATE"], scoring_metric, -1)
    if model_name not in model_configs:
        raise ValueError(
            f"Model {model_name} not found in available models: {list(model_configs.keys())}"
        )

    model_config = model_configs[model_name]
    base_model = model_config["model"]

    # Set hyperparameters on the model
    for param_name, param_value in hyperparameters.items():
        if hasattr(base_model, param_name):
            setattr(base_model, param_name, param_value)
            logger.info(f"Set {param_name} = {param_value}")
        else:
            logger.warning(f"Parameter {param_name} not found in {model_name} model")

    # Setup cross-validation
    cv_strategy = KFold(
        n_splits=config["cv_folds"], shuffle=True, random_state=config["RANDOM_STATE"]
    )

    # Train and evaluate with cross-validation
    with mlflow.start_run(
        run_name=f"SingleModel_{model_name}_{feature_set_name}"
    ) as run:
        logger.info(f"MLflow Run ID: {run.info.run_id}")

        # Log basic information - matching trainer.py
        mlflow.set_tag("run_type", "single_model_run")
        mlflow.set_tag("experiment_version", config.get("experiment_version", "v1"))
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("feature_set", feature_set_name)
        mlflow.set_tag("scoring_metric", scoring_metric)
        mlflow.set_tag("training_mode", "single_model_no_tuning")

        # Log model name as parameter (matching trainer.py)
        mlflow.log_param("model_name", model_name)

        # Log hyperparameters with  prefix (matching trainer.py)
        for param_name, param_value in hyperparameters.items():
            mlflow.log_param(f"{param_name}", param_value)

        # Log feature information - matching trainer.py format
        mlflow.log_param("total_features", feature_info["total_features"])
        mlflow.log_param("tabular_features", feature_info["tabular_features"])
        mlflow.log_param("image_features", feature_info["image_features"])
        mlflow.log_param("selected_tabular_features", feature_info["selected_tabular_features"])
        mlflow.log_param("transformer_type", feature_info["transformer_type"])
        mlflow.log_param("should_include_images", feature_info["should_include_images"])

        # Log feature selection metadata with fs_meta_ prefix (matching trainer.py)
        for param, value in feature_info["feature_selection_metadata"].items():
            mlflow.log_param(f"fs_meta_{param}", value)

        # Log configuration parameters (matching trainer.py)
        mlflow.log_param("cv_folds", config["cv_folds"])
        mlflow.log_param("random_state", config["RANDOM_STATE"])
        # For single model, we don't have Optuna trials, so set to 0
        mlflow.log_param("n_trials_optuna", 0)
        mlflow.log_param("tuning_scoring_metric", scoring_metric)
        mlflow.log_param("optuna_direction", "minimize" if scoring_metric in ["mape", "rmse"] else "maximize")

        # Cross-validation training using pre-transformed data
        cv_scores = []
        fold_train_metrics = []
        fold_val_metrics = []

        # Get log-transformed target once
        y_train_log = data_manager.get_log_transformed_target()
        y_train_orig = data_manager.Y_train_raw[data_manager.target_variable]

        for fold_idx, (train_idx, val_idx) in enumerate(
            cv_strategy.split(data_manager.X_train_full_raw, y_train_log)
        ):
            logger.info(f"Training fold {fold_idx + 1}/{config['cv_folds']}")

            # Use pre-transformed data for cross-validation (no re-transformation!)
            X_fold_train = X_train_transformed.iloc[train_idx]
            X_fold_val = X_train_transformed.iloc[val_idx]

            # Get corresponding target values
            y_fold_train = y_train_log.iloc[train_idx]
            y_fold_val = y_train_log.iloc[val_idx]

            # Create a fresh model instance for this fold
            fold_model = model_configs[model_name]["model"]
            for param_name, param_value in hyperparameters.items():
                if hasattr(fold_model, param_name):
                    setattr(fold_model, param_name, param_value)

            # Build pipeline
            model_pipeline = build_model_pipeline(fold_model)

            # Train model
            model_pipeline.fit(X_fold_train, y_fold_train)

            # Make predictions
            y_train_pred_log = model_pipeline.predict(X_fold_train)
            y_val_pred_log = model_pipeline.predict(X_fold_val)

            # Convert predictions back to original price scale for evaluation
            y_train_pred_orig = np.exp(y_train_pred_log)
            y_val_pred_orig = np.exp(y_val_pred_log)

            # Get original price targets for evaluation
            y_fold_train_orig = y_train_orig.iloc[train_idx]
            y_fold_val_orig = y_train_orig.iloc[val_idx]

            # Evaluate on original price scale
            train_metrics = evaluate_model(
                y_fold_train_orig, y_train_pred_orig, "train"
            )
            val_metrics = evaluate_model(y_fold_val_orig, y_val_pred_orig, "val")

            fold_train_metrics.append(train_metrics)
            fold_val_metrics.append(val_metrics)

            # Store CV score based on scoring metric
            if scoring_metric == "mape":
                cv_scores.append(val_metrics["val_mape"])
            elif scoring_metric == "rmse":
                cv_scores.append(val_metrics["val_rmse"])
            else:
                cv_scores.append(val_metrics["val_r2"])

        # Calculate mean CV metrics
        mean_cv_score = np.mean(cv_scores)
        std_cv_score = np.std(cv_scores)

        logger.info(
            f"Cross-validation {scoring_metric}: {mean_cv_score:.4f} ± {std_cv_score:.4f}"
        )

        # Log CV results
        mlflow.log_metric(f"cv_{scoring_metric}_mean", mean_cv_score)
        mlflow.log_metric(f"cv_{scoring_metric}_std", std_cv_score)

        # Log best CV score (matching trainer.py format)
        mlflow.log_metric(f"best_optuna_cv_{scoring_metric}", mean_cv_score)
        
        # Log the MLflow callback metric (matching trainer.py format)
        mlflow.log_metric(config.get("MLFLOW_CALLBACK_METRIC_NAME", f"cv_{scoring_metric}"), mean_cv_score)

        # Calculate and log mean metrics across folds
        for metric_name in ["rmse", "mape", "r2"]:
            train_values = [
                metrics[f"train_{metric_name}"] for metrics in fold_train_metrics
            ]
            val_values = [metrics[f"val_{metric_name}"] for metrics in fold_val_metrics]

            mlflow.log_metric(f"train_{metric_name}_mean", np.mean(train_values))
            mlflow.log_metric(f"val_{metric_name}_mean", np.mean(val_values))
            mlflow.log_metric(f"train_{metric_name}_std", np.std(train_values))
            mlflow.log_metric(f"val_{metric_name}_std", np.std(val_values))

        # Train final model on all training data and evaluate on test set
        logger.info("Training final model on full training set...")

        X_train_final = X_train_transformed
        X_test_final = X_test_transformed

        y_train = data_manager.get_log_transformed_target()
        y_test = np.log(data_manager.get_test_target())

        # Align test target to match test data indices
        y_test = y_test.loc[X_test_final.index]

        # Create final model
        final_model = model_configs[model_name]["model"]
        for param_name, param_value in hyperparameters.items():
            if hasattr(final_model, param_name):
                setattr(final_model, param_name, param_value)

        final_pipeline = build_model_pipeline(final_model)

        # Train final model
        final_pipeline.fit(X_train_final, y_train)

        # Make final predictions (in log space)
        y_train_pred_final_log = final_pipeline.predict(X_train_final)
        y_test_pred_final_log = final_pipeline.predict(X_test_final)

        # Convert predictions back to original price scale for evaluation
        y_train_pred_final_orig = np.exp(y_train_pred_final_log)
        y_test_pred_final_orig = np.exp(y_test_pred_final_log)

        y_train_orig = data_manager.Y_train_raw[data_manager.target_variable]
        y_test_orig = data_manager.get_test_target().loc[X_test_final.index]

        # Evaluate final model on original price scale
        final_train_metrics = evaluate_model(
            y_train_orig, y_train_pred_final_orig, "final_train"
        )
        final_test_metrics = evaluate_model(
            y_test_orig, y_test_pred_final_orig, "final_test"
        )

        # Log final metrics
        for metric_name, metric_value in final_train_metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        for metric_name, metric_value in final_test_metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Log model
        model_step_in_pipeline = final_pipeline.named_steps["model"]
        try:
            if isinstance(model_step_in_pipeline, xgb.XGBModel):
                mlflow.xgboost.log_model(
                    model_step_in_pipeline,
                    f"{model_name}_model",
                    input_example=X_train_final.head(10),
                    pip_requirements=None,
                    extra_pip_requirements=None,
                )
            elif isinstance(model_step_in_pipeline, lgb.LGBMModel):
                mlflow.lightgbm.log_model(
                    model_step_in_pipeline,
                    f"{model_name}_model",
                    input_example=X_train_final.head(10),
                    pip_requirements=None, 
                    extra_pip_requirements=None,
                )
            else:
                mlflow.sklearn.log_model(
                    final_pipeline,
                    f"{model_name}_pipeline",
                    input_example=X_train_final.head(10),
                    pip_requirements=None,  # Disable automatic pip requirement detection
                    extra_pip_requirements=None,
                )
        except Exception as e:
            logger.error(f"Error logging model {model_name}: {e}")

        # Log feature names and metadata
        mlflow.log_text(
            json.dumps(X_train_final.columns.tolist()), "used_features.json"
        )
        mlflow.log_text(
            json.dumps(feature_info["feature_selection_metadata"]),
            "feature_selection_metadata.json",
        )
        mlflow.log_text(json.dumps(feature_info), "feature_info.json")

        # Log feature list as artifact (matching trainer.py format)
        feature_list_path = "final_selected_features_list.txt"
        with open(feature_list_path, "w") as f:
            for feature in X_train_final.columns:
                f.write(f"{feature}\n")
        mlflow.log_artifact(feature_list_path, artifact_path="feature_info")
        if os.path.exists(feature_list_path):
            os.remove(feature_list_path)

        # Log duration metrics (matching trainer.py format)
        end_time = time.time()
        total_duration = end_time - start_time
        mlflow.log_metric("tuning_duration_sec", 0)  # No tuning for single model
        mlflow.log_metric("total_pipeline_duration_sec", total_duration)
        mlflow.log_metric("training_duration_seconds", total_duration)  # Keep existing metric

        logger.info(f"✅ Training completed in {total_duration:.1f} seconds")
        logger.info(
            f"📊 Final test {scoring_metric}: {final_test_metrics[f'final_test_{scoring_metric}']:.4f}"
        )
        logger.info(f"🔗 MLflow Run: {run.info.run_id}")


def main():
    """
    Single Model Training Script

    This script trains a single model without hyperparameter tuning for testing pipelines
    with new feature sets. It supports two modes:
    1. Manual hyperparameters: Pass hyperparameters as arguments
    2. Load from best run: Use hyperparameters from a previous MLflow run

    Usage:
        # Manual hyperparameters
        python train_single_model.py --feature_set rfecv_all_nfeat_158_umap_count_loc.json \
                                    --model LightGBM \
                                    --params '{"objective": "huber", "learning_rate": 0.00922908558352886, "n_estimators": 4800, "num_leaves": 300, "max_depth": 33, "min_child_samples": 16, "subsample": 0.2177655313067815, "colsample_bytree": 0.2339444644166288, "reg_alpha": 2.517432528090951e-07, "reg_lambda": 9.360235437478908e-07}'

        # Load from MLflow run
        python train_single_model.py --feature_set rfecv_all_nfeat_158_umap_count_loc.json \
                                    --model LightGBM \
                                    --mlflow_run_id 

        # Use default hyperparameters
        python train_single_model.py --feature_set rfecv_all_nfeat_158_umap_count_loc.json \
                                    --model LightGBM \
                                    --use_defaults
    """

    parser = argparse.ArgumentParser(
        description="Train a single model without hyperparameter tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--feature_set",
        type=str,
        required=True,
        help="Path to feature set JSON file (e.g., 'feature_sets/rfecv_all_nfeat_158_umap_count_loc.json')",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["LightGBM", "XGBoost", "XGBoostQuantile", "RandomForest"],
        help="Model to train",
    )

    parser.add_argument(
        "--params",
        type=str,
        help='JSON string of hyperparameters (e.g., \'{"n_estimators": 1000, "max_depth": 10}\')',
    )

    parser.add_argument(
        "--mlflow_run_id", type=str, help="MLflow run ID to load hyperparameters from"
    )

    parser.add_argument(
        "--use_defaults", action="store_true", help="Use default hyperparameters"
    )

    parser.add_argument(
        "--scoring_metric",
        type=str,
        default="mape",
        choices=["mape", "rmse", "r2"],
        help="Scoring metric to use (default: mape)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config file (default: config/config.yaml)",
    )

    args = parser.parse_args()

    param_sources = [
        args.params is not None,
        args.mlflow_run_id is not None,
        args.use_defaults,
    ]
    if sum(param_sources) != 1:
        parser.error(
            "Exactly one of --params, --mlflow_run_id, or --use_defaults must be specified"
        )

    if not os.path.exists(args.feature_set):
        feature_set_path = os.path.join("feature_sets", args.feature_set)
        if not os.path.exists(feature_set_path):
            parser.error(f"Feature set file not found: {args.feature_set}")
        args.feature_set = feature_set_path

    config = load_config(args.config)

    if args.use_defaults:
        hyperparameters = get_default_hyperparameters(args.model)
        logger.info(f"Using default hyperparameters for {args.model}")
    elif args.mlflow_run_id:
        hyperparameters = load_hyperparameters_from_mlflow(args.mlflow_run_id)
        logger.info(f"Loaded hyperparameters from MLflow run {args.mlflow_run_id}")
    else:
        try:
            hyperparameters = json.loads(args.params)
            logger.info("Using provided hyperparameters")
        except json.JSONDecodeError as e:
            parser.error(f"Invalid JSON in --params: {e}")

    if not hyperparameters:
        parser.error(f"No hyperparameters found for model {args.model}")

    # Train the model
    try:
        train_single_model(
            model_name=args.model,
            hyperparameters=hyperparameters,
            feature_set_path=args.feature_set,
            config=config,
            scoring_metric=args.scoring_metric,
        )
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
