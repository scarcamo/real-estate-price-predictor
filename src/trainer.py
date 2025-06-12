import json
import logging
import os
import time
from typing import Dict, Any, Tuple, Optional

import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import optuna
import xgboost as xgb
import lightgbm as lgb
from optuna.integration import MLflowCallback
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
import pandas as pd

from src.model_factory import get_model_params, build_model_pipeline
from src.data_manager import DataManager

def evaluate_model(y_true, y_pred, prefix=""):
    """Evaluate model performance using multiple metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    metrics = {
        f"{prefix}_rmse": rmse,
        f"{prefix}_mape": mape,
        f"{prefix}_r2": r2,
    }
    logging.info(
        f"{prefix.capitalize()} Metrics: RMSE={rmse:.4f}, MAPE={mape:.4f}, R2={r2:.4f}"
    )
    return metrics

def get_selected_features(selected_features_filename_with_path: str) -> Tuple[list, dict]:
    """Load selected features and metadata from JSON file"""
    logging.info(f"\nLoading selected feature names from {selected_features_filename_with_path}...")
    selected_features_names = []
    feature_selection_metadata = {}
    
    try:
        with open(selected_features_filename_with_path, "r") as f:
            feature_data = json.load(f)
            selected_features_names = feature_data.get("selected_features", [])
            feature_selection_metadata = feature_data.get("metadata", {})
            
            if not selected_features_names:
                logging.warning(f"No 'selected_features' array found or array is empty in {selected_features_filename_with_path}.")
            if not feature_selection_metadata:
                logging.warning(f"No 'metadata' dictionary found or dictionary is empty in {selected_features_filename_with_path}.")
            else:
                logging.info(f"Loaded feature selection metadata: {feature_selection_metadata}")

    except FileNotFoundError:
        logging.error(
            f"Feature selection file not found at {selected_features_filename_with_path}. Cannot proceed with this feature set."
        )
        raise
    except (json.JSONDecodeError, ValueError) as e:
        logging.error(f"Error loading or validating feature selection metadata from {selected_features_filename_with_path}: {e}")
        raise

    return selected_features_names, feature_selection_metadata

class ModelTrainer:
    def __init__(self, config: Dict[str, Any], data_manager: DataManager, feature_set_name: str):
        self.config = config
        self.data_manager = data_manager
        self.model_prefix = "model__"
        self.feature_set_name = feature_set_name
        self.experiment_version = config.get("experiment_version", "v1")
        
    def train_evaluate_log(
        self,
        model_name: str,
        model_config: Dict[str, Any],
        selected_feature_names: list,
        cv_strategy: KFold,
        parent_run_id: Optional[str] = None,
        feature_selection_metadata: Optional[Dict] = None,
    ) -> Tuple[Optional[str], Optional[optuna.Study]]:
        """Train, evaluate and log a model with hyperparameter tuning"""
        logging.info(f"Tuning {model_name} with Optuna...")
        start_time = time.time()

        # Fit preprocessor once on all training data
        if feature_selection_metadata:
            apply_scaling = feature_selection_metadata.get("apply_scale_transform", self.config["APPLY_SCALE_TRANSFORM"])
            apply_pca = feature_selection_metadata.get("apply_pca_img_transform", self.config["APPLY_PCA_IMG_TRANSFORM"])
            n_pca_components = feature_selection_metadata.get("n_pca_components", self.config["N_PCA_COMPONENTS"])
            logging.info(f"Using feature selection metadata for transformer - "
                        f"scaling={apply_scaling}, pca={apply_pca}, n_components={n_pca_components}")
        else:
            apply_scaling = self.config["APPLY_SCALE_TRANSFORM"]
            apply_pca = self.config["APPLY_PCA_IMG_TRANSFORM"]
            n_pca_components = self.config["N_PCA_COMPONENTS"]
            logging.warning(f"No feature selection metadata available, using config defaults")

        # Create and fit transformer on all training data
        data_transformer = self.data_manager.create_data_transformer(
            apply_scaling=apply_scaling,
            apply_pca=apply_pca,
            n_pca_components=n_pca_components
        )
        data_transformer.fit(self.data_manager.X_train_full_raw.copy(), self.data_manager.get_log_transformed_target())

        # Get the valid feature names that exist after transformation
        sample_transformed = data_transformer.transform(self.data_manager.X_train_full_raw.head(1).copy())
        transformed_feature_names = data_transformer.get_feature_names_out()
        sample_df = pd.DataFrame(sample_transformed, columns=transformed_feature_names)
        
        valid_selected_features = [
            name for name in selected_feature_names
            if name in sample_df.columns
        ]
        
        missing_features = [
            name for name in selected_feature_names
            if name not in sample_df.columns
        ]
        
        if missing_features:
            logging.warning(f"Missing {len(missing_features)} features from selected list: {missing_features[:10]}{'...' if len(missing_features) > 10 else ''}")
            logging.warning(f"This is likely due to OneHotEncoder min_frequency filtering or rare categories. Continuing with {len(valid_selected_features)} available features.")
            raise ValueError(f"Missing {len(missing_features)} features from selected list: {missing_features[:10]}{'...' if len(missing_features) > 10 else ''}")
        
        if not valid_selected_features:
            logging.error(f"CRITICAL ERROR - No valid selected features remaining after transformation!")
            return None, None

        def objective(trial):
            fold_scores = []
            for fold_idx, (train_idx, val_idx) in enumerate(cv_strategy.split(
                self.data_manager.X_train_full_raw, 
                self.data_manager.get_log_transformed_target()
            )):
                try:
                    fold_score = self._process_fold(
                        trial, fold_idx, train_idx, val_idx, model_name, model_config, valid_selected_features, data_transformer
                    )
                    if fold_score is not None:
                        fold_scores.append(fold_score)
                except optuna.TrialPruned:
                    logging.warning(f"Trial pruned for fold {fold_idx}")
                    return float("inf") if self.config["optuna"]["direction"] == "minimize" else float("-inf")
                except Exception as e:
                    logging.error(f"Trial fold failed for {model_name}: {e}", exc_info=True)
                    return float("inf") if self.config["optuna"]["direction"] == "minimize" else float("-inf")

            if not fold_scores:
                logging.warning(f"No fold scores recorded for trial in {model_name}. Returning non-optimal score.")
                return float("inf") if self.config["optuna"]["direction"] == "minimize" else float("-inf")
            
            return np.mean(fold_scores)

        # Setup and run Optuna study
        study = self._setup_optuna_study(model_name)
        mlflow_callback = self._setup_mlflow_callback()
        
        study.optimize(
            objective,
            n_trials=self.config["optuna"]["n_trials"],
            # callbacks=[mlflow_callback],
            n_jobs=-1,
            gc_after_trial=True,
        )
        
        tuning_duration = time.time() - start_time
        logging.info(f"Optuna hyperparameter tuning for {model_name} finished in {tuning_duration:.2f} seconds.")

        # Train final model with best parameters
        final_run_id = self._train_final_model(
            model_name,
            model_config,
            study.best_trial.params,
            study.best_trial.value,
            valid_selected_features,
            tuning_duration,
            start_time,
            parent_run_id,
            data_transformer
        )

        return final_run_id, study

    def _process_fold(self, trial, fold_idx, train_idx, val_idx, model_name, model_config, selected_feature_names, data_transformer):
        """Process a single cross-validation fold"""
        # Get data for this fold
        X_fold_train_raw = self.data_manager.X_train_full_raw.iloc[train_idx]
        y_fold_train_log = self.data_manager.get_log_transformed_target().iloc[train_idx]
        X_fold_val_raw = self.data_manager.X_train_full_raw.iloc[val_idx]
        y_fold_val_log = self.data_manager.get_log_transformed_target().iloc[val_idx]

        # Transform fold data using the pre-fitted transformer
        X_fold_train_processed_np = data_transformer.transform(X_fold_train_raw.copy())
        X_fold_val_processed_np = data_transformer.transform(X_fold_val_raw.copy())
        
        # Convert numpy arrays to DataFrames with proper column names
        try:
            transformed_feature_names = data_transformer.get_feature_names_out()
            X_fold_train_processed_df = pd.DataFrame(
                X_fold_train_processed_np,
                columns=transformed_feature_names,
                index=X_fold_train_raw.index,
            )
            X_fold_val_processed_df = pd.DataFrame(
                X_fold_val_processed_np,
                columns=transformed_feature_names,
                index=X_fold_val_raw.index,
            )
        except Exception as e:
            logging.error(f"Fold {fold_idx}: Error getting feature names from transformer: {e}. Skipping fold.")
            return None

        # Use the pre-validated selected features
        X_fold_train_selected = X_fold_train_processed_df[selected_feature_names]
        X_fold_val_selected = X_fold_val_processed_df[selected_feature_names]

        # Get parameters and train model
        params = get_model_params(model_name, trial)
        trial_model_instance = model_config["model"]
        model_pipeline = build_model_pipeline(trial_model_instance)
        model_pipeline.set_params(**params)

        model_pipeline.fit(X_fold_train_selected, y_fold_train_log)
        y_pred_val_log = model_pipeline.predict(X_fold_val_selected)
        
        # Calculate metrics
        y_pred_val_orig = np.exp(y_pred_val_log)
        y_fold_val_orig = np.exp(y_fold_val_log)

        if self.config["optuna"]["scoring_metric"] == "mape":
            score = mean_absolute_percentage_error(y_fold_val_orig, y_pred_val_orig)
        elif self.config["optuna"]["scoring_metric"] == "rmse":
            score = np.sqrt(mean_squared_error(y_fold_val_orig, y_pred_val_orig))
        else:  # Default to mape
            score = mean_absolute_percentage_error(y_fold_val_orig, y_pred_val_orig)

        trial.report(score, fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()
            
        return score

    def _setup_optuna_study(self, model_name: str) -> optuna.Study:
        """Setup Optuna study"""
        scoring_metric = self.config["optuna"]["scoring_metric"]
        study_name = f"{model_name}_{self.feature_set_name}_{scoring_metric}_{self.experiment_version}_study"
        logging.info(f"Using Optuna study '{study_name}' with storage '{self.config['optuna']['study_db_path']}'")

        study = optuna.create_study(
            storage=self.config["optuna"]["study_db_path"],
            study_name=study_name,
            direction=self.config["optuna"]["direction"],
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(),
        )

        study.set_user_attr("run_type", "optuna_run")
        study.set_user_attr("model_name", model_name)
        study.set_user_attr("feature_set_name", self.feature_set_name)
        study.set_user_attr("experiment_version", self.experiment_version)
        study.set_user_attr("scoring_metric", scoring_metric)
        study.set_user_attr("cv_folds", self.config["CV_FOLDS"])
        study.set_user_attr("random_state", self.config["RANDOM_STATE"])
        study.set_user_attr("n_trials_optuna", self.config["optuna"]["n_trials"])
        study.set_user_attr("tuning_scoring_metric", self.config["optuna"]["scoring_metric"])
        study.set_user_attr("optuna_direction", self.config["optuna"]["direction"])

        return study

    def _setup_mlflow_callback(self) -> MLflowCallback:
        """Setup MLflow callback for Optuna"""
        return MLflowCallback(
            tracking_uri=mlflow.get_tracking_uri(),
            create_experiment=False,
            metric_name=self.config["MLFLOW_CALLBACK_METRIC_NAME"],
            tag_study_user_attrs=True,
            tag_trial_user_attrs=True,
            mlflow_kwargs={"nested": True},
        )

    def _train_final_model(
        self,
        model_name: str,
        model_config: Dict[str, Any],
        best_params: Dict[str, Any],
        best_score: float,
        selected_feature_names: list,
        tuning_duration: float,
        start_time: float,
        parent_run_id: Optional[str] = None,
        data_transformer=None,
    ) -> Optional[str]:
        """Train and log the final model using the best parameters"""
        logging.info(f"Training final model for {model_name} using best parameters...")

        # Transform data using the pre-fitted transformer
        X_train_processed_final_df, X_test_processed_final_df = self.data_manager.get_transformed_data(data_transformer)

        # Use the pre-validated selected features
        X_train_selected_final = X_train_processed_final_df[selected_feature_names]
        X_test_selected_final = X_test_processed_final_df[selected_feature_names]

        # Train final model
        final_model_instance = model_config["model"]
        best_model_pipeline = build_model_pipeline(final_model_instance)
        
        final_params = best_params.copy()
        final_params[f"{self.model_prefix}n_jobs"] = -1
        logging.info("Set model n_jobs to -1 for final training.")

        best_model_pipeline.set_params(**final_params)
        
        best_model_pipeline.set_params(**best_params)
        best_model_pipeline.fit(X_train_selected_final, self.data_manager.get_log_transformed_target())

        # Make predictions
        Y_predict_train_log = best_model_pipeline.predict(X_train_selected_final)
        Y_predict_test_log = best_model_pipeline.predict(X_test_selected_final)
        Y_predict_train_orig = np.exp(Y_predict_train_log)
        Y_predict_test_orig = np.exp(Y_predict_test_log)

        # Calculate metrics
        train_metrics = evaluate_model(
            np.exp(self.data_manager.get_log_transformed_target()),
            Y_predict_train_orig,
            prefix="train_final"
        )
        test_metrics = evaluate_model(
            self.data_manager.get_test_target(),
            Y_predict_test_orig,
            prefix="test_final"
        )

        # Log to MLflow
        run_tags = {}
        if parent_run_id:
            run_tags["mlflow.parentRunId"] = parent_run_id

        with mlflow.start_run(
            run_name=f"{model_name}_{self.feature_set_name}_{self.config['optuna']['scoring_metric']}_{self.experiment_version}_BEST",
            tags=run_tags,
            nested=True,
        ) as run:
            self._log_to_mlflow(
                run,
                model_name,
                best_params,
                best_score,
                train_metrics,
                test_metrics,
                tuning_duration,
                time.time() - start_time,
                best_model_pipeline,
                X_train_selected_final,
                selected_feature_names,
                None
            )
            return run.info.run_id

    def _log_to_mlflow(
        self,
        run,
        model_name: str,
        best_params: Dict[str, Any],
        best_score: float,
        train_metrics: Dict[str, float],
        test_metrics: Dict[str, float],
        tuning_duration: float,
        total_duration: float,
        model_pipeline,
        input_example_data,
        selected_feature_names: list,
        feature_selection_metadata: Optional[Dict] = None,
    ) -> None:
        """Log all relevant information to MLflow"""
        mlflow.set_tag("run_type", "best_run")
        mlflow.set_tag("experiment_version", self.experiment_version)
        mlflow.log_param("model_name", model_name)
        mlflow.set_tag("feature_set", self.feature_set_name)
        mlflow.log_param("num_selected_features", len(selected_feature_names))
        mlflow.log_params({k: v for k, v in best_params.items()})

        # Log configuration parameters
        mlflow.log_param("cv_folds", self.config["CV_FOLDS"])
        mlflow.log_param("random_state", self.config["RANDOM_STATE"])
        mlflow.log_param("n_trials_optuna", self.config["optuna"]["n_trials"])
        mlflow.log_param("tuning_scoring_metric", self.config["optuna"]["scoring_metric"])
        mlflow.log_param("optuna_direction", self.config["optuna"]["direction"])

        # Log feature selection metadata
        if feature_selection_metadata:
            for key, value in feature_selection_metadata.items():
                if key in ('rfe_step_size', 'n_estimators', 'method'):
                    mlflow.log_param(f"fs_{key}", value)
                else:
                    mlflow.log_param(f"{key}", value)

        # Log metrics
        mlflow.log_metric(self.config["MLFLOW_CALLBACK_METRIC_NAME"], best_score)
        mlflow.log_metric(f"best_optuna_cv_{self.config['optuna']['scoring_metric']}", best_score)
        mlflow.log_metrics(train_metrics)
        mlflow.log_metrics(test_metrics)
        mlflow.log_metric("tuning_duration_sec", tuning_duration)
        mlflow.log_metric("total_pipeline_duration_sec", total_duration)

        # Log model
        model_step_in_pipeline = model_pipeline.named_steps["model"]
        try:
            if isinstance(model_step_in_pipeline, xgb.XGBModel):
                mlflow.xgboost.log_model(
                    model_step_in_pipeline,
                    f"{model_name}_model",
                    input_example=input_example_data,
                    pip_requirements=None, 
                    extra_pip_requirements=None,
                )
            elif isinstance(model_step_in_pipeline, lgb.LGBMModel):
                mlflow.lightgbm.log_model(
                    model_step_in_pipeline,
                    f"{model_name}_model",
                    input_example=input_example_data,
                    pip_requirements=None,  # Disable automatic pip requirement detection
                    extra_pip_requirements=None,
                )
            else:
                mlflow.sklearn.log_model(
                    model_pipeline,
                    f"{model_name}_pipeline",
                    input_example=input_example_data,
                    pip_requirements=None,  # Disable automatic pip requirement detection
                    extra_pip_requirements=None,
                )
        except Exception as e:
            logging.error(f"Error logging model {model_name}: {e}")

        # Log feature list
        feature_list_path = "final_selected_features_list.txt"
        with open(feature_list_path, "w") as f:
            for feature in selected_feature_names:
                f.write(f"{feature}\n")
        mlflow.log_artifact(feature_list_path, artifact_path="feature_info")
        if os.path.exists(feature_list_path):
            os.remove(feature_list_path) 