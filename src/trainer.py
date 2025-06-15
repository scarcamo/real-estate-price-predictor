import json
import logging
import os
import pickle
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

        # Get feature set path from the feature set name
        feature_set_path = f"feature_sets/{self.feature_set_name}.json"
        
        # Use centralized DataManager transformation
        logging.info(f"🔄 Loading and transforming data using centralized DataManager")
        X_train_transformed, X_test_transformed, feature_info = self.data_manager.get_transformed_data_from_feature_set(feature_set_path)
        
        logging.info(f"✅ Data transformation completed")
        logging.info(f"📊 Training data shape: {X_train_transformed.shape}")
        logging.info(f"📊 Test data shape: {X_test_transformed.shape}")
        logging.info(f"📋 Feature info: {feature_info}")
        
        # Get the actual features being used (may include image features)
        actual_features = X_train_transformed.columns.tolist()
        logging.info(f"📋 Using {len(actual_features)} features (may include image features)")

        def objective(trial):
            fold_scores = []
            for fold_idx, (train_idx, val_idx) in enumerate(cv_strategy.split(
                self.data_manager.X_train_full_raw, 
                self.data_manager.get_log_transformed_target()
            )):
                try:
                    fold_score = self._process_fold_optimized(
                        trial, fold_idx, train_idx, val_idx, model_name, model_config, 
                        actual_features, X_train_transformed
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

        current_completed_trials = len(study.trials)
        logging.info(f"DEBUG: Optuna study '{study.study_name}' loaded with {current_completed_trials} existing trials.")

        total_desired_trials = self.config['optuna']['n_trials']

        trials_to_run = max(0, total_desired_trials - current_completed_trials)
        logging.info(f"DEBUG: Study already has {current_completed_trials} trials. Aiming for {total_desired_trials} total trials. Will run {trials_to_run} new trials.")

        mlflow_callback = self._setup_mlflow_callback()
        

        if trials_to_run > 0:
            study.optimize(
                objective,
                n_trials=trials_to_run,
                n_jobs=-1,
                gc_after_trial=True,
            )
            
            sampler_path = study.user_attrs.get("sampler_path", f"optuna_samplers/{study.study_name}_sampler.pkl")
            pruner_path = study.user_attrs.get("pruner_path", f"optuna_pruners/{study.study_name}_pruner.pkl")
            self._save_sampler_and_pruner(study, sampler_path, pruner_path)
        else:
            logging.info(f"DEBUG: Study already has {current_completed_trials} trials, which meets or exceeds the target of {total_desired_trials}. No new trials will be run in this call.")

        tuning_duration = time.time() - start_time
        logging.info(f"Optuna hyperparameter tuning for {model_name} finished in {tuning_duration:.2f} seconds.")

        # Train final model with best parameters
        final_run_id = self._train_final_model(
            model_name,
            model_config,
            study.best_trial.params,
            study.best_trial.value,
            actual_features,
            tuning_duration,
            start_time,
            parent_run_id,
            X_train_transformed,
            X_test_transformed,
            feature_info
        )

        return final_run_id, study

    def _process_fold_optimized(self, trial, fold_idx, train_idx, val_idx, model_name, model_config, selected_feature_names, X_train_transformed):
        """Process a single cross-validation fold using pre-transformed data"""
        # Get data for this fold using pre-transformed data
        X_fold_train_selected = X_train_transformed.iloc[train_idx]
        X_fold_val_selected = X_train_transformed.iloc[val_idx]
        
        # Get corresponding target values
        y_fold_train_log = self.data_manager.get_log_transformed_target().iloc[train_idx]
        y_fold_val_log = self.data_manager.get_log_transformed_target().iloc[val_idx]

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
        """Setup Optuna study with sampler and pruner persistence"""
        scoring_metric = self.config["optuna"]["scoring_metric"]
        study_name = f"{model_name}_{self.feature_set_name}_{scoring_metric}_{self.experiment_version}_study"
        logging.info(f"Using Optuna study '{study_name}' with storage '{self.config['optuna']['study_db_path']}'")

        # Use absolute paths for stability
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        samplers_dir = os.path.join(project_root, "optuna_samplers")
        pruners_dir = os.path.join(project_root, "optuna_pruners")
        
        sampler_path = os.path.join(samplers_dir, f"{study_name}_sampler.pkl")
        pruner_path = os.path.join(pruners_dir, f"{study_name}_pruner.pkl")
        
        os.makedirs(samplers_dir, exist_ok=True)
        os.makedirs(pruners_dir, exist_ok=True)

        sampler = self._load_or_create_sampler(sampler_path)
        pruner = self._load_or_create_pruner(pruner_path)

        study = optuna.create_study(
            storage=self.config["optuna"]["study_db_path"],
            study_name=study_name,
            direction=self.config["optuna"]["direction"],
            load_if_exists=True,
            sampler=sampler,
            pruner=pruner,
        )

        self._save_sampler_and_pruner(study, sampler_path, pruner_path)

        study.set_user_attr("run_type", "optuna_run")
        study.set_user_attr("model_name", model_name)
        study.set_user_attr("feature_set_name", self.feature_set_name)
        study.set_user_attr("experiment_version", self.experiment_version)
        study.set_user_attr("scoring_metric", scoring_metric)
        study.set_user_attr("cv_folds", self.config["cv_folds"])
        study.set_user_attr("random_state", self.config["RANDOM_STATE"])
        study.set_user_attr("n_trials_optuna", self.config["optuna"]["n_trials"])
        study.set_user_attr("tuning_scoring_metric", self.config["optuna"]["scoring_metric"])
        study.set_user_attr("optuna_direction", self.config["optuna"]["direction"])
        study.set_user_attr("sampler_path", sampler_path)
        study.set_user_attr("pruner_path", pruner_path)

        return study

    def _load_or_create_sampler(self, sampler_path: str) -> optuna.samplers.BaseSampler:
        """Load existing sampler or create a new one"""
        if os.path.exists(sampler_path):
            try:
                logging.info(f"Loading existing sampler from {sampler_path}")
                with open(sampler_path, "rb") as f:
                    sampler = pickle.load(f)
                logging.info("Successfully loaded existing sampler for study resumption")
                return sampler
            except Exception as e:
                logging.warning(f"Failed to load sampler from {sampler_path}: {e}. Creating new sampler.")
        
        logging.info("Creating new TPESampler with seed for reproducibility")
        sampler = optuna.samplers.TPESampler(seed=self.config['RANDOM_STATE'])
        return sampler

    def _load_or_create_pruner(self, pruner_path: str) -> optuna.pruners.BasePruner:
        """Load existing pruner or create a new one"""
        if os.path.exists(pruner_path):
            try:
                logging.info(f"Loading existing pruner from {pruner_path}")
                with open(pruner_path, "rb") as f:
                    pruner = pickle.load(f)
                logging.info("Successfully loaded existing pruner for study resumption")
                return pruner
            except Exception as e:
                logging.warning(f"Failed to load pruner from {pruner_path}: {e}. Creating new pruner.")
        
        logging.info("Creating new MedianPruner")
        pruner = optuna.pruners.MedianPruner()
        return pruner

    def _save_sampler_and_pruner(self, study: optuna.Study, sampler_path: str, pruner_path: str) -> None:
        """Save the sampler and pruner for future resumption"""
        try:
            # Save sampler
            with open(sampler_path, "wb") as f:
                pickle.dump(study.sampler, f)
            logging.info(f"Sampler saved to {sampler_path} for future study resumption")
            
            # Save pruner
            with open(pruner_path, "wb") as f:
                pickle.dump(study.pruner, f)
            logging.info(f"Pruner saved to {pruner_path} for future study resumption")
            
        except Exception as e:
            logging.error(f"Failed to save sampler/pruner: {e}")

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
        X_train_transformed=None,
        X_test_transformed=None,
        feature_info=None,
    ) -> Optional[str]:
        """Train and log the final model using the best parameters"""
        logging.info(f"Training final model for {model_name} using best parameters...")

        # Train final model
        final_model_instance = model_config["model"]
        best_model_pipeline = build_model_pipeline(final_model_instance)
        
        final_params = best_params.copy()
        final_params[f"{self.model_prefix}n_jobs"] = -1
        logging.info("Set model n_jobs to -1 for final training.")

        best_model_pipeline.set_params(**final_params)
        
        best_model_pipeline.set_params(**best_params)
        best_model_pipeline.fit(X_train_transformed, self.data_manager.get_log_transformed_target())

        # Make predictions
        Y_predict_train_log = best_model_pipeline.predict(X_train_transformed)
        Y_predict_test_log = best_model_pipeline.predict(X_test_transformed)
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
                X_train_transformed,
                selected_feature_names,
                feature_info
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
        feature_info: Optional[Dict] = None,
    ) -> None:
        """Log all relevant information to MLflow"""
        mlflow.set_tag("run_type", "best_run")
        mlflow.set_tag("experiment_version", self.experiment_version)
        mlflow.log_param("model_name", model_name)
        mlflow.set_tag("feature_set", self.feature_set_name)
        
        # Log feature information from centralized approach
        if feature_info:
            mlflow.log_param("total_features", feature_info["total_features"])
            mlflow.log_param("tabular_features", feature_info["tabular_features"])
            mlflow.log_param("image_features", feature_info["image_features"])
            mlflow.log_param("selected_tabular_features", feature_info["selected_tabular_features"])
            mlflow.log_param("transformer_type", feature_info["transformer_type"])
            mlflow.log_param("should_include_images", feature_info["should_include_images"])
            
            # Log feature selection metadata
            for key, value in feature_info["feature_selection_metadata"].items():
                mlflow.log_param(f"fs_meta_{key}", value)
        else:
            # Fallback to old approach
            mlflow.log_param("num_selected_features", len(selected_feature_names))
        
        mlflow.log_params({k: v for k, v in best_params.items()})

        # Log configuration parameters
        mlflow.log_param("cv_folds", self.config["cv_folds"])
        mlflow.log_param("random_state", self.config["RANDOM_STATE"])
        mlflow.log_param("n_trials_optuna", self.config["optuna"]["n_trials"])
        mlflow.log_param("tuning_scoring_metric", self.config["optuna"]["scoring_metric"])
        mlflow.log_param("optuna_direction", self.config["optuna"]["direction"])

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