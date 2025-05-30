# %% Imports
import json
import logging
import os
import random
import time
import warnings
from typing import Tuple

import lightgbm as lgb

# MLflow import
import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from optuna.integration import MLflowCallback
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from config import load_config
from preprocessor import create_data_transformer_pipeline
from split_data import get_train_test_data, get_train_test_img

logging.basicConfig(
    level=logging.INFO, 
    handlers=[logging.StreamHandler()],
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.getLogger("lightgbm").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
pd.set_option("display.float_format", "{:.4f}".format)

# %% Configuration Constants
config = load_config()

TARGET_VARIABLE = config.get("TARGET_VARIABLE")
RANDOM_STATE = config.get("RANDOM_STATE")
N_TRIALS_OPTUNA = config.get("N_TRIALS_OPTUNA")
CV_FOLDS = config.get("CV_FOLDS")
TUNING_SCORING_METRIC = config.get("TUNING_SCORING_METRIC")
OPTUNA_DIRECTION = config.get("OPTUNA_DIRECTION")
MLFLOW_CALLBACK_METRIC_NAME = f"cv_{TUNING_SCORING_METRIC}"

# Defaults values and can be overridden by feature JSON metadata
APPLY_SCALE_TRANSFORM = config.get("APPLY_SCALE_TRANSFORM")
APPLY_PCA_IMG_TRANSFORM = config.get("APPLY_PCA_IMG_TRANSFORM")
N_PCA_COMPONENTS = config.get("N_PCA_COMPONENTS")

# used for feature selection, log the parameters
N_ESTIMATORS = config.get("N_ESTIMATORS")
RFE_STEP_SIZE = config.get("RFE_STEP_SIZE")

optuna_dir = config.get("optuna_dir")
db_file = "tuning.db"
STUDY_DB_PATH = f"sqlite:///{os.path.join(optuna_dir, db_file)}"
ARTIFACT_PATH = config.get("ARTIFACT_PATH")
SELECTED_FEATURES_FOLDER = config.get("SELECTED_FEATURES_DIR")

SELECTED_FEATURES_FILENAME = "rfecv_all_nfeat_44_pca_scaled_count"

SELECTED_FEATURES_PATH = f"{SELECTED_FEATURES_FILENAME}.json"

os.makedirs(ARTIFACT_PATH, exist_ok=True)
os.makedirs(optuna_dir, exist_ok=True)


random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def build_model_pipeline(model):
    pipeline = Pipeline([("model", model)])
    return pipeline


def evaluate_model(y_true, y_pred, prefix=""):
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


# %% Model Definitions and Hyperparameter Spaces
MODEL_PREFIX = "model__"


def get_model_configs():
    """Returns a dictionary of models and their hyperparameter search spaces."""
    configs = {
        "LightGBM": {
            "model": lgb.LGBMRegressor(
                random_state=RANDOM_STATE, 
                objective="regression",
                metric="mape",
                n_jobs=-1,
                verbose=-1,
            ),
        },
        "XGBoost": {
            "model": xgb.XGBRegressor(
                random_state=RANDOM_STATE, 
                tree_method="hist",
                objective="reg:squarederror",
                n_jobs=-1,
            ),
        },
        "XGBoostQuantile": {
            "model": xgb.XGBRegressor(
                random_state=RANDOM_STATE, 
                tree_method="hist",
                objective="reg:quantileerror",
                quantile_alpha=0.5,
                n_jobs=-1,
            ),
        },
        "RandomForest": {
            "model": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1), 
        },
    }
    return configs


# %% Main Training Function


def get_model_params(model_name, trial):
    params = {}
    if model_name == "DecisionTree":
        params[f"{MODEL_PREFIX}max_depth"] = trial.suggest_int(
            f"{MODEL_PREFIX}max_depth", 5, 50
        )
        params[f"{MODEL_PREFIX}min_samples_split"] = trial.suggest_int(
            f"{MODEL_PREFIX}min_samples_split", 5, 101
        )
        params[f"{MODEL_PREFIX}min_samples_leaf"] = trial.suggest_int(
            f"{MODEL_PREFIX}min_samples_leaf", 5, 51
        )
        params[f"{MODEL_PREFIX}max_features"] = trial.suggest_categorical(
            f"{MODEL_PREFIX}max_features", ["sqrt", "log2", None]
        )
        params[f"{MODEL_PREFIX}ccp_alpha"] = trial.suggest_float(
            f"{MODEL_PREFIX}ccp_alpha", 0.0, 0.05
        )

    elif model_name == "RandomForest":
        params[f"{MODEL_PREFIX}n_estimators"] = trial.suggest_int(
            f"{MODEL_PREFIX}n_estimators", 100, 1000, step=50
        )
        params[f"{MODEL_PREFIX}max_depth"] = trial.suggest_int(
            f"{MODEL_PREFIX}max_depth", 5, 20
        )
        params[f"{MODEL_PREFIX}min_samples_split"] = trial.suggest_int(
            f"{MODEL_PREFIX}min_samples_split", 5, 50
        )
        params[f"{MODEL_PREFIX}min_samples_leaf"] = trial.suggest_int(
            f"{MODEL_PREFIX}min_samples_leaf", 5, 50
        )
        params[f"{MODEL_PREFIX}max_features"] = trial.suggest_categorical(
            f"{MODEL_PREFIX}max_features", ["sqrt", "log2", 0.5, 0.7, 1.0]
        )
        params[f"{MODEL_PREFIX}bootstrap"] = trial.suggest_categorical(
            f"{MODEL_PREFIX}bootstrap", [True, False]
        )

    elif model_name in ["XGBoost", "XGBoostQuantile"]:
        params[f"{MODEL_PREFIX}max_depth"] = trial.suggest_int(
            f"{MODEL_PREFIX}max_depth",
            3,
            8,
        )
        params[f"{MODEL_PREFIX}learning_rate"] = trial.suggest_float(
            f"{MODEL_PREFIX}learning_rate", 0.01, 0.3, log=True
        )
        params[f"{MODEL_PREFIX}n_estimators"] = trial.suggest_int(
            f"{MODEL_PREFIX}n_estimators", 100, 1000, step=50
        )
        params[f"{MODEL_PREFIX}min_child_weight"] = trial.suggest_int(
            f"{MODEL_PREFIX}min_child_weight",
            5,
            50,
        )
        params[f"{MODEL_PREFIX}subsample"] = trial.suggest_float(
            f"{MODEL_PREFIX}subsample", 0.6, 1.0
        )
        params[f"{MODEL_PREFIX}colsample_bytree"] = trial.suggest_float(
            f"{MODEL_PREFIX}colsample_bytree", 0.6, 1.0
        )
        params[f"{MODEL_PREFIX}gamma"] = trial.suggest_float(
            f"{MODEL_PREFIX}gamma",
            0,
            5.0,
        )
        params[f"{MODEL_PREFIX}reg_lambda"] = trial.suggest_float(
            f"{MODEL_PREFIX}reg_lambda", 1e-2, 50.0, log=True
        )
        params[f"{MODEL_PREFIX}reg_alpha"] = trial.suggest_float(
            f"{MODEL_PREFIX}reg_alpha", 1e-2, 50.0, log=True
        )

    elif model_name == "LightGBM":
        params[f"{MODEL_PREFIX}learning_rate"] = trial.suggest_float(
            f"{MODEL_PREFIX}learning_rate", 0.01, 0.3, log=True
        )
        params[f"{MODEL_PREFIX}n_estimators"] = trial.suggest_int(
            f"{MODEL_PREFIX}n_estimators", 400, 1500, step=50
        )
        params[f"{MODEL_PREFIX}num_leaves"] = trial.suggest_int(
            f"{MODEL_PREFIX}num_leaves", 20, 100
        )
        params[f"{MODEL_PREFIX}max_depth"] = trial.suggest_int(
            f"{MODEL_PREFIX}max_depth", 5, 30
        )
        params[f"{MODEL_PREFIX}min_child_samples"] = trial.suggest_int(
            f"{MODEL_PREFIX}min_child_samples", 5, 51
        )
        params[f"{MODEL_PREFIX}subsample"] = trial.suggest_float(
            f"{MODEL_PREFIX}subsample", 0.6, 1.0
        )
        params[f"{MODEL_PREFIX}colsample_bytree"] = trial.suggest_float(
            f"{MODEL_PREFIX}colsample_bytree", 0.5, 1.0
        )
        params[f"{MODEL_PREFIX}reg_alpha"] = trial.suggest_float(
            f"{MODEL_PREFIX}reg_alpha", 1e-2, 50.0, log=True
        )
        params[f"{MODEL_PREFIX}reg_lambda"] = trial.suggest_float(
            f"{MODEL_PREFIX}reg_lambda", 1e-2, 50.0, log=True
        )

    elif model_name == "AdaBoost":
        params[f"{MODEL_PREFIX}n_estimators"] = trial.suggest_int(
            f"{MODEL_PREFIX}n_estimators", 50, 500, step=25
        )
        params[f"{MODEL_PREFIX}learning_rate"] = trial.suggest_float(
            f"{MODEL_PREFIX}learning_rate", 0.01, 1.0, log=True
        )
        params[f"{MODEL_PREFIX}estimator__max_depth"] = trial.suggest_int(
            f"{MODEL_PREFIX}estimator__max_depth", 3, 11
        )
        params[f"{MODEL_PREFIX}estimator__min_samples_split"] = trial.suggest_int(
            f"{MODEL_PREFIX}estimator__min_samples_split", 5, 30
        )
        params[f"{MODEL_PREFIX}estimator__min_samples_leaf"] = trial.suggest_int(
            f"{MODEL_PREFIX}estimator__min_samples_leaf", 5, 30
        )
    elif model_name in ["Lasso", "Ridge"]:
        params[f"{MODEL_PREFIX}alpha"] = trial.suggest_float(
            f"{MODEL_PREFIX}alpha", 1e-4, 10.0, log=True
        )
    elif model_name == "ElasticNet":
        params[f"{MODEL_PREFIX}alpha"] = trial.suggest_float(
            f"{MODEL_PREFIX}alpha", 1e-4, 10.0, log=True
        )
        params[f"{MODEL_PREFIX}l1_ratio"] = trial.suggest_float(
            f"{MODEL_PREFIX}l1_ratio", 0.05, 0.95
        )

    return params


def train_evaluate_log(
    model_name,
    model_config,
    X_train_raw_full,
    Y_train_log,
    X_test_raw_full,
    Y_test,
    numeric_cols_original,
    categorical_cols_original,
    img_cols_original,
    district_col_name_original,
    outlier_col_name_original,
    selected_feature_names,
    cv_strategy,
    parent_run_id=None,
    feature_selection_metadata=None,
):
    logging.info(f"Tuning {model_name} with Optuna using feature set: {SELECTED_FEATURES_PATH}")
    start_time = time.time()

    # APPLY_SCALE_TRANSFORM and APPLY_PCA_IMG_TRANSFORM should have been updated in the main block
    def objective(trial):
        fold_scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(
            cv_strategy.split(X_train_raw_full, Y_train_log)
        ):
            X_fold_train_raw = X_train_raw_full.iloc[train_idx]
            y_fold_train_log = Y_train_log.iloc[train_idx]
            X_fold_val_raw = X_train_raw_full.iloc[val_idx]
            y_fold_val_log = Y_train_log.iloc[val_idx]

            current_data_transformer = create_data_transformer_pipeline(
                numeric_cols=numeric_cols_original,
                categorical_cols=categorical_cols_original,
                img_feature_cols=img_cols_original,
                district_group_col=district_col_name_original,
                outlier_indicator_col=outlier_col_name_original,
                apply_scaling_and_transform=APPLY_SCALE_TRANSFORM, 
                apply_pca=APPLY_PCA_IMG_TRANSFORM, 
                n_pca_components=N_PCA_COMPONENTS
            )
            current_data_transformer.fit(X_fold_train_raw.copy(), y_fold_train_log)
            X_fold_train_processed_np = current_data_transformer.transform(
                X_fold_train_raw.copy()
            )
            X_fold_val_processed_np = current_data_transformer.transform(
                X_fold_val_raw.copy()
            )

            try:
                transformed_feature_names_fold = (
                    current_data_transformer.get_feature_names_out()
                )
                X_fold_train_processed_df = pd.DataFrame(
                    X_fold_train_processed_np,
                    columns=transformed_feature_names_fold,
                    index=X_fold_train_raw.index,
                )
                X_fold_val_processed_df = pd.DataFrame(
                    X_fold_val_processed_np,
                    columns=transformed_feature_names_fold,
                    index=X_fold_val_raw.index,
                )
            except Exception as e:
                logging.error(
                    f"Fold {fold_idx}: Error getting feature names from transformer: {e}. Skipping fold."
                )
                continue

            valid_selected_names_fold = [
                name
                for name in selected_feature_names
                if name in X_fold_train_processed_df.columns
            ]
            if len(valid_selected_names_fold) != len(selected_feature_names):
                missing_features = set(selected_feature_names) - set(valid_selected_names_fold)
                logging.warning(f"Fold {fold_idx}: Some selected features not in transformed data. Missing: {missing_features}")
                if not valid_selected_names_fold:
                    logging.error(f"Fold {fold_idx}: No valid selected features after name validation. Skipping.")
                    continue 


            if not valid_selected_names_fold:
                logging.warning(
                    f"Fold {fold_idx}: No valid selected features remaining after transformation. Skipping fold."
                )
                continue

            X_fold_train_selected = X_fold_train_processed_df[valid_selected_names_fold]
            X_fold_val_selected = X_fold_val_processed_df[valid_selected_names_fold]

            params = get_model_params(model_name, trial)
            trial_model_instance = model_config["model"]
            model_pipeline = build_model_pipeline(trial_model_instance)
            model_pipeline.set_params(**params)

            try:
                model_pipeline.fit(X_fold_train_selected, y_fold_train_log)
                y_pred_val_log = model_pipeline.predict(X_fold_val_selected)
                y_pred_val_orig = np.exp(y_pred_val_log)
                y_fold_val_orig = np.exp(y_fold_val_log)

                if TUNING_SCORING_METRIC == "mape":
                    score = mean_absolute_percentage_error(
                        y_fold_val_orig, y_pred_val_orig
                    )
                elif TUNING_SCORING_METRIC == "rmse":
                    score = np.sqrt(
                        mean_squared_error(y_fold_val_orig, y_pred_val_orig)
                    )
                else: # Default to mape
                    score = mean_absolute_percentage_error(
                        y_fold_val_orig, y_pred_val_orig
                    )

                trial.report(score, fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                fold_scores.append(score)

            except optuna.TrialPruned:
                logging.warning(
                    f"Trial pruned for fold {fold_idx}. With score {score:.4f}"
                )
                return float("inf") if OPTUNA_DIRECTION == "minimize" else float("-inf")
            except Exception as e:
                logging.error(f"Trial fold failed for {model_name}: {e}", exc_info=True)
                return float("inf") if OPTUNA_DIRECTION == "minimize" else float("-inf")

        if not fold_scores:
            logging.warning(f"No fold scores recorded for trial in {model_name}. Returning non-optimal score.")
            return float("inf") if OPTUNA_DIRECTION == "minimize" else float("-inf")
        return np.mean(fold_scores)

    # Optuna Study Setup
    study_name = f"{model_name}_{SELECTED_FEATURES_FILENAME}_study"
    logging.info(f"Using Optuna study '{study_name}' with storage '{STUDY_DB_PATH}'") 

    mlflow_callback = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(),
        create_experiment=False, 
        metric_name=MLFLOW_CALLBACK_METRIC_NAME,
        tag_study_user_attrs=True,
        tag_trial_user_attrs=True,
        mlflow_kwargs={"nested": True},
    )

    study = optuna.create_study(
        storage=STUDY_DB_PATH,
        study_name=study_name,
        direction=OPTUNA_DIRECTION,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(),
    )

    study.set_user_attr("feature_set_name", SELECTED_FEATURES_FILENAME) 

    if feature_selection_metadata:
        for key, value in feature_selection_metadata.items():
            if key in ('rfe_step_size', 'n_estimators', 'method'):
                mlflow.log_param(f"fs_{key}", value)
            else:
                mlflow.log_param(f"{key}", value)


    logging.info(f"Starting Optuna optimization for {model_name} with {N_TRIALS_OPTUNA} trials...") 
    optuna.logging.set_verbosity(optuna.logging.WARNING) 
    study.optimize(
        objective,
        n_trials=N_TRIALS_OPTUNA,
        callbacks=[mlflow_callback],
        n_jobs=1,
        gc_after_trial=True,
    )
    tuning_duration = time.time() - start_time
    logging.info(f"Optuna hyperparameter tuning for {model_name} finished in {tuning_duration:.2f} seconds.") 

    best_trial = study.best_trial
    best_params_optuna = best_trial.params
    best_score_optuna = best_trial.value
    logging.info(f"Best Optuna Trial Score for {model_name} ({TUNING_SCORING_METRIC}): {best_score_optuna:.4f}") 
    logging.info(f"Best Hyperparameters for {model_name} found by Optuna: {best_params_optuna}") 

    logging.info(f"Training final model for {model_name} using best parameters...") 

    final_data_transformer = create_data_transformer_pipeline(
        numeric_cols=numeric_cols_original,
        categorical_cols=categorical_cols_original,
        img_feature_cols=img_cols_original,
        district_group_col=district_col_name_original,
        outlier_indicator_col=outlier_col_name_original,
        apply_scaling_and_transform=APPLY_SCALE_TRANSFORM, 
        apply_pca=APPLY_PCA_IMG_TRANSFORM, 
        n_pca_components=N_PCA_COMPONENTS
    )
    final_data_transformer.fit(X_train_raw_full.copy(), Y_train_log)

    X_train_processed_final_np = final_data_transformer.transform(
        X_train_raw_full.copy()
    )
    X_test_processed_final_np = final_data_transformer.transform(X_test_raw_full.copy())

    try:
        final_transformed_feature_names = final_data_transformer.get_feature_names_out()
        X_train_processed_final_df = pd.DataFrame(
            X_train_processed_final_np,
            columns=final_transformed_feature_names,
            index=X_train_raw_full.index,
        )
        X_test_processed_final_df = pd.DataFrame(
            X_test_processed_final_np,
            columns=final_transformed_feature_names,
            index=X_test_raw_full.index,
        )
    except Exception as e:
        logging.error(f"Final model for {model_name}: Error getting feature names from transformer: {e}") 
        return None, None

    final_valid_selected_names = [
        name
        for name in selected_feature_names
        if name in X_train_processed_final_df.columns
    ]
    if not final_valid_selected_names:
        logging.error(f"Error: No valid selected features for the final model {model_name}.") 
        return None, None
    
    if len(final_valid_selected_names) != len(selected_feature_names):
        missing_final = set(selected_feature_names) - set(final_valid_selected_names)
        logging.warning(f"Final model {model_name}: Some selected features not in final transformed data. Missing: {missing_final}")


    X_train_selected_final = X_train_processed_final_df[final_valid_selected_names]
    X_test_selected_final = X_test_processed_final_df[final_valid_selected_names]

    final_model_instance = model_config["model"]
    best_model_pipeline = build_model_pipeline(final_model_instance)
    best_model_pipeline.set_params(**best_params_optuna)
    best_model_pipeline.fit(X_train_selected_final, Y_train_log)

    logging.info(f"Final model training for {model_name} complete.") 

    Y_predict_train_log = best_model_pipeline.predict(X_train_selected_final)
    Y_predict_test_log = best_model_pipeline.predict(X_test_selected_final)
    Y_predict_train_orig = np.exp(Y_predict_train_log)
    Y_predict_test_orig = np.exp(Y_predict_test_log)
    Y_train_orig = np.exp(Y_train_log)

    train_metrics = evaluate_model(
        Y_train_orig, Y_predict_train_orig, prefix="train_final"
    )
    test_metrics = evaluate_model(Y_test, Y_predict_test_orig, prefix="test_final")
    total_duration = time.time() - start_time

    run_tags = {}
    if parent_run_id:
        run_tags["mlflow.parentRunId"] = parent_run_id

    with mlflow.start_run(
        run_name=f"{model_name}_{SELECTED_FEATURES_FILENAME}_BEST", 
        tags=run_tags,
        nested=True,
    ) as run:
        final_run_id_logged = run.info.run_id
        mlflow.set_tag("run_type", "best_run")

        mlflow.log_param("model_name", model_name)
        mlflow.set_tag("feature_set", SELECTED_FEATURES_FILENAME) 
        mlflow.log_param("num_selected_features", len(final_valid_selected_names))
        mlflow.log_params({k: v for k, v in best_params_optuna.items()})

        mlflow.log_param("cv_folds", CV_FOLDS)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("n_trials_optuna", N_TRIALS_OPTUNA)
        mlflow.log_param("tuning_scoring_metric", TUNING_SCORING_METRIC)
        mlflow.log_param("optuna_direction", OPTUNA_DIRECTION)
    

        # Log metadata from the feature selection JSON file
        if feature_selection_metadata:
            for key, value in feature_selection_metadata.items():
                if key in ('rfe_step_size', 'n_estimators', 'method'):
                    mlflow.log_param(f"fs_{key}", value)
                else:
                    mlflow.log_param(f"{key}", value)

        mlflow.log_metric(MLFLOW_CALLBACK_METRIC_NAME, best_score_optuna) 
        mlflow.log_metric(f"best_optuna_cv_{TUNING_SCORING_METRIC}", best_score_optuna)
        mlflow.log_metrics(train_metrics)
        mlflow.log_metrics(test_metrics)
        mlflow.log_metric("tuning_duration_sec", tuning_duration)
        mlflow.log_metric("total_pipeline_duration_sec", total_duration) 

        input_example_data = X_train_selected_final.head(5)
        model_step_in_pipeline = best_model_pipeline.named_steps["model"]

        try:
            if isinstance(model_step_in_pipeline, xgb.XGBModel):
                mlflow.xgboost.log_model(
                    model_step_in_pipeline,
                    f"{model_name}_model",
                    input_example=input_example_data,
                )
            elif isinstance(model_step_in_pipeline, lgb.LGBMModel):
                mlflow.lightgbm.log_model(
                    model_step_in_pipeline,
                    f"{model_name}_model",
                    input_example=input_example_data,
                )
            else:
                mlflow.sklearn.log_model(
                    best_model_pipeline, 
                    f"{model_name}_pipeline",
                    input_example=input_example_data,
                )
        except Exception as e:
            logging.error(f"Error logging model {model_name}: {e}")


        feature_list_path = "final_selected_features_list.txt"
        with open(feature_list_path, "w") as f:
            for feature in final_valid_selected_names:
                f.write(f"{feature}\n")
        mlflow.log_artifact(feature_list_path, artifact_path="feature_info")
        if os.path.exists(feature_list_path): 
            os.remove(feature_list_path)

    return final_run_id_logged, study


def get_selected_features(selected_features_filename_with_path) -> Tuple[list, dict]:
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


# %% Main Execution Block
if __name__ == "__main__":
    logging.info("--- Starting Experiment ---") 

    full_selected_features_path = os.path.join(SELECTED_FEATURES_FOLDER, SELECTED_FEATURES_PATH)
    logging.info(f"Attempting to load feature set: {full_selected_features_path}")


    try:
        selected_features_names, selected_features_metadata = get_selected_features(
            full_selected_features_path 
        )
    except Exception as e:
        logging.error(f"Could not load feature selection file {full_selected_features_path}. Exiting. Error: {e}")
        exit() 


    # Override global configurations using metadata
    if selected_features_metadata:
        logging.info("Overriding configurations with values from feature selection metadata...")
        
        original_apply_scale = APPLY_SCALE_TRANSFORM
        original_apply_pca = APPLY_PCA_IMG_TRANSFORM
        original_n_pca_components = N_PCA_COMPONENTS
        original_random_state = RANDOM_STATE

        APPLY_SCALE_TRANSFORM = selected_features_metadata.get(
            "apply_scale_transform", APPLY_SCALE_TRANSFORM
        )
        APPLY_PCA_IMG_TRANSFORM = selected_features_metadata.get(
            "apply_pca_img_transform", APPLY_PCA_IMG_TRANSFORM
        )
        RANDOM_STATE = selected_features_metadata.get("random_state", RANDOM_STATE)

        N_PCA_COMPONENTS = selected_features_metadata.get(
            "n_pca_components", N_PCA_COMPONENTS
        )
        
        if original_apply_scale != APPLY_SCALE_TRANSFORM:
            logging.info(f"APPLY_SCALE_TRANSFORM overridden by metadata: {APPLY_SCALE_TRANSFORM} (was {original_apply_scale})")
        if original_apply_pca != APPLY_PCA_IMG_TRANSFORM:
            logging.info(f"APPLY_PCA_IMG_TRANSFORM overridden by metadata: {APPLY_PCA_IMG_TRANSFORM} (was {original_apply_pca})")
        if original_random_state != RANDOM_STATE:
            logging.info(f"RANDOM_STATE overridden by metadata: {RANDOM_STATE} (was {original_random_state})")
            logging.info(f"Re-seeding with new RANDOM_STATE: {RANDOM_STATE}")
            np.random.seed(RANDOM_STATE)
            random.seed(RANDOM_STATE)
        if original_n_pca_components != N_PCA_COMPONENTS:
            logging.info(f"N_PCA_COMPONENTS overriden by metadata: {N_PCA_COMPONENTS} (was {original_n_pca_components})")
    else:
        logging.warning("No metadata found in feature selection file. Using default configurations.")


    feature_set_identifier = SELECTED_FEATURES_FILENAME.replace(".json", "")
    MLFLOW_EXPERIMENT_NAME = (
        f"Flat price predictor - {TARGET_VARIABLE} - {feature_set_identifier}"
    )
    logging.info(f"Using MLflow Experiment: '{MLFLOW_EXPERIMENT_NAME}'")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


    # Load data
    X_train_full_raw, X_test_full_raw, Y_train_raw, Y_test_raw = get_train_test_data()
    img_train_raw, img_test_raw = get_train_test_img()

    logging.info( 
        f"Raw Train Data: {X_train_full_raw.shape}, Raw Test Data: {X_test_full_raw.shape}"
    )

    district_col_name = "district"
    outlier_col_name = "outlier"

    original_img_cols = img_train_raw.columns.tolist()
    original_numeric_cols = X_train_full_raw.select_dtypes(
        include=np.number
    ).columns.tolist()
    original_numeric_cols = [
        col
        for col in original_numeric_cols
        if col != TARGET_VARIABLE
        and col != outlier_col_name
        and col not in original_img_cols
    ]
    original_categorical_cols = X_train_full_raw.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    original_categorical_cols = [
        col for col in original_categorical_cols if col != district_col_name
    ]

    logging.info(f"Original Numeric Cols: {len(original_numeric_cols)}") 
    logging.info(f"Original Categorical Cols: {len(original_categorical_cols)}") 
    logging.info(f"Original Image Cols: {len(original_img_cols)}") 


    Y_train_log_transformed = np.log(Y_train_raw[TARGET_VARIABLE])
    cv_strategy = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE) 
    
    model_configs = get_model_configs()


    with mlflow.start_run(run_name=f"MainRun_{feature_set_identifier}") as main_run:
        logging.info(f"Main Run ID for {feature_set_identifier}: {main_run.info.run_id}") 
        mlflow.set_tag("feature_set_file", SELECTED_FEATURES_FILENAME)

        # Log effective parameters used for this main run
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("cv_folds", CV_FOLDS)
        mlflow.log_param("apply_scale_transform", APPLY_SCALE_TRANSFORM)
        mlflow.log_param("apply_pca_img_transform", APPLY_PCA_IMG_TRANSFORM)
        mlflow.log_param("n_pca_components", N_PCA_COMPONENTS) 
        mlflow.log_param("n_trials_optuna", N_TRIALS_OPTUNA)
        mlflow.log_param("tuning_scoring_metric", TUNING_SCORING_METRIC)
        mlflow.log_param("optuna_direction", OPTUNA_DIRECTION)
        
        # Log details from the feature selection metadata itself
        if selected_features_metadata:
            for key, value in selected_features_metadata.items():
                 mlflow.log_param(f"fs_meta_{key}", value) 

        mlflow.log_param("num_initial_numeric_features", len(original_numeric_cols))
        mlflow.log_param(
            "num_initial_categorical_features", len(original_categorical_cols)
        )
        mlflow.log_param("num_initial_image_features", len(original_img_cols))
        mlflow.log_param(
            "num_loaded_selected_feature_names", len(selected_features_names)
        )
        mlflow.log_param("selected_features_file_path", full_selected_features_path)


        for model_name, model_specific_config in model_configs.items(): 
            try:
                train_evaluate_log(
                    model_name=model_name,
                    model_config=model_specific_config,
                    X_train_raw_full=X_train_full_raw,
                    Y_train_log=Y_train_log_transformed,
                    X_test_raw_full=X_test_full_raw,
                    Y_test=Y_test_raw[TARGET_VARIABLE],
                    numeric_cols_original=original_numeric_cols,
                    categorical_cols_original=original_categorical_cols,
                    img_cols_original=original_img_cols,
                    district_col_name_original=district_col_name,
                    outlier_col_name_original=outlier_col_name,
                    selected_feature_names=selected_features_names,
                    cv_strategy=cv_strategy,
                    parent_run_id=main_run.info.run_id,
                    feature_selection_metadata=selected_features_metadata,
                )
            except Exception as e:
                logging.error(f"* FATAL ERROR training {model_name} for feature set {feature_set_identifier}: {e} !!!!", exc_info=True)
                mlflow.log_param(f"ERROR_{model_name}", str(e)) 
                continue

    logging.info("\n--- Experiment Finished ---") 
