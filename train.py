# %% Imports
import json
import logging
import os
import random
import time
import warnings

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
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from preprocessor import create_data_transformer_pipeline
from split_data import get_train_test_data, get_train_test_img

logging.basicConfig(
    level=logging.WARNING,
    handlers=[logging.StreamHandler()],
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.getLogger("lightgbm").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
pd.set_option("display.float_format", "{:.4f}".format)

# %% Configuration Constants
RANDOM_STATE = 42
N_TRIALS_OPTUNA = 100
CV_FOLDS = 5
TARGET_VARIABLE = "price"
TUNING_SCORING_METRIC = "mape"
OPTUNA_DIRECTION = "minimize"
MLFLOW_CALLBACK_METRIC_NAME = f"cv_{TUNING_SCORING_METRIC}"


APPLY_SCALE_TRANSFORM_IN_PIPELINE = False

optuna_dir = "optuna_studies"
db_file = "tuning.db"
STUDY_DB_PATH = f"sqlite:///{os.path.join(optuna_dir, db_file)}"
ARTIFACT_PATH = "models_and_artifacts"
MLFLOW_EXPERIMENT_NAME = "Real Estate Price Prediction"
SELECTED_FEATURES_PATH = "selected_features.json"

os.makedirs(ARTIFACT_PATH, exist_ok=True)
os.makedirs(optuna_dir, exist_ok=True)


random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# Simplified build_pipeline, only contains the model
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
    print(
        f"{prefix.capitalize()} Metrics: RMSE={rmse:.4f}, MAPE={mape:.4f}, R2={r2:.4f}"
    )
    return metrics


# %% Model Definitions and Hyperparameter Spaces
MODEL_PREFIX = "model__"


def get_model_configs():
    """Returns a dictionary of models and their hyperparameter search spaces."""
    configs = {
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
        "LightGBM": {
            "model": lgb.LGBMRegressor(
                random_state=RANDOM_STATE,
                objective="regression",
                metric="mape",  # LGBM internal metric
                n_jobs=-1,
                verbose=-1,
            ),
        },
        "AdaBoost": {
            "model": AdaBoostRegressor(
                estimator=DecisionTreeRegressor(max_depth=5), random_state=RANDOM_STATE
            ),
        },
        "RandomForest": {
            "model": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        },
        "DecisionTree": {
            "model": DecisionTreeRegressor(random_state=RANDOM_STATE),
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
        )  # Original: 5, 101
        params[f"{MODEL_PREFIX}min_samples_leaf"] = trial.suggest_int(
            f"{MODEL_PREFIX}min_samples_leaf", 5, 51
        )  # Original: 5, 51
        params[f"{MODEL_PREFIX}max_features"] = trial.suggest_categorical(
            f"{MODEL_PREFIX}max_features", ["sqrt", "log2", None]
        )
        params[f"{MODEL_PREFIX}ccp_alpha"] = trial.suggest_float(
            f"{MODEL_PREFIX}ccp_alpha", 0.0, 0.05
        )

    elif model_name == "RandomForest":
        params[f"{MODEL_PREFIX}n_estimators"] = trial.suggest_int(
            f"{MODEL_PREFIX}n_estimators", 100, 1000, step=50
        )  # Original: 100, 1000, step=50
        params[f"{MODEL_PREFIX}max_depth"] = trial.suggest_int(
            f"{MODEL_PREFIX}max_depth", 5, 50
        )
        params[f"{MODEL_PREFIX}min_samples_split"] = trial.suggest_int(
            f"{MODEL_PREFIX}min_samples_split", 5, 30
        )
        params[f"{MODEL_PREFIX}min_samples_leaf"] = trial.suggest_int(
            f"{MODEL_PREFIX}min_samples_leaf", 5, 30
        )
        params[f"{MODEL_PREFIX}max_features"] = trial.suggest_categorical(
            f"{MODEL_PREFIX}max_features", ["sqrt", "log2", 0.5, 0.7, 1.0]
        )
        params[f"{MODEL_PREFIX}bootstrap"] = trial.suggest_categorical(
            f"{MODEL_PREFIX}bootstrap", [True, False]
        )

    elif model_name in ["XGBoost", "XGBoostQuantile"]:
        params[f"{MODEL_PREFIX}max_depth"] = trial.suggest_int(
            f"{MODEL_PREFIX}max_depth", 3, 12 # TODO: test lower like 3,8/10
        )
        params[f"{MODEL_PREFIX}learning_rate"] = trial.suggest_float(
            f"{MODEL_PREFIX}learning_rate", 0.01, 0.3, log=True
        )  # Original: 0.01, 0.5
        params[f"{MODEL_PREFIX}n_estimators"] = trial.suggest_int(
            f"{MODEL_PREFIX}n_estimators", 100, 1000, step=50
        )  # Original: 100, 1000, step=50
        params[f"{MODEL_PREFIX}min_child_weight"] = trial.suggest_int(
            f"{MODEL_PREFIX}min_child_weight", 5, 25
        )
        params[f"{MODEL_PREFIX}subsample"] = trial.suggest_float(
            f"{MODEL_PREFIX}subsample", 0.6, 1.0
        )
        params[f"{MODEL_PREFIX}colsample_bytree"] = trial.suggest_float(
            f"{MODEL_PREFIX}colsample_bytree", 0.6, 1.0
        )
        params[f"{MODEL_PREFIX}gamma"] = trial.suggest_float(
            f"{MODEL_PREFIX}gamma", 0, 2.0 # TODO: test up to 5
        )
        params[f"{MODEL_PREFIX}reg_lambda"] = trial.suggest_float(
            f"{MODEL_PREFIX}reg_lambda", 1e-3, 10.0, log=True
        )
        params[f"{MODEL_PREFIX}reg_alpha"] = trial.suggest_float(
            f"{MODEL_PREFIX}reg_alpha", 1e-3, 10.0, log=True
        )
        # if model_name == "XGBoostQuantile":
        #     params[f'{MODEL_PREFIX}quantile_alpha'] = trial.suggest_float(f'{MODEL_PREFIX}quantile_alpha', 0.4, 0.6)

    elif model_name == "LightGBM":
        params[f"{MODEL_PREFIX}learning_rate"] = trial.suggest_float(
            f"{MODEL_PREFIX}learning_rate", 0.01, 0.2, log=True
        )  # Original: 0.01, 0.2
        params[f"{MODEL_PREFIX}n_estimators"] = trial.suggest_int(
            f"{MODEL_PREFIX}n_estimators", 100, 1000, step=50
        )  # Original: 100, 1000
        params[f"{MODEL_PREFIX}num_leaves"] = trial.suggest_int(
            f"{MODEL_PREFIX}num_leaves", 20, 100
        )  # Original: 20, 100
        params[f"{MODEL_PREFIX}max_depth"] = trial.suggest_categorical(
            f"{MODEL_PREFIX}max_depth", [-1, 5, 10, 15, 20, 30]
        )
        params[f"{MODEL_PREFIX}min_child_samples"] = trial.suggest_int(
            f"{MODEL_PREFIX}min_child_samples", 5, 51
        )
        params[f"{MODEL_PREFIX}subsample"] = trial.suggest_float(
            f"{MODEL_PREFIX}subsample", 0.6, 1.0
        )
        params[f"{MODEL_PREFIX}colsample_bytree"] = trial.suggest_float(
            f"{MODEL_PREFIX}colsample_bytree", 0.6, 1.0
        )
        params[f"{MODEL_PREFIX}reg_alpha"] = trial.suggest_float(
            f"{MODEL_PREFIX}reg_alpha", 1e-3, 10.0, log=True
        )
        params[f"{MODEL_PREFIX}reg_lambda"] = trial.suggest_float(
            f"{MODEL_PREFIX}reg_lambda", 1e-3, 10.0, log=True
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
        params[f"{MODEL_PREFIX}estimator__min_samples_split"] = (
            trial.suggest_int(
                f"{MODEL_PREFIX}estimator__min_samples_split", 5, 30
            )
        )
        params[f"{MODEL_PREFIX}estimator__min_samples_leaf"] = (
            trial.suggest_int(
                f"{MODEL_PREFIX}estimator__min_samples_leaf", 5, 30
            )
        )

    return params

def train_evaluate_log(
    model_name,
    model_config,
    # Raw data and column definitions are passed now
    X_train_raw_full,  # Combined raw features (numeric, cat, image)
    Y_train_log,
    X_test_raw_full,  # Combined raw features for test
    Y_test,  # Original scale Y_test for final evaluation
    # Column names for the preprocessor
    numeric_cols_original,
    categorical_cols_original,
    img_cols_original,
    district_col_name_original,
    outlier_col_name_original,
    selected_feature_names,  # Loaded from JSON
    cv_strategy,
    parent_run_id=None,
):
    logging.info(f"Tuning {model_name} with Optuna")
    start_time = time.time()

    base_model = model_config["model"]

    # Define Optuna Objective Function
    def objective(trial):
        fold_scores = []
        
        # Manual CV loop to handle per-fold preprocessing
        for fold_idx, (train_idx, val_idx) in enumerate(
            cv_strategy.split(X_train_raw_full, Y_train_log)
        ):
            X_fold_train_raw = X_train_raw_full.iloc[train_idx]
            y_fold_train_log = Y_train_log.iloc[train_idx]
            X_fold_val_raw = X_train_raw_full.iloc[val_idx]
            y_fold_val_log = Y_train_log.iloc[val_idx]

            # 1. Create and fit data_transformer for the current fold's training data
            current_data_transformer = create_data_transformer_pipeline(
                numeric_cols=numeric_cols_original,
                categorical_cols=categorical_cols_original,
                img_feature_cols=img_cols_original,
                district_group_col=district_col_name_original,
                outlier_indicator_col=outlier_col_name_original,
                apply_scaling_and_transform=APPLY_SCALE_TRANSFORM_IN_PIPELINE,
            )
            current_data_transformer.fit(X_fold_train_raw.copy(), y_fold_train_log)

            # 2. Transform fold data
            X_fold_train_processed_np = current_data_transformer.transform(
                X_fold_train_raw.copy()
            )
            X_fold_val_processed_np = current_data_transformer.transform(
                X_fold_val_raw.copy()
            )

            # 3. Get feature names and convert to DataFrame
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
                print(
                    f"Fold {fold_idx}: Error getting feature names from transformer: {e}. Skipping fold."
                )
                continue

            # 4. Apply selected features (validate names match output of transformer)
            valid_selected_names_fold = [
                name
                for name in selected_feature_names
                if name in X_fold_train_processed_df.columns
            ]
            if len(valid_selected_names_fold) != len(selected_feature_names):
                raise ValueError(
                    f"Fold {fold_idx}: Some selected features are not present in the transformed data. "
                    f"Expected {len(selected_feature_names)}, found {len(valid_selected_names_fold)}."
                )

            if not valid_selected_names_fold:
                print(
                    f"Fold {fold_idx}: No valid selected features remaining after transformation. Skipping fold."
                )
                continue

            X_fold_train_selected = X_fold_train_processed_df[valid_selected_names_fold]
            X_fold_val_selected = X_fold_val_processed_df[valid_selected_names_fold]

            # 5. Define model and pipeline for Optuna trial parameters
            params = get_model_params(model_name, trial)

            trial_model_instance = model_config["model"]  # Fresh instance
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
                else:
                    score = mean_absolute_percentage_error(
                        y_fold_val_orig, y_pred_val_orig
                    )

                trial.report(score, fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                fold_scores.append(score)

            except optuna.TrialPruned:
                logging.warning(f"Trial pruned for fold {fold_idx}.")
                return float("inf") if OPTUNA_DIRECTION == "minimize" else float("-inf")
            except Exception as e:
                logging.error(f"Trial fold failed: {e}", exc_info=True)
                trial.report(
                    float("inf") if OPTUNA_DIRECTION == "minimize" else float("-inf"),
                    fold_idx,
                )
                return float("inf") if OPTUNA_DIRECTION == "minimize" else float("-inf")

        if not fold_scores:
            return float("inf") if OPTUNA_DIRECTION == "minimize" else float("-inf")
        return np.mean(fold_scores)

    # Optuna Study Setup
    study_name = f"{model_name}_selected_features_study"
    print(f"Using Optuna study '{study_name}' with storage '{STUDY_DB_PATH}'")

    mlflow_callback = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(),
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

    print(f"Starting Optuna optimization with {N_TRIALS_OPTUNA} trials...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(
        objective,
        n_trials=N_TRIALS_OPTUNA,
        callbacks=[mlflow_callback],
        n_jobs=1,
        gc_after_trial=True,
    )
    tuning_duration = time.time() - start_time
    print(f"Optuna hyperparameter tuning finished in {tuning_duration:.2f} seconds.")

    best_trial = study.best_trial
    best_params_optuna = best_trial.params
    best_score_optuna = best_trial.value
    print(f"Best Optuna Trial Score ({TUNING_SCORING_METRIC}): {best_score_optuna:.4f}")
    print("Best Hyperparameters found by Optuna:", best_params_optuna)

    # Train final model with best params on full training data
    print("Training final model using best parameters...")

    # 1. Create and fit the data_transformer on training data
    final_data_transformer = create_data_transformer_pipeline(
        numeric_cols=numeric_cols_original,
        categorical_cols=categorical_cols_original,
        img_feature_cols=img_cols_original,
        district_group_col=district_col_name_original,
        outlier_indicator_col=outlier_col_name_original,
        apply_scaling_and_transform=APPLY_SCALE_TRANSFORM_IN_PIPELINE,
    )
    final_data_transformer.fit(X_train_raw_full.copy(), Y_train_log)

    # 2. Transform sets
    X_train_processed_final_np = final_data_transformer.transform(
        X_train_raw_full.copy()
    )
    X_test_processed_final_np = final_data_transformer.transform(X_test_raw_full.copy())

    # 3. Get feature names and convert to df
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
        print(f"Final model: Error getting feature names from transformer: {e}")
        return None, None

    # 4. Apply selected features
    final_valid_selected_names = [
        name
        for name in selected_feature_names
        if name in X_train_processed_final_df.columns
    ]
    if not final_valid_selected_names:
        print("Error: No valid selected features for the final model.")
        return None, None

    X_train_selected_final = X_train_processed_final_df[final_valid_selected_names]
    X_test_selected_final = X_test_processed_final_df[final_valid_selected_names]

    # 5. Build and fit the final model pipeline
    final_model_instance = model_config["model"]
    best_model_pipeline = build_model_pipeline(final_model_instance)
    best_model_pipeline.set_params(**best_params_optuna)
    best_model_pipeline.fit(X_train_selected_final, Y_train_log)

    print("Final model training complete.")

    # --- Evaluate Final Model ---
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

    # MLflow logging
    run_tags = {}
    if parent_run_id:
        run_tags["mlflow.parentRunId"] = parent_run_id

    with mlflow.start_run(
        run_name=f"{model_name}_SelectedFeatures_PCA_in_CV_BEST",
        tags=run_tags,
        nested=True,
    ) as run:
        final_run_id_logged = run.info.run_id
        mlflow.set_tag("run_type", "best_run_pca_in_cv")
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("num_selected_features", len(final_valid_selected_names))
        mlflow.log_params(
            {k.replace(MODEL_PREFIX, ""): v for k, v in best_params_optuna.items()}
        )
        mlflow.log_metric(f"best_optuna_cv_{TUNING_SCORING_METRIC}", best_score_optuna)
        mlflow.log_metrics(train_metrics)
        mlflow.log_metrics(test_metrics)
        mlflow.log_metric("tuning_duration_sec", tuning_duration)
        mlflow.log_metric("total_pipeline_duration_sec", total_duration)

        input_example_data = X_train_selected_final.head(5)
        model_step_in_pipeline = best_model_pipeline.named_steps["model"]

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

        # Log selected feature list
        feature_list_path = "final_selected_features_list.txt"
        with open(feature_list_path, "w") as f:
            for feature in final_valid_selected_names:
                f.write(f"{feature}\n")
        mlflow.log_artifact(feature_list_path, artifact_path="feature_info")
        os.remove(feature_list_path)

    return final_run_id_logged, study


# %% Main Execution Block
if __name__ == "__main__":
    print("--- Starting Experiment ---")

    # load data
    X_train_full_raw, X_test_full_raw, Y_train_raw, Y_test_raw = get_train_test_data()
    img_train_raw, img_test_raw = get_train_test_img()

    print(
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

    print(f"Original Numeric Cols: {len(original_numeric_cols)}")
    print(f"Original Categorical Cols: {len(original_categorical_cols)}")
    print(f"Original Image Cols: {len(original_img_cols)}")

    print(f"\nLoading selected feature names from {SELECTED_FEATURES_PATH}...")
    try:
        with open(SELECTED_FEATURES_PATH, "r") as f:
            selected_features_names = json.load(f)
        if not (
            isinstance(selected_features_names, list)
            and all(isinstance(f, str) for f in selected_features_names)
        ):
            raise ValueError(
                "Selected features file should contain a JSON list of strings (feature names)."
            )
        print(f"Loaded {len(selected_features_names)} selected feature names.")

    except FileNotFoundError:
        print(f"Error: Selected features file not found at {SELECTED_FEATURES_PATH}.")
        exit()
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error loading or validating selected features: {e}")
        exit()

    Y_train_log_transformed = np.log(Y_train_raw[TARGET_VARIABLE])

    cv_strategy = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    model_configs = get_model_configs()

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    print(f"Using MLflow Experiment: '{MLFLOW_EXPERIMENT_NAME}'")

    with mlflow.start_run(run_name="Main_Experiment_Run_PCA_in_CV") as main_run:
        print(f"Main Run ID: {main_run.info.run_id}")
        mlflow.log_param("RANDOM_STATE", RANDOM_STATE)
        mlflow.log_param("CV_FOLDS", CV_FOLDS)
        mlflow.log_param(
            "APPLY_SCALE_TRANSFORM_IN_PIPELINE", APPLY_SCALE_TRANSFORM_IN_PIPELINE
        )
        mlflow.log_param("N_TRIALS_OPTUNA", N_TRIALS_OPTUNA)

        mlflow.log_param("num_initial_numeric_features", len(original_numeric_cols))
        mlflow.log_param(
            "num_initial_categorical_features", len(original_categorical_cols)
        )
        mlflow.log_param("num_initial_image_features", len(original_img_cols))
        mlflow.log_param(
            "num_loaded_selected_feature_names", len(selected_features_names)
        )

        for model_name, config in model_configs.items():
            try:
                train_evaluate_log(
                    model_name=model_name,
                    model_config=config,
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
                )
            except Exception as e:
                logging.error(f"* ERROR training {model_name}: {e} !!!!", exc_info=True)
                mlflow.log_param(f"ERROR_{model_name}", str(e))
                continue

    print("\n--- Experiment Finished ---")
