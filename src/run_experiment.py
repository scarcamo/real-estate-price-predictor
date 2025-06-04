import logging
import os
import random
from typing import Dict, Any, List

import mlflow
import numpy as np
from sklearn.model_selection import KFold

from src.config import load_config
from src.data_manager import DataManager
from src.model_factory import get_model_configs
from src.trainer import ModelTrainer, get_selected_features

logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.getLogger("lightgbm").setLevel(logging.ERROR)

def merge_configs(config1: Dict[str, Any], config2: Dict[str, Any]) -> Dict[str, Any]:
    """Merge configurations with proper key mapping"""
    # Create a new config with experiment config as base
    merged = config1.copy()
    
    # Map keys from config2 to config1 format
    key_mapping = {
        "TARGET_VARIABLE": "target_variable",
    }
    
    # Helper function to set nested dictionary values
    def set_nested(d: dict, path: str, value: Any):
        parts = path.split('.')
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    
    # Apply mappings
    for old_key, new_key in key_mapping.items():
        if old_key in config2:
            set_nested(merged, new_key, config2[old_key])
    
    # Ensure required keys exist with defaults if not present
    if "models_to_run" not in merged:
        merged["models_to_run"] = ["LightGBM", "XGBoost", "XGBoostQuantile", "RandomForest"]
    
    if "feature_sets" not in merged:
        merged["feature_sets"] = ["rfecv_all_nfeat_44_pca_scaled_count.json"]
    
    return merged

def setup_experiment(config: Dict[str, Any], feature_set: str) -> None:
    """Setup MLflow experiment"""
    feature_set_identifier = feature_set.replace(".json", "")
    experiment_name = f"RE {config['target_variable']} predictor - {feature_set_identifier}"
    logging.info(f"Using MLflow Experiment: '{experiment_name}'")
    mlflow.set_experiment(experiment_name)

def load_and_validate_features(config: Dict[str, Any], feature_set: str) -> tuple:
    """Load and validate selected features"""
    features_dir = config.get("feature_sets_dir", config.get("SELECTED_FEATURES_DIR", "feature_sets"))
    full_selected_features_path = os.path.join(
        features_dir,
        feature_set
    )
    logging.info(f"Attempting to load feature set: {full_selected_features_path}")

    try:
        selected_features_names, selected_features_metadata = get_selected_features(
            full_selected_features_path
        )
    except Exception as e:
        logging.error(f"Could not load feature selection file {full_selected_features_path}. Error: {e}")
        raise

    return selected_features_names, selected_features_metadata

def update_config_from_metadata(config: Dict[str, Any], metadata: Dict[str, Any]) -> None:
    """Update configuration with values from feature selection metadata"""
    if not metadata:
        logging.warning("No metadata found in feature selection file. Using default configurations.")
        return

    logging.info("Overriding configurations with values from feature selection metadata...")
    
    config_updates = {
        "apply_scale_transform": "apply_scale_transform",
        "apply_pca_img_transform": "apply_pca_img_transform",
        "random_state": "random_state",
        "n_pca_components": "n_pca_components"
    }
    
    for config_key, metadata_key in config_updates.items():
        if metadata_key in metadata:
            original_value = config.get(config_key)
            config[config_key] = metadata[metadata_key]
            if original_value != config[config_key]:
                logging.info(f"{config_key} overridden by metadata: {config[config_key]} (was {original_value})")

    if config["RANDOM_STATE"] != metadata.get("random_state", config["RANDOM_STATE"]):
        logging.info(f"Re-seeding with new random_state: {config['RANDOM_STATE']}")
        np.random.seed(config["RANDOM_STATE"])
        random.seed(config["RANDOM_STATE"])

def run_experiment_for_feature_set(config: Dict[str, Any], feature_set: str) -> None:
    """Run experiment for a single feature set"""
    logging.info(f"\n=== Starting experiment for feature set: {feature_set} ===")
    
    # Get feature set name without .json extension for trainer
    feature_set_name = feature_set.replace(".json", "")
    
    try:
        # Load and validate features
        selected_features_names, selected_features_metadata = load_and_validate_features(config, feature_set)
        
        # Update configuration from metadata
        update_config_from_metadata(config, selected_features_metadata)
        
        # Setup MLflow experiment
        setup_experiment(config, feature_set)
        
        # Initialize data manager and load data
        data_manager = DataManager(config)
        data_manager.load_data()
        
        # Setup cross-validation
        cv_strategy = KFold(
            n_splits=config["cv_folds"],
            shuffle=True,
            random_state=config["RANDOM_STATE"]
        )
        
        # Get model configurations for selected models
        all_model_configs = get_model_configs(config["RANDOM_STATE"])
        model_configs = {
            name: cfg for name, cfg in all_model_configs.items()
            if name in config["models_to_run"]
        }
        
        if not model_configs:
            logging.warning(f"No valid models found in config['models_to_run']: {config['models_to_run']}")
            return
        
        # Initialize trainer with feature set name
        trainer = ModelTrainer(config, data_manager, feature_set_name)
        
        # Main experiment run
        with mlflow.start_run(
            run_name=f"MainRun_{feature_set_name}"
        ) as main_run:
            logging.info(f"Main Run ID for {feature_set}: {main_run.info.run_id}")
            mlflow.set_tag("feature_set_file", feature_set)
            
            # Log configuration parameters
            config_params = {
                "random_state": config["RANDOM_STATE"],
                "cv_folds": config["cv_folds"],
                "apply_scale_transform": config["APPLY_SCALE_TRANSFORM"],
                "apply_pca_img_transform": config["APPLY_PCA_IMG_TRANSFORM"],
                "n_pca_components": config["n_pca_components"],
                "n_trials_optuna": config["optuna"]["n_trials"],
                "tuning_scoring_metric": config["optuna"]["scoring_metric"],
                "optuna_direction": config["optuna"]["direction"]
            }
            mlflow.log_params(config_params)
            
            # Log feature selection metadata
            if selected_features_metadata:
                for key, value in selected_features_metadata.items():
                    mlflow.log_param(f"fs_meta_{key}", value)
            
            # Log data statistics
            mlflow.log_param("num_initial_numeric_features", len(data_manager.original_numeric_cols))
            mlflow.log_param("num_initial_categorical_features", len(data_manager.original_categorical_cols))
            mlflow.log_param("num_initial_image_features", len(data_manager.original_img_cols))
            mlflow.log_param("num_loaded_selected_feature_names", len(selected_features_names))
            mlflow.log_param("feature_set", feature_set)
            
            # Train and evaluate each model
            for model_name, model_specific_config in model_configs.items():
                try:
                    trainer.train_evaluate_log(
                        model_name=model_name,
                        model_config=model_specific_config,
                        selected_feature_names=selected_features_names,
                        cv_strategy=cv_strategy,
                        parent_run_id=main_run.info.run_id,
                        feature_selection_metadata=selected_features_metadata,
                    )
                except Exception as e:
                    logging.error(
                        f"* FATAL ERROR training {model_name} for feature set {feature_set}: {e} !!!!",
                        exc_info=True
                    )
                    mlflow.log_param(f"ERROR_{model_name}", str(e))
                    continue
    except Exception as e:
        logging.error(f"Error in experiment for feature set {feature_set}: {e}", exc_info=True)
        raise

def main():
    logging.info("--- Starting Experiments ---")
    
    # Load configuration using simplified approach
    try:
        config = load_config()
    except Exception as e:
        logging.error(f"Error loading configuration: {e}", exc_info=True)
        raise
    
    # Run experiments for each feature set
    for feature_set in config["feature_sets"]:
        try:
            run_experiment_for_feature_set(config, feature_set)
        except Exception as e:
            logging.error(f"Failed to run experiment for feature set {feature_set}: {e}", exc_info=True)
            continue
    
    logging.info("\n--- All Experiments Finished ---")

if __name__ == "__main__":
    main() 