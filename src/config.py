import os
import yaml

# Default paths to the two config files
default_config_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
default_experiment_config_path = os.path.join(os.path.dirname(__file__), "..", "config", "experiment_config.yaml")


def load_config(file_path=None, experiment_config_path=None):
    """
    Loads and merges YAML configurations from two files.
    
    Args:
        file_path: Path to main config file (config.yaml). If None, uses default.
        experiment_config_path: Path to experiment config file (experiment_config.yaml). If None, uses default.
    
    Returns:
        dict: Merged configuration with main config taking priority over experiment config.
    """
    # Use default paths if not provided
    if file_path is None:
        file_path = default_config_path
    if experiment_config_path is None:
        experiment_config_path = default_experiment_config_path
    
    # Load main config file (config.yaml)
    try:
        with open(file_path, "r") as stream:
            main_config = yaml.safe_load(stream)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The main config file {file_path} was not found.")
    except yaml.YAMLError as exc:
        raise ValueError(f"Error parsing main config YAML file: {exc}")
    
    # Load experiment config file (experiment_config.yaml)  
    try:
        with open(experiment_config_path, "r") as stream:
            experiment_config = yaml.safe_load(stream)
    except FileNotFoundError:
        # If experiment config doesn't exist, just use main config
        print(f"Warning: Experiment config file {experiment_config_path} not found. Using only main config.")
        experiment_config = {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Error parsing experiment config YAML file: {exc}")
    
    # Merge configs: start with experiment config, then override with main config values
    merged_config = {}
    if experiment_config:
        merged_config.update(experiment_config)
    if main_config:
        merged_config.update(main_config)
    
    # Handle nested structures - ensure optuna section exists
    if "optuna" not in merged_config:
        merged_config["optuna"] = {}
    
    # Map some keys from main config to the expected format
    key_mappings = {
        "TARGET_VARIABLE": "target_variable",
        "APPLY_SCALE_TRANSFORM": "apply_scale_transform",
        "APPLY_PCA_IMG_TRANSFORM": "apply_pca_img_transform",
        "N_PCA_COMPONENTS": "n_pca_components",
        "RANDOM_STATE": "random_state",
        "SELECTED_FEATURES_DIR": "feature_sets_dir",
        "CV_FOLDS": "cv_folds"
    }
    
    # Apply key mappings
    for old_key, new_key in key_mappings.items():
        if old_key in merged_config:
            if isinstance(new_key, tuple):
                # Handle nested keys like ("optuna", "n_trials")
                section, key = new_key
                if section not in merged_config:
                    merged_config[section] = {}
                merged_config[section][key] = merged_config[old_key]
            else:
                # Handle simple key mapping
                merged_config[new_key] = merged_config[old_key]
    
    # Ensure required keys exist with defaults
    if "feature_sets" not in merged_config:
        merged_config["feature_sets"] = ["rfecv_all_nfeat_44_pca_scaled_count.json"]
    
    if "models_to_run" not in merged_config:
        merged_config["models_to_run"] = ["LightGBM", "XGBoost", "XGBoostQuantile", "RandomForest"]
    
    if "feature_sets_dir" not in merged_config:
        merged_config["feature_sets_dir"] = "feature_sets"
    
    if "optuna" not in merged_config:
        merged_config["optuna"] = {}
    if "mlflow" not in merged_config:
        merged_config["mlflow"] = {}



    optuna_dir = merged_config.get("optuna_dir", "optuna_studies")
    db_file = merged_config.get("optuna", {}).get("db_file", "tuning.db")

    optuna_dir_path = merged_config.get("optuna_dir", "optuna_studies") 
    
    merged_config["optuna"]["study_db_path"] = f"sqlite:///{os.path.join(optuna_dir_path, db_file)}"


    uppercase_keys = {
        "CV_FOLDS": merged_config.get("cv_folds", 5),

    }
    
    merged_config.update(uppercase_keys)
    
    return merged_config
