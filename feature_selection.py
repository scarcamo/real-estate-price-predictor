import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, RFECV, SelectFromModel
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold

from config import load_config
from preprocessor import create_data_transformer_pipeline
from split_data import get_train_test_data, get_train_test_img

config = load_config()

TARGET_VARIABLE = config.get("TARGET_VARIABLE")
RANDOM_STATE = config.get("RANDOM_STATE")
APPLY_SCALE_TRANSFORM = config.get("APPLY_SCALE_TRANSFORM")
N_ESTIMATORS = config.get("N_ESTIMATORS")
OUTPUT_DIR = config.get("SELECTED_FEATURES_DIR")
APPLY_PCA_IMG_TRANSFORM = config.get("APPLY_PCA_IMG_TRANSFORM")
VERBOSE = 1

N_FEATURES = config.get("N_FEATURES", 100)
RFE_STEP_SIZE = config.get("RFE_STEP_SIZE")
INCLUDE_COUNT = config.get("INCLUDE_COUNT", True)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def export_selected_features(
    output_filename, final_selected_feature_names, metadata=None
):
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_path = os.path.join(OUTPUT_DIR, output_filename)
        data_to_save = {
            "selected_features": final_selected_feature_names,
            "metadata": metadata if metadata else {},
        }
        with open(save_path, "w") as f:
            json.dump(data_to_save, f, indent=4)
        print(f"\nSelected feature names/indices and metadata saved to {save_path}")
    except Exception as e:
        print(f"Error saving selected features: {e}.")
        print("Selected features:", final_selected_feature_names)
        print("Metadata:", metadata)

    print("Exporting selected features...")


def get_feature_subsets(X_data):
    """Identifies and returns lists of feature names for different categories."""
    cols = X_data.columns

    img_substrings = ["img_", "interior_", "exterior_", "vector_", "feature_", "pca_"]
    img_cols = [col for col in cols if any([sub in col for sub in img_substrings])]

    poi_substrings = ["play_areas_", "park_areas_", "dist_", "count_"]
    poi_cols = [col for col in cols if any([sub in col for sub in poi_substrings])]

    pano_cols = [col for col in cols if "pano_" in col]

    base_cols = [col for col in cols if col not in img_cols + poi_cols + pano_cols]

    return {
        "all": cols.tolist(),
        "base": base_cols,
        "pano": pano_cols,
        "poi": poi_cols,
        "img_cols": img_cols,
        "base_img": base_cols + img_cols,
        "base_pano": base_cols + pano_cols,
        "base_poi": base_cols + poi_cols,
        "base_poi_pano": base_cols + poi_cols + pano_cols,
        "base_poi_img": base_cols + poi_cols + img_cols,
        "base_pano_img": base_cols + pano_cols + img_cols,
        "base_poi_pano_img": base_cols + poi_cols + pano_cols + img_cols,
        "poi_pano_img": poi_cols + pano_cols + img_cols,
    }


def get_output_filename(method, feature_subset, **kwargs):
    """
    Generates unique filename for the feature selection output.
    """
    filename_parts = [method, feature_subset]

    if method == "rf":
        threshold = kwargs.get("threshold", "median")
        filename_parts.append(f"thresh_{str(threshold).replace('.', 'p')}")
    elif method == "rfe" or method == "rfecv" or method == "permutation_importance":
        n_features = kwargs.get("n_features")
        if n_features:
            filename_parts.append(f"nfeat_{n_features}")
    elif method == "lasso":
        alpha_lasso = kwargs.get("alpha_lasso")
        if alpha_lasso is not None:
            filename_parts.append(f"alpha_{str(alpha_lasso).replace('.', 'p')}")

    if APPLY_PCA_IMG_TRANSFORM:
        filename_parts.append("pca")
    else:
        filename_parts.append("nonpca")

    if APPLY_SCALE_TRANSFORM:
        filename_parts.append("scaled")

    if INCLUDE_COUNT:
        filename_parts.append("count")


    # timestamp = int(time.time())
    # filename_parts.append(str(timestamp))

    return "_".join(filename_parts) + ".json"


def run_feature_selection(
    method="rf",
    threshold="median",
    n_features=None,
    alpha_lasso=None,
    output_dir="feature_sets",
    feature_subset="all",
):
    if method == "rfe" and not n_features:
        raise ValueError("n_features must be specified for RFE method.")
    if method == "lasso" and not alpha_lasso:
        raise ValueError("alpha_lasso must be specified for Lasso method.")

    X_train_full, X_test_full, y_train, y_test = get_train_test_data(
        include_count=INCLUDE_COUNT
    )
    img_train, img_test = get_train_test_img()

    print(f"Feature Selection: X_train_full shape: {X_train_full.shape}")
    print(f"Feature Selection: y_train shape: {y_train.shape}")

    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    feature_sets = get_feature_subsets(X_train_full)
    if feature_subset not in feature_sets:
        raise ValueError(
            f"Invalid feature_subset: {feature_subset}. Choose from {list(feature_sets.keys())}"
        )

    cols_to_use = feature_sets[feature_subset]
    X_train = X_train_full[cols_to_use].copy()
    X_test = X_test_full[cols_to_use].copy()

    print(
        f"Using feature subset '{feature_subset}'. X_train shape for selection: {X_train.shape}"
    )

    district_col_name = "district"
    outlier_col_name = "outlier"

    # Identify image columns that are present in the current subset
    img_cols_in_subset = [
        col for col in img_train.columns.to_list() if col in X_train.columns
    ]

    numeric_cols_list = X_train.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols_list = [
        col for col in numeric_cols_list if col not in img_cols_in_subset
    ]

    categorical_cols_list = X_train.select_dtypes(
        include=[object, "category"]
    ).columns.tolist()

    data_transformer = create_data_transformer_pipeline(
        numeric_cols=numeric_cols_list,
        categorical_cols=categorical_cols_list,
        img_feature_cols=img_cols_in_subset,  # Pass only relevant img cols
        district_group_col=district_col_name,
        outlier_indicator_col=outlier_col_name,
        apply_scaling_and_transform=APPLY_SCALE_TRANSFORM,
        apply_pca=APPLY_PCA_IMG_TRANSFORM,
    )

    print("Fitting data transformer on X_train for feature selection...")
    data_transformer.fit(X_train.copy(), y_train)
    print("Data transformer fitted.")

    print("Transforming X_train to get processed features...")
    X_train_processed_np = data_transformer.transform(X_train.copy())
    X_test_processed_np = data_transformer.transform(X_test.copy())

    processed_feature_names = None

    try:
        processed_feature_names = data_transformer.get_feature_names_out()
        X_train_processed = pd.DataFrame(
            X_train_processed_np, columns=processed_feature_names, index=X_train.index
        )

        X_test_processed = pd.DataFrame(
            X_test_processed_np, columns=processed_feature_names, index=X_test.index
        )

        print(
            f"Processed training data for selection - shape: {X_train_processed.shape}"
        )
        print(
            "Example processed feature names:",
            processed_feature_names[:10].tolist(),
            "...",
        )
    except Exception as e:
        print(f"Could not get feature names from pipeline. Error: {e}")
        print("Feature selection will proceed with indices.")
        X_train_processed = X_train_processed_np
        X_test_processed = X_test_processed_np

    print(f"X_train_processed shape: {X_train_processed.shape}")
    print(f"Performing feature selection {method}...")

    if method == "rf":
        selector_estimator = RandomForestRegressor(
            n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1
        )
        selector = SelectFromModel(
            selector_estimator,
            threshold=threshold,
            max_features=n_features,
            prefit=False,
        )
        selector.fit(X_train_processed, y_train)
        selected_features_mask = selector.get_support()

    elif method == "rfe":
        selector_estimator = RandomForestRegressor(
            n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1
        )
        selector = RFE(
            selector_estimator,
            n_features_to_select=n_features,
            step=RFE_STEP_SIZE,
            verbose=VERBOSE,
        )
        selector.fit(X_train_processed, y_train)
        selected_features_mask = selector.get_support()

    elif method == "rfecv":
        rf_estimator = RandomForestRegressor(
            n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1
        )
        cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)
        selector = RFECV(
            estimator=rf_estimator,
            step=RFE_STEP_SIZE,
            cv=cv_strategy,
            scoring="neg_mean_squared_error",
            min_features_to_select=1,
            n_jobs=-1,
            verbose=1,
        )
        print("Starting RFECV to find optimal number of features...")
        selector.fit(X_train_processed, y_train)
        print("RFECV completed.")
        n_features = selector.n_features_
        print(f"\nOptimal number of features found: {n_features}")
        selected_features_mask = selector.get_support()

    else:
        raise ValueError(f"Unknown feature selection method: {method}")

    if processed_feature_names is not None and isinstance(
        X_train_processed, pd.DataFrame
    ):
        final_selected_feature_names = X_train_processed.columns[
            selected_features_mask
        ].tolist()
        print(
            f"\nSelected {len(final_selected_feature_names)} features (out of {X_train_processed.shape[1]}):"
        )
    else:
        final_selected_feature_names = np.where(selected_features_mask)[0].tolist()
        print(
            f"\nSelected {len(final_selected_feature_names)} feature INDICES (out of {X_train_processed.shape[1]}):"
        )

    output_filename = get_output_filename(
        method=method,
        feature_subset=feature_subset,
        threshold=threshold,
        n_features=n_features,
        alpha_lasso=alpha_lasso,
    )

    metadata = {
        "method": method,
        "feature_subset_used": feature_subset,
        "n_features_selected": len(final_selected_feature_names),
        "apply_pca_img_transform": APPLY_PCA_IMG_TRANSFORM,
        "apply_scale_transform": APPLY_SCALE_TRANSFORM,
        "include_count": INCLUDE_COUNT,
        "n_estimators": N_ESTIMATORS,
        "random_state": RANDOM_STATE,
        **({"threshold": threshold} if method == "rf" else {}),
        **({"rfe_step_size": RFE_STEP_SIZE} if method in ["rfe", "rfecv"] else {}),
    }

    export_selected_features(output_filename, final_selected_feature_names, metadata)

    if method in ["rfecv"]: # Add other methods if they have similar results
        cv_results_filename = output_filename.replace(".json", "_cv_results.json")
        try:
            with open(os.path.join(output_dir, cv_results_filename), 'w') as f:
                # Convert numpy arrays in cv_results_ to lists for JSON serialization
                serializable_cv_results = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in selector.cv_results_.items()}
                json.dump(serializable_cv_results, f, indent=4)
            print(f"CV results saved to {os.path.join(output_dir, cv_results_filename)}")
        except Exception as e:
            print(f"Error saving CV results: {e}")

    print("\nFeature selection script finished using training data.")


def make_features():
    """
    Export predefined feature sets
    """
    X_train, _, _, _ = get_train_test_data()
    feature_subsets = get_feature_subsets(X_train)

    for name, cols in feature_subsets.items():
        export_selected_features(
            f"{name}_cols.json",
            cols,
            metadata={"subset_type": name, "source": "initial_definition"},
        )


if __name__ == "__main__":

    run_feature_selection(
        method="rfecv",
        output_dir=OUTPUT_DIR,
        feature_subset="base_img",
    )

    run_feature_selection(
        method="rfecv",
        output_dir=OUTPUT_DIR,
        feature_subset="poi_pano_img",
    )

    run_feature_selection(
        method="rfecv",
        output_dir=OUTPUT_DIR,
        feature_subset="base",
    )

    run_feature_selection(
        method="rfecv",
        output_dir=OUTPUT_DIR,
        feature_subset="base_pano",
    )

    run_feature_selection(
        method="rfecv",
        output_dir=OUTPUT_DIR,
        feature_subset="all",
    )

    run_feature_selection(
        method="rfecv",
        output_dir=OUTPUT_DIR,
        feature_subset="base_poi",
    )

    run_feature_selection(
        method="rfecv",
        output_dir=OUTPUT_DIR,
        feature_subset="base_poi_pano",
    )