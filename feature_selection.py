import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, SelectFromModel, RFECV

from sklearn.inspection import permutation_importance

from preprocessor import create_data_transformer_pipeline
from split_data import get_train_test_data, get_train_test_img

TARGET_COLUMN = "price"
RANDOM_STATE = 42
APPLY_SCALE_TRANSFORM = True
N_ESTIMATORS = 100
OUTPUT_DIR = "selected_features"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def export_selected_features(output_filename, final_selected_feature_names):
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_path = os.path.join(OUTPUT_DIR, output_filename)
        with open(save_path, "w") as f:
            json.dump(final_selected_feature_names, f)
        print(f"\nSelected feature names/indices saved to {save_path}")
    except Exception as e:
        print(f"Error saving selected features: {e}.")
        print("Selected features:", final_selected_feature_names)

    print("Exporting selected features...")

def run_feature_selection(
    method="rf",
    threshold="median",
    n_features=None,
    alpha_lasso=None,
    output_dir="feature_sets",
):
    if method == "rfe" and not n_features:
        raise ValueError("n_features must be specified for RFE method.")
    if method == "lasso" and not alpha_lasso:
        raise ValueError("alpha_lasso must be specified for Lasso method.")

    X_train, X_test, y_train, y_test = get_train_test_data(include_count=False)
    img_train, img_test = get_train_test_img()

    print(f"Feature Selection: X_train shape: {X_train.shape}")
    print(f"Feature Selection: y_train shape: {y_train.shape}")

    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    district_col_name = "district"
    outlier_col_name = "outlier"

    img_cols = img_train.columns.to_list()
    numeric_cols_list = X_train.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols_list = [col for col in numeric_cols_list if col not in img_cols]

    categorical_cols_list = X_train.select_dtypes(
        include=[object, "category"]
    ).columns.tolist()

    data_transformer = create_data_transformer_pipeline(
        numeric_cols=numeric_cols_list,
        categorical_cols=categorical_cols_list,
        img_feature_cols=img_cols,
        district_group_col=district_col_name,
        outlier_indicator_col=outlier_col_name,
        apply_scaling_and_transform=APPLY_SCALE_TRANSFORM,
    )

    print("Fitting data transformer on X_train for feature selection...")
    data_transformer.fit(X_train.copy(), y_train)
    print("Data transformer fitted.")

    print("Transforming X_train to get processed features...")
    X_train_processed_np = data_transformer.transform(X_train.copy())
    X_test_processed_np = data_transformer.transform(X_test.copy())

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
        processed_feature_names = None

    print(f"Performing feature selection {method}...")

    ##### DO feature selection #####
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

        output_filename = (
            f"rf_thresh_{str(threshold).replace('.', 'p')}.json"
        )
    elif method == "rfe":
        selector_estimator = RandomForestRegressor(
            n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1
        )
        selector = RFE(selector_estimator, n_features_to_select=n_features, step=1)

        selector.fit(X_train_processed, y_train)
        selected_features_mask = selector.get_support()

        output_filename = f"rfe_{n_features}.json"

    elif method == "permutation_importance":
        model_for_permutation = RandomForestRegressor(
            n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1
        )
        model_for_permutation.fit(X_train_processed, y_train)

        perm_importance_result = permutation_importance(
            model_for_permutation,
            X_test_processed,
            y_test,
            n_repeats=10,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            scoring="neg_mean_squared_error",
        )

        sorted_idx = perm_importance_result.importances_mean.argsort()[::-1]

        # 3. Select top N features based on permutation importance
        num_features_to_select_perm = n_features if n_features else 70
        selected_indices_perm = sorted_idx[:num_features_to_select_perm]

        selected_features_mask = np.zeros(X_train_processed.shape[1], dtype=bool)
        selected_features_mask[selected_indices_perm] = True

        output_filename = f"permutation_importance_{num_features_to_select_perm}.json"

    elif method == "lasso":
        output_filename = (
            f"lasso_alpha_{str(alpha_lasso).replace('.', 'p')}.json"
        )
        raise ValueError("Lasso method not implemented yet.")
    else:
        pass

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

    export_selected_features(output_filename, final_selected_feature_names)

    print("\nFeature selection script finished using training data.")


def make_features():
    X_train, X_test, y_train, y_test = get_train_test_data()
    cols = X_train.columns

    img_substrings = ["img_", "interior_", "exterior_", "vector_", "feature_"]
    img_cols = [col for col in cols if any([sub in col for sub in img_substrings])]

    poi_substrings = ["play_areas_", "park_areas_", "dist_", "count_"]
    poi_cols = [col for col in cols if any([sub in col for sub in poi_substrings])]
    
    pano_cols = [col for col in cols if "pano_" in col]

    base_cols = [col for col in cols if col not in img_cols + poi_cols + pano_cols]

    export_selected_features("base_cols.json", base_cols)
    export_selected_features("base_poi.json", base_cols + poi_cols)
    export_selected_features("base_pano.json", base_cols + pano_cols)


if __name__ == "__main__":
    run_feature_selection(
        method="rf",
        threshold="median",
        output_dir=OUTPUT_DIR,
    )

    run_feature_selection(
        method="rfe",
        n_features=100,
        output_dir=OUTPUT_DIR,
    )

    