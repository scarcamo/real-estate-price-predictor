import logging
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, RFECV, SelectFromModel
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.model_selection import KFold

from src.config import load_config
from src.preprocessor import create_data_transformer_pipeline
from src.split_data import get_train_test_data, get_train_test_img

config = load_config()

TARGET_VARIABLE = config.get("TARGET_VARIABLE")
RANDOM_STATE = config.get("RANDOM_STATE")
APPLY_SCALE_TRANSFORM = config.get("APPLY_SCALE_TRANSFORM")
N_ESTIMATORS = config.get("N_ESTIMATORS")
OUTPUT_DIR = config.get("feature_sets_dir")
APPLY_PCA_IMG_TRANSFORM = config.get("APPLY_PCA_IMG_TRANSFORM")
VERBOSE = 1

N_FEATURES = config.get("N_FEATURES", 100)
RFE_STEP_SIZE = config.get("RFE_STEP_SIZE")
INCLUDE_COUNT = config.get("INCLUDE_COUNT", True)
N_PCA_COMPONENTS = config.get("N_PCA_COMPONENTS", 10)
INCLUDE_LOCATION_FEATURES = config.get("INCLUDE_LOCATION_FEATURES", False)

os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def export_selected_features(
    output_filename, final_selected_feature_names, metadata=None, ranking=None
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
    # remove rooms_number from base_cols
    base_cols = [col for col in base_cols if col != "rooms_number"]

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

    if kwargs.get("n_features"):
        n_features = kwargs.get("n_features")
        if n_features:
            filename_parts.append(f"nfeat_{n_features}")

    if method == "rf":
        threshold = kwargs.get("threshold", "median")
        filename_parts.append(f"thresh_{str(threshold).replace('.', 'p')}")

    elif method == "lasso":
        alpha_lasso = kwargs.get("alpha_lasso")
        if alpha_lasso is not None:
            filename_parts.append(f"alpha_{str(alpha_lasso).replace('.', 'p')}")
    elif method == "elasticnet":
        # For ElasticNetCV, alpha is determined automatically
        filename_parts.append("cv")

    apply_pca_img_transform = kwargs.get("apply_pca_img_transform", APPLY_PCA_IMG_TRANSFORM)
    apply_scale_transform = kwargs.get("apply_scale_transform", APPLY_SCALE_TRANSFORM)
    include_count = kwargs.get("include_count", INCLUDE_COUNT)

    # PCA is only relevant for feature subsets that include images
    img_subsets = ["all", "base_img", "base_poi_img", "base_pano_img", "base_poi_pano_img"]
    should_apply_pca = apply_pca_img_transform and feature_subset in img_subsets

    if should_apply_pca:
        filename_parts.append("pca")
    else:
        filename_parts.append("nonpca")

    if apply_scale_transform:
        filename_parts.append("scaled")

    if include_count:
        filename_parts.append("count")

    include_location_features = kwargs.get("include_location_features", INCLUDE_LOCATION_FEATURES)
    if include_location_features:
        filename_parts.append("loc")

    # timestamp = int(time.time())
    # filename_parts.append(str(timestamp))

    return "_".join(filename_parts) + ".json"

def filter_image_features_with_lasso(X_train_processed, y_train, alpha_range=None):
    """
    Filter image features using LassoCV, return filtered dataset with tabular + selected image features.
    
    Parameters:
    -----------
    X_train_processed : pd.DataFrame
        Processed training data
    y_train : array-like
        Target variable
    alpha_range : array-like, optional
        Range of alpha values for LassoCV
    
    Returns:
    --------
    X_filtered : pd.DataFrame
        Dataset with tabular features + selected image features
    selected_img_features : list
        List of selected image feature names
    """
    # Identify image columns
    img_substrings = ["img_", "interior_", "exterior_", "vector_", "feature_", "pca_"]
    img_cols = [col for col in X_train_processed.columns if any([sub in col for sub in img_substrings])]
    
    # Identify non-image (tabular) columns
    tabular_cols = [col for col in X_train_processed.columns if col not in img_cols]
    
    if not img_cols:
        print("No image features found - returning original dataset")
        return X_train_processed, []
    
    print(f"Found {len(img_cols)} image features and {len(tabular_cols)} tabular features")
    print("Applying LassoCV to filter image features...")
    
    # Extract image features for Lasso selection
    X_img_features = X_train_processed[img_cols]
    
    # Scale ONLY the image features for better Lasso convergence
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_img_features_scaled = scaler.fit_transform(X_img_features)
    
    # Default alpha range if not provided
    if alpha_range is None:
        alpha_range = np.logspace(-4, 1, 100)
    
    # Apply LassoCV to scaled image features only
    lasso_cv = LassoCV(
        alphas=alpha_range,
        cv=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_iter=20000,
        tol=1e-4
    )
    
    y_train_log = np.log(y_train)
    lasso_cv.fit(X_img_features_scaled, y_train_log)
    
    # Get selected image features (non-zero coefficients)
    selected_mask = lasso_cv.coef_ != 0
    selected_img_features = [col for col, selected in zip(img_cols, selected_mask) if selected]
    
    print(f"LassoCV selected {len(selected_img_features)} image features out of {len(img_cols)}")
    print(f"Best alpha: {lasso_cv.alpha_}")
    
    # Combine tabular features with selected image features
    final_cols = tabular_cols + selected_img_features
    X_filtered = X_train_processed[final_cols]
    
    print(f"Final dataset shape: {X_filtered.shape} (tabular: {len(tabular_cols)}, selected images: {len(selected_img_features)})")
    
    return X_filtered, selected_img_features



def run_feature_selection(
    method="rf",
    threshold="median",
    n_features=None,
    alpha_lasso=None,
    output_dir="feature_sets",
    feature_subset="all",
    apply_scale_transform=None,
    apply_pca_img_transform=None,
    n_pca_components=None,
    rfe_step_size=None,
    include_location_features=None,
    alpha_range=None,
):
    # Use config defaults if parameters are not provided
    if apply_scale_transform is None:
        apply_scale_transform = APPLY_SCALE_TRANSFORM
    if apply_pca_img_transform is None:
        apply_pca_img_transform = APPLY_PCA_IMG_TRANSFORM
    if n_pca_components is None:
        n_pca_components = N_PCA_COMPONENTS
    if rfe_step_size is None:
        rfe_step_size = RFE_STEP_SIZE
    if include_location_features is None:
        include_location_features = INCLUDE_LOCATION_FEATURES

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
    y_train_log = np.log(y_train)  # Use log1p for better numerical stability

    feature_sets = get_feature_subsets(X_train_full)
    if feature_subset not in feature_sets:
        raise ValueError(
            f"Invalid feature_subset: {feature_subset}. Choose from {list(feature_sets.keys())}"
        )

    cols_to_use = feature_sets[feature_subset]
    
    district_col_name = "district"
    outlier_col_name = "outlier"
    
    # Add district temporarily for preprocessing if not already in subset
    cols_with_district = cols_to_use.copy()
    district_added_temporarily = False
    if district_col_name in X_train_full.columns and district_col_name not in cols_to_use:
        cols_with_district.append(district_col_name)
        district_added_temporarily = True
        print(f"Adding '{district_col_name}' temporarily for grouped preprocessing")
    
    X_train = X_train_full[cols_with_district].copy()
    X_test = X_test_full[cols_with_district].copy()

    print(
        f"Using feature subset '{feature_subset}'. X_train shape for preprocessing: {X_train.shape}"
    )

    # Identify image columns that are present in the current subset
    img_subsets = [
        "all",
        "base_img",
        "base_poi_img",
        "base_pano_img",
        "base_poi_pano_img",
    ]

    if feature_subset in img_subsets:
        print(f"Feature subset '{feature_subset}' includes image features.")
        img_cols_in_subset = [
         col for col in img_train.columns.to_list() if col in X_train.columns
        ]
    else:
        print(f"Feature subset '{feature_subset}' does not include image features.")
        img_cols_in_subset = []

    numeric_cols_list = X_train.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols_list = [
        col for col in numeric_cols_list if col not in img_cols_in_subset
    ]

    

    categorical_cols_list = X_train.select_dtypes(
        include=[object, "category"]
    ).columns.tolist()
    
    # Handle location features based on toggle
    location_features = [district_col_name, "neighborhood"]
    
    if include_location_features:
        print(f"Including location features in feature selection: {location_features}")
        # Add location features to categorical columns if they exist in the data
        for loc_col in location_features:
            if loc_col in X_train.columns and loc_col not in categorical_cols_list:
                categorical_cols_list.append(loc_col)
    else:
        print(f"Excluding location features from feature selection: {location_features}")
        # Remove location features from both numeric and categorical lists
        categorical_cols_list = [col for col in categorical_cols_list if col not in location_features]

        # PCA should only be applied if we have image features AND the parameter is True
    should_apply_pca = apply_pca_img_transform and bool(img_cols_in_subset)
    

    data_transformer = create_data_transformer_pipeline(
        numeric_cols=numeric_cols_list,
        categorical_cols=categorical_cols_list,
        img_feature_cols=img_cols_in_subset, 
        district_group_col=district_col_name,  
        outlier_indicator_col=outlier_col_name,
        apply_scaling_and_transform=apply_scale_transform,
        apply_pca=should_apply_pca,
        n_pca_components=n_pca_components if should_apply_pca else None,
        include_location_features=include_location_features,
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
        X_train_processed_full = pd.DataFrame(
            X_train_processed_np, columns=processed_feature_names, index=X_train.index
        )

        # Remove district column from processed features if it was added temporarily
        if district_added_temporarily:
            district_features = [f for f in processed_feature_names if district_col_name in f]
            features_to_keep = [f for f in processed_feature_names if f not in district_features]
            X_train_processed = X_train_processed_full[features_to_keep]
            print(f"Removed {len(district_features)} district-related features from processed data")
        else:
            X_train_processed = X_train_processed_full

        X_test_processed = pd.DataFrame(
            X_test_processed_np, columns=processed_feature_names, index=X_test.index
        )

        print(
            f"Processed training data for selection - shape: {X_train_processed.shape}"
        )
        print(
            "Example processed feature names:",
            list(X_train_processed.columns)[:10],
            "...",
        )
    except Exception as e:
        print(f"Could not get feature names from pipeline. Error: {e}")
        print("Feature selection will proceed with indices.")
        X_train_processed = X_train_processed_np
        X_test_processed = X_test_processed_np

    print(f"X_train_processed shape: {X_train_processed.shape}")
    print(f"Performing feature selection {method}...")

    # Track if filter was performed for proper feature name extraction
    was_filtered = False

    if method == "rf":
        selector_estimator = RandomForestRegressor(
            n_estimators=N_ESTIMATORS, 
            random_state=RANDOM_STATE, n_jobs=-1
        )
        selector = SelectFromModel(
            selector_estimator,
            threshold=threshold,
            max_features=n_features,
            prefit=False,
        )
        selector.fit(X_train_processed, y_train_log)
        selected_features_mask = selector.get_support()

    elif method == "rfe":
        selector_estimator = RandomForestRegressor(
            n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1
        )
        selector = RFE(
            selector_estimator,
            n_features_to_select=n_features,
            step=rfe_step_size,
            verbose=VERBOSE,
        )
        selector.fit(X_train_processed, y_train_log)
        selected_features_mask = selector.get_support()

    elif method == "rfecv":
        # Pre-filter image features with Lasso if they exist
        if img_cols_in_subset:
            print("Pre-filtering image features with LassoCV before RFECV...")
            X_train_processed, selected_img_features = filter_image_features_with_lasso(
                X_train_processed, y_train, alpha_range=alpha_range
            )
            print(f"Proceeding with RFECV on {X_train_processed.shape[1]} features (including {len(selected_img_features)} selected image features)")
        
        rf_estimator = RandomForestRegressor(
            n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1
        )
        
        # min select 20% of the features
        min_features_to_select = int(0.2 * X_train_processed.shape[1])
        print(f"Min features to select: {min_features_to_select}")

        cv_strategy = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        selector = RFECV(
            estimator=rf_estimator,
            step=rfe_step_size,
            cv=cv_strategy,
            scoring="neg_mean_absolute_percentage_error", # mape
            min_features_to_select=min_features_to_select, # equal to pai_poi_pano
            n_jobs=-1,
            verbose=1,
        )
        print("Starting RFECV to find optimal number of features...")
        selector.fit(X_train_processed, y_train_log)
        print("RFECV completed.")
        n_features = selector.n_features_
        print(f"\nOptimal number of features found: {n_features}")
        selected_features_mask = selector.get_support()

    elif method == "elasticnet":
        # ElasticNetCV for entire dataset
        if alpha_range is None:
            alpha_range = np.logspace(-4, 1, 100)
        
        l1_ratio_range = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99] 
        
        # Standardize target variable only for ElasticNet for better convergence
        y_train_scaled = (y_train_log - np.mean(y_train_log)) / np.std(y_train_log)
        
        elasticnet_cv = ElasticNetCV(
            alphas=alpha_range,
            l1_ratio=l1_ratio_range,
            cv=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            max_iter=20000,
            tol=1e-4,

        )
        
        elasticnet_cv.fit(X_train_processed, y_train_scaled)
        
        # Get selected features (non-zero coefficients)
        selected_features_mask = elasticnet_cv.coef_ != 0
        n_features = np.sum(selected_features_mask)
        
        print(f"ElasticNetCV selected {n_features} features out of {X_train_processed.shape[1]}")
        print(f"Best alpha: {elasticnet_cv.alpha_}")
        print(f"Best l1_ratio: {elasticnet_cv.l1_ratio_}")
        
        selector = elasticnet_cv

    else:
        raise ValueError(f"Unknown feature selection method: {method}")

    # Extract feature names or indices based on data type
    if isinstance(X_train_processed, pd.DataFrame):
        final_selected_feature_names = X_train_processed.columns[selected_features_mask].tolist()
        print(
            f"\nSelected {len(final_selected_feature_names)} features (out of {X_train_processed.shape[1]}):"
        )
        print("First 10 selected features:", final_selected_feature_names[:10])
    else:
        final_selected_feature_names = np.where(selected_features_mask)[0].tolist()
        print(
            f"\nSelected {len(final_selected_feature_names)} feature INDICES (out of {X_train_processed.shape[1]}):"
        )
        print("First 10 selected indices:", final_selected_feature_names[:10])

    output_filename = get_output_filename(
        method=method,
        feature_subset=feature_subset,
        threshold=threshold,
        n_features=n_features,
        alpha_lasso=alpha_lasso,
        apply_pca_img_transform=should_apply_pca,
        apply_scale_transform=apply_scale_transform,
        include_count=INCLUDE_COUNT,
        include_location_features=include_location_features,
    )

    img_subsets = [
            "all",
            "base_img",
            "base_poi_img",
            "base_pano_img",
            "base_poi_pano_img",
        ]
    
    metadata = {
        "method": method,
        "feature_subset_used": feature_subset,
        "n_features_selected": len(final_selected_feature_names),
        "apply_pca_img_transform": should_apply_pca,
        "n_pca_components": n_pca_components if should_apply_pca else None,
        "apply_scale_transform": apply_scale_transform,
        "include_count": INCLUDE_COUNT,
        "include_location_features": include_location_features,
        "n_estimators": N_ESTIMATORS,
        "random_state": RANDOM_STATE,
        **({"threshold": threshold} if method == "rf" else {}),
        **({"rfe_step_size": rfe_step_size} if method in ["rfe", "rfecv"] else {}),
        **({"alpha": getattr(selector, 'alpha_', None)} if method == "elasticnet" else {}),
        **({"l1_ratio": getattr(selector, 'l1_ratio_', None)} if method == "elasticnet" else {}),
    }


    # Only pass ranking if selector has it (RFE and RFECV methods)
    ranking = getattr(selector, 'ranking_', None)
    export_selected_features(output_filename, final_selected_feature_names, metadata, ranking)

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
    return output_filename





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
        method="elasticnet",
        output_dir=OUTPUT_DIR,
        feature_subset="all",
        apply_scale_transform=True,
        apply_pca_img_transform=True,
        include_location_features=True,
        n_pca_components=0.80,
    )

    run_feature_selection(
        method="elasticnet",
        output_dir=OUTPUT_DIR,
        feature_subset="all",
        apply_scale_transform=True,
        apply_pca_img_transform=False,
        include_location_features=True,
    )

    run_feature_selection(
        method="elasticnet",
        output_dir=OUTPUT_DIR,
        feature_subset="img_cols",
        apply_scale_transform=True,
        apply_pca_img_transform=False,
    )

    run_feature_selection(
        method="elasticnet",
        output_dir=OUTPUT_DIR,
        feature_subset="img_cols",
        apply_scale_transform=True,
        apply_pca_img_transform=True,
        n_pca_components=0.80,
    )

    run_feature_selection(
        method="rfecv",
        output_dir=OUTPUT_DIR,
        feature_subset="all",
        apply_scale_transform=False,
        apply_pca_img_transform=False,
        include_location_features=True,
        rfe_step_size=0.08
    )

    run_feature_selection(
        method="rfecv",
        output_dir=OUTPUT_DIR,
        feature_subset="all",
        apply_scale_transform=False,
        include_location_features=True,
        apply_pca_img_transform=True,
        n_pca_components=0.80,
        rfe_step_size=0.08
    )