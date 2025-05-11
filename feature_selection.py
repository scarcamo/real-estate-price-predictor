import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

from preprocessor import create_data_transformer_pipeline
from split_data import get_train_test_data, get_train_test_img

TARGET_COLUMN = "price"
RANDOM_STATE = 42
APPLY_SCALE_TRANSFORM_BEFORE_FS = False
N_ESTIMATORS = 100

X_train, X_test, y_train, y_test = get_train_test_data()
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
    apply_scaling_and_transform=APPLY_SCALE_TRANSFORM_BEFORE_FS,
)

print("Fitting data transformer on X_train for feature selection...")
data_transformer.fit(X_train.copy(), y_train)
print("Data transformer fitted.")

print("Transforming X_train to get processed features...")
X_train_processed_np = data_transformer.transform(X_train.copy())

try:
    processed_feature_names = data_transformer.get_feature_names_out()
    X_train_processed = pd.DataFrame(
        X_train_processed_np, columns=processed_feature_names, index=X_train.index
    )
    print(f"Processed training data for selection - shape: {X_train_processed.shape}")
    print(
        "Example processed feature names:", processed_feature_names[:10].tolist(), "..."
    )
except Exception as e:
    print(f"Could not get feature names from pipeline. Error: {e}")
    print(
        "Feature selection will proceed with indices."
    )
    X_train_processed = X_train_processed_np
    processed_feature_names = None


print("Performing feature selection...")

selector_estimator = RandomForestRegressor(
    n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1
)
selector = SelectFromModel(selector_estimator, threshold="median", prefit=False)

print("Fitting feature selector on processed training data...")
selector.fit(X_train_processed, y_train)
print("Feature selector fitted.")


selected_features_mask = selector.get_support()

if processed_feature_names is not None and isinstance(X_train_processed, pd.DataFrame):
    final_selected_feature_names = X_train_processed.columns[
        selected_features_mask
    ].tolist()
    print(
        f"\nSelected {len(final_selected_feature_names)} features (out of {X_train_processed.shape[1]}):"
    )
else:
    final_selected_feature_names = np.where(selected_features_mask)[
        0
    ].tolist()  
    print(
        f"\nSelected {len(final_selected_feature_names)} feature INDICES (out of {X_train_processed.shape[1]}):"
    )


SELECTED_FEATURES_PATH = "selected_features.json"

try:
    import json

    with open(SELECTED_FEATURES_PATH, "w") as f:
        json.dump(final_selected_feature_names, f)
    print(f"\nSelected feature names/indices saved to {SELECTED_FEATURES_PATH}")
except Exception as e:
    print(f"Error saving selected features: {e}.")
    print("Selected features:", final_selected_feature_names)

print("\nFeature selection script finished using training data.")