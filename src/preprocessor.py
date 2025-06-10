import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)

from src.config import load_config

config = load_config()

N_PCA_COMPONENTS = config.get("N_PCA_COMPONENTS")


class GroupedMedianImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing values in specified numeric columns using the median of each group,
    optionally excluding outliers when calculating the median.
    """

    def __init__(self, group_col, feature_cols_to_impute, outlier_col=None):
        self.group_col = group_col
        self.feature_cols_to_impute = feature_cols_to_impute
        self.outlier_col = outlier_col
        self.group_medians_ = {}
        self.global_medians_ = {}
        self._feature_names_in_ = None

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("GroupedMedianImputer expects a pandas DataFrame.")
        self._feature_names_in_ = list(X.columns)

        if self.group_col not in X.columns:
            raise ValueError(
                f"Group column '{self.group_col}' not found in input DataFrame."
            )

        df_for_stats = X.copy()
        if self.outlier_col and self.outlier_col in df_for_stats.columns:
            try:
                outlier_mask = df_for_stats[self.outlier_col].astype(bool)
                df_no_outliers = df_for_stats[~outlier_mask]
                if not df_no_outliers.empty:
                    df_for_stats = df_no_outliers
                else:
                    warnings.warn(
                        f"DataFrame empty after outlier filter for '{self.outlier_col}'. Using original data for stats."
                    )
            except Exception as e:
                warnings.warn(
                    f"Outlier filter failed for '{self.outlier_col}': {e}. Using original data for stats."
                )

        for col in self.feature_cols_to_impute:
            if col not in X.columns:
                warnings.warn(
                    f"Fit: Feature column '{col}' for grouped imputation not found. Skipping."
                )
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                self.group_medians_[col] = df_for_stats.groupby(self.group_col)[
                    col
                ].median()

            self.global_medians_[col] = df_for_stats[col].median()
            if pd.isna(self.global_medians_[col]):
                self.global_medians_[col] = X[col].median()
            if pd.isna(self.global_medians_[col]):
                self.global_medians_[col] = 0  # Fallback to 0 if all values are NaN

            if col in self.group_medians_:  # Fill NaNs in calculated group medians
                self.group_medians_[col] = self.group_medians_[col].fillna(
                    self.global_medians_[col]
                )

        self.fitted_ = True
        return self

    def transform(self, X):
        if not hasattr(self, "fitted_"):
            raise RuntimeError("Transformer has not been fitted yet.")
        if not isinstance(X, pd.DataFrame):
            raise ValueError(
                "GroupedMedianImputer expects a pandas DataFrame for transform."
            )
        Xt = X.copy()
        for col in self.feature_cols_to_impute:
            if col not in Xt.columns or col not in self.group_medians_:
                continue
            group_map = self.group_medians_.get(col, pd.Series(dtype="float64"))
            global_fallback = self.global_medians_.get(col, 0)
            imputation_values = (
                Xt[self.group_col].map(group_map).fillna(global_fallback)
            )
            Xt[col] = Xt[col].fillna(imputation_values)
            if Xt[col].isnull().any():
                Xt[col].fillna(global_fallback, inplace=True)
        return Xt

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        if self._feature_names_in_ is not None:
            return np.asarray(self._feature_names_in_, dtype=object)
        raise ValueError(
            "Cannot determine output feature names. Fit the transformer or provide input_features."
        )


def create_data_transformer_pipeline(
    numeric_cols,
    categorical_cols,
    img_feature_cols,
    district_group_col,
    outlier_indicator_col,
    apply_scaling_and_transform=False,
    apply_pca=True,
    n_pca_components=N_PCA_COMPONENTS,
    include_location_features=True,
):
    """
    Pipeline that handles all data preprocessing:

    1. Grouped median imputation for numeric columns.
    2. Optional scaling/transformation.
    3. One-hot encoding for categorical columns.
    """
    grouped_imputer = GroupedMedianImputer(
        group_col=district_group_col,
        feature_cols_to_impute=numeric_cols,
        outlier_col=outlier_indicator_col,
    )

    numeric_transformer_steps = [
        ("median_imputer_safeguard", SimpleImputer(strategy="median"))
    ]
    if apply_scaling_and_transform:
        numeric_transformer_steps.extend(
            [
                ("robust_scaler", RobustScaler()),
                (
                    "power_transformer",
                    PowerTransformer(method="yeo-johnson", standardize=True),
                ),
            ]
        )
    numeric_pipeline = Pipeline(numeric_transformer_steps)
    
    categorical_pipeline = Pipeline(
        [
            (
                "imputer_missing_cat",
                SimpleImputer(strategy="constant", fill_value="missing_value"),
            ),
            (
                "one_hot_encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    min_frequency=0.01,
                    drop=None,
                ),
            ),
        ]
    )

    # Image features pipeline

    image_pipeline_steps = [
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("scaler", StandardScaler()),
    ]

    if apply_pca:
        logging.debug(
            "Applying PCA to image features with n_components = %d", N_PCA_COMPONENTS
        )
        n_pca_components = N_PCA_COMPONENTS
        image_pipeline_steps.append(("pca", PCA(n_components=n_pca_components)))

    image_pipeline = Pipeline(image_pipeline_steps)

    # Combine all transformers into a list for ColumnTransformer
    transformers = [
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ]

    if img_feature_cols and len(img_feature_cols) > 0:
        transformers.append(("img", image_pipeline, img_feature_cols))

    # ColumnTransformer processes the output of GroupedMedianImputer
    feature_processor_ct = ColumnTransformer(
        transformers=transformers,
        remainder="drop",  # Drops columns not in numeric_cols or categorical_cols (e.g., outlier_col)
        verbose_feature_names_out=False,
    )

    # The full preprocessing pipeline
    data_transformer_pipeline = Pipeline(
        [
            ("grouped_imputation", grouped_imputer),
            ("column_transformations", feature_processor_ct),
        ]
    )

    return data_transformer_pipeline
