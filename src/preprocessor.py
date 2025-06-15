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
from umap import UMAP

from src.config import load_config

warnings.filterwarnings("ignore", category=FutureWarning)

config = load_config()

N_PCA_COMPONENTS = config.get("N_PCA_COMPONENTS")
N_UMAP_COMPONENTS = config.get("N_UMAP_COMPONENTS", 50)


def validate_dataframe(X: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    Validate DataFrame to ensure data integrity.
    """
    if not isinstance(X, pd.DataFrame):
        raise ValueError(f"{name} must be a pandas DataFrame, got {type(X)}")
    
    # Check for duplicate columns
    duplicate_cols = X.columns[X.columns.duplicated()].tolist()
    if duplicate_cols:
        raise ValueError(
            f"{name} contains duplicate column names: {duplicate_cols}. "
            "This can cause issues with pandas operations. Please remove duplicates."
        )
    
    # Check index name
    if X.index.name != "id":
        raise ValueError(
            f"{name} index name must be 'id', got '{X.index.name}'. "
            "This is required for proper data handling."
        )
    
    # Check for empty DataFrame
    if X.empty:
        raise ValueError(f"{name} is empty")
    
    # Check for all-null columns
    null_cols = X.columns[X.isnull().all()].tolist()
    if null_cols and X.shape[0] > 500:
        warnings.warn(f"{name} contains columns with all null values: {null_cols}")
    
    logging.debug(f"{name} validation passed: {X.shape[0]} rows, {X.shape[1]} columns")


class ImageFeatureProcessor(BaseEstimator, TransformerMixin):
    """
    Processes image features separately by type (interior, interior_building, exterior).
    Applies dimensionality reduction (PCA or UMAP) to each type separately.
    """
    
    def __init__(self, 
                 use_pca=True, 
                 use_umap=False, 
                 n_pca_components=N_PCA_COMPONENTS,
                 n_umap_components=N_UMAP_COMPONENTS,
                 random_state=42):
        self.use_pca = use_pca
        self.use_umap = use_umap
        self.n_pca_components = n_pca_components
        self.n_umap_components = n_umap_components
        self.random_state = random_state
        
        self.imputers_ = {}
        self.scalers_ = {}
        self.reducers_ = {}
        self.feature_names_in_ = None
        self.feature_names_out_ = None

        logging.info(f"📷 ImageFeatureProcessor initialized with {self.use_pca} PCA and {self.use_umap} UMAP")
        
    def _identify_image_types(self, feature_names):
        """Identify different types of image features"""
        image_types = {
            'interior': [col for col in feature_names if 'interior_' in col and 'building' not in col],
            'interior_building': [col for col in feature_names if 'interior_building_' in col],
            'exterior': [col for col in feature_names if 'exterior_' in col]
        }
        return image_types
    
    def fit(self, X, y=None):
        """Fit the image feature processor"""
        # Validate input data if it's a DataFrame
        if isinstance(X, pd.DataFrame):
            validate_dataframe(X, "ImageFeatureProcessor input")
            self.feature_names_in_ = X.columns.tolist()
            X_array = X.values
        else:
            self.feature_names_in_ = [f"feature_{i}" for i in range(X.shape[1])]
            X_array = X
            
        image_types = self._identify_image_types(self.feature_names_in_)
        
        self.feature_names_out_ = []
        
        for img_type, cols in image_types.items():
            if not cols:
                continue
                
            print(f"Processing {img_type} image features: {len(cols)} features")
            
            # Get column indices
            col_indices = [self.feature_names_in_.index(col) for col in cols]
            X_type = X_array[:, col_indices]
            
            # Impute missing values first
            imputer = SimpleImputer(strategy="constant", fill_value=0)
            X_imputed = imputer.fit_transform(X_type)
            self.imputers_[img_type] = imputer
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_imputed)
            self.scalers_[img_type] = scaler
            
            # Apply dimensionality reduction
            if self.use_umap:
                n_components = min(self.n_umap_components, X_scaled.shape[1], X_scaled.shape[0] - 1)
                reducer = UMAP(
                    n_components=n_components,
                    random_state=self.random_state,
                    n_jobs=1  # UMAP can be unstable with multiple jobs
                )
                reducer.fit(X_scaled)
                self.reducers_[img_type] = reducer
                self.feature_names_out_.extend([f"{img_type}_umap_{i}" for i in range(n_components)])
                print(f"Applied UMAP to {img_type}: {X_scaled.shape[1]} -> {n_components}")
                
            elif self.use_pca:
                if isinstance(self.n_pca_components, float) and self.n_pca_components < 1.0:
                    # Variance-based PCA
                    n_components = self.n_pca_components
                else:
                    # Fixed number of components
                    n_components = min(int(self.n_pca_components), X_scaled.shape[1], X_scaled.shape[0] - 1)
                
                reducer = PCA(n_components=n_components, random_state=self.random_state)
                reducer.fit(X_scaled)
                self.reducers_[img_type] = reducer
                
                actual_components = reducer.n_components_
                self.feature_names_out_.extend([f"{img_type}_pca_{i}" for i in range(actual_components)])
                print(f"Applied PCA to {img_type}: {X_scaled.shape[1]} -> {actual_components}")
            else:
                # No dimensionality reduction
                self.reducers_[img_type] = None
                self.feature_names_out_.extend([f"{img_type}_{col}" for col in cols])
                print(f"No dimensionality reduction for {img_type}: keeping {len(cols)} features")
        
        return self
    
    def transform(self, X):
        """Transform image features"""
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
            
        image_types = self._identify_image_types(self.feature_names_in_)
        
        transformed_parts = []
        
        for img_type, cols in image_types.items():

            logging.info(f"📷 Processing {img_type} image features: {len(cols)} features")

            if not cols or img_type not in self.scalers_:
                continue
                
            # Get column indices
            col_indices = [self.feature_names_in_.index(col) for col in cols]
            X_type = X_array[:, col_indices]
            
            # Impute missing values first
            X_imputed = self.imputers_[img_type].transform(X_type)
            
            # Scale features
            X_scaled = self.scalers_[img_type].transform(X_imputed)
            
            # Apply dimensionality reduction
            if self.reducers_[img_type] is not None:
                X_reduced = self.reducers_[img_type].transform(X_scaled)
                transformed_parts.append(X_reduced)
            else:
                transformed_parts.append(X_scaled)
        
        if not transformed_parts:
            # No image features found
            return np.empty((X_array.shape[0], 0))
        
        return np.concatenate(transformed_parts, axis=1)
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names"""
        if self.feature_names_out_ is not None:
            return np.array(self.feature_names_out_, dtype=object)
        else:
            return np.array([], dtype=object)


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
        validate_dataframe(X, "GroupedMedianImputer input")
        
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
            if col not in df_for_stats.columns:
                continue
            group_medians = df_for_stats.groupby(self.group_col)[col].median()
            global_median = df_for_stats[col].median()
            self.group_medians_[col] = group_medians
            self.global_medians_[col] = global_median

        self.fitted_ = True
        return self

    def transform(self, X):
        if not hasattr(self, "fitted_"):
            raise RuntimeError("Transformer has not been fitted yet.")
        
        validate_dataframe(X, "GroupedMedianImputer transform input")
        
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
    4. Image feature processing (if any) - deprecated in favor of separate image handling
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

    # Combine all transformers into a list for ColumnTransformer
    transformers = [
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ]

    # Legacy image handling - kept for backward compatibility but discouraged
    if img_feature_cols and len(img_feature_cols) > 0:
        warnings.warn("Direct image feature inclusion in preprocessor is deprecated. Use separate image handling.", DeprecationWarning)
        image_pipeline_steps = [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
        ]

        if apply_pca:
            logging.debug(
                "Applying PCA to image features with n_components = %d", n_pca_components
            )
            image_pipeline_steps.append(("pca", PCA(n_components=n_pca_components)))

        image_pipeline = Pipeline(image_pipeline_steps)
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


def create_enhanced_data_transformer_pipeline(
    numeric_cols,
    categorical_cols,
    img_feature_cols,
    district_group_col,
    outlier_indicator_col,
    apply_scaling_and_transform=False,
    apply_pca_img_transform=False,
    apply_umap_img_transform=False,
    n_pca_components=N_PCA_COMPONENTS,
    n_umap_components=N_UMAP_COMPONENTS,
    include_location_features=True,
    random_state=42,
):
    """
    Enhanced pipeline that handles tabular and image features separately.
    
    Returns:
    --------
    tabular_transformer : Pipeline
        Transformer for non-image features
    image_transformer : ImageFeatureProcessor or None
        Transformer for image features (if any)
    """
    # Create tabular data transformer (same as before)
    tabular_transformer = create_data_transformer_pipeline(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        img_feature_cols=[],  # No images in tabular transformer
        district_group_col=district_group_col,
        outlier_indicator_col=outlier_indicator_col,
        apply_scaling_and_transform=apply_scaling_and_transform,
        apply_pca=False,  # No PCA for tabular data
        include_location_features=include_location_features,
    )
    
    # Create image transformer if needed
    image_transformer = None
    if img_feature_cols and len(img_feature_cols) > 0:
        use_pca = apply_pca_img_transform and not apply_umap_img_transform
        use_umap = apply_umap_img_transform
        
        image_transformer = ImageFeatureProcessor(
            use_pca=use_pca,
            use_umap=use_umap,
            n_pca_components=n_pca_components,
            n_umap_components=n_umap_components,
            random_state=random_state
        )
        
        print(f"Created image transformer: PCA={use_pca}, UMAP={use_umap}")
    
    return tabular_transformer, image_transformer


class CombinedFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Combines tabular and image transformers into a single transformer.
    Handles feature selection metadata to decide whether to include images.
    """
    
    def __init__(self, 
                 tabular_transformer,
                 image_transformer=None,
                 selected_features=None,
                 should_include_images=False):
        self.tabular_transformer = tabular_transformer
        self.image_transformer = image_transformer
        self.selected_features = selected_features or []
        self.should_include_images = should_include_images
        self.feature_names_out_ = None

        logging.info(f"CombinedFeatureTransformer initialized with {len(self.selected_features)} selected features")
        
    def fit(self, X_tabular, X_images=None, y=None):
        """Fit both transformers"""
        validate_dataframe(X_tabular, "CombinedFeatureTransformer tabular input")
        
        if X_images is not None:
            validate_dataframe(X_images, "CombinedFeatureTransformer image input")
        
        # Fit tabular transformer
        self.tabular_transformer.fit(X_tabular, y)
        
        # Fit image transformer if needed
        if self.should_include_images and self.image_transformer is not None and X_images is not None:
            self.image_transformer.fit(X_images, y)
            print("Fitted image transformer")
        
        # Build feature names
        self._build_feature_names()
        
        print(f"CombinedFeatureTransformer fitted with {len(self.selected_features)} selected features")
        
        return self
    
    def transform(self, X, X_images=None):
        """Transform features and combine
        
        Parameters:
        -----------
        X : pd.DataFrame
            Tabular features (when used as standard sklearn transformer)
            or X_tabular when called with keyword arguments
        X_images : pd.DataFrame, optional
            Image features (only used when called with keyword arguments)
        """
        # Handle both sklearn standard signature and our custom dual-input signature
        if X_images is not None:
            # Called with explicit X_images, X is the tabular data
            return self._transform_impl(X, X_images)
        else:
            # Standard sklearn transform call, X should be tabular data
            # For image features, we need to get them from somewhere else
            # This case should not happen in our current setup but handle it gracefully
            return self._transform_impl(X, None)
    
    def _transform_impl(self, X_tabular, X_images=None):
        """Internal transform implementation"""
        validate_dataframe(X_tabular, "CombinedFeatureTransformer transform tabular input")
        
        if X_images is not None:
            validate_dataframe(X_images, "CombinedFeatureTransformer transform image input")
        
        # Transform tabular features
        X_tabular_transformed = self.tabular_transformer.transform(X_tabular)
        
        # Get tabular feature names
        try:
            tabular_feature_names = self.tabular_transformer.get_feature_names_out()
        except:
            tabular_feature_names = [f"tabular_{i}" for i in range(X_tabular_transformed.shape[1])]
        
        # Create tabular DataFrame
        if isinstance(X_tabular_transformed, np.ndarray):
            X_tabular_df = pd.DataFrame(X_tabular_transformed, columns=tabular_feature_names, index=X_tabular.index)
        else:
            X_tabular_df = X_tabular_transformed
        
        # Filter selected tabular features
        available_tabular_features = [f for f in self.selected_features if f in X_tabular_df.columns]
        X_selected_tabular = X_tabular_df[available_tabular_features]
        
        # Transform and add image features if needed
        if self.should_include_images and self.image_transformer is not None and X_images is not None:
            X_images_transformed = self.image_transformer.transform(X_images)
            image_feature_names = self.image_transformer.get_feature_names_out()
            
            # Create image DataFrame and reindex to match tabular data, filling missing with 0
            X_images_df = pd.DataFrame(X_images_transformed, columns=image_feature_names, index=X_images.index)
            X_images_df = X_images_df.reindex(X_tabular.index).fillna(0)
            
            # Combine tabular and image features
            X_combined = pd.concat([X_selected_tabular, X_images_df], axis=1)
            print(f"Combined features: {X_selected_tabular.shape[1]} tabular + {X_images_df.shape[1]} image = {X_combined.shape[1]} total")
        else:
            X_combined = X_selected_tabular
            print(f"Using tabular features only: {X_combined.shape[1]} features")
        
        return X_combined
    
    def _build_feature_names(self):
        """Build output feature names"""
        # Get tabular feature names
        try:
            tabular_features = self.tabular_transformer.get_feature_names_out().tolist()
        except:
            tabular_features = [f"tabular_{i}" for i in range(100)]  # Fallback
        
        # Filter selected tabular features
        available_tabular_features = [f for f in self.selected_features if f in tabular_features]
        
        self.feature_names_out_ = available_tabular_features.copy()
        
        # Add image features if needed
        if self.should_include_images and self.image_transformer is not None:
            try:
                image_features = self.image_transformer.get_feature_names_out().tolist()
                self.feature_names_out_.extend(image_features)
            except:
                pass  # No image features
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names"""
        if self.feature_names_out_ is not None:
            return np.array(self.feature_names_out_, dtype=object)
        else:
            return np.array([], dtype=object)
