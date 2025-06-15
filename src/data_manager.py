import logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
import json

from src.split_data import get_train_test_data, _get_data
from src.preprocessor import (
    create_data_transformer_pipeline, 
    create_enhanced_data_transformer_pipeline,
    CombinedFeatureTransformer
)

class DataManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.target_variable = config.get("TARGET_VARIABLE")
        self.district_col_name = "district"
        self.outlier_col_name = "outlier"
        self.include_location_features = config.get("INCLUDE_LOCATION_FEATURES", False)
        
        self._data = None
        self.X_train_full_raw = None
        self.X_test_full_raw = None
        self.Y_train_raw = None
        self.Y_test_raw = None
        self.img_train_raw = None
        self.img_test_raw = None
        
        self.original_img_cols = None
        self.original_numeric_cols = None
        self.original_categorical_cols = None

    def load_data(self) -> None:
        """Load and prepare the raw data"""
        if self._data is None:
            logging.info("Loading and cleaning raw data for the first time...")
            self._data = _get_data()
        
        logging.info("Splitting data into train/test sets...")
        self.X_train_full_raw, self.X_test_full_raw, self.Y_train_raw, self.Y_test_raw, self.img_train_raw, self.img_test_raw = get_train_test_data(data=self._data)

        logging.info(
            f"Raw Train Data: {self.X_train_full_raw.shape}, Raw Test Data: {self.X_test_full_raw.shape}"
        )
        logging.info(
            f"Raw Image Train Data: {self.img_train_raw.shape}, Raw Image Test Data: {self.img_test_raw.shape}"
        )

        self._prepare_feature_columns()

    def _prepare_feature_columns(self) -> None:
        """Prepare feature columns by type"""
        self.original_img_cols = self.img_train_raw.columns.tolist()
        self.original_numeric_cols = self.X_train_full_raw.select_dtypes(
            include=np.number
        ).columns.tolist()
        self.original_numeric_cols = [
            col
            for col in self.original_numeric_cols
            if col != self.target_variable
            and col != self.outlier_col_name
            and col not in self.original_img_cols
        ]
        self.original_categorical_cols = self.X_train_full_raw.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        
        # Handle location features based on setting
        location_features = [self.district_col_name, "neighborhood"]
        
        if not self.include_location_features:
            # Remove location features from categorical columns if not including them
            self.original_categorical_cols = [
                col for col in self.original_categorical_cols if col not in location_features
            ]
            logging.info(f"Excluding location features {location_features} from categorical columns")
        else:
            logging.info(f"Including location features {location_features} in categorical columns")

        logging.info(f"Original Numeric Cols: {len(self.original_numeric_cols)}")
        logging.info(f"Original Categorical Cols: {len(self.original_categorical_cols)}")
        logging.info(f"Original Image Cols: {len(self.original_img_cols)}")

    def get_transformed_data(self, data_transformer) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Transform data using the provided transformer"""
        X_train_processed_np = data_transformer.transform(self.X_train_full_raw.copy())
        X_test_processed_np = data_transformer.transform(self.X_test_full_raw.copy())
        
        transformed_feature_names = data_transformer.get_feature_names_out()
        
        X_train_processed_df = pd.DataFrame(
            X_train_processed_np,
            columns=transformed_feature_names,
            index=self.X_train_full_raw.index,
        )
        X_test_processed_df = pd.DataFrame(
            X_test_processed_np,
            columns=transformed_feature_names,
            index=self.X_test_full_raw.index,
        )
        
        return X_train_processed_df, X_test_processed_df
    
    def get_combined_transformed_data(self, combined_transformer) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Transform data using the combined transformer (tabular + images)"""
        X_train_combined = combined_transformer.transform(
            self.X_train_full_raw.copy(), 
            self.img_train_raw.copy()
        )
        X_test_combined = combined_transformer.transform(
            self.X_test_full_raw.copy(), 
            self.img_test_raw.copy()
        )
        
        return X_train_combined, X_test_combined

    def get_log_transformed_target(self) -> pd.Series:
        """Get log-transformed target variable"""
        return np.log(self.Y_train_raw[self.target_variable])

    def get_test_target(self) -> pd.Series:
        """Get test target variable"""
        return self.Y_test_raw[self.target_variable]

    def create_data_transformer(self, apply_scaling: bool, apply_pca: bool, n_pca_components: int):
        """Create a data transformer pipeline (legacy method)"""
        # ALWAYS use district for grouped imputation
        return create_data_transformer_pipeline(
            numeric_cols=self.original_numeric_cols,
            categorical_cols=self.original_categorical_cols,
            img_feature_cols=[],  # No images in legacy transformer
            district_group_col=self.district_col_name, 
            outlier_indicator_col=self.outlier_col_name,
            apply_scaling_and_transform=apply_scaling,
            apply_pca=apply_pca,
            n_pca_components=n_pca_components
        )
    
    def create_enhanced_transformer(self, 
                                  apply_scaling: bool, 
                                  apply_pca_img: bool = False,
                                  apply_umap_img: bool = False,
                                  n_pca_components: int = 0.8,
                                  n_umap_components: int = 50) -> Tuple:
        """Create enhanced transformers for tabular and image data separately"""
        
        tabular_transformer, image_transformer = create_enhanced_data_transformer_pipeline(
            numeric_cols=self.original_numeric_cols,
            categorical_cols=self.original_categorical_cols,
            img_feature_cols=self.original_img_cols,
            district_group_col=self.district_col_name,
            outlier_indicator_col=self.outlier_col_name,
            apply_scaling_and_transform=apply_scaling,
            apply_pca_img_transform=apply_pca_img,
            apply_umap_img_transform=apply_umap_img,
            n_pca_components=n_pca_components,
            n_umap_components=n_umap_components,
            include_location_features=self.include_location_features,
            random_state=self.config.get("RANDOM_STATE", 42)
        )
        
        return tabular_transformer, image_transformer
    
    def create_combined_transformer_from_metadata(self, 
                                                feature_selection_metadata: Dict,
                                                selected_features: list) -> CombinedFeatureTransformer:
        """Create a combined transformer based on feature selection metadata"""
        
        # Extract parameters from metadata
        apply_scaling = feature_selection_metadata.get("apply_scale_transform", False)
        should_include_images = feature_selection_metadata.get("should_include_images", False)
        apply_pca_img = feature_selection_metadata.get("apply_pca_img_transform", False)
        apply_umap_img = feature_selection_metadata.get("apply_umap_img_transform", False)
        n_pca_components = feature_selection_metadata.get("n_pca_components", 0.8)
        n_umap_components = feature_selection_metadata.get("n_umap_components", 50)
        
        logging.info(f"Creating combined transformer - Images: {should_include_images}, "
                    f"PCA: {apply_pca_img}, UMAP: {apply_umap_img}")
        
        # Create separate transformers
        tabular_transformer, image_transformer = self.create_enhanced_transformer(
            apply_scaling=apply_scaling,
            apply_pca_img=apply_pca_img,
            apply_umap_img=apply_umap_img,
            n_pca_components=n_pca_components,
            n_umap_components=n_umap_components
        )
        
        # Create combined transformer
        combined_transformer = CombinedFeatureTransformer(
            tabular_transformer=tabular_transformer,
            image_transformer=image_transformer,
            selected_features=selected_features,
            should_include_images=should_include_images
        )
        
        return combined_transformer
    
    def load_feature_selection_metadata(self, feature_set_path: str) -> Tuple[list, Dict]:
        """Load feature selection results and metadata"""
        try:
            with open(feature_set_path, 'r') as f:
                data = json.load(f)
            
            selected_features = data.get("selected_features", [])
            metadata = data.get("metadata", {})
            
            logging.info(f"Loaded {len(selected_features)} selected features from {feature_set_path}")
            logging.info(f"Feature selection metadata: {metadata}")
            
            return selected_features, metadata
        except Exception as e:
            logging.error(f"Error loading feature selection data from {feature_set_path}: {e}")
            raise
    
    def get_transformed_data_from_feature_set(self, feature_set_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Complete transformation pipeline based on feature selection metadata.
        This is the single source of truth for all training scripts.
        
        Returns:
        --------
        X_train_transformed : pd.DataFrame
            Transformed training data with selected features (and image features if applicable)
        X_test_transformed : pd.DataFrame
            Transformed test data with selected features (and image features if applicable)
        feature_info : Dict
            Information about the features being used
        """
        # Load feature selection metadata
        selected_features, feature_selection_metadata = self.load_feature_selection_metadata(feature_set_path)
        
        # Check if this is a new approach feature set
        is_new_approach = feature_selection_metadata.get("images_handled_separately", False)
        
        if is_new_approach:
            logging.info("🆕 Using new combined transformer approach")
            
            # Create and fit combined transformer
            combined_transformer = self.create_combined_transformer_from_metadata(
                feature_selection_metadata=feature_selection_metadata,
                selected_features=selected_features
            )
            
            # Fit the combined transformer
            combined_transformer.fit(
                X_tabular=self.X_train_full_raw.copy(),
                X_images=self.img_train_raw.copy(),
                y=self.get_log_transformed_target()
            )
            
            # Transform training and test data
            X_train_transformed = combined_transformer.transform(
                X=self.X_train_full_raw.copy(),
                X_images=self.img_train_raw.copy()
            )
            
            X_test_transformed = combined_transformer.transform(
                X=self.X_test_full_raw.copy(),
                X_images=self.img_test_raw.copy()
            )
            
            # Analyze features
            all_features = X_train_transformed.columns.tolist()
            image_features = [f for f in all_features if any(img_prefix in f for img_prefix in ['umap_', 'pca_', 'img_', 'interior_', 'exterior_', 'vector_', 'feature_'])]
            tabular_features = [f for f in all_features if f not in image_features]
            
            feature_info = {
                "total_features": len(all_features),
                "tabular_features": len(tabular_features),
                "image_features": len(image_features),
                "selected_tabular_features": len(selected_features),
                "feature_selection_metadata": feature_selection_metadata,
                "transformer_type": "combined",
                "should_include_images": feature_selection_metadata.get("should_include_images", False)
            }
            
            logging.info(f"🖼️ Combined transformer: {len(tabular_features)} tabular + {len(image_features)} image = {len(all_features)} total features")
            if image_features:
                logging.info(f"🖼️ Image feature examples: {image_features[:5]}{'...' if len(image_features) > 5 else ''}")
            
        else:
            logging.info("🔧 Using legacy transformer approach")
            
            # Legacy approach
            apply_scaling = feature_selection_metadata.get("apply_scale_transform", False)
            apply_pca = feature_selection_metadata.get("apply_pca_img_transform", False)
            n_pca_components = feature_selection_metadata.get("n_pca_components", 0.8)
            
            # Create legacy transformer
            data_transformer = self.create_data_transformer(
                apply_scaling=apply_scaling,
                apply_pca=apply_pca,
                n_pca_components=n_pca_components
            )
            
            # Fit transformer
            data_transformer.fit(self.X_train_full_raw.copy(), self.get_log_transformed_target())
            
            # Transform data
            X_train_transformed_np = data_transformer.transform(self.X_train_full_raw.copy())
            X_test_transformed_np = data_transformer.transform(self.X_test_full_raw.copy())
            
            # Get feature names
            feature_names = data_transformer.get_feature_names_out()
            
            # Create DataFrames
            X_train_transformed = pd.DataFrame(
                X_train_transformed_np,
                columns=feature_names,
                index=self.X_train_full_raw.index
            )
            X_test_transformed = pd.DataFrame(
                X_test_transformed_np,
                columns=feature_names,
                index=self.X_test_full_raw.index
            )
            
            # Filter to selected features
            X_train_transformed = X_train_transformed[selected_features]
            X_test_transformed = X_test_transformed[selected_features]
            
            feature_info = {
                "total_features": len(selected_features),
                "tabular_features": len(selected_features),
                "image_features": 0,
                "selected_tabular_features": len(selected_features),
                "feature_selection_metadata": feature_selection_metadata,
                "transformer_type": "legacy",
                "should_include_images": False
            }
            
            logging.info(f"🔧 Legacy transformer: {len(selected_features)} features")
        
        return X_train_transformed, X_test_transformed, feature_info 