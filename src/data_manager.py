import logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

from src.split_data import get_train_test_data, get_train_test_img
from src.preprocessor import create_data_transformer_pipeline

class DataManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.target_variable = config.get("TARGET_VARIABLE")
        self.district_col_name = "district"
        self.outlier_col_name = "outlier"
        self.include_location_features = config.get("INCLUDE_LOCATION_FEATURES", False)
        
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
        logging.info("Loading raw data...")
        self.X_train_full_raw, self.X_test_full_raw, self.Y_train_raw, self.Y_test_raw = get_train_test_data()
        self.img_train_raw, self.img_test_raw = get_train_test_img()

        logging.info(
            f"Raw Train Data: {self.X_train_full_raw.shape}, Raw Test Data: {self.X_test_full_raw.shape}"
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

    def get_log_transformed_target(self) -> pd.Series:
        """Get log-transformed target variable"""
        return np.log(self.Y_train_raw[self.target_variable])

    def get_test_target(self) -> pd.Series:
        """Get test target variable"""
        return self.Y_test_raw[self.target_variable]

    def create_data_transformer(self, apply_scaling: bool, apply_pca: bool, n_pca_components: int):
        """Create a data transformer pipeline"""
        # ALWAYS use district for grouped imputation
        return create_data_transformer_pipeline(
            numeric_cols=self.original_numeric_cols,
            categorical_cols=self.original_categorical_cols,
            img_feature_cols=self.original_img_cols,
            district_group_col=self.district_col_name, 
            outlier_indicator_col=self.outlier_col_name,
            apply_scaling_and_transform=apply_scaling,
            apply_pca=apply_pca,
            n_pca_components=n_pca_components
        ) 