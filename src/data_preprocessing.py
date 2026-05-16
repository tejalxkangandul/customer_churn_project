"""
Data Preprocessing Module
Handles data cleaning, handling missing values, and data preparation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Class to handle data preprocessing"""
    
    def __init__(self):
        """Initialize the preprocessor"""
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.columns_to_drop = []
        
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            df: Input dataframe
            strategy: Strategy to handle missing values ('mean', 'median', 'drop')
            
        Returns:
            DataFrame: Data with missing values handled
        """
        df = df.copy()
        
        missing_cols = df.columns[df.isnull().any()].tolist()
        
        if len(missing_cols) == 0:
            logger.info("No missing values found")
            return df
        
        logger.info(f"Found missing values in columns: {missing_cols}")
        
        for col in missing_cols:
            if df[col].dtype in ['float64', 'int64']:
                if strategy == 'mean':
                    df[col].fillna(df[col].mean(), inplace=True)
                elif strategy == 'median':
                    df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        logger.info("Missing values handled")
        return df
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame: Data without duplicates
        """
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed = initial_rows - len(df)
        logger.info(f"Removed {removed} duplicate rows")
        return df
    
    def remove_outliers_iqr(self, df: pd.DataFrame, column: str, multiplier: float = 1.5) -> pd.DataFrame:
        """
        Remove outliers using IQR method
        
        Args:
            df: Input dataframe
            column: Column to check for outliers
            multiplier: IQR multiplier for outlier detection
            
        Returns:
            DataFrame: Data without outliers
        """
        df = df.copy()
        
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        initial_rows = len(df)
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        removed = initial_rows - len(df)
        
        logger.info(f"Removed {removed} outliers from {column}")
        return df
    
    def encode_categorical(self, df: pd.DataFrame, categorical_cols: list, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical variables
        
        Args:
            df: Input dataframe
            categorical_cols: List of categorical columns to encode
            fit: Whether to fit new encoders (True for training, False for prediction)
            
        Returns:
            DataFrame: Data with encoded categorical variables
        """
        df = df.copy()
        
        for col in categorical_cols:
            if col in df.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    if col in self.label_encoders:
                        df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        logger.info(f"Encoded {len(categorical_cols)} categorical columns")
        return df
    
    def scale_features(self, df: pd.DataFrame, numeric_cols: list, fit: bool = True) -> pd.DataFrame:
        """
        Scale numeric features
        
        Args:
            df: Input dataframe
            numeric_cols: List of numeric columns to scale
            fit: Whether to fit new scaler (True for training, False for prediction)
            
        Returns:
            DataFrame: Data with scaled features
        """
        df = df.copy()
        
        cols_to_scale = [col for col in numeric_cols if col in df.columns]
        
        if fit:
            df[cols_to_scale] = self.scaler.fit_transform(df[cols_to_scale])
        else:
            df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])
        
        logger.info(f"Scaled {len(cols_to_scale)} numeric columns")
        return df
    
    def preprocess(self, df: pd.DataFrame, target_col: str, categorical_cols: list, 
                   numeric_cols: list, fit: bool = True) -> tuple:
        """
        Complete preprocessing pipeline
        
        Args:
            df: Input dataframe
            target_col: Name of target column
            categorical_cols: List of categorical columns
            numeric_cols: List of numeric columns
            fit: Whether to fit transformers
            
        Returns:
            tuple: (processed_features, target_values)
        """
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Remove duplicates
        df = self.remove_duplicates(df)
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col].copy()
        
        # Encode categorical variables
        if categorical_cols:
            X = self.encode_categorical(X, categorical_cols, fit=fit)
        
        # Scale numeric features
        if numeric_cols:
            X = self.scale_features(X, numeric_cols, fit=fit)
        
        logger.info("Preprocessing completed successfully")
        return X, y
