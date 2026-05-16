"""
Unit tests for data preprocessing module
"""

import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_preprocessing import DataPreprocessor


class TestDataPreprocessor(unittest.TestCase):
    """Test cases for DataPreprocessor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.preprocessor = DataPreprocessor()
        
        # Create sample data
        self.sample_df = pd.DataFrame({
            'age': [25, 30, np.nan, 45, 50],
            'tenure': [12, 24, 36, 48, np.nan],
            'monthly_charges': [50.0, 75.0, 60.0, 80.0, 90.0],
            'category': ['A', 'B', 'A', 'C', 'B'],
            'target': [0, 1, 0, 1, 0]
        })
    
    def test_handle_missing_values(self):
        """Test missing value handling"""
        df_filled = self.preprocessor.handle_missing_values(self.sample_df)
        
        # Check no missing values remain
        self.assertEqual(df_filled.isnull().sum().sum(), 0)
    
    def test_remove_duplicates(self):
        """Test duplicate removal"""
        df_dup = self.sample_df.copy()
        df_dup = pd.concat([df_dup, df_dup.iloc[0:1]], ignore_index=True)
        
        df_clean = self.preprocessor.remove_duplicates(df_dup)
        
        # Check duplicates removed
        self.assertEqual(len(df_clean), len(self.sample_df))
    
    def test_encode_categorical(self):
        """Test categorical encoding"""
        df_encoded = self.preprocessor.encode_categorical(
            self.sample_df,
            categorical_cols=['category'],
            fit=True
        )
        
        # Check encoding worked
        self.assertIsInstance(df_encoded['category'].iloc[0], (int, np.integer))
    
    def test_scale_features(self):
        """Test feature scaling"""
        numeric_cols = ['age', 'monthly_charges']
        df_scaled = self.preprocessor.scale_features(
            self.sample_df,
            numeric_cols=numeric_cols,
            fit=True
        )
        
        # Check scaling applied
        # Scaled values should be around 0 mean and 1 std
        self.assertLess(abs(df_scaled[numeric_cols].mean().mean()), 0.1)


if __name__ == '__main__':
    unittest.main()
