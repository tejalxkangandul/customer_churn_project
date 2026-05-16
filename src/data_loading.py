"""
Data Loading Module
Handles loading and initial exploration of customer data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Class to handle data loading operations"""
    
    def __init__(self, data_path: str):
        """
        Initialize DataLoader
        
        Args:
            data_path: Path to the data file
        """
        self.data_path = Path(data_path)
        self.df = None
    
    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Returns:
            DataFrame: Loaded customer data
        """
        try:
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            return self.df
        except FileNotFoundError:
            logger.error(f"File not found: {self.data_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def explore_data(self) -> dict:
        """
        Explore and return basic statistics about the data
        
        Returns:
            dict: Statistics and information about the dataset
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        exploration = {
            'shape': self.df.shape,
            'columns': self.df.columns.tolist(),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'duplicates': self.df.duplicated().sum(),
            'numeric_stats': self.df.describe().to_dict(),
        }
        
        logger.info(f"Data exploration completed: {exploration['shape'][0]} rows, {exploration['shape'][1]} columns")
        return exploration
    
    def get_data(self) -> pd.DataFrame:
        """
        Get loaded dataframe
        
        Returns:
            DataFrame: The loaded customer data
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return self.df.copy()
