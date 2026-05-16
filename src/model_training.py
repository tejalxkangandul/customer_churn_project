"""
Model Training Module
Handles training of machine learning models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import logging
import pickle

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Class to handle model training"""
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize ModelTrainer
        
        Args:
            test_size: Proportion of data to use for testing
            random_state: Random state for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        """
        Split data into training and testing sets
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y  # Maintain class distribution
        )
        
        logger.info(f"Data split: {len(self.X_train)} training, {len(self.X_test)} testing samples")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_logistic_regression(self, **kwargs) -> object:
        """
        Train Logistic Regression model
        
        Args:
            **kwargs: Additional parameters for LogisticRegression
            
        Returns:
            Trained LogisticRegression model
        """
        logger.info("Training Logistic Regression model...")
        
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
            **kwargs
        )
        self.model.fit(self.X_train, self.y_train)
        
        logger.info("Logistic Regression model trained successfully")
        return self.model
    
    def train_random_forest(self, n_estimators: int = 100, **kwargs) -> object:
        """
        Train Random Forest model
        
        Args:
            n_estimators: Number of trees in the forest
            **kwargs: Additional parameters for RandomForestClassifier
            
        Returns:
            Trained RandomForestClassifier model
        """
        logger.info("Training Random Forest model...")
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
            **kwargs
        )
        self.model.fit(self.X_train, self.y_train)
        
        logger.info("Random Forest model trained successfully")
        return self.model
    
    def train_gradient_boosting(self, n_estimators: int = 100, **kwargs) -> object:
        """
        Train Gradient Boosting model
        
        Args:
            n_estimators: Number of boosting stages to perform
            **kwargs: Additional parameters for GradientBoostingClassifier
            
        Returns:
            Trained GradientBoostingClassifier model
        """
        logger.info("Training Gradient Boosting model...")
        
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            random_state=self.random_state,
            **kwargs
        )
        self.model.fit(self.X_train, self.y_train)
        
        logger.info("Gradient Boosting model trained successfully")
        return self.model
    
    def train_svm(self, kernel: str = 'rbf', **kwargs) -> object:
        """
        Train Support Vector Machine model
        
        Args:
            kernel: Kernel type for SVM
            **kwargs: Additional parameters for SVC
            
        Returns:
            Trained SVC model
        """
        logger.info("Training SVM model...")
        
        self.model = SVC(
            kernel=kernel,
            random_state=self.random_state,
            probability=True,
            **kwargs
        )
        self.model.fit(self.X_train, self.y_train)
        
        logger.info("SVM model trained successfully")
        return self.model
    
    def get_model(self) -> object:
        """
        Get the trained model
        
        Returns:
            Trained model
        """
        if self.model is None:
            raise ValueError("No model has been trained yet")
        return self.model
    
    def save_model(self, filepath: str) -> None:
        """
        Save trained model to file
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> object:
        """
        Load a trained model from file
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded model
        """
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        
        logger.info(f"Model loaded from {filepath}")
        return self.model
