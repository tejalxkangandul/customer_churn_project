"""
Customer Churn Analysis and Prediction Package
"""

from .data_loading import DataLoader
from .data_preprocessing import DataPreprocessor
from .feature_engineering import FeatureEngineer
from .model_training import ModelTrainer
from .model_evaluation import ModelEvaluator
from .prediction import ChurnPredictor
from .utils import setup_logging, load_config, save_config

__version__ = '1.0.0'
__author__ = 'Data Science Team'

__all__ = [
    'DataLoader',
    'DataPreprocessor',
    'FeatureEngineer',
    'ModelTrainer',
    'ModelEvaluator',
    'ChurnPredictor',
    'setup_logging',
    'load_config',
    'save_config'
]
