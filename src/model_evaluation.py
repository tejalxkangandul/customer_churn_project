"""
Model Evaluation Module
Handles evaluation of trained models
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, auc, precision_recall_curve
)
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Class to handle model evaluation"""
    
    def __init__(self, model: object):
        """
        Initialize ModelEvaluator
        
        Args:
            model: Trained model object
        """
        self.model = model
        self.results = {}
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Comprehensive model evaluation
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        # Get predictions
        y_pred = self.model.predict(X_test)
        
        # Get probability predictions if available
        if hasattr(self.model, 'predict_proba'):
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        self.results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred),
        }
        
        # Add ROC-AUC if probability predictions are available
        if y_pred_proba is not None:
            self.results['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        
        logger.info("Model evaluation completed")
        logger.info(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                   f"Recall: {recall:.4f}, F1-Score: {f1:.4f}")
        
        return self.results
    
    def print_results(self) -> None:
        """Print evaluation results"""
        if not self.results:
            logger.warning("No evaluation results available")
            return
        
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        
        print(f"\nAccuracy:  {self.results['accuracy']:.4f}")
        print(f"Precision: {self.results['precision']:.4f}")
        print(f"Recall:    {self.results['recall']:.4f}")
        print(f"F1-Score:  {self.results['f1_score']:.4f}")
        
        if 'roc_auc' in self.results:
            print(f"ROC-AUC:   {self.results['roc_auc']:.4f}")
        
        print("\nConfusion Matrix:")
        print(self.results['confusion_matrix'])
        
        print("\nClassification Report:")
        print(self.results['classification_report'])
        print("="*50 + "\n")
    
    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """
        Get feature importance from the model
        
        Args:
            feature_names: List of feature names
            
        Returns:
            DataFrame: Feature importance scores
        """
        if not hasattr(self.model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_ attribute")
            return None
        
        importances = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df
    
    def get_results(self) -> dict:
        """
        Get evaluation results
        
        Returns:
            dict: Evaluation metrics
        """
        return self.results
