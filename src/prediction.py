"""
Prediction Module
Handles making predictions on new data
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ChurnPredictor:
    """Class to handle churn predictions"""
    
    def __init__(self, model: object, preprocessor: object):
        """
        Initialize ChurnPredictor
        
        Args:
            model: Trained model object
            preprocessor: DataPreprocessor object with fitted transformers
        """
        self.model = model
        self.preprocessor = preprocessor
    
    def predict(self, X: pd.DataFrame) -> np.array:
        """
        Make predictions on new data
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            np.array: Predicted labels (0 or 1)
        """
        predictions = self.model.predict(X)
        logger.info(f"Made predictions for {len(X)} samples")
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.array:
        """
        Get prediction probabilities
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            np.array: Probability predictions
        """
        if not hasattr(self.model, 'predict_proba'):
            logger.warning("Model does not support probability predictions")
            return None
        
        probabilities = self.model.predict_proba(X)
        logger.info(f"Got probabilities for {len(X)} samples")
        return probabilities
    
    def predict_with_confidence(self, X: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """
        Make predictions with confidence scores
        
        Args:
            X: Feature matrix for prediction
            threshold: Confidence threshold for churn prediction
            
        Returns:
            DataFrame: Predictions with confidence and risk level
        """
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        
        if probabilities is None:
            results = pd.DataFrame({
                'prediction': predictions,
                'churn_label': ['Will Churn' if p == 1 else 'Will Not Churn' for p in predictions]
            })
        else:
            churn_prob = probabilities[:, 1]
            
            results = pd.DataFrame({
                'prediction': predictions,
                'churn_probability': churn_prob,
                'confidence': np.max(probabilities, axis=1),
                'churn_label': ['Will Churn' if p == 1 else 'Will Not Churn' for p in predictions],
                'risk_level': ['High Risk' if prob >= 0.7 else 'Medium Risk' if prob >= threshold else 'Low Risk' 
                               for prob in churn_prob]
            })
        
        logger.info("Predictions with confidence generated")
        return results
    
    def batch_predict(self, data_path: str, categorical_cols: list, numeric_cols: list) -> pd.DataFrame:
        """
        Predict churn for a batch of customers from a file
        
        Args:
            data_path: Path to the data file
            categorical_cols: List of categorical columns
            numeric_cols: List of numeric columns
            
        Returns:
            DataFrame: Original data with predictions
        """
        # Load data
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} samples from {data_path}")
        
        # Preprocess
        X, _ = self.preprocessor.preprocess(
            df, 
            target_col='churn',  # Dummy target
            categorical_cols=categorical_cols,
            numeric_cols=numeric_cols,
            fit=False
        )
        
        # Remove target column if it exists
        if 'churn' in X.columns:
            X = X.drop(columns=['churn'])
        
        # Make predictions
        predictions = self.predict_with_confidence(X)
        
        # Add predictions to original data
        result_df = df.copy()
        result_df['predicted_churn'] = predictions['churn_label']
        
        if 'churn_probability' in predictions.columns:
            result_df['churn_probability'] = predictions['churn_probability']
            result_df['risk_level'] = predictions['risk_level']
        
        logger.info("Batch prediction completed")
        return result_df
    
    def identify_at_risk_customers(self, X: pd.DataFrame, customer_ids: list = None, 
                                   risk_threshold: float = 0.5) -> pd.DataFrame:
        """
        Identify customers at risk of churning
        
        Args:
            X: Feature matrix
            customer_ids: Optional list of customer IDs
            risk_threshold: Threshold for identifying at-risk customers
            
        Returns:
            DataFrame: At-risk customers with probabilities
        """
        predictions = self.predict_with_confidence(X, threshold=risk_threshold)
        
        # Filter at-risk customers
        if 'churn_probability' in predictions.columns:
            at_risk = predictions[predictions['churn_probability'] >= risk_threshold].copy()
        else:
            at_risk = predictions[predictions['prediction'] == 1].copy()
        
        if customer_ids is not None:
            at_risk.insert(0, 'customer_id', [customer_ids[i] for i in at_risk.index])
        
        at_risk = at_risk.sort_values('churn_probability', ascending=False, 
                                      na_position='last') if 'churn_probability' in at_risk.columns else at_risk
        
        logger.info(f"Identified {len(at_risk)} at-risk customers")
        return at_risk
