"""
Feature Engineering Module
Handles creation and transformation of features
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Class to handle feature engineering"""

    @staticmethod
    def create_tenure_groups(df: pd.DataFrame, tenure_col: str = 'tenure') -> pd.DataFrame:
        """
        Create tenure groups

        Args:
            df: Input dataframe
            tenure_col: Name of tenure column

        Returns:
            DataFrame: Data with tenure groups
        """
        df = df.copy()

        if tenure_col in df.columns:
            df['tenure_group'] = pd.cut(df[tenure_col],
                                        bins=[0, 12, 24, 48, float('inf')],
                                        labels=['0-1 year', '1-2 years', '2-4 years', '4+ years'])
            logger.info("Created tenure groups")

        return df

    @staticmethod
    def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features

        Args:
            df: Input dataframe

        Returns:
            DataFrame: Data with interaction features
        """
        df = df.copy()

        # Monthly charges * tenure interaction
        if 'monthly_charges' in df.columns and 'tenure' in df.columns:
            df['charges_tenure_interaction'] = df['monthly_charges'] * df['tenure']

        # Total charges per month
        if 'total_charges' in df.columns and 'tenure' in df.columns:
            df['avg_monthly_charges'] = df['total_charges'] / \
                (df['tenure'] + 1)

        logger.info("Created interaction features")
        return df

    @staticmethod
    def create_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create behavioral features from contract and service usage

        Args:
            df: Input dataframe

        Returns:
            DataFrame: Data with behavioral features
        """
        df = df.copy()

        # Number of services subscribed
        service_cols = [col for col in df.columns if col.endswith(
            '_service') or col.endswith('_support')]
        if service_cols:
            df['num_services'] = (df[service_cols] == 'Yes').sum(axis=1)

        logger.info("Created behavioral features")
        return df

    @staticmethod
    def create_polynomial_features(df: pd.DataFrame, columns: list, degree: int = 2) -> pd.DataFrame:
        """
        Create polynomial features

        Args:
            df: Input dataframe
            columns: Columns to create polynomial features from
            degree: Degree of polynomial

        Returns:
            DataFrame: Data with polynomial features
        """
        df = df.copy()

        for col in columns:
            if col in df.columns:
                for d in range(2, degree + 1):
                    df[f'{col}_degree{d}'] = df[col] ** d

        logger.info(f"Created polynomial features with degree {degree}")
        return df

    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete feature engineering pipeline

        Args:
            df: Input dataframe

        Returns:
            DataFrame: Data with engineered features
        """
        # Create tenure groups - COMMENTED OUT (causes encoding issues)
        # df = FeatureEngineer.create_tenure_groups(df)

        # Create interaction features
        df = FeatureEngineer.create_interaction_features(df)

        # Create behavioral features
        df = FeatureEngineer.create_behavioral_features(df)

        # Create polynomial features for tenure
        if 'tenure' in df.columns:
            df = FeatureEngineer.create_polynomial_features(
                df, ['tenure'], degree=2)

        logger.info("Feature engineering pipeline completed")
        return df
