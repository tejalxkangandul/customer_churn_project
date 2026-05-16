"""
Prediction Script for Customer Churn
"""

from src import ChurnPredictor, setup_logging, load_config
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import pickle
import argparse
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """Main prediction pipeline"""

    parser = argparse.ArgumentParser(description='Predict customer churn')
    parser.add_argument('--input', type=str, default='data/raw/customer_data.csv',
                        help='Input data file path')
    parser.add_argument('--model', type=str, default='models/random_forest_model.pkl',
                        help='Trained model file path')
    parser.add_argument('--output', type=str, default='data/predictions.csv',
                        help='Output predictions file path')
    parser.add_argument('--risk_threshold', type=float, default=0.5,
                        help='Risk threshold for identifying at-risk customers')

    args = parser.parse_args()

    # Setup logging
    setup_logging(log_file='logs/prediction.log')

    print("\n" + "="*60)
    print("CUSTOMER CHURN PREDICTION")
    print("="*60 + "\n")

    # Load configuration
    try:
        config = load_config('config/config.yaml')
    except:
        config = {
            'categorical_columns': ['gender', 'internet_service', 'contract_type',
                                    'payment_method', 'online_security', 'online_backup',
                                    'device_protection', 'tech_support', 'streaming_tv',
                                    'streaming_movies', 'phone_service', 'paperless_billing'],
            'numeric_columns': ['age', 'tenure', 'monthly_charges', 'total_charges']
        }

    try:
        # Step 1: Load Model and Preprocessor
        print("Step 1: Loading Model and Preprocessor...")
        print("-" * 60)

        with open(args.model, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Model loaded from {args.model}")

        with open('models/preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        print(f"✓ Preprocessor loaded from models/preprocessor.pkl\n")

        # Step 2: Initialize Predictor
        print("Step 2: Initializing Predictor...")
        print("-" * 60)
        predictor = ChurnPredictor(model, preprocessor)
        print("✓ Predictor initialized\n")

        # Step 3: Load and Prepare Data
        print("Step 3: Loading and Preparing Data...")
        print("-" * 60)

        df = pd.read_csv(args.input)
        print(f"Loaded {len(df)} customer records")

        # IMPORT AND APPLY FEATURE ENGINEERING
        from src.feature_engineering import FeatureEngineer
        engineer = FeatureEngineer()
        df = engineer.engineer_features(df)

        # Drop customer_id
        if 'customer_id' in df.columns:
            df = df.drop(columns=['customer_id'])

        # Separate features and labels if churn exists
        if 'churn' in df.columns:
            y_actual = df['churn']
            has_actual = True
        else:
            has_actual = False

        # Preprocess data
        X, _ = preprocessor.preprocess(
            df.copy(),
            target_col='churn',
            categorical_cols=config['categorical_columns'],
            numeric_cols=config['numeric_columns'],
            fit=False
        )

        # Remove target column if exists
        if 'churn' in X.columns:
            X = X.drop('churn', axis=1)

        print(f"Data prepared. Feature shape: {X.shape}\n")

        # Step 4: Make Predictions
        print("Step 4: Making Predictions...")
        print("-" * 60)

        predictions = predictor.predict_with_confidence(
            X, threshold=args.risk_threshold)

        print(f"✓ Predictions made for {len(predictions)} customers")
        print(f"  Will Churn: {(predictions['prediction'] == 1).sum()}")
        print(f"  Will Not Churn: {(predictions['prediction'] == 0).sum()}\n")

        # Step 5: Create Results DataFrame
        print("Step 5: Creating Results...")
        print("-" * 60)

        results_df = df.copy()
        results_df['predicted_churn'] = predictions['churn_label'].values

        if 'churn_probability' in predictions.columns:
            results_df['churn_probability'] = predictions['churn_probability'].values
            results_df['confidence'] = predictions['confidence'].values
            results_df['risk_level'] = predictions['risk_level'].values

        # Add actual churn if available
        if has_actual:
            results_df['actual_churn'] = y_actual.values

            # Calculate prediction accuracy
            accuracy = (predictions['prediction'] == y_actual.values).mean()
            print(f"Prediction Accuracy: {accuracy:.4f}")

        # Step 6: Identify At-Risk Customers
        print("\nIdentifying At-Risk Customers...")
        at_risk_df = predictor.identify_at_risk_customers(
            X,
            customer_ids=list(range(len(df))),
            risk_threshold=args.risk_threshold
        )

        at_risk_count = len(at_risk_df)
        print(
            f"Found {at_risk_count} at-risk customers (probability >= {args.risk_threshold})\n")

        # Step 7: Save Results
        print("Step 7: Saving Results...")
        print("-" * 60)

        # Save all predictions
        results_df.to_csv(args.output, index=False)
        print(f"✓ All predictions saved to {args.output}")

        # Save at-risk customers
        at_risk_output = args.output.replace('.csv', '_at_risk.csv')
        at_risk_df.to_csv(at_risk_output, index=False)
        print(f"✓ At-risk customers saved to {at_risk_output}")

        # Save summary statistics
        summary = {
            'total_customers': len(results_df),
            'predicted_churn': int((predictions['prediction'] == 1).sum()),
            'predicted_retention': int((predictions['prediction'] == 0).sum()),
            'at_risk_customers': int(at_risk_count),
            'churn_rate': float((predictions['prediction'] == 1).mean()),
            'at_risk_rate': float(at_risk_count / len(results_df)),
        }

        if has_actual:
            summary['actual_churn'] = int(y_actual.sum())
            summary['actual_retention'] = int((1 - y_actual).sum())
            summary['prediction_accuracy'] = float(accuracy)

        summary_output = args.output.replace('.csv', '_summary.json')
        with open(summary_output, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"✓ Summary statistics saved to {summary_output}\n")

        # Step 8: Display Summary
        print("="*60)
        print("PREDICTION SUMMARY")
        print("="*60)
        print(f"Total Customers: {summary['total_customers']}")
        print(
            f"Predicted to Churn: {summary['predicted_churn']} ({summary['churn_rate']*100:.2f}%)")
        print(
            f"Predicted to Retain: {summary['predicted_retention']} ({(1-summary['churn_rate'])*100:.2f}%)")
        print(
            f"At-Risk Customers: {summary['at_risk_customers']} ({summary['at_risk_rate']*100:.2f}%)")

        if has_actual:
            print(f"\nActual Churn: {summary['actual_churn']}")
            print(f"Prediction Accuracy: {summary['prediction_accuracy']:.4f}")

        print("\n" + "="*60)
        print("PREDICTION COMPLETED SUCCESSFULLY!")
        print("="*60 + "\n")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure to train the model first using: python scripts/train_model.py")
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise


if __name__ == '__main__':
    main()
