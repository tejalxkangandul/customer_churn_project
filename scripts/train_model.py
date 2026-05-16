"""
Main Training Script for Customer Churn Prediction
"""

from src import (
    DataLoader, DataPreprocessor, FeatureEngineer,
    ModelTrainer, ModelEvaluator, setup_logging, load_config
)
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """Main training pipeline"""

    # Setup logging
    setup_logging(log_file='logs/training.log')

    print("\n" + "="*60)
    print("CUSTOMER CHURN PREDICTION - TRAINING PIPELINE")
    print("="*60 + "\n")

    # Load configuration
    try:
        config = load_config('config/config.yaml')
    except:
        config = {
            'data_path': 'data/raw/customer_data.csv',
            'test_size': 0.2,
            'models': ['random_forest', 'gradient_boosting'],
            'categorical_columns': ['gender', 'internet_service', 'contract_type',
                                    'payment_method', 'online_security', 'online_backup',
                                    'device_protection', 'tech_support', 'streaming_tv',
                                    'streaming_movies', 'phone_service', 'paperless_billing',
                                    'tenure_group'],
            'numeric_columns': ['age', 'tenure', 'monthly_charges', 'total_charges']
        }

    # Step 1: Load Data
    print("Step 1: Loading Data...")
    print("-" * 60)
    loader = DataLoader(config['data_path'])
    df = loader.load_data()
    exploration = loader.explore_data()
    print(f"Loaded data shape: {exploration['shape']}")
    print(f"Columns: {exploration['columns']}")
    print(f"Missing values: {sum(exploration['missing_values'].values())}\n")

    # Remove customer_id (not a feature)
    df = df.drop(columns=['customer_id'])

    # Step 2: Feature Engineering
    print("Step 2: Feature Engineering...")
    # Step 2: Feature Engineering
    print("Step 2: Feature Engineering...")
    print("-" * 60)
    engineer = FeatureEngineer()
    df = engineer.engineer_features(df)
    print(f"New features created. Shape after engineering: {df.shape}\n")

    # Step 3: Data Preprocessing
    print("Step 3: Data Preprocessing...")
    print("-" * 60)
    preprocessor = DataPreprocessor()

    X, y = preprocessor.preprocess(
        df,
        target_col='churn',
        categorical_cols=config['categorical_columns'],
        numeric_cols=config['numeric_columns'],
        fit=True
    )

    print(f"Features shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}\n")

    # Step 4: Train Multiple Models
    print("Step 4: Training Models...")
    print("-" * 60)

    results = {}
    models_trained = []

    # Create model trainer
    trainer = ModelTrainer(test_size=config['test_size'])
    X_train, X_test, y_train, y_test = trainer.split_data(X, y)

    # Train Random Forest
    if 'random_forest' in config['models']:
        print("\nTraining Random Forest...")
        rf_model = trainer.train_random_forest(n_estimators=100, max_depth=10)
        models_trained.append(('Random Forest', rf_model))
        trainer.save_model('models/random_forest_model.pkl')
        print("✓ Random Forest model trained and saved")

    # Train Gradient Boosting
    if 'gradient_boosting' in config['models']:
        print("\nTraining Gradient Boosting...")
        gb_model = trainer.train_gradient_boosting(
            n_estimators=100, max_depth=5)
        models_trained.append(('Gradient Boosting', gb_model))
        trainer.save_model('models/gradient_boosting_model.pkl')
        print("✓ Gradient Boosting model trained and saved")

    # Train Logistic Regression
    if 'logistic_regression' in config['models']:
        print("\nTraining Logistic Regression...")
        lr_model = trainer.train_logistic_regression()
        models_trained.append(('Logistic Regression', lr_model))
        trainer.save_model('models/logistic_regression_model.pkl')
        print("✓ Logistic Regression model trained and saved")

    # Step 5: Evaluate Models
    print("\n\nStep 5: Evaluating Models...")
    print("-" * 60)

    evaluation_results = {}
    best_model = None
    best_score = 0

    for model_name, model in models_trained:
        print(f"\nEvaluating {model_name}...")
        evaluator = ModelEvaluator(model)
        results = evaluator.evaluate(X_test, y_test)

        evaluation_results[model_name] = {
            'accuracy': float(results['accuracy']),
            'precision': float(results['precision']),
            'recall': float(results['recall']),
            'f1_score': float(results['f1_score']),
        }

        if 'roc_auc' in results:
            evaluation_results[model_name]['roc_auc'] = float(
                results['roc_auc'])

        print(f"  Accuracy:  {results['accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall:    {results['recall']:.4f}")
        print(f"  F1-Score:  {results['f1_score']:.4f}")
        if 'roc_auc' in results:
            print(f"  ROC-AUC:   {results['roc_auc']:.4f}")

        # Track best model
        if results['f1_score'] > best_score:
            best_score = results['f1_score']
            best_model = model_name

        # Get feature importance if available
        feature_importance = evaluator.get_feature_importance(
            X_train.columns.tolist())
        if feature_importance is not None:
            print(f"\n  Top 5 Important Features:")
            for idx, row in feature_importance.head().iterrows():
                print(f"    {row['feature']}: {row['importance']:.4f}")

    # Step 6: Save Results
    print("\n\nStep 6: Saving Results...")
    print("-" * 60)

    # Save evaluation results
    with open('logs/evaluation_results.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)

    # Save preprocessor
    import pickle
    with open('models/preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)

    print(f"Results saved to logs/evaluation_results.json")
    print(f"Preprocessor saved to models/preprocessor.pkl")

    # Step 7: Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\nBest Model: {best_model} (F1-Score: {best_score:.4f})")
    print(f"\nAll models have been trained and saved in 'models/' directory")
    print(f"Evaluation results saved in 'logs/evaluation_results.json'\n")


if __name__ == '__main__':
    main()
