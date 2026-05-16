# Customer Churn Analysis and Prediction System

A comprehensive machine learning project for analyzing customer behavior and predicting churn. This system leverages advanced data preprocessing, feature engineering, and multiple machine learning models to identify customers at risk of leaving.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Results](#results)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

This project provides a complete pipeline for:

1. **Data Loading & Exploration** - Load and analyze customer data
2. **Data Preprocessing** - Clean data and handle missing values
3. **Feature Engineering** - Create meaningful features for prediction
4. **Model Training** - Train multiple ML models (Random Forest, Gradient Boosting, Logistic Regression)
5. **Model Evaluation** - Comprehensive evaluation metrics and visualizations
6. **Churn Prediction** - Identify customers at risk of churning

## ✨ Features

- **Multiple ML Models**: Random Forest, Gradient Boosting, Logistic Regression, SVM
- **Comprehensive Preprocessing**: Handles missing values, duplicates, outliers
- **Feature Engineering**: Creates interaction features, tenure groups, polynomial features
- **Easy Configuration**: YAML-based configuration file
- **Batch Prediction**: Process multiple customers at once
- **Risk Assessment**: Categorize customers by churn risk level
- **Logging & Monitoring**: Detailed logging for all operations
- **Modular Design**: Easy to extend and customize

## 📁 Project Structure

```
customer_churn_project/
├── src/                          # Source code modules
│   ├── __init__.py
│   ├── data_loading.py           # Data loading utilities
│   ├── data_preprocessing.py     # Data cleaning and preprocessing
│   ├── feature_engineering.py    # Feature creation
│   ├── model_training.py         # Model training
│   ├── model_evaluation.py       # Model evaluation
│   ├── prediction.py             # Prediction utilities
│   └── utils.py                  # Helper functions
│
├── scripts/                      # Executable scripts
│   ├── generate_sample_data.py   # Generate sample dataset
│   ├── train_model.py            # Main training script
│   └── predict_churn.py          # Prediction script
│
├── data/                         # Data directory
│   ├── raw/                      # Raw data
│   └── processed/                # Processed data
│
├── models/                       # Trained models
│   ├── random_forest_model.pkl
│   ├── gradient_boosting_model.pkl
│   └── preprocessor.pkl
│
├── logs/                         # Log files
│   ├── training.log
│   └── prediction.log
│
├── config/                       # Configuration files
│   └── config.yaml
│
├── notebooks/                    # Jupyter notebooks
│   └── eda_and_analysis.ipynb
│
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
└── README.md                     # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone or Download the Project

```bash
cd customer_churn_project
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install the Package

```bash
pip install -e .
```

## 🎬 Quick Start

### 1. Generate Sample Data

```bash
python scripts/generate_sample_data.py
```

This creates a sample dataset with 5000 customer records.

**Output**: `data/raw/customer_data.csv`

### 2. Train Models

```bash
python scripts/train_model.py
```

This trains multiple models and evaluates their performance.

**Output**:
- Trained models in `models/` directory
- Evaluation results in `logs/evaluation_results.json`

### 3. Make Predictions

```bash
python scripts/predict_churn.py
```

This makes predictions on the sample data.

**Output**:
- Predictions in `data/predictions.csv`
- At-risk customers in `data/predictions_at_risk.csv`
- Summary statistics in `data/predictions_summary.json`

## 💻 Usage

### Basic Usage

```python
from src import DataLoader, DataPreprocessor, ModelTrainer, ModelEvaluator

# Load data
loader = DataLoader('data/raw/customer_data.csv')
df = loader.load_data()

# Preprocess
preprocessor = DataPreprocessor()
X, y = preprocessor.preprocess(df, target_col='churn', 
                                categorical_cols=['gender', 'contract_type'],
                                numeric_cols=['age', 'tenure'])

# Train model
trainer = ModelTrainer()
X_train, X_test, y_train, y_test = trainer.split_data(X, y)
model = trainer.train_random_forest(n_estimators=100)

# Evaluate
evaluator = ModelEvaluator(model)
results = evaluator.evaluate(X_test, y_test)
print(f"Accuracy: {results['accuracy']:.4f}")
```

### Advanced Usage - Custom Configuration

Edit `config/config.yaml` to customize:

```yaml
# Add or remove models
models:
  - random_forest
  - gradient_boosting

# Adjust model parameters
random_forest_params:
  n_estimators: 200
  max_depth: 15

# Add more feature columns
categorical_columns:
  - gender
  - contract_type
  # Add your columns here
```

### Making Predictions on New Data

```bash
# Using default model
python scripts/predict_churn.py --input data/new_customers.csv

# Using specific model
python scripts/predict_churn.py \
  --input data/new_customers.csv \
  --model models/gradient_boosting_model.pkl \
  --output data/new_predictions.csv \
  --risk_threshold 0.6
```

### Available Command-line Options

```
--input PATH              Path to input data file (default: data/raw/customer_data.csv)
--model PATH              Path to trained model (default: models/random_forest_model.pkl)
--output PATH             Path to save predictions (default: data/predictions.csv)
--risk_threshold FLOAT    Risk threshold (default: 0.5)
```

## ⚙️ Configuration

### config/config.yaml

Key configuration parameters:

```yaml
# Data paths
data_path: data/raw/customer_data.csv

# Train-test split
test_size: 0.2

# Feature columns
categorical_columns: [...]
numeric_columns: [...]

# Models to train
models:
  - random_forest
  - gradient_boosting
  - logistic_regression

# Model-specific parameters
random_forest_params:
  n_estimators: 100
  max_depth: 10
```

## 📊 Results

After running the training pipeline, you'll get:

### Model Performance Metrics

- **Accuracy**: Proportion of correct predictions
- **Precision**: Proportion of positive predictions that were correct
- **Recall**: Proportion of actual positives correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve

### Sample Output

```
CUSTOMER CHURN PREDICTION - TRAINING PIPELINE
============================================================

Step 1: Loading Data...
Loaded data shape: (5000, 16)

Step 4: Training Models...

Evaluating Random Forest...
  Accuracy:  0.8234
  Precision: 0.7891
  Recall:    0.7654
  F1-Score:  0.7771
  ROC-AUC:   0.8912

Evaluating Gradient Boosting...
  Accuracy:  0.8412
  Precision: 0.8123
  Recall:    0.7989
  F1-Score:  0.8055
  ROC-AUC:   0.9034

TRAINING COMPLETED SUCCESSFULLY!
Best Model: Gradient Boosting (F1-Score: 0.8055)
```

## 🔧 Troubleshooting

### Issue: FileNotFoundError for data

**Solution**: Make sure to generate sample data first:
```bash
python scripts/generate_sample_data.py
```

### Issue: Module not found errors

**Solution**: Ensure you've installed the package:
```bash
pip install -e .
```

### Issue: Missing dependencies

**Solution**: Install all requirements:
```bash
pip install -r requirements.txt
```

### Issue: "No module named 'config'"

**Solution**: Make sure you're running scripts from the project root directory:
```bash
cd customer_churn_project
python scripts/train_model.py
```

### Issue: YAML parsing error

**Solution**: Ensure `config/config.yaml` has proper YAML syntax. Use an online YAML validator if needed.

## 📈 Next Steps

1. **Explore Your Data**: Use `notebooks/eda_and_analysis.ipynb` for detailed analysis
2. **Fine-tune Models**: Adjust parameters in `config/config.yaml`
3. **Add Features**: Extend `src/feature_engineering.py` with domain-specific features
4. **Deploy**: Package the trained model for production use
5. **Monitor**: Track model performance over time with new data

## 📚 Model Descriptions

### Random Forest
- Ensemble of decision trees
- Good for capturing non-linear relationships
- Fast training and inference
- Provides feature importance

### Gradient Boosting
- Sequential ensemble method
- Often achieves better performance
- Slower training but better predictions
- Handles complex patterns well

### Logistic Regression
- Linear classification model
- Fast and interpretable
- Good baseline model
- Works well with scaled features

## 🤝 Contributing

To extend this project:

1. Add new features in `src/feature_engineering.py`
2. Implement new models in `src/model_training.py`
3. Add evaluation metrics in `src/model_evaluation.py`
4. Update configuration in `config/config.yaml`


