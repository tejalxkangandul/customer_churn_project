# EXECUTION GUIDE - Customer Churn Analysis and Prediction

## 📋 Complete Step-by-Step Instructions

This guide provides detailed instructions for setting up and running the entire Customer Churn Analysis and Prediction project.

---

## Part 1: Initial Setup

### Step 1.1: Navigate to Project Directory

```bash
cd customer_churn_project
```

### Step 1.2: Create Virtual Environment

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the beginning of your terminal prompt.

### Step 1.3: Upgrade pip

```bash
pip install --upgrade pip
```

### Step 1.4: Install Dependencies

```bash
pip install -r requirements.txt
```

Expected output will show installation of:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- pyyaml
- jupyter
- And other dependencies

### Step 1.5: Install Package in Development Mode

```bash
pip install -e .
```

This allows you to use the project as a package.

---

## Part 2: Data Generation

### Step 2.1: Generate Sample Data

```bash
python scripts/generate_sample_data.py
```

**What this does:**
- Creates 5000 sample customer records
- Generates realistic churn patterns
- Saves to `data/raw/customer_data.csv`

**Expected output:**
```
Generating 5000 customer records...

Data generated successfully!
File saved to: data/raw/customer_data.csv

Dataset Statistics:
Total samples: 5000
Churned customers: 847 (16.94%)
Retained customers: 4153 (83.06%)

Columns: ['customer_id', 'gender', 'age', 'tenure', 'contract_type', 
           'monthly_charges', 'total_charges', 'internet_service', ...]
```

### Step 2.2: Verify Generated Data

```bash
python -c "import pandas as pd; df = pd.read_csv('data/raw/customer_data.csv'); print(df.head()); print(df.info())"
```

---

## Part 3: Model Training

### Step 3.1: Train Models

```bash
python scripts/train_model.py
```

**What this does:**
- Loads and explores the data
- Performs feature engineering
- Preprocesses all features
- Trains multiple models (Random Forest, Gradient Boosting, Logistic Regression)
- Evaluates each model
- Saves trained models and preprocessor

**Expected output:**
```
============================================================
CUSTOMER CHURN PREDICTION - TRAINING PIPELINE
============================================================

Step 1: Loading Data...
Loaded data shape: (5000, 16)
Columns: ['customer_id', 'gender', 'age', ...]
Missing values: 0

Step 2: Feature Engineering...
New features created. Shape after engineering: (5000, 22)

Step 3: Data Preprocessing...
Features shape: (5000, 35)
Target distribution:
0    4153
1     847

Step 4: Training Models...

Training Random Forest...
✓ Random Forest model trained and saved

Training Gradient Boosting...
✓ Gradient Boosting model trained and saved

Training Logistic Regression...
✓ Logistic Regression model trained and saved

Step 5: Evaluating Models...

Evaluating Random Forest...
  Accuracy:  0.8234
  Precision: 0.7891
  Recall:    0.7654
  F1-Score:  0.7771
  ROC-AUC:   0.8912

  Top 5 Important Features:
    tenure: 0.1523
    monthly_charges: 0.1245
    contract_type: 0.0987
    age: 0.0856
    total_charges: 0.0743

Evaluating Gradient Boosting...
  Accuracy:  0.8412
  Precision: 0.8123
  Recall:    0.7989
  F1-Score:  0.8055
  ROC-AUC:   0.9034

  Top 5 Important Features:
    tenure: 0.1654
    contract_type: 0.1321
    monthly_charges: 0.1187
    total_charges: 0.0954
    age: 0.0832

Evaluating Logistic Regression...
  Accuracy:  0.7987
  Precision: 0.7654
  Recall:    0.7523
  F1-Score:  0.7588
  ROC-AUC:   0.8654

============================================================
TRAINING COMPLETED SUCCESSFULLY!
============================================================

Best Model: Gradient Boosting (F1-Score: 0.8055)

All models have been trained and saved in 'models/' directory
Evaluation results saved in 'logs/evaluation_results.json'
```

### Step 3.2: Check Training Results

```bash
cat logs/evaluation_results.json
```

You should see detailed metrics for each model:
```json
{
    "Random Forest": {
        "accuracy": 0.8234,
        "precision": 0.7891,
        "recall": 0.7654,
        "f1_score": 0.7771,
        "roc_auc": 0.8912
    },
    ...
}
```

### Step 3.3: Verify Saved Models

```bash
ls -la models/
```

You should see:
- `random_forest_model.pkl`
- `gradient_boosting_model.pkl`
- `logistic_regression_model.pkl`
- `preprocessor.pkl`

---

## Part 4: Making Predictions

### Step 4.1: Predict Churn (Using Default Model)

```bash
python scripts/predict_churn.py
```

**What this does:**
- Loads the trained Random Forest model
- Loads and preprocesses customer data
- Makes churn predictions
- Identifies at-risk customers
- Saves results

**Expected output:**
```
============================================================
CUSTOMER CHURN PREDICTION
============================================================

Step 1: Loading Model and Preprocessor...
✓ Model loaded from models/random_forest_model.pkl
✓ Preprocessor loaded from models/preprocessor.pkl

Step 2: Initializing Predictor...
✓ Predictor initialized

Step 3: Loading and Preparing Data...
Loaded 5000 customer records
Data prepared. Feature shape: (5000, 35)

Step 4: Making Predictions...
✓ Predictions made for 5000 customers
  Will Churn: 847
  Will Not Churn: 4153

Step 5: Creating Results...
Prediction Accuracy: 0.8234

Identifying At-Risk Customers...
Found 1245 at-risk customers (probability >= 0.5)

============================================================
PREDICTION SUMMARY
============================================================
Total Customers: 5000
Predicted to Churn: 847 (16.94%)
Predicted to Retain: 4153 (83.06%)
At-Risk Customers: 1245 (24.90%)

Actual Churn: 847
Prediction Accuracy: 0.8234

============================================================
PREDICTION COMPLETED SUCCESSFULLY!
============================================================
```

### Step 4.2: Predict Using Different Model

```bash
python scripts/predict_churn.py --model models/gradient_boosting_model.pkl
```

### Step 4.3: Predict with Custom Risk Threshold

```bash
python scripts/predict_churn.py --risk_threshold 0.6
```

This identifies customers with >60% churn probability as at-risk.

### Step 4.4: Predict on New Data

```bash
python scripts/predict_churn.py --input your_new_data.csv --output new_predictions.csv
```

### Step 4.5: View Prediction Results

```bash
# View all predictions
head -20 data/predictions.csv

# View at-risk customers
head -20 data/predictions_at_risk.csv

# View summary statistics
cat data/predictions_summary.json
```

Expected summary output:
```json
{
    "total_customers": 5000,
    "predicted_churn": 847,
    "predicted_retention": 4153,
    "at_risk_customers": 1245,
    "churn_rate": 0.1694,
    "at_risk_rate": 0.249,
    "actual_churn": 847,
    "actual_retention": 4153,
    "prediction_accuracy": 0.8234
}
```

---

## Part 5: Advanced Usage

### Step 5.1: Run Unit Tests

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

### Step 5.2: Use the Package Programmatically

```python
from src import DataLoader, DataPreprocessor, ModelTrainer, ModelEvaluator

# Load data
loader = DataLoader('data/raw/customer_data.csv')
df = loader.load_data()

# Preprocess
preprocessor = DataPreprocessor()
X, y = preprocessor.preprocess(
    df,
    target_col='churn',
    categorical_cols=['gender', 'contract_type'],
    numeric_cols=['age', 'tenure']
)

# Train
trainer = ModelTrainer()
X_train, X_test, y_train, y_test = trainer.split_data(X, y)
model = trainer.train_random_forest(n_estimators=100)

# Evaluate
evaluator = ModelEvaluator(model)
results = evaluator.evaluate(X_test, y_test)
print(f"Accuracy: {results['accuracy']:.4f}")
```

### Step 5.3: Custom Configuration

Edit `config/config.yaml`:

```yaml
# Increase number of estimators
random_forest_params:
  n_estimators: 200
  max_depth: 15

# Add more categorical columns
categorical_columns:
  - gender
  - internet_service
  - contract_type
  - payment_method
  - your_custom_column
```

Then run training again:
```bash
python scripts/train_model.py
```

---

## Part 6: Troubleshooting

### Issue: Module Import Error

**Error:** `ModuleNotFoundError: No module named 'src'`

**Solution:**
```bash
# Make sure you're in the project root directory
cd customer_churn_project

# Reinstall the package
pip install -e .
```

### Issue: File Not Found

**Error:** `FileNotFoundError: [Errno 2] No such file or directory: 'data/raw/customer_data.csv'`

**Solution:**
```bash
# Generate the sample data first
python scripts/generate_sample_data.py
```

### Issue: Missing Dependencies

**Error:** `ModuleNotFoundError: No module named 'sklearn'`

**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: YAML Parsing Error

**Error:** `yaml.scanner.ScannerError`

**Solution:**
- Check `config/config.yaml` has proper indentation
- Use YAML validator online
- Ensure no tabs are used (only spaces)

### Issue: Permission Denied on scripts

**Error:** `PermissionError: [Errno 13] Permission denied`

**Solution:**
```bash
# On macOS/Linux
chmod +x scripts/*.py

# Or just use python to run
python scripts/generate_sample_data.py
```

---

## Part 7: Output Files Reference

After running the complete pipeline, you'll have:

### Data Files
- `data/raw/customer_data.csv` - Original customer data
- `data/predictions.csv` - All predictions with probabilities
- `data/predictions_at_risk.csv` - Only at-risk customers
- `data/predictions_summary.json` - Summary statistics

### Models
- `models/random_forest_model.pkl` - Trained Random Forest
- `models/gradient_boosting_model.pkl` - Trained Gradient Boosting
- `models/logistic_regression_model.pkl` - Trained Logistic Regression
- `models/preprocessor.pkl` - Fitted preprocessor for feature transformation

### Logs
- `logs/training.log` - Training execution log
- `logs/prediction.log` - Prediction execution log
- `logs/evaluation_results.json` - Model evaluation metrics

---

## Part 8: Performance Optimization

### For Large Datasets

Edit `scripts/train_model.py` and increase `random_state`:

```python
trainer = ModelTrainer(test_size=0.1)  # Use less test data
```

### For Faster Training

Use fewer estimators:

```python
rf_model = trainer.train_random_forest(n_estimators=50)
```

### For Better Results

Increase estimators and tune parameters in `config/config.yaml`:

```yaml
random_forest_params:
  n_estimators: 300
  max_depth: 20
  min_samples_split: 3
```

---

## Part 9: Deployment Checklist

- [ ] Sample data generated successfully
- [ ] All models trained and saved
- [ ] Predictions generated on test data
- [ ] Accuracy metrics acceptable (>75% recommended)
- [ ] Logs show no errors
- [ ] Configuration file reviewed and finalized
- [ ] Documentation reviewed
- [ ] Tests passing

---

## 🎉 You're All Set!

Your Customer Churn Analysis and Prediction system is now fully operational. Start making predictions and identifying at-risk customers!

For more information, refer to `README.md`.

---

**Questions or Issues?**

Check:
1. README.md for general information
2. config/config.yaml for configuration options
3. logs/ directory for detailed execution logs
4. Script docstrings for detailed parameter information

---

Last Updated: May 2024
