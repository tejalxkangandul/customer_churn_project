# Customer Churn Analysis and Prediction System

A production-ready machine learning framework for analyzing customer behavior, understanding retention dynamics, and predicting churn risk. This system processes raw behavioral data, engineers high-signal features, evaluates multiple classification models, and outputs risk-stratified customer insights alongside interactive analytical dashboards.

---

## 📋 Table of Contents
- [🎯 Business Case & Workflow](#-business-case--workflow)
- [🛠️ Tech Stack](#️-tech-stack)
- [✨ Key Features](#-key-features)
- [📁 Project Structure & File Guide](#-project-structure--file-guide)
- [🚀 Installation & Environment Setup](#-installation--environment-setup)
- [🎬 End-to-End Execution Guide](#-end-to-end-execution-guide)
- [📊 Performance Metrics & Output Reference](#-performance-metrics--output-reference)
- [🔧 Troubleshooting](#-troubleshooting)

---

## 🎯 Business Case & Workflow

The core objective of this system is to replace reactive customer retention strategies with a proactive framework. By analyzing tenure patterns, billing anomalies, and account profiles, the system flags at-risk contracts before they reach defection.

### Core Architecture Pipeline

1. **Ingestion & Validation:** Ingests tabular business records and enforces type conformity.
2. **Preprocessing & Quality Gate:** Handles structural anomalies, drops duplicate profiles, and imputes missing indicators without data leakage.
3. **Engineering & Transformation:** Creates interaction terms, groups continuous variables (e.g., tenure buckets), and scales numerical matrices.
4. **Ensemble Modeling:** Trains, tunes, and stacks distinct mathematical models to identify the optimal configuration based on F1-Score.
5. **Inference & Risk Tiering:** Generates customer-level churn probabilities, grouping accounts into actionable risk segments.
6. **Reporting Analytics:** Compiles evaluation metrics and spins up web-ready interactive visual matrices.

---

## 🛠️ Tech Stack

This framework is built natively on a modern Python data science ecosystem designed for efficiency, scannability, and modularity:

- **Core Engine:** Python 3.8+
- **Data Wrangling & Processing:** `pandas`, `numpy`
- **Machine Learning Modeling:** `scikit-learn` (Ensemble methods, linear classifiers, validation splits)
- **Analytical Graphics & Dashboards:** `matplotlib`, `seaborn`, `plotly`
- **Configuration Architecture:** `PyYAML` (YAML structural parsing)
- **Defensive Test Framework:** `unittest`

---

## ✨ Key Features

- **Multi-Model Benchmarking Engine:** Runs automated parallel evaluations across Random Forests, Gradient Boosting Machines, and Logistic Regression models.
- **Leakage-Protected Data Pipelines:** Implements isolated preprocessors (`preprocessor.pkl`) calibrated strictly on training folds to prevent out-of-sample data leaks during runtime inference.
- **Granular Behavioral Feature Engineering:** Derives non-linear features, calculates charge-to-age interaction ratios, and splits lifecycle metrics into categorical groups.
- **Dynamic Risk Threshold Profiling:** Supports floating-point classification thresholds to allow business teams to balance Precision and Recall manually based on budget realities.
- **Dual-Layer Visualizations:** Generates both high-resolution static charts for static reports and interactive HTML plots for deep-dive customer analysis.
- **Developer-Ready Testing Framework:** Contains dedicated modular unit tests to isolate and verify preprocessing routines before production deployments.

---

## 📁 Project Structure & File Guide

### Root Workspace Files

- `setup.py` — Configuration script used to build and distribute the system as an executable, system-wide local package (`pip install -e .`).
- `requirements.txt` — Explicit list of pinned third-party dependencies required to ensure environment stability.
- `README.md` — The file you are currently reading; acts as the primary systemic landing page.
- `.gitignore` — Prevents large datasets (`data/`), binaries (`models/`), and operational track records (`logs/`) from entering version control.

### 📜 Detailed Directory Map

```text
customer_churn_project/
│
├── scripts/                        # Executable entrypoints for workflows
│   ├── generate_sample_data.py     # Generates mock telemetry data with synthetic correlations
│   ├── train_model.py              # Orchestrates data processing, model training, and metrics logging
│   ├── predict_churn.py            # Executes batch prediction runs on new/unlabeled datasets
│   ├── visualize_results.py        # Generates standard static diagnostic charts (ROC curves, confusion matrices)
│   └── visualize_interactive.py    # Builds web-ready interactive visual profiling dashboards
│
├── src/                            # Core application package (Reusable Library Code)
│   ├── __init__.py                 # Binds submodules to clean import hooks for external consumption
│   ├── data_loading.py             # Safeguards data stream access points and verifies file existence
│   ├── data_preprocessing.py       # Manages type conversion, missing value imputations, and categorical dummy arrays
│   ├── feature_engineering.py      # Computes advanced structural features (interaction vectors, tenure bins)
│   ├── model_training.py           # Encapsulates model tuning and instantiates estimators
│   ├── model_evaluation.py         # Generates validation statistics (F1, Precision, Recall, AUC scores)
│   ├── prediction.py               # Houses runtime prediction logic and risk classification boundaries
│   └── utils.py                    # Global system loggers, exception handlers, and configuration parsers
│
└── tests/                          # Automated Quality Assurance Layer
    ├── __init__.py                 # Initializes the testing directory scope
    └── test_preprocessing.py       # Runs isolated verification testing assertions against data filters
```

---

## 🚀 Installation & Environment Setup

To keep the system isolated from systemic global dependency conflicts, always deploy within a virtual environment.

### Step 1: Navigate to the Project Root

```bash
cd customer_churn_project
```

### Step 2: Initialize and Activate the Virtual Environment

#### On Windows (Command Prompt / PowerShell)

```bash
python -m venv venv
venv\Scripts\activate
```

#### On macOS / Linux

```bash
python -m venv venv
source venv/bin/activate
```

Verify your shell prompt now shows the `(venv)` prefix.

### Step 3: Upgrade Package Installer & Ingest Framework Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Install the Local Package in Editable Mode

```bash
pip install -e .
```

This binds the `src/` folder directly to your virtual environment's path, allowing package-wide imports across execution scripts seamlessly.

---

## 🎬 End-to-End Execution Guide

Follow these sequential terminal commands to execute the pipeline from scratch.

### Step 1: Generate Synthetic Workspace Data

Simulate an enterprise database containing 5,000 customer profiles with realistic retention telemetry metrics.

```bash
python scripts/generate_sample_data.py
```

**Expected Output:** Confirms generation and populates raw targets under `data/raw/customer_data.csv`.

### Step 2: Run the Model Training Pipeline

Orchestrate data cleaning, engineering, model training, hyperparameter calculation, and final evaluation storage.

```bash
python scripts/train_model.py
```

**Expected Output:** Logs processing milestones on-screen, outputs performance metrics for comparison, and exports binaries (`models/preprocessor.pkl`, `models/gradient_boosting_model.pkl`) alongside metric matrices (`logs/evaluation_results.json`).

### Step 3: Compile Analytical Plots & Dashboards

Process performance profiles to create visual distributions.

```bash
python scripts/visualize_results.py
python scripts/visualize_interactive.py
```

**Expected Output:** Saves static diagnostic graphics and spins up an interactive analytical web view.

### Step 4: Execute Mass Batch Inference

Generate predictions from your customer pools and isolate accounts meeting critical retention thresholds.

```bash
python scripts/predict_churn.py
```

**Expected Output:** Exports the complete assessment matrix to `data/predictions.csv` and highlights target high-risk profiles directly inside `data/predictions_at_risk.csv`.

### Step 5: Run Automated Software Unit Tests

Confirm code refactoring didn't introduce logic breaks or algorithmic drift across the pipeline.

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

---

## 📊 Performance Metrics & Output Reference

The following baseline metrics reflect performance evaluations calculated over the synthetic reference configuration data.

### 🏆 Model Comparison Matrix

| Model Objective | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-----------------|---------------|-----------|--------|----------|---------|
| Gradient Boosting | 84.12% | 81.23% | 79.89% | 0.8055 | 0.9034 |
| Random Forest | 82.34% | 78.91% | 76.54% | 0.7771 | 0.8912 |
| Logistic Regression | 79.87% | 76.54% | 75.23% | 0.7588 | 0.8654 |

---

## 🛠️ Key CLI Overrides (Prediction Framework)

Tailor runtime batch predictions directly via command-line arguments:

```bash
python scripts/predict_churn.py \
  --input data/unlabeled_marketing_dump.csv \
  --model models/gradient_boosting_model.pkl \
  --risk_threshold 0.65 \
  --output data/high_risk_retention_targets.csv
```

---

## 🔧 Troubleshooting

### Encountering `ModuleNotFoundError: No module named 'src'`

**Root Cause:**  
Python cannot locate the root core package because it was not added to the local site-packages tree.

**Resolution:**  
Make sure you are inside the parent directory `customer_churn_project/` and execute the packaging link:

```bash
pip install -e .
```

---

### Experiencing `FileNotFoundError` on Script Initialization

**Root Cause:**  
A script is searching for raw data layers or trained weights before they have been generated.

**Resolution:**  
The pipeline must run sequentially. You cannot execute `train_model.py` without first running `generate_sample_data.py`. Follow the explicit order in the Execution Guide.

---
