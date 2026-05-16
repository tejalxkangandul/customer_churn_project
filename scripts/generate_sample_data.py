"""
Generate Sample Customer Data for Churn Analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import setup_logging

def generate_sample_data(n_samples: int = 5000, output_path: str = 'data/raw/customer_data.csv') -> None:
    """
    Generate sample customer data
    
    Args:
        n_samples: Number of samples to generate
        output_path: Path to save the data
    """
    setup_logging()
    
    np.random.seed(42)
    
    print(f"Generating {n_samples} customer records...")
    
    data = {
        'customer_id': [f'CUST_{i:05d}' for i in range(n_samples)],
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'age': np.random.randint(18, 80, n_samples),
        'tenure': np.random.exponential(24, n_samples).astype(int) + 1,
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], 
                                          n_samples, p=[0.5, 0.25, 0.25]),
        'monthly_charges': np.random.exponential(75, n_samples) + 20,
        'total_charges': np.random.exponential(3000, n_samples) + 100,
        'internet_service': np.random.choice(['Fiber optic', 'DSL', 'No'], n_samples, p=[0.4, 0.4, 0.2]),
        'online_security': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
        'online_backup': np.random.choice(['Yes', 'No'], n_samples, p=[0.35, 0.65]),
        'device_protection': np.random.choice(['Yes', 'No'], n_samples, p=[0.32, 0.68]),
        'tech_support': np.random.choice(['Yes', 'No'], n_samples, p=[0.35, 0.65]),
        'streaming_tv': np.random.choice(['Yes', 'No'], n_samples, p=[0.4, 0.6]),
        'streaming_movies': np.random.choice(['Yes', 'No'], n_samples, p=[0.38, 0.62]),
        'phone_service': np.random.choice(['Yes', 'No'], n_samples, p=[0.6, 0.4]),
        'paperless_billing': np.random.choice(['Yes', 'No'], n_samples, p=[0.55, 0.45]),
        'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 
                                            'Credit card (automatic)'], n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create churn target variable with realistic patterns
    # Higher churn for month-to-month contracts
    churn_prob = np.zeros(n_samples)
    
    # Base churn probability
    churn_prob = 0.2
    
    # Increase churn for month-to-month contracts
    month_to_month = df['contract_type'] == 'Month-to-month'
    churn_prob = np.where(month_to_month, 0.35, churn_prob)
    
    # Decrease churn for longer tenure
    churn_prob = np.where(df['tenure'] > 24, churn_prob * 0.5, churn_prob)
    
    # Increase churn for high monthly charges
    churn_prob = np.where(df['monthly_charges'] > df['monthly_charges'].quantile(0.75), 
                          churn_prob * 1.3, churn_prob)
    
    # Decrease churn for customers with tech support
    churn_prob = np.where(df['tech_support'] == 'Yes', churn_prob * 0.7, churn_prob)
    
    # Ensure probabilities are between 0 and 1
    churn_prob = np.clip(churn_prob, 0, 1)
    
    df['churn'] = (np.random.random(n_samples) < churn_prob).astype(int)
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save data
    df.to_csv(output_path, index=False)
    
    print(f"\nData generated successfully!")
    print(f"File saved to: {output_path}")
    print(f"\nDataset Statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Churned customers: {df['churn'].sum()} ({df['churn'].mean()*100:.2f}%)")
    print(f"Retained customers: {(1-df['churn']).sum()} ({(1-df['churn']).mean()*100:.2f}%)")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())


if __name__ == '__main__':
    # Generate sample data
    generate_sample_data(n_samples=5000)
