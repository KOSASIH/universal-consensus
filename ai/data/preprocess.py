# preprocess.py
import pandas as pd
import numpy as np

def load_dataset(file_path):
    """Load a dataset from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(data):
    """Preprocess the data by handling missing values and scaling."""
    # Handle missing values
    data.fillna(data.mean(), inplace=True)
    
    # Scale the data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    data[['feature1', 'feature2', ...]] = scaler.fit_transform(data[['feature1', 'feature2', ...]])
    
    return data
