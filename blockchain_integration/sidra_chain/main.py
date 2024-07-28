# -*- coding: utf-8 -*-

# Import necessary libraries
import os
import sys
import time
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.stats import norm
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import custom libraries
from config import Config
from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from feature_engineer import FeatureEngineer
from model_trainer import ModelTrainer
from model_evaluator import ModelEvaluator
from model_saver import ModelSaver
from model_loader import ModelLoader
from api_integration import APIIntegration

# Define constants
DATA_DIR = 'data'
MODELS_DIR = 'models'
RESULTS_DIR = 'results'
API_URL = 'https://api.sidrachain.com'
API_KEY = 'your_api_key'

# Define the main function
def main():
    # Initialize the configuration
    config = Config()

    # Load the data
    data_loader = DataLoader(DATA_DIR)
    data = data_loader.load_data()

    # Preprocess the data
    data_preprocessor = DataPreprocessor()
    data = data_preprocessor.preprocess_data(data)

    # Engineer features
    feature_engineer = FeatureEngineer()
    data = feature_engineer.engineer_features(data)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1), data['target'], test_size=0.2, random_state=42)

    # Train the model
    model_trainer = ModelTrainer()
    model = model_trainer.train_model(X_train, y_train)

    # Evaluate the model
    model_evaluator = ModelEvaluator()
    metrics = model_evaluator.evaluate_model(model, X_test, y_test)

    # Save the model
    model_saver = ModelSaver(MODELS_DIR)
    model_saver.save_model(model)

    # Load the saved model
    model_loader = ModelLoader(MODELS_DIR)
    loaded_model = model_loader.load_model()

    # Make predictions using the loaded model
    predictions = loaded_model.predict(X_test)

    # Evaluate the loaded model
    loaded_metrics = model_evaluator.evaluate_model(loaded_model, X_test, y_test)

    # Integrate with the API
    api_integration = APIIntegration(API_URL, API_KEY)
    api_integration.send_predictions(predictions)

    # Print the results
    print('Metrics:')
    print(metrics)
    print('Loaded Metrics:')
    print(loaded_metrics)

if __name__ == '__main__':
    main()
