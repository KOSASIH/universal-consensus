# helpers.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def load_data(file_path):
    """Loads data from a CSV file"""
    return pd.read_csv(file_path)

def split_data(X, y, test_size=0.2, random_state=42):
    """Splits data into training and testing sets"""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def scale_data(X_train, X_test):
    """Scales data using StandardScaler"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def handle_imbalanced_data(X_train, y_train):
    """Handles imbalanced data using SMOTE and RandomUnderSampler"""
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    rus = RandomUnderSampler(random_state=42)
    X_train_res, y_train_res = rus.fit_resample(X_train_res, y_train_res)
    return X_train_res, y_train_res

def get_class_weights(y_train):
    """Calculates class weights"""
    class_weights = {}
    for i in np.unique(y_train):
        class_weights[i] = len(y_train) / (len(np.unique(y_train)) * len(y_train[y_train == i]))
    return class_weights

def get_metrics(y_test, y_pred):
    """Calculates accuracy, precision, recall, and F1 score"""
    accuracy = np.sum(y_test == y_pred) / len(y_test)
    precision = np.sum(y_test[y_pred == 1] == 1) / np.sum(y_pred == 1)
    recall = np.sum(y_test[y_test == 1] == y_pred[y_test == 1]) / np.sum(y_test == 1)
    f1_score = 2 * precision * recall / (precision + recall)
    return accuracy, precision, recall, f1_score

def plot_confusion_matrix(y_test, y_pred):
    """Plots a confusion matrix"""
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.show()
