import numpy as np
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split

# ============ EXISTING IRIS FUNCTIONS ============

def load_data():
    """
    Load the Iris dataset and return the features and target values.
    Returns:
        X (numpy.ndarray): The features of the Iris dataset.
        y (numpy.ndarray): The target values of the Iris dataset.
    """
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y

def split_data(X, y):
    """
    Split the data into training and testing sets.
    Args:
        X (numpy.ndarray): The features of the dataset.
        y (numpy.ndarray): The target values of the dataset.
    Returns:
        X_train, X_test, y_train, y_test (tuple): The split dataset.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)
    return X_train, X_test, y_train, y_test

# ============ NEW WINE FUNCTIONS ============

def load_wine_data():
    """
    Load the Wine dataset and return the features and target values.
    Returns:
        X (numpy.ndarray): The features of the Wine dataset.
        y (numpy.ndarray): The target values of the Wine dataset.
    """
    wine = load_wine()
    X = wine.data
    y = wine.target
    return X, y

def split_wine_data(X, y):
    """
    Split the wine data into training and testing sets.
    Args:
        X (numpy.ndarray): The features of the dataset.
        y (numpy.ndarray): The target values of the dataset.
    Returns:
        X_train, X_test, y_train, y_test (tuple): The split dataset.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test