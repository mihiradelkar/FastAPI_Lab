import joblib
import numpy as np
import os
from typing import Dict, Any

# ============ (BACKWARD COMPATIBLE) ============

def predict_data(X):
    """
    EXISTING FUNCTION - Kept for backward compatibility
    Predict the class labels for the input data.
    Args:
        X (numpy.ndarray): Input data for which predictions are to be made.
    Returns:
        y_pred (numpy.ndarray): Predicted class labels.
    """
    model = joblib.load("../model/iris_model.pkl")
    y_pred = model.predict(X)
    return y_pred

# ============ NEW MULTI-MODEL FUNCTIONS ============

# Global model storage to avoid reloading
_models = {}

def load_model(model_name: str):
    """
    Load a model if not already loaded
    Args:
        model_name: 'iris' or 'wine'
    Returns:
        Loaded model or None if not found
    """
    if model_name in _models:
        return _models[model_name]
    
    model_path = f"../model/{model_name}_model.pkl"
    
    if not os.path.exists(model_path):
        print(f"⚠️ {model_name} model not found at {model_path}")
        return None
    
    try:
        model_data = joblib.load(model_path)
        _models[model_name] = model_data
        print(f"✅ {model_name} model loaded successfully")
        return model_data
    except Exception as e:
        print(f"❌ Error loading {model_name} model: {e}")
        return None

def predict_iris_from_dict(data: Dict) -> Dict[str, Any]:
    """
    Make Iris prediction from dictionary input
    Args:
        data: Dictionary with keys: sepal_length, sepal_width, petal_length, petal_width
    Returns:
        Dictionary with prediction results
    """
    # Load model
    model = load_model('iris')
    if model is None:
        raise ValueError("Iris model not loaded")
    
    # Prepare input
    X = np.array([[
        data['sepal_length'],
        data['sepal_width'],
        data['petal_length'],
        data['petal_width']
    ]])
    
    # Make prediction
    prediction = model.predict(X)[0]
    
    # Get confidence if model supports it
    confidence = None
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)[0]
        confidence = float(max(probs))
    
    # Map to species name
    species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    species = species_map[int(prediction)]
    
    return {
        'prediction': int(prediction),
        'species': species,
        'confidence': confidence
    }

def predict_wine_from_dict(data: Dict) -> Dict[str, Any]:
    """
    Make Wine prediction from dictionary input
    Args:
        data: Dictionary with 13 wine chemical features
    Returns:
        Dictionary with prediction results
    """
    # Load model (might be a dict with model and scaler)
    model_data = load_model('wine')
    if model_data is None:
        raise ValueError("Wine model not loaded")
    
    # Handle case where we saved model with scaler
    if isinstance(model_data, dict):
        model = model_data['model']
        scaler = model_data.get('scaler', None)
    else:
        model = model_data
        scaler = None
    
    # Prepare input
    X = np.array([[
        data['alcohol'],
        data['malic_acid'],
        data['ash'],
        data['alcalinity_of_ash'],
        data['magnesium'],
        data['total_phenols'],
        data['flavanoids'],
        data['nonflavanoid_phenols'],
        data['proanthocyanins'],
        data['color_intensity'],
        data['hue'],
        data['od280_od315_of_diluted_wines'],
        data['proline']
    ]])
    
    # Scale if scaler available
    if scaler:
        X = scaler.transform(X)
    
    # Make prediction
    prediction = model.predict(X)[0]
    
    # Get probabilities if available
    confidence = None
    probabilities = None
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)[0]
        confidence = float(max(probs))
        probabilities = probs.tolist()
    
    # Map to wine class
    wine_map = {0: 'Barolo', 1: 'Grignolino', 2: 'Barbera'}
    wine_class = wine_map[int(prediction)]
    
    return {
        'prediction': int(prediction),
        'wine_class': wine_class,
        'confidence': confidence,
        'probabilities': probabilities
    }

def get_loaded_models():
    """Return list of available models"""
    models = []
    if os.path.exists("../model/iris_model.pkl"):
        models.append('iris')
    if os.path.exists("../model/wine_model.pkl"):
        models.append('wine')
    return models

# Pre-load models when module is imported
print("Checking for available models...")
for model in get_loaded_models():
    load_model(model)