# src/train_iris.py
import joblib
import os
import sys
import json
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from data_loader import load_data, split_data

def train_iris_model(timestamp=None):
    """
    Train and save Iris classification model with metrics
    """
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    
    # Load and split data
    print("Loading Iris dataset...")
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train model
    print("Training Decision Tree Classifier...")
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        "model": "iris",
        "timestamp": timestamp,
        "train_accuracy": accuracy_score(y_train, y_pred_train),
        "test_accuracy": accuracy_score(y_test, y_pred_test),
        "test_f1": f1_score(y_test, y_pred_test, average='macro'),
        "test_precision": precision_score(y_test, y_pred_test, average='macro'),
        "test_recall": recall_score(y_test, y_pred_test, average='macro')
    }
    
    print(f"\nModel Performance:")
    print(f"Training accuracy: {metrics['train_accuracy']:.3f}")
    print(f"Test accuracy: {metrics['test_accuracy']:.3f}")
    print(f"F1 Score: {metrics['test_f1']:.3f}")
    
    # Create directories
    os.makedirs('../model', exist_ok=True)
    os.makedirs('../metrics', exist_ok=True)
    
    # Save model
    model_path = f'../model/iris_model_{timestamp}.pkl'
    joblib.dump(model, model_path)
    print(f"✅ Model saved to {model_path}")
    
    # Save metrics
    metrics_path = f'../metrics/iris_{timestamp}_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Metrics saved to {metrics_path}")
    
    return model, metrics

if __name__ == "__main__":
    # Check for timestamp argument
    timestamp = sys.argv[2] if len(sys.argv) > 2 and '--timestamp' in sys.argv else None
    train_iris_model(timestamp)