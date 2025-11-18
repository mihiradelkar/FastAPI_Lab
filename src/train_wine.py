# src/train_wine.py
import joblib
import os
import sys
import json
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from data_loader import load_wine_data, split_wine_data  # Import from data_loader

def train_wine_model(timestamp=None):
    """
    Train and save Wine classification model with metrics
    """
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    
    # Load and split data using data_loader functions
    print("Loading Wine dataset...")
    X, y = load_wine_data()
    X_train, X_test, y_train, y_test = split_wine_data(X, y)
    
    # Train model
    print("Training Decision Tree Classifier for Wine...")
    model = DecisionTreeClassifier(random_state=42, max_depth=3)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        "model": "wine",
        "timestamp": timestamp,
        "train_accuracy": accuracy_score(y_train, y_pred_train),
        "test_accuracy": accuracy_score(y_test, y_pred_test),
        "test_f1": f1_score(y_test, y_pred_test, average='macro'),
        "test_precision": precision_score(y_test, y_pred_test, average='macro'),
        "test_recall": recall_score(y_test, y_pred_test, average='macro')
    }
    
    print(f"\nWine Model Performance:")
    print(f"Training accuracy: {metrics['train_accuracy']:.3f}")
    print(f"Test accuracy: {metrics['test_accuracy']:.3f}")
    print(f"F1 Score: {metrics['test_f1']:.3f}")
    
    # Create directories
    os.makedirs('../model', exist_ok=True)
    os.makedirs('../metrics', exist_ok=True)
    
    # Save model with timestamp
    model_path = f'../model/wine_model_{timestamp}.pkl'
    joblib.dump(model, model_path)
    print(f"✅ Wine model saved to {model_path}")
    
    # Also save as latest (for API)
    latest_path = '../model/wine_model.pkl'
    joblib.dump(model, latest_path)
    print(f"✅ Latest wine model saved to {latest_path}")
    
    # Save metrics
    metrics_path = f'../metrics/wine_{timestamp}_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Metrics saved to {metrics_path}")
    
    return model, metrics

if __name__ == "__main__":
    # Check for timestamp argument
    timestamp = sys.argv[2] if len(sys.argv) > 2 and '--timestamp' in sys.argv else None
    train_wine_model(timestamp)