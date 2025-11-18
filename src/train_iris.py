import joblib
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from data_loader import load_data, split_data

def train_iris_model():
    """
    Train and save Iris classification model using joblib
    """
    
    # Load data using existing functions
    print("Loading Iris dataset...")
    X, y = load_data()
    
    # Split data using existing function
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train model
    print("Training Decision Tree Classifier...")
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    print(f"\nModel Performance:")
    print(f"Training accuracy: {train_accuracy:.3f}")
    print(f"Test accuracy: {test_accuracy:.3f}")
    
    # Create model directory if it doesn't exist
    model_dir = '../model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Save model using joblib
    model_path = os.path.join(model_dir, 'iris_model.pkl')
    joblib.dump(model, model_path)
    
    print(f"\n✅ Iris model saved to {model_path}")
    
    return model

if __name__ == "__main__":
    train_iris_model()