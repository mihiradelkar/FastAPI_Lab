import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from data_loader import load_wine_data, split_wine_data

def train_wine_model():
    """
    Train and save Wine classification model using joblib
    """
    
    # Load data
    print("Loading Wine dataset...")
    X, y = load_wine_data()
    
    # Split data
    X_train, X_test, y_train, y_test = split_wine_data(X, y)
    
    # Scale features (important for wine dataset)
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    print(f"\nModel Performance:")
    print(f"Training accuracy: {train_accuracy:.3f}")
    print(f"Test accuracy: {test_accuracy:.3f}")
    
    # Create model directory if it doesn't exist
    model_dir = '../model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Save model and scaler together using joblib
    model_path = os.path.join(model_dir, 'wine_model.pkl')
    joblib.dump({
        'model': model,
        'scaler': scaler
    }, model_path)
    
    print(f"\n✅ Wine model saved to {model_path}")
    
    return model, scaler

if __name__ == "__main__":
    train_wine_model()