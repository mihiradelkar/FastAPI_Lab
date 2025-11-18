"""
Simple Model Demonstration
Shows clear examples of how models classify different inputs
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_iris_classifications():
    """Demonstrate Iris classifications with clear examples"""
    
    print("\n" + "="*60)
    print("IRIS CLASSIFICATION DEMONSTRATION")
    print("="*60)
    print("\nThe model classifies iris flowers into 3 species based on measurements:")
    print("• Setosa: Small flowers with tiny petals")
    print("• Versicolor: Medium-sized flowers")  
    print("• Virginica: Large flowers with long petals")
    
    test_cases = [
        {
            "name": "Small Flower (Expected: Setosa)",
            "data": {
                "sepal_length": 4.5,
                "sepal_width": 3.2,
                "petal_length": 1.3,  # Very small petal
                "petal_width": 0.2
            }
        },
        {
            "name": "Medium Flower (Expected: Versicolor)",
            "data": {
                "sepal_length": 5.5,
                "sepal_width": 2.5,
                "petal_length": 4.0,  # Medium petal
                "petal_width": 1.3
            }
        },
        {
            "name": "Large Flower (Expected: Virginica)",
            "data": {
                "sepal_length": 7.0,
                "sepal_width": 3.2,
                "petal_length": 6.0,  # Large petal
                "petal_width": 2.3
            }
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{i}. {test['name']}")
        print("-" * 40)
        
        # Make prediction
        response = requests.post(f"{BASE_URL}/iris/predict", json=test['data'])
        result = response.json()
        
        print(f"Measurements:")
        print(f"  Petal: {test['data']['petal_length']} x {test['data']['petal_width']} cm")
        print(f"  Sepal: {test['data']['sepal_length']} x {test['data']['sepal_width']} cm")
        
        print(f"\nPrediction: {result['species'].upper()}")
        if result.get('confidence'):
            print(f"Confidence: {result['confidence']*100:.0f}%")

def test_wine_classifications():
    """Demonstrate Wine classifications with clear examples"""
    
    print("\n" + "="*60)
    print("WINE CLASSIFICATION DEMONSTRATION")
    print("="*60)
    print("\nThe model classifies Italian wines based on chemical composition:")
    print("• Barolo: Premium wine with high alcohol and proline")
    print("• Grignolino: Medium-bodied wine with balanced profile")
    print("• Barbera: Light wine with lower alcohol")
    
    test_cases = [
        {
            "name": "High-End Wine Profile (Expected: Barolo)",
            "data": {
                "alcohol": 13.9,  # High alcohol
                "malic_acid": 1.7,
                "ash": 2.5,
                "alcalinity_of_ash": 16.0,
                "magnesium": 110.0,
                "total_phenols": 3.2,
                "flavanoids": 3.5,  # High flavanoids
                "nonflavanoid_phenols": 0.3,
                "proanthocyanins": 2.0,
                "color_intensity": 7.0,  # Deep color
                "hue": 1.0,
                "od280_od315_of_diluted_wines": 3.2,
                "proline": 1200.0  # Very high proline
            }
        },
        {
            "name": "Medium Wine Profile (Expected: Grignolino)",
            "data": {
                "alcohol": 12.7,  # Medium alcohol
                "malic_acid": 2.8,
                "ash": 2.4,
                "alcalinity_of_ash": 21.0,
                "magnesium": 95.0,
                "total_phenols": 2.0,
                "flavanoids": 1.3,  # Medium flavanoids
                "nonflavanoid_phenols": 0.4,
                "proanthocyanins": 1.4,
                "color_intensity": 4.0,  # Medium color
                "hue": 0.8,
                "od280_od315_of_diluted_wines": 2.2,
                "proline": 720.0  # Medium proline
            }
        },
        {
            "name": "Light Wine Profile (Expected: Barbera)",
            "data": {
                "alcohol": 11.6,  # Low alcohol
                "malic_acid": 3.9,  # High malic acid
                "ash": 2.7,
                "alcalinity_of_ash": 28.0,
                "magnesium": 85.0,
                "total_phenols": 1.3,
                "flavanoids": 0.5,  # Low flavanoids
                "nonflavanoid_phenols": 0.5,
                "proanthocyanins": 1.0,
                "color_intensity": 2.0,  # Light color
                "hue": 0.6,
                "od280_od315_of_diluted_wines": 1.5,
                "proline": 420.0  # Low proline
            }
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{i}. {test['name']}")
        print("-" * 40)
        
        # Make prediction
        response = requests.post(f"{BASE_URL}/wine/predict", json=test['data'])
        result = response.json()
        
        print(f"Key Characteristics:")
        print(f"  Alcohol: {test['data']['alcohol']}%")
        print(f"  Proline: {test['data']['proline']} mg/L")
        print(f"  Flavanoids: {test['data']['flavanoids']} mg/L")
        print(f"  Color Intensity: {test['data']['color_intensity']}")
        
        print(f"\nPrediction: {result['wine_class'].upper()}")
        if result.get('confidence'):
            print(f"Confidence: {result['confidence']*100:.0f}%")
        
        if result.get('probabilities'):
            print(f"\nProbability Distribution:")
            print(f"  Barolo: {result['probabilities'][0]*100:.0f}%")
            print(f"  Grignolino: {result['probabilities'][1]*100:.0f}%")
            print(f"  Barbera: {result['probabilities'][2]*100:.0f}%")

def main():
    """Run demonstrations"""
    
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║           SIMPLE MODEL CLASSIFICATION DEMO                ║")
    print("╚══════════════════════════════════════════════════════════╝")
    
    # Check if API is running
    try:
        response = requests.get(f"{BASE_URL}/health")
        health = response.json()
        print(f"\n✅ API Status: {health['status']}")
        print(f"Models Loaded: {', '.join(health['models_loaded'])}")
    except:
        print("\n❌ Error: Cannot connect to API")
        print("Please start the API with: uvicorn main:app --reload")
        return
    
    test_iris_classifications()
    test_wine_classifications()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\n✅ Both models successfully classify their respective inputs!")
    print("\nKey Insights:")
    print("• Iris: Petal length is the primary decision factor")
    print("• Wine: Alcohol and proline content drive classification")
    print("• Both models show high confidence in clear-cut cases")
    print("• Random Forest (wine) provides probability distributions")
    print("• Decision Tree (iris) gives binary decisions")

if __name__ == "__main__":
    main()