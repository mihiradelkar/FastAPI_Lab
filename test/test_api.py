import requests
import json

base_url = "http://localhost:8000"

# Check health
print("=== Health Check ===")
response = requests.get(f"{base_url}/health")
print(json.dumps(response.json(), indent=2))

# Test Iris
print("\n=== Testing Iris ===")
iris_data = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}
response = requests.post(f"{base_url}/iris/predict", json=iris_data)
print("Iris prediction:", response.json())

# Test Wine
print("\n=== Testing Wine ===")
# Get example data
response = requests.get(f"{base_url}/wine/example")
wine_data = response.json()["example_input"]
# Make prediction
response = requests.post(f"{base_url}/wine/predict", json=wine_data)
print("Wine prediction:", response.json())

# Test legacy endpoint
print("\n=== Testing Legacy Endpoint ===")
response = requests.post(f"{base_url}/predict", json=iris_data)
print("Legacy prediction:", response.json())