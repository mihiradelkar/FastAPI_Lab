# FastAPI Multi-Model ML API Lab

---
- Video Explanation: [FastAPI lab](https://www.youtube.com/watch?v=KReburHqRIQ&list=PLcS4TrUUc53LeKBIyXAaERFKBJ3dvc9GZ&index=4)
- Blog: [FastAPI Lab-1](https://www.mlwithramin.com/blog/fastapi-lab1)

---

## Overview

In this enhanced lab, we expose **multiple ML models** as APIs using [FastAPI](https://fastapi.tiangolo.com/) and [uvicorn](https://www.uvicorn.org/).

### Models Included:
1. **Iris Classifier**: Decision Tree model for classifying iris flowers (3 species)
2. **Wine Classifier**: Random Forest model for classifying Italian wines (3 cultivars)

### Technologies:
- **FastAPI**: Modern, fast web framework for building APIs with Python
- **uvicorn**: ASGI web server implementation for Python
- **Docker**: Containerization for easy deployment
- **scikit-learn**: Machine learning library for model training
- **joblib**: Efficient model persistence

## Project Structure

```
FastAPI_Labs/
├── Dockerfile              # Docker configuration
├── .dockerignore          # Docker ignore file
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── model/                # Trained models
│   ├── iris_model.pkl    # Decision Tree for Iris
│   └── wine_model.pkl    # Random Forest for Wine
└── src/                  # Source code
    ├── __init__.py
    ├── data.py           # Original data loading
    ├── data_loader.py    # Enhanced data loading for both datasets
    ├── schemas.py        # Pydantic models for API
    ├── train.py          # Original Iris training
    ├── train_iris.py     # Enhanced Iris training
    ├── train_wine.py     # Wine model training
    ├── predict.py        # Multi-model prediction logic
    └── main.py           # FastAPI application

```

## Setup Instructions

### Local Development Setup

1. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   
   # Windows
   .\venv\Scripts\activate.bat
   
   # Linux/Mac
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the models:**
   ```bash
   cd src
   
   # Train Iris model
   python train.py
   
   # Train Wine model  
   python train_wine.py
   ```

4. **Run the API:**
   ```bash
   # From src directory
   uvicorn main:app --reload
   
   # From project root
   uvicorn src.main:app --reload
   ```

5. **Access the API:**
   - Documentation: http://localhost:8000/docs
   - Health check: http://localhost:8000/health

### Docker Setup

1. **Build the Docker image:**
   ```bash
   docker build -t ml-api .
   ```

2. **Run the container:**
   ```bash
   docker run -d -p 8000:8000 --name ml-api-container ml-api
   ```

3. **Check if running:**
   ```bash
   docker ps
   docker logs ml-api-container
   ```

4. **Stop and remove:**
   ```bash
   docker stop ml-api-container
   docker rm ml-api-container
   ```

## API Endpoints

### General Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Welcome message & available models |
| GET | `/health` | API health check |
| GET | `/docs` | Interactive API documentation |

### Iris Classification Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/iris/predict` | Predict iris species |
| POST | `/predict` | Legacy iris prediction (backward compatible) |
| GET | `/iris/example` | Get example input |

**Example Request:**
```bash
curl -X POST "http://localhost:8000/iris/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```

**Example Response:**
```json
{
  "prediction": 0,
  "species": "setosa",
  "confidence": 1.0
}
```

### Wine Classification Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/wine/predict` | Predict wine type |
| GET | `/wine/example` | Get example input |

**Example Request:**
```bash
curl -X POST "http://localhost:8000/wine/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "alcohol": 13.2,
    "malic_acid": 1.78,
    "ash": 2.14,
    "alcalinity_of_ash": 11.2,
    "magnesium": 100.0,
    "total_phenols": 2.65,
    "flavanoids": 2.76,
    "nonflavanoid_phenols": 0.26,
    "proanthocyanins": 1.28,
    "color_intensity": 4.38,
    "hue": 1.05,
    "od280_od315_of_diluted_wines": 3.4,
    "proline": 1050.0
  }'
```

**Example Response:**
```json
{
  "prediction": 0,
  "wine_class": "Barolo",
  "confidence": 0.98,
  "probabilities": [0.98, 0.02, 0.0]
}
```

## Model Details

### Iris Classifier
- **Algorithm**: Decision Tree Classifier
- **Features**: 4 (sepal length/width, petal length/width)
- **Classes**: 3 (Setosa, Versicolor, Virginica)
- **Key Feature**: Petal length (most discriminative)
- **Typical Accuracy**: ~95%

### Wine Classifier
- **Algorithm**: Random Forest Classifier (100 trees)
- **Features**: 13 chemical properties
- **Classes**: 3 (Barolo, Grignolino, Barbera)
- **Key Features**: Alcohol content, Proline
- **Typical Accuracy**: ~97%

## Classification Logic

### Iris Classification Rules:
- **Setosa**: Petal length < 2.5 cm (small flowers)
- **Versicolor**: Petal length 2.5-5.0 cm (medium flowers)
- **Virginica**: Petal length > 5.0 cm (large flowers)

### Wine Classification Rules:
- **Barolo**: High alcohol (13-14%), high proline (>1000 mg/L)
- **Grignolino**: Medium alcohol (12-13%), medium proline (600-900 mg/L)
- **Barbera**: Low alcohol (11-12%), low proline (<600 mg/L)

## Testing the API

### Using Python:
```python
import requests

# Test Iris
iris_data = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}
response = requests.post("http://localhost:8000/iris/predict", json=iris_data)
print("Iris:", response.json())

# Test Wine
wine_data = {
    "alcohol": 13.2,
    "malic_acid": 1.78,
    # ... (include all 13 features)
}
response = requests.post("http://localhost:8000/wine/predict", json=wine_data)
print("Wine:", response.json())
```

### Using PowerShell:
```powershell
# Test health
Invoke-RestMethod -Uri "http://localhost:8000/health"

# Test Iris
$body = @{
    sepal_length = 5.1
    sepal_width = 3.5
    petal_length = 1.4
    petal_width = 0.2
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri "http://localhost:8000/iris/predict" `
    -Body $body -ContentType "application/json"
```

## FastAPI Features Implemented

### Data Models (Pydantic)

**Request Models:**
- `IrisData`: Validates iris flower measurements
- `WineData`: Validates wine chemical properties

**Response Models:**
- `IrisResponse`: Returns prediction, species, confidence
- `WineResponse`: Returns prediction, wine class, confidence, probabilities

### Key Features:
1. **Automatic Validation**: Input validation using Pydantic
2. **Type Hints**: Full type annotation support
3. **Interactive Docs**: Auto-generated Swagger UI
4. **Async Support**: Non-blocking request handling
5. **Error Handling**: HTTPException for proper error responses
6. **Health Checks**: Monitoring endpoint for deployment

## Docker Deployment

### Dockerfile Features:
- Python 3.9 slim base image
- Efficient layer caching
- Minimal image size
- Health check included

### Build and Run:
```bash
# Build image
docker build -t ml-api .

# Run container
docker run -d \
  --name ml-api \
  -p 8000:8000 \
  ml-api

# View logs
docker logs ml-api

# Stop container
docker stop ml-api
```

### Testing Docker Container:
```bash
# From host machine
curl http://localhost:8000/health

# Inside container
docker exec ml-api curl http://localhost:8000/health
```

## Development Tips

### Adding a New Model:
1. Create training script: `train_[model].py`
2. Add Pydantic schemas in `schemas.py`
3. Update `predict.py` with prediction function
4. Add endpoints in `main.py`
5. Retrain and save model to `model/` directory

### Debugging:
```bash
# Check loaded models
curl http://localhost:8000/health

# View API documentation
open http://localhost:8000/docs

# Check Docker logs
docker logs -f ml-api-container
```

## Performance Metrics

| Model | Training Samples | Test Accuracy | Inference Time |
|-------|-----------------|---------------|----------------|
| Iris | 150 | ~95% | <10ms |
| Wine | 178 | ~97% | <15ms |

## Future Enhancements

- [ ] Add more classification models (Digits, Breast Cancer)
- [ ] Implement batch prediction endpoints
- [ ] Add model versioning
- [ ] Create simple web UI
- [ ] Add PostgreSQL for prediction logging
- [ ] Implement CI/CD pipeline
- [ ] Add Kubernetes deployment configs
- [ ] Include A/B testing capabilities

## Troubleshooting

### Models not loading:
```bash
# Check if model files exist
ls -la model/

# Verify joblib installation
pip show joblib
```

### Port already in use:
```bash
# Find process using port 8000
netstat -tulnp | grep 8000

# Kill the process or use different port
uvicorn main:app --port 8001
```

### Docker issues:
```bash
# Rebuild without cache
docker build --no-cache -t ml-api .

# Check container logs
docker logs ml-api-container

# Enter container for debugging
docker exec -it ml-api-container bash
```

## Contributors

- Original Lab: [Dhanush Kumar Shankar](https://github.com/dhanush)
- Blog Credits: [Sai Akhilesh Ande](https://github.com/saiakhilesh)
- Multi-Model Enhancement: Nidhi Surve

## License

MIT License - Feel free to use for educational purposes

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Uvicorn Documentation](https://www.uvicorn.org/)
- [Docker Documentation](https://docs.docker.com/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pydantic Documentation](https://docs.pydantic.dev/)