from fastapi import FastAPI, HTTPException
import numpy as np

# Import schemas
from schemas import (
    IrisData, IrisResponse,
    WineData, WineResponse,
    HealthCheck
)

# Import prediction functions
from predict import (
    predict_data,  # Your existing function
    predict_iris_from_dict,
    predict_wine_from_dict,
    get_loaded_models
)

# Create FastAPI app
app = FastAPI(
    title="ML Model API",
    description="API serving Iris and Wine classification models",
    version="1.0.0"
)

# ============ ROOT & HEALTH ENDPOINTS ============

@app.get("/")
async def root():
    """Welcome endpoint"""
    return {
        "message": "ML Model API - Iris & Wine Classification",
        "available_models": get_loaded_models(),
        "endpoints": [
            "/iris/predict - Iris flower classification",
            "/wine/predict - Wine type classification",
            "/health - API health check",
            "/docs - Interactive documentation"
        ]
    }

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Check API and models health"""
    loaded_models = get_loaded_models()
    return HealthCheck(
        status="healthy" if loaded_models else "no models loaded",
        models_loaded=loaded_models
    )

# ============ IRIS ENDPOINTS ============

@app.post("/iris/predict", response_model=IrisResponse)
async def predict_iris(data: IrisData):
    """
    Predict Iris species based on flower measurements
    """
    try:
        # Convert Pydantic model to dict
        data_dict = data.dict()
        
        # Use new prediction function
        result = predict_iris_from_dict(data_dict)
        return IrisResponse(**result)
        
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Keep backward compatibility endpoint
@app.post("/predict")
async def predict_iris_legacy(data: IrisData):
    """
    Legacy endpoint - predict Iris species (backward compatible)
    """
    try:
        # Convert to numpy array for existing predict_data function
        X = np.array([[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]])
        
        # Use existing predict_data function
        prediction = predict_data(X)[0]
        
        # Map to species
        species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        species = species_map[int(prediction)]
        
        return {
            "prediction": int(prediction),
            "species": species
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============ WINE ENDPOINTS ============

@app.post("/wine/predict", response_model=WineResponse)
async def predict_wine(data: WineData):
    """
    Predict Wine classification based on chemical features
    """
    try:
        # Convert Pydantic model to dict
        data_dict = data.dict()
        
        # Use wine prediction function
        result = predict_wine_from_dict(data_dict)
        return WineResponse(**result)
        
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# ============ EXAMPLE ENDPOINTS ============

@app.get("/iris/example")
async def iris_example():
    """Get example Iris input"""
    return {
        "example_input": {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        },
        "expected_output": {
            "prediction": 0,
            "species": "setosa"
        }
    }

@app.get("/wine/example")
async def wine_example():
    """Get example Wine input"""
    return {
        "example_input": {
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
            "od280_od315_of_diluted_wines": 3.40,
            "proline": 1050.0
        },
        "expected_output": {
            "prediction": 0,
            "wine_class": "Barolo"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)