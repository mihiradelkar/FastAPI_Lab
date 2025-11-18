from pydantic import BaseModel
from typing import List, Optional

# ============ IRIS MODELS ============

class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class IrisResponse(BaseModel):
    prediction: int
    species: str
    confidence: Optional[float] = None

# ============ WINE MODELS ============

class WineData(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float

class WineResponse(BaseModel):
    prediction: int
    wine_class: str
    confidence: Optional[float] = None
    probabilities: Optional[List[float]] = None

# ============ GENERAL MODELS ============

class HealthCheck(BaseModel):
    status: str
    models_loaded: List[str]