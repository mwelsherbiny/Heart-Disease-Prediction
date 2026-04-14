from pydantic import BaseModel, Field
from typing import Literal

class PredictRequest(BaseModel):
    age: int = Field(..., description="Age of the patient in years")
    trestbps: int = Field(..., description="Resting blood pressure in mm Hg")
    chol: int = Field(..., description="Serum cholesterol in mg/dl")
    thalch: int = Field(..., description="Maximum heart rate achieved")
    oldpeak: float = Field(..., description="ST depression induced by exercise relative to rest")
    ca: int = Field(..., ge=0, le=3, description="Number of major vessels (0–3) colored by fluoroscopy")
    cp: Literal["typical angina", "atypical angina", "non-anginal", "asymptomatic"] = Field(..., description="Chest pain type: typical angina < atypical angina < non-anginal < asymptomatic")
    slope: Literal["upsloping", "flat", "downsloping"] = Field(..., description="Slope of the peak exercise ST segment: upsloping < flat < downsloping")
    thal: Literal["normal", "fixed defect", "reversable defect"] = Field(..., description="Heart defect")
    sex: Literal["Male", "Female"] = Field(..., description="Male/Female")
    fbs: bool = Field(..., description="Fasting blood sugar > 120 mg/dl")
    restecg: Literal["normal", "stt abnormality", "lv hypertrophy"] = Field(..., description="Resting electrocardiographic results")
    exang: bool = Field(..., description="Exercise-induced angina")
