from pydantic import BaseModel, Field

class PredictResponse(BaseModel):
    has_heart_disease: bool
    probability: float = Field(..., ge=0.0, le=1.0, description="Probability of having heart disease, between 0 and 1")