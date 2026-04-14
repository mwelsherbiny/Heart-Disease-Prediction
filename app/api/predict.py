from fastapi import APIRouter

from app.schemas import PredictRequest
from app.schemas.PredictResponse import PredictResponse
from app.services.PredictService import PredictService

predict_service = PredictService()

router = APIRouter(
    prefix="/predict",
    tags=["Prediction"]
)

@router.post("/", response_model=PredictResponse)
async def predict(predict_request: PredictRequest):
    return predict_service.predict(predict_request)
    
