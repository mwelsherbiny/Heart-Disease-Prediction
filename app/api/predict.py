from fastapi import APIRouter
import pandas as pd

from app.schemas.PredictRequest import PredictRequest
from app.schemas.PredictResponse import PredictResponse
from app.services.PredictService import PredictService

from ml.constants import ORDERED_COLS

predict_service = PredictService()

router = APIRouter(
    prefix="/predict",
    tags=["Prediction"]
)

@router.post("/", response_model=PredictResponse)
async def predict(predict_request: PredictRequest):
    data = predict_request.model_dump()

    df = pd.DataFrame([data])
    df = df[ORDERED_COLS]

    return predict_service.predict(df)
