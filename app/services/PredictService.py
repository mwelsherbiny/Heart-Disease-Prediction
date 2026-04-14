import ml.preprocessing
import pandas as pd
import joblib

class PredictService:
    def __init__(self, model_path='Random_Forest.pkl'):
        self.model = joblib.load(model_path)

    def predict(self, data):
        df = pd.DataFrame([data])
        prediction = self.model.predict(df)
        proba = self.model.predict_proba(df)

        return {
            'has_heart_disease': bool(prediction[0]),
            'probability': float(proba[0][1])
        }