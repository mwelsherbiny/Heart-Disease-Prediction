import joblib
import pandas as pd
import numpy as np
import shap

class PredictService:
    def __init__(self, model_path='models/Random_Forest.pkl'):
        self.model_pipeline = joblib.load(model_path)
        self.model = self.model_pipeline.named_steps["model"]
        self.preprocessor = self.model_pipeline.named_steps["preprocessor"]
        self.feature_names = self.preprocessor.get_feature_names_out()  
        self.explainer = shap.TreeExplainer(self.model)

    def predict(self, df, top_k=5):
        X_transformed = self.preprocessor.transform(df)

        # 2. prediction
        pred = self.model.predict(X_transformed)
        proba = self.model.predict_proba(X_transformed)

        explanation = self.explainer(X_transformed)
        
        results = []

        for i in range(len(df)):
            if len(explanation.values.shape) == 3:
                row_values = explanation.values[i][:, 1]
            else:
                row_values = explanation.values[i]

            top_idx = np.argsort(np.abs(row_values))[::-1][:top_k]

            top_features = [
                {
                    "feature": self.feature_names[j],
                    "impact": float(row_values[j])
                }
                for j in top_idx
            ]

            results.append(top_features)

        return {
            "prediction": int(pred[0]),
            "probability": float(proba[0][1]),
            "top_features": results[0]
        }