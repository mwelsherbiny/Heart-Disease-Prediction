# Heart Disease Prediction

A machine learning project that predicts the likelihood of heart disease using patient clinical data.

The project includes data preprocessing, model training, model evaluation, and a FastAPI endpoint for predictions.

## Features

- Predicts heart disease risk from clinical attributes
- Includes preprocessing and feature engineering
- Provides model evaluation workflow
- Serves predictions through a FastAPI API
- Returns prediction probability
- Includes SHAP-based feature impact explanations

## Project Structure

```text
Heart-Disease-Prediction/
├── app/
│   ├── api/
│   ├── schemas/
│   ├── services/
│   └── main.py
├── ml/
├── models/
├── Heart_Disease_Prediction_Model.ipynb
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/mwelsherbiny/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment:

```bash
# Windows
venv\Scripts\activate
```

```bash
# macOS / Linux
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the FastAPI server:

```bash
uvicorn app.main:app --reload
```

Open the API documentation in your browser:

```text
http://127.0.0.1:8000/docs
```

## Prediction Endpoint

```http
POST /api/predict/
```

### Example Request

```json
{
  "age": 52,
  "trestbps": 130,
  "chol": 250,
  "thalch": 150,
  "oldpeak": 1.2,
  "ca": 0,
  "cp": "typical angina",
  "slope": "flat",
  "thal": "normal",
  "sex": "Male",
  "fbs": false,
  "restecg": "normal",
  "exang": false
}
```

### Example Response

```json
{
  "prediction": 0,
  "probability": 0.23,
  "top_features": [
    {
      "feature": "normal_num__thalch",
      "impact": -0.12
    },
    {
      "feature": "ordinal__cp",
      "impact": 0.09
    }
  ]
}
```

## Model

The API loads the trained model from:

```text
models/Random_Forest.pkl
```

The model predicts a binary target:

```text
0 = No heart disease
1 = Heart disease
```

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- FastAPI
- Uvicorn
- SHAP
- Joblib

## Disclaimer

This project is for educational and experimental purposes only.

It should not be used as a substitute for professional medical diagnosis or clinical decision-making.