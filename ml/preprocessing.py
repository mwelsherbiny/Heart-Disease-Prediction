from sklearn.preprocessing import FunctionTransformer
import numpy as np

def clean_data(df):
    df = df.copy()

    # Replace invalid zeros with NaN
    df['chol'] = df['chol'].replace(0, np.nan)
    df['oldpeak'] = df['oldpeak'].clip(lower=0)
    df['oldpeak'] = df['oldpeak'].replace(0, np.nan)
    df['trestbps'] = df['trestbps'].replace(0, np.nan)

    return df

cleaning_transformer = FunctionTransformer(
    clean_data,
    feature_names_out='one-to-one'
)

def feature_engineering(df):
    df = df.copy()

    df['age_thalch'] = df['age'] * df['thalch']
    df['stress_index'] = df['oldpeak'] / (df['thalch'] + 1)
    df['bp_age'] = df['trestbps'] * df['age']
    df['metabolic_risk'] = (
      (df['chol'] / 200) *
      np.where(df['fbs'] > 0, 2, 1) *
      (df['age'] / 50)
    )

    return df

def feature_names_out(transformer, input_features):
    return list(input_features) + ['age_thalch', 'stress_index','bp_age', 'metabolic_risk']

features_transformer = FunctionTransformer(
    feature_engineering,
    feature_names_out=feature_names_out
)