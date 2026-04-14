from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder, 
    PowerTransformer, 
    StandardScaler
)
import numpy as np
import pandas as pd

from ml.constants import *

def build_preprocessing_pipeline():
    cleaning_transformer = FunctionTransformer(
        clean_data,
        feature_names_out='one-to-one'
    )

    features_transformer = FunctionTransformer(
        feature_engineering,
        feature_names_out=get_updated_features
    )

    normal_num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    skewed_num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('power_transformer', PowerTransformer(method='yeo-johnson')),
        ('scaler', StandardScaler())
    ])

    one_hot_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first')),
    ])

    ordinal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(categories=ORDINAL_CAT)),
        ('scaler', StandardScaler())
    ])

    column_tranformer = ColumnTransformer([
        ('ordinal', ordinal_pipeline, ORDINAL_COLS),
        ('one_hot', one_hot_pipeline, ONE_HOT_COLS),
        ('normal_num', normal_num_pipeline, NORMAL_TRAIN_NUM_COLS),
        ('skewed_num', skewed_num_pipeline, SKEWED_TRAIN_NUM_COLS)
    ])

    preprocessor = Pipeline([
        ('cleaning', cleaning_transformer),
        ('features', features_transformer),
        ('column_tranformor', column_tranformer)
    ])

    return preprocessor

def clean_data(df):
    df = df.copy()

    # Replace invalid zeros with NaN
    df['chol'] = df['chol'].replace(0, np.nan)
    df['oldpeak'] = df['oldpeak'].clip(lower=0)
    df['oldpeak'] = df['oldpeak'].replace(0, np.nan)
    df['trestbps'] = df['trestbps'].replace(0, np.nan)

    return df

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

def get_updated_features(transformer, input_features):
    return list(input_features) + EXTRACTED_COLS