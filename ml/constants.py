from pathlib import Path

import numpy as np

GOOGLE_DRIVE_FILE_URL = "https://drive.google.com/uc?id=1iNQ1Fy-0LQjktOeNBLv1N4TxqM_Snkfo"
CSV_PATH_STR = Path("datasets/heart_disease_uci.csv")

SEED = 42
TEST_PERCENT = 0.2

TARGET_COL = "num"

IRRELEVANT_COLS = ['id', 'dataset']

STRATIFIED_COLS = ['cp', 'num']

EXTRACTED_COLS = ['age_thalch', 'stress_index','bp_age', 'metabolic_risk']

NUM_COLS = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
TRAIN_NUM_COLS = NUM_COLS + EXTRACTED_COLS

SKEWED_TRAIN_NUM_COLS = ['oldpeak', 'chol', 'stress_index', 'trestbps']
NORMAL_TRAIN_NUM_COLS = [col for col in TRAIN_NUM_COLS if col not in SKEWED_TRAIN_NUM_COLS]

CAT_COLS = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'num']
TRAIN_CAT_COLS = CAT_COLS[:-1]

ONE_HOT_COLS = ['restecg', 'sex', 'fbs', 'exang']

ORDINAL_COLS = ['cp', 'slope', 'thal', 'ca']
ORDINAL_CAT = [
    ['non-anginal', 'atypical angina', 'typical angina', 'asymptomatic'],
    ['upsloping', 'flat', 'downsloping'],
    ['normal', 'reversable defect','fixed defect'],
    [0.0, 1.0, 2.0, 3.0]
]