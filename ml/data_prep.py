from pathlib import Path

from ml.constants import GOOGLE_DRIVE_FILE_URL, CSV_PATH_STR
import gdown
import pandas as pd
from sklearn.model_selection import train_test_split

def load_drive_csv_data(url, csv_path_str):
    csv_path = Path(csv_path_str)

    if not csv_path.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(url, str(csv_path), quiet=False)

    df = pd.read_csv(csv_path)
    return df

def binarize_target(df, target_col):
    df = df.copy()
    df[target_col] = (df[target_col] > 0).astype(int)
    return df

def drop_irrelevant_features(df, cols_to_drop):    
    return df.drop(columns=cols_to_drop)

def feature_target_split(df, target_col):
    x = df.drop(columns=[target_col])
    y = df[target_col]

    return x, y


def stratified_train_test_split(
    df,
    stratify_cols,
    test_size,
    random_state,
):
    df = df.copy()
    df["_stratify_col"] = df[stratify_cols].astype(str).agg("_".join, axis=1)

    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["_stratify_col"]
    )

    train = train.drop(columns=["_stratify_col"])
    test = test.drop(columns=["_stratify_col"])

    return train, test