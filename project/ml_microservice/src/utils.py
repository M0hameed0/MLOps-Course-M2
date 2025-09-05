import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

def read_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_data(csv_path):
    df = pd.read_csv(csv_path)

    # Conversion des colonnes numériques en float (sécurité pour MLflow)
    numeric_cols = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)

    return df

def split_data(df, target, test_size=0.2, random_state=42):
    X = df.drop(columns=[target])
    y = df[target].map({"yes": 1, "no": 0})  # convert yes/no -> 1/0
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
