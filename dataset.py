import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_dataset(path="dataset_clean.csv"):
    df = pd.read_csv(path)
    # Transformation en composantes vectorielles (u, v)
    # On suppose que DD_cos et DD_sin sont déjà présents
    df["wind_u"] = df["FF"] * df["DD_cos"]
    df["wind_v"] = df["FF"] * df["DD_sin"]
    return df

def make_windows(X, y, window=24, horizon=1):
    Xs, ys = [], []
    for i in range(window, len(X) - horizon + 1):
        Xs.append(X[i-window:i])
        ys.append(y[i + horizon - 1])
    return np.array(Xs), np.array(ys)

def build_dataset(window=24):
    df = load_dataset()

    # On prédit maintenant les composantes u et v
    target_cols = ["wind_u", "wind_v"]
    features_cols = [c for c in df.columns if c not in target_cols and c != "FF"]

    X = df[features_cols].values
    y = df[target_cols].values

    # Amélioration 1 : Normalisation des features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    Xw, yw = make_windows(X, y, window)

    return Xw, yw, len(features_cols)