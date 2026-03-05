import numpy as np
import pandas as pd

def load_dataset(path="dataset_clean.csv"):
    df = pd.read_csv(path)
    return df

def make_windows(X, y, window=24, horizon=1):
    Xs = []
    ys = []

    for i in range(window, len(X) - horizon + 1):
        Xs.append(X[i-window:i])
        ys.append(y[i + horizon - 1])

    return np.array(Xs), np.array(ys)

def build_dataset(window=24):
    df = load_dataset()

    target = "FF"
    features = [c for c in df.columns if c != target]

    X = df[features].values
    y = df[target].values

    Xw, yw = make_windows(X, y, window)

    return Xw, yw, len(features)