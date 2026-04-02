import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_dataset(path="dataset_clean.csv"):
    df = pd.read_csv(path)

    # Targets vectoriels
    df["wind_u"] = df["FF"] * df["DD_cos"]
    df["wind_v"] = df["FF"] * df["DD_sin"]

    return df


def make_windows(X, y, window=24, horizon=1):
    Xs, ys = [], []
    for i in range(window, len(X) - horizon + 1):
        Xs.append(X[i - window:i])
        ys.append(y[i + horizon - 1])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def temporal_split(df, train_ratio=0.7, val_ratio=0.15):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def build_datasets(path="dataset_clean.csv", window=24, horizon=1):
    df = load_dataset(path)

    target_cols = ["wind_u", "wind_v"]
    feature_cols = [c for c in df.columns if c not in target_cols + ["FF"]]

    train_df, val_df, test_df = temporal_split(df)

    X_train_raw = train_df[feature_cols].values
    X_val_raw = val_df[feature_cols].values
    X_test_raw = test_df[feature_cols].values

    y_train_raw = train_df[target_cols].values
    y_val_raw = val_df[target_cols].values
    y_test_raw = test_df[target_cols].values

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train_raw)
    X_val = scaler_X.transform(X_val_raw)
    X_test = scaler_X.transform(X_test_raw)

    y_train = scaler_y.fit_transform(y_train_raw)
    y_val = scaler_y.transform(y_val_raw)
    y_test = scaler_y.transform(y_test_raw)

    X_train_w, y_train_w = make_windows(X_train, y_train, window=window, horizon=horizon)
    X_val_w, y_val_w = make_windows(X_val, y_val, window=window, horizon=horizon)
    X_test_w, y_test_w = make_windows(X_test, y_test, window=window, horizon=horizon)

    return {
        "X_train": X_train_w,
        "y_train": y_train_w,
        "X_val": X_val_w,
        "y_val": y_val_w,
        "X_test": X_test_w,
        "y_test": y_test_w,
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
    }