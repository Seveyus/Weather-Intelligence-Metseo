import numpy as np
import pandas as pd

# =============================
# Colonnes à charger
# =============================

cols = [
    "AAAAMMJJHH",
    "NOM_USUEL",
    "T", "TD", "U",
    "RR1", "DRR1",
    "PSTAT",
    "N", "VV", "WW",
    "GLO", "INS",
    "FF", "DD", "FXY", "FXI"
]

# =============================
# Chargement des fichiers
# =============================

df_prev = pd.read_csv(
    "H_42_previous-2020-2024.csv",
    sep=";",
    usecols=cols,
    dtype={"AAAAMMJJHH": str}
)

df_last = pd.read_csv(
    "H_42_latest-2025-2026.csv",
    sep=";",
    usecols=cols,
    dtype={"AAAAMMJJHH": str}
)

df = pd.concat([df_prev, df_last], ignore_index=True)

# =============================
# Conversion date
# =============================

df["date"] = pd.to_datetime(df["AAAAMMJJHH"], format="%Y%m%d%H")

# =============================
# Conversion vent m/s → km/h
# =============================

df["FF"] = df["FF"] * 3.6
df["FXY"] = df["FXY"] * 3.6
df["FXI"] = df["FXI"] * 3.6

# =============================
# Features temporelles
# =============================

df["hour"] = df["date"].dt.hour
df["dayofweek"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["season"] = (df["month"] % 12 + 3) // 3

# =============================
# Sélection station
# =============================

df_se = df[df["NOM_USUEL"] == "SAINT-ETIENNE"].copy()
df_bt = df[df["NOM_USUEL"].str.contains("BOUTHEON", na=False)].copy()

# Pression Bouthéon pour compléter PSTAT
df_bt = df_bt[["date", "PSTAT"]].rename(columns={"PSTAT": "PSTAT_BOUTHEON"})
df_se = df_se.merge(df_bt, on="date", how="left")
df_se["PSTAT"] = df_se["PSTAT"].fillna(df_se["PSTAT_BOUTHEON"])
df_se = df_se.drop(columns=["PSTAT_BOUTHEON"])

# =============================
# Dataset complet
# =============================

dataset = df_se.copy()

# =============================
# Supprimer colonnes trop vides
# =============================

nan_ratio = dataset.isna().mean()

print("\nPourcentage de NaN par colonne :\n")
print(nan_ratio)

cols_to_keep = nan_ratio[nan_ratio < 0.3].index
dataset = dataset[cols_to_keep]

print("\nColonnes conservées :\n")
print(dataset.columns)

# =============================
# Direction du vent en sin/cos
# =============================

dataset["DD_sin"] = np.sin(np.deg2rad(dataset["DD"]))
dataset["DD_cos"] = np.cos(np.deg2rad(dataset["DD"]))
dataset = dataset.drop(columns=["DD"])

# =============================
# Features physiques utiles
# =============================

dataset["PSTAT_diff1"] = dataset["PSTAT"].diff(1)
dataset["PSTAT_diff3"] = dataset["PSTAT"].diff(3)
dataset["PSTAT_diff6"] = dataset["PSTAT"].diff(6)

dataset["T_roll3"] = dataset["T"].rolling(3).mean()
dataset["U_roll3"] = dataset["U"].rolling(3).mean()
dataset["PSTAT_roll3"] = dataset["PSTAT"].rolling(3).mean()
dataset["RR1_roll6"] = dataset["RR1"].rolling(6).mean()

# =============================
# Lags explicites vent / rafales
# =============================

dataset["FF_lag1"] = dataset["FF"].shift(1)
dataset["FF_lag3"] = dataset["FF"].shift(3)
dataset["FF_lag6"] = dataset["FF"].shift(6)

dataset["FXY_lag1"] = dataset["FXY"].shift(1)
dataset["FXY_lag3"] = dataset["FXY"].shift(3)
dataset["FXY_lag6"] = dataset["FXY"].shift(6)

dataset["FXI_lag1"] = dataset["FXI"].shift(1)
dataset["FXI_lag3"] = dataset["FXI"].shift(3)

dataset["T_lag1"] = dataset["T"].shift(1)
dataset["U_lag1"] = dataset["U"].shift(1)
dataset["PSTAT_lag1"] = dataset["PSTAT"].shift(1)

# =============================
# Nettoyage NaN
# =============================

dataset = dataset.ffill()

drop_cols = [c for c in ["NOM_USUEL", "AAAAMMJJHH", "date"] if c in dataset.columns]
dataset_model = dataset.drop(columns=drop_cols)

dataset_model = dataset_model.dropna().reset_index(drop=True)

# =============================
# Vérification
# =============================

pd.set_option("display.max_columns", None)

print("\nDataset modèle final :\n")
print(dataset_model.head(20))

print("\nShape :", dataset_model.shape)
print("\nColonnes utilisées :")
print(dataset_model.columns)

# =============================
# Sauvegarde
# =============================

dataset_model.to_csv("dataset_clean.csv", index=False)