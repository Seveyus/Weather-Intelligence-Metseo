import numpy as np
import pandas as pd

# =============================

# Colonnes à charger

# =============================

cols = [
"AAAAMMJJHH",
"NOM_USUEL",
"T","TD","U",
"RR1","DRR1",
"PSTAT",
"N","VV","WW",
"GLO","INS",
"FF","DD","FXY","FXI"
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

# Sélection stations

# =============================

df_se = df[df["NOM_USUEL"] == "SAINT-ETIENNE"].copy()
df_bt = df[df["NOM_USUEL"].str.contains("BOUTHEON", na=False)].copy()

# récupérer pression Bouthéon

df_bt = df_bt[["date", "PSTAT"]].rename(columns={"PSTAT": "PSTAT_BOUTHEON"})

# fusion

df_se = df_se.merge(df_bt, on="date", how="left")

# compléter pression manquante

df_se["PSTAT"] = df_se["PSTAT"].fillna(df_se["PSTAT_BOUTHEON"])

# supprimer colonne inutile

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

# Transformation direction vent

# =============================

dataset["DD_sin"] = np.sin(np.deg2rad(dataset["DD"]))
dataset["DD_cos"] = np.cos(np.deg2rad(dataset["DD"]))

dataset = dataset.drop(columns=["DD"])

# =============================

# Features physiques utiles

# =============================

dataset["PSTAT_diff1"] = dataset["PSTAT"].diff()
dataset["PSTAT_diff3"] = dataset["PSTAT"].diff(3)

dataset["T_roll3"] = dataset["T"].rolling(3).mean()
dataset["U_roll3"] = dataset["U"].rolling(3).mean()
dataset["PSTAT_roll3"] = dataset["PSTAT"].rolling(3).mean()

# remplir premiers NaN créés

dataset = dataset.ffill()

# =============================

# Supprimer colonnes non utiles

# =============================

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
