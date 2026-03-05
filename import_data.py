import pandas as pd

# -----------------------------

# Colonnes à charger

# -----------------------------

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

# -----------------------------

# Chargement des fichiers

# -----------------------------

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

# -----------------------------

# Conversion date

# -----------------------------

df["date"] = pd.to_datetime(df["AAAAMMJJHH"], format="%Y%m%d%H")

# -----------------------------

# Conversion vent m/s → km/h

# -----------------------------

df["FF"] = df["FF"] * 3.6
df["FXY"] = df["FXY"] * 3.6
df["FXI"] = df["FXI"] * 3.6

# -----------------------------

# Features temporelles

# -----------------------------

df["hour"] = df["date"].dt.hour
df["dayofweek"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month

# saison

df["season"] = (df["month"] % 12 + 3) // 3

# -----------------------------

# Sélection stations

# -----------------------------

df_se = df[df["NOM_USUEL"] == "SAINT-ETIENNE"].copy()
df_bt = df[df["NOM_USUEL"].str.contains("BOUTHEON", na=False)].copy()

# pression Bouthéon

df_bt = df_bt[["date", "PSTAT"]].rename(columns={"PSTAT": "PSTAT_BOUTHEON"})

# fusion

df_se = df_se.merge(df_bt, on="date", how="left")

# remplir pression manquante

df_se["PSTAT"] = df_se["PSTAT"].fillna(df_se["PSTAT_BOUTHEON"])

df_se = df_se.drop(columns=["PSTAT_BOUTHEON"])

# -----------------------------

# Définition du dataset IA

# -----------------------------

target = "FF"

features = [
"T","TD","U",
"RR1","DRR1",
"PSTAT",
"N","VV","WW",
"GLO","INS",
"DD","FXY","FXI",
"hour","dayofweek","month","season"
]

X = df_se[features]
y = df_se[target]

# -----------------------------

# Nettoyage valeurs manquantes

# -----------------------------

X = X.ffill()
y = y.ffill()

# -----------------------------

# Vérification

# -----------------------------

# print("Shape dataset:", X.shape)
# print("Features:", features)

# print(df_se.head(20))


# -----------------------------
# Dataset complet
# -----------------------------

dataset = df_se.copy()

# Pourcentage de NaN par colonne
nan_ratio = dataset.isna().mean()

print("\nPourcentage de NaN par colonne :\n")
print(nan_ratio)

# Garder colonnes avec moins de 30% de NaN
cols_to_keep = nan_ratio[nan_ratio < 0.3].index

dataset = dataset[cols_to_keep]

print("\nColonnes conservées :\n")
print(dataset.columns)

# -----------------------------
# Définition cible et features
# -----------------------------

target = "FF"

features = dataset.columns.drop(target)

X = dataset[features]
y = dataset[target]

# Remplir les trous restants
X = X.ffill()
y = y.ffill()

# -----------------------------
# Vérification
# -----------------------------

dataset_final = pd.concat([X, y], axis=1)
pd.set_option("display.max_columns", None)

print("\nDataset final utilisé pour l'IA :\n")
print(dataset_final.head(20))

print("\nShape du dataset :", dataset_final.shape)
print("\nColonnes utilisées :")
print(dataset_final.columns)