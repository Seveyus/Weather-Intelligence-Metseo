# AI_m-t-o

Projet de prévision du vent à **Saint-Étienne** : réseau de neurones récurrent à **+24 / +48 / +72 h**, comparé à une baseline de correction météo par arbres. L'ancien BiGRU à +1 h est conservé séparément.

## État actuel : un BiLSTM entraîné, baseline pas encore dépassée

Trois architectures récurrentes ont été entraînées avec deux graines chacune : **BiGRU**, **BiLSTM**, **BiGRU avec attention**. Un contrôle neutralise la séquence tout en conservant les variables historiques du contexte. Le meilleur candidat récurrent sur validation 2025 est le **BiLSTM**, avec 72 h d'historique et trois sorties directes. Aucun résultat des arbres ne sert d'entrée ou de cible neuronale.

Sur les **748 mêmes cibles** par échéance du 4 janvier au 4 février 2026 :

| Échéance | MAE arbres | MAE BiLSTM | RMSE arbres | RMSE BiLSTM |
| --- | ---: | ---: | ---: | ---: |
| +24 h | **2,76** | 2,86 | **3,54** | 3,65 |
| +48 h | **3,08** | 3,22 | **3,88** | 4,07 |
| +72 h | **3,11** | 3,30 | **3,90** | 4,12 |

Unités : **km/h**. Le BiLSTM améliore la météo brute de 17,3 / 14,2 / 9,8 % en MAE, mais perd face aux arbres. L'attention n'améliore pas la validation ; l'apport de la séquence supplémentaire n'est pas démontré par l'ablation. Les résultats négatifs sont conservés.

**Le mois 2026 avait déjà été consulté : cette comparaison est rétrospective, sur un test réutilisé et hivernal.** Elle ne garantit pas les performances sur d'autres saisons. La sélection du réseau et de ses époques a été commitée avant le calcul de ses scores 2026.

[Rapport complet](reports/wind_neural/RAPPORT.md) · [Architecture et références](reports/wind_neural/ARCHITECTURE.md) · [Protocole préalable](reports/wind_neural/PROTOCOLE.md) · [Courbes d'apprentissage](reports/wind_neural/apprentissage.png)

![Résultats réels du modèle neuronal](reports/wind_neural/resultats_24_48_72h.png)

## Exécuter l'expérience neuronale

Environnement testé : **Python 3.12**, PyTorch 2.8.0 CUDA 12.6, NVIDIA GTX 1650 Max-Q. Python **3.11 minimum** pour le nouveau pipeline. Le CPU est également supporté. Utiliser un environnement Linux distinct de celui de Windows, même lorsque les sources sont sur le même SSD.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
# Construction GPU testée sur cette machine
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-neural.txt
```

Pour une installation CPU, remplacer l'index PyTorch par `https://download.pytorch.org/whl/cpu`. Placer à la racine les deux exports Météo-France décrits plus bas. Sur une nouvelle copie, récupérer les archives météo avant d'entraîner :

```bash
python analysis/fetch_nwp_archive.py
python analysis/fetch_nwp_archive.py --horizons 96 --output-dir reports/wind_advanced/data/day4
python analysis/train_wind_neural.py --phase select
python analysis/train_wind_neural.py --phase refit
python analysis/train_wind_neural.py --phase evaluate
python analysis/plot_wind_neural.py
python -m unittest discover -s tests -v
```

Les commandes réutilisent les caches téléchargés. Les petites tables de prédictions des arbres sont versionnées pour figer les cas comparés. Les phases neuronales remplacent les artefacts existants : copier le dossier de résultats avant une nouvelle expérience.

Le pipeline réindexe les observations heure par heure, masque les cibles manquantes, apprend les normalisations sur l'apprentissage uniquement et purge les frontières temporelles jusqu'à +72 h. Les bulletins météo conservent une marge de disponibilité de 24 h ; le réseau apprend leur correction locale.

Après entraînement, rejouer une émission historique :

```bash
python analysis/train_wind_neural.py --predict-at 2026-02-01T00:00:00Z
```

Cette émission prévoit les vitesses aux **2, 3 et 4 février à 00:00 UTC**. Un graphique présentant des centaines d'heures juxtapose des émissions successives ; il n'est pas une unique prévision de cette durée. Les sorties actuelles sont la vitesse du vent, pas les rafales ni la direction.

| Fichier | Rôle |
| --- | --- |
| `analysis/neural_data.py` | Séquences causales, dates, masques, normalisations |
| `analysis/neural_models.py` | BiGRU, BiLSTM, attention et correction positive |
| `analysis/train_wind_neural.py` | Sélection, réentraînement, calibration, évaluation et rejeu |
| `analysis/plot_wind_neural.py` | Graphiques calculés à partir des résultats |
| `reports/wind_neural/` | Protocole, rapports, métriques, historiques et prédictions |
| `tests/test_neural_*.py` | Contrôles de causalité, modèles et entraînement |

Les poids et normalisations sont **locaux et ignorés par Git**. Un clone neuf doit les recréer en entraînant. Les checkpoints sauvegardent leur configuration ; `selection.json` fixe les graines/époques et `run_config.json` les empreintes du code et des données. Le rejeu utilise les archives présentes : **aucun service de prévision en direct n'est encore implémenté**.

## Frozen tree baseline at +24 / +48 / +72 hours

The new pipeline compares PCA + Ridge, Ridge, boosted trees, raw forecasts and forecast ensembles. Models are selected on January–September 2025, uncertainty intervals calibrated on October–December 2025, and final errors measured on January 4–February 4, 2026 (748 shared observations).

| Horizon | Raw NWP reference MAE | Selected model MAE | Selected model RMSE |
| --- | ---: | ---: | ---: |
| +24 h | 3.46 km/h | **2.76 km/h** | **3.54 km/h** |
| +48 h | 3.76 km/h | **3.08 km/h** | **3.88 km/h** |
| +72 h | 3.66 km/h | **3.11 km/h** | **3.90 km/h** |

The selected models combine direct and residual gradient boosting using ECMWF, ICON Global, ICON Europe and GFS forecast inputs. PCA was tested and slightly improved linear validation scores, but was not part of the selected ensembles.

Archived NWP offsets use initialization rather than actual publication time. This experiment conservatively uses 48/72/96-hour archive offsets for 24/48/72-hour predictions, allowing a full day for forecast availability. The same restriction applies to the raw NWP reference. Historical publication outages are not verified. These are **short winter-test results**, not a year-round operational guarantee; comparison with the older GRU's different test period would be misleading.

[Detailed report in French](reports/wind_advanced/RAPPORT.md) · [Results chart](reports/wind_advanced/resultats_24_48_72h.png)

```bash
python -m pip install -r requirements-wind.txt
python analysis/fetch_nwp_archive.py
python analysis/fetch_nwp_archive.py --horizons 96 --output-dir reports/wind_advanced/data/day4
python analysis/train_wind_advanced.py
python analysis/plot_wind_advanced.py
python -m unittest discover -s tests -p "test_wind_advanced.py" -v
```

With the raw observation CSVs and cached forecasts present, replay a historical forecast using the saved models:

```bash
python analysis/train_wind_advanced.py --predict-at 2026-02-01T00:00:00Z
```

This returns three point forecasts and calibrated intervals for the specified historical issue time. It does not fetch current station observations or provide a live forecast service. Models, preprocessing and feature schema are saved in `reports/wind_advanced/wind_models.joblib`; downloaded data and that artifact are ignored by Git.

## Original GRU: what the model predicts

The original training configuration is:

| Setting | Value | Meaning |
| --- | ---: | --- |
| `window` | 24 | Use the previous 24 observations as input |
| `horizon` | 1 | Predict the next observation |
| Source frequency | Hourly | Consecutive rows are normally one hour apart |

Therefore, the original checkpoint is a **one-hour-ahead model using about 24 hours of history**.

For an input window ending at time `t`, `make_windows()` selects the target one row later when `horizon=1`:

```text
input:  t-23, t-22, ..., t
target: t+1
```

This is not an 80-hour or 300-hour forecast. For example, `pred_vs_real.png` displays 300 consecutive test targets. Each point is a separate one-hour forecast made from its own overlapping 24-hour input window; the chart spans about 300 hours of the test set.

Features such as `PSTAT_diff3`, `FF_lag3`, and three-row rolling averages look back three observations. They do **not** change the forecast horizon. To predict about three hours ahead with hourly data, set `horizon` to `3` and retrain the model.

The older source file contains a few missing-hour gaps. Because preprocessing does not reindex or validate the timeline, a small number of adjacent rows are more than one hour apart. Thus, `horizon=1` means exactly “next available row” and normally means one hour.

## Original GRU pipeline

1. `import_data.py` reads the two raw semicolon-separated Météo-France exports.
2. It selects `SAINT-ETIENNE`, fills missing station pressure from `ST ETIENNE-BOUTHEON`, converts wind speeds from m/s to km/h, and creates calendar, direction, difference, rolling, and lag features.
3. It writes `dataset_clean.csv`.
4. `dataset.py` converts speed and direction into the targets:

   ```text
   wind_u = FF * cos(direction)
   wind_v = FF * sin(direction)
   ```

5. The rows are split chronologically into 70% training, 15% validation, and 15% test sets. `StandardScaler` instances are fitted only on the training split.
6. `train.py` trains the GRU, keeps the checkpoint with the best validation loss, evaluates the test split, compares it with a persistence baseline, and writes plots and metrics under `checkpoints/`.

The original model has two bidirectional GRU layers, a hidden size of 128, dropout of 0.2, layer normalization, and a two-value regression head for `(wind_u, wind_v)`. Bidirectional processing only runs across the 24 historical input rows; it does not expose the model to the future target. Training uses AdamW, MSE loss, automatic mixed precision on CUDA, gradient clipping, learning-rate reduction, and early stopping. CPU execution is supported automatically.

## Original GRU recorded result

The committed `checkpoints/metrics.json` records the following test result:

| Metric | GRU | Persistence baseline |
| --- | ---: | ---: |
| Wind-speed MAE | **2.670 km/h** | 2.699 km/h |
| Wind-speed RMSE | **3.535 km/h** | 3.606 km/h |

The GRU improves MAE by about 1.1% and RMSE by about 2.0% over the persistence baseline. This is a small improvement, so the project should still be treated as an experiment rather than a production forecast system.

The best validation MSE for standardized `(u, v)` targets was `0.3166`. The recorded run used `cuda:1`, 60 maximum epochs, batch size 2048, and seed 42.

## Data required

These files must be placed at the repository root:

```text
H_42_previous-2020-2024.csv
H_42_latest-2025-2026.csv
```

They are intentionally excluded from Git, as is the generated `dataset_clean.csv`. The files inspected for the current project contain hourly Saint-Etienne observations from January 2020 through February 4, 2026.

For the original GRU pipeline, the local `dataset_clean.csv` must be regenerated before training. The new +24/+48/+72 h pipelines read the raw exports directly and do not use this file. The copy currently present on the development SSD was produced by an older preprocessing version and has 18 model inputs, while the current preprocessing code and committed GRU checkpoint use 31 inputs.

## Installation

For the original GRU, Python 3.10 or newer is recommended. The new neural experiment above requires Python 3.11 or newer.

### Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The default PyTorch package can run on CPU. For NVIDIA GPU training, install the PyTorch build matching the machine's CUDA setup, then install the remaining requirements.

Do not share one virtual environment between Windows and Linux. Create a separate environment for each operating system.

## Run the original one-hour GRU

Generate the current feature set:

```bash
python import_data.py
```

Train and evaluate the model:

```bash
python train.py
```

The main settings, including `window` and `horizon`, are in the `CONFIG` dictionary at the top of `train.py`. Changing either setting requires retraining.

Training creates or updates:

```text
checkpoints/gru_wind_best.pth
checkpoints/loss_curve.png
checkpoints/pred_vs_real.png
checkpoints/metrics.json
```

## Repository files

| Path | Purpose |
| --- | --- |
| `import_data.py` | Raw-data cleaning and feature engineering |
| `dataset.py` | Scaling, temporal splitting, and sequence-window creation |
| `model.py` | Current bidirectional GRU definition |
| `train.py` | Training, validation, test evaluation, plots, and metrics |
| `checkpoints/gru_wind_best.pth` | Current GRU state dictionary |
| `checkpoints/metrics.json` | Configuration and recorded evaluation metrics |
| `model_wind_v1.pth` | Older model checkpoint kept for reference |
| `résultat_entrainement.txt` | Older training output kept for reference |

## Limitations specific to the original GRU

- The saved checkpoint contains model weights only. Feature and target scalers are rebuilt from the dataset and are not saved with the checkpoint.
- There is no inference command yet for supplying the latest observations and producing a dated forecast.
- Timestamps are removed from `dataset_clean.csv`, so training does not reject windows that cross a missing-hour gap.
- The evaluation tests one-step forecasts with observed history. It does not evaluate recursive multi-hour forecasting.
- Wind-direction metrics are not currently reported, even though the model predicts vector components.
- `model_wind_v1.pth`, `Entrainemen_2.png`, and `résultat_entrainement.txt` describe an older experiment and should not be confused with the files under `checkpoints/`.
