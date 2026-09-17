# AI_m-t-o

Experimental wind forecasting for the **SAINT-ETIENNE** weather station. The repository contains the original one-hour GRU experiment and a new evaluated pipeline for **+24, +48 and +72 hours** using archived numerical weather forecasts with local machine learning corrections.

## New: forecasts at +24 / +48 / +72 hours

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
python -m unittest discover -s tests -v
```

With the raw observation CSVs and cached forecasts present, replay a historical forecast using the saved models:

```bash
python analysis/train_wind_advanced.py --predict-at 2026-02-01T00:00:00Z
```

This returns three point forecasts and calibrated intervals for the specified historical issue time. It does not fetch current station observations or provide a live forecast service. Models, preprocessing and feature schema are saved in `reports/wind_advanced/wind_models.joblib`; downloaded data and that artifact are ignored by Git.

## Original GRU: what the model predicts

The current training configuration is:

| Setting | Value | Meaning |
| --- | ---: | --- |
| `window` | 24 | Use the previous 24 observations as input |
| `horizon` | 1 | Predict the next observation |
| Source frequency | Hourly | Consecutive rows are normally one hour apart |

Therefore, the current checkpoint is a **one-hour-ahead model using about 24 hours of history**.

For an input window ending at time `t`, `make_windows()` selects the target one row later when `horizon=1`:

```text
input:  t-23, t-22, ..., t
target: t+1
```

This is not an 80-hour or 300-hour forecast. For example, `pred_vs_real.png` displays 300 consecutive test targets. Each point is a separate one-hour forecast made from its own overlapping 24-hour input window; the chart spans about 300 hours of the test set.

Features such as `PSTAT_diff3`, `FF_lag3`, and three-row rolling averages look back three observations. They do **not** change the forecast horizon. To predict about three hours ahead with hourly data, set `horizon` to `3` and retrain the model.

The older source file contains a few missing-hour gaps. Because preprocessing does not reindex or validate the timeline, a small number of adjacent rows are more than one hour apart. Thus, `horizon=1` means exactly “next available row” and normally means one hour.

## Pipeline

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

The current model has two bidirectional GRU layers, a hidden size of 128, dropout of 0.2, layer normalization, and a two-value regression head for `(wind_u, wind_v)`. Bidirectional processing only runs across the 24 historical input rows; it does not expose the model to the future target. Training uses AdamW, MSE loss, automatic mixed precision on CUDA, gradient clipping, learning-rate reduction, and early stopping. CPU execution is supported automatically.

## Current recorded result

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

The local `dataset_clean.csv` must be regenerated before training. The copy currently present on the development SSD was produced by an older preprocessing version and has 18 model inputs, while the current preprocessing code and committed GRU checkpoint use 31 inputs.

## Installation

Python 3.10 or newer is recommended.

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

## Run the project

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

## Known limitations

- The saved checkpoint contains model weights only. Feature and target scalers are rebuilt from the dataset and are not saved with the checkpoint.
- There is no inference command yet for supplying the latest observations and producing a dated forecast.
- Timestamps are removed from `dataset_clean.csv`, so training does not reject windows that cross a missing-hour gap.
- The evaluation tests one-step forecasts with observed history. It does not evaluate recursive multi-hour forecasting.
- Wind-direction metrics are not currently reported, even though the model predicts vector components.
- `model_wind_v1.pth`, `Entrainemen_2.png`, and `résultat_entrainement.txt` describe an older experiment and should not be confused with the files under `checkpoints/`.
