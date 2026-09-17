# AI_m-t-o

Experimental hourly wind forecasting for the **SAINT-ETIENNE** weather station. The project prepares historical station observations, trains a bidirectional GRU, and predicts wind as two vector components before converting the result back to wind speed.

## What the model currently predicts

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
