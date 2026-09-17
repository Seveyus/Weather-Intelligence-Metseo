"""Exploratory direct wind-speed forecasts; never overwrite the GRU artifacts.

Run from the repository root with: .venv/bin/python analysis/wind_horizon_study.py
Uses observed labels, exact hourly timestamps, and chronological model selection.
"""

from pathlib import Path
import hashlib
import json
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "wind_24h"
HORIZONS = (1, 3, 6, 12, 24)
SOURCE_FILES = ("H_42_previous-2020-2024.csv", "H_42_latest-2025-2026.csv")
VARIABLES = ("FF", "DD", "FXY", "FXI", "T", "TD", "U", "PSTAT", "RR1")
TRAIN_END = pd.Timestamp("2024-01-01")
TEST_START = pd.Timestamp("2025-01-01")
TEST_END = pd.Timestamp("2026-01-01")
STAGES = (50, 100, 200)


def load_observations():
    parts = []
    columns = ["AAAAMMJJHH", "NOM_USUEL", "QFF", *VARIABLES]
    for name in SOURCE_FILES:
        print(f"Reading {name}", flush=True)
        for chunk in pd.read_csv(ROOT / name, sep=";", usecols=columns,
                                 dtype={"AAAAMMJJHH": str}, chunksize=100_000):
            parts.append(chunk.loc[chunk.NOM_USUEL.isin(
                ["SAINT-ETIENNE", "ST ETIENNE-BOUTHEON"]
            )].copy())
    raw = pd.concat(parts, ignore_index=True)
    raw["time"] = pd.to_datetime(raw.AAAAMMJJHH, format="%Y%m%d%H")

    def station(name):
        frame = raw.loc[raw.NOM_USUEL == name].set_index("time").sort_index()
        if not frame.index.is_unique:
            raise ValueError(f"Duplicate timestamps for {name}; investigate before scoring")
        return frame

    se = station("SAINT-ETIENNE")
    bt = station("ST ETIENNE-BOUTHEON")
    # An hourly grid makes a shift of 24 rows exactly 24 hours, including gaps.
    index = pd.date_range(se.index.min(), se.index.max(), freq="h")
    data = se[list(VARIABLES)].reindex(index)
    data["PSTAT"] = data.PSTAT.fillna(bt.PSTAT.reindex(index))
    for name in ("FF", "FXY", "FXI"):
        data[name] *= 3.6
    data.loc[data.FF < 0, "FF"] = np.nan
    audit = {
        "raw_station_rows": len(se),
        "grid_rows": len(data),
        "absent_hourly_rows": len(data) - len(se),
        "missing_ff_in_raw_rows": int(se.FF.isna().sum()),
        "target_imputation": "none; only originally observed nonnegative FF scored",
        "quality_flags": {str(k): int(v) for k, v in se.QFF.value_counts(dropna=False).items()},
        "quality_policy": "Nonmissing nonnegative FF; no additional quality-flag filter",
        "pressure_source": "SAINT-ETIENNE, falling back to BOUTHEON at the same time",
        "timestamp_policy": "Exact source timestamps, sorted and reindexed hourly; no timezone conversion",
    }
    return data, audit


def make_features(data):
    # No backfill or interpolation: all information is from the issue time or earlier.
    history = data.ffill(limit=3).copy()
    angle = np.deg2rad(history.pop("DD"))
    history["wind_u"] = -history.FF * np.sin(angle)
    history["wind_v"] = -history.FF * np.cos(angle)
    cols = {}
    for name in history:
        for lag in (0, 1, 3, 6, 12, 24, 48, 72):
            cols[f"{name}_lag{lag}"] = history[name].shift(lag)
    for name in ("FF", "wind_u", "wind_v", "T", "PSTAT"):
        for size in (6, 24, 72):
            cols[f"{name}_mean{size}"] = history[name].rolling(size, min_periods=size).mean()
        cols[f"{name}_diff3"] = history[name] - history[name].shift(3)
    cols["FF_std24"] = history.FF.rolling(24, min_periods=24).std()
    return pd.DataFrame(cols, index=data.index)


def score(y, pred):
    error = np.asarray(pred) - np.asarray(y)
    return {"mae_kmh": float(np.abs(error).mean()),
            "rmse_kmh": float(np.sqrt(np.mean(error ** 2))),
            "bias_kmh": float(error.mean()), "n": int(len(error))}


def skill_uncertainty(y, baseline, prediction, block=168, replicates=1000):
    """Paired approximately weekly blocks; exploratory uncertainty only."""
    a, b = np.abs(y - baseline), np.abs(y - prediction)
    blocks = [slice(i, min(i + block, len(a))) for i in range(0, len(a), block)]
    sa = np.array([a[s].sum() for s in blocks])
    sb = np.array([b[s].sum() for s in blocks])
    count = np.array([len(a[s]) for s in blocks])
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(blocks), size=(replicates, len(blocks)))
    ea = sa[draws].sum(axis=1) / count[draws].sum(axis=1)
    eb = sb[draws].sum(axis=1) / count[draws].sum(axis=1)
    return {"mae_skill_percent": float(100 * (1 - b.mean() / a.mean())),
            "approx_95pct_block_bootstrap_interval_percent":
                (100 * np.quantile(1 - eb / ea, [0.025, 0.975])).tolist(),
            "block_observations": block, "replicates": replicates}


def benchmark(data, features):
    # Same verifying timestamps for every horizon and method, so errors are comparable.
    # Every issue time must follow validation, including the first +24h forecast.
    common = (data.index >= TEST_START + pd.Timedelta(hours=max(HORIZONS)))
    common &= (data.index < TEST_END) & data.FF.notna()
    for horizon in HORIZONS:
        common &= data.FF.shift(horizon).notna()
    test_times = data.index[common]
    rows, details, predictions = [], {}, {}
    train_values = data.FF.loc[data.index < TRAIN_END].dropna()
    climate = train_values.groupby([train_values.index.month, train_values.index.hour]).mean()
    global_mean = float(train_values.mean())
    high_threshold = float(train_values.quantile(0.9))
    for horizon in HORIZONS:
        print(f"Training direct +{horizon}h model", flush=True)
        valid_times = data.index + pd.Timedelta(hours=horizon)
        target = data.FF.shift(-horizon)
        X = features.copy()
        X["target_hour_sin"] = np.sin(2 * np.pi * valid_times.hour / 24)
        X["target_hour_cos"] = np.cos(2 * np.pi * valid_times.hour / 24)
        X["target_year_sin"] = np.sin(2 * np.pi * valid_times.dayofyear / 365.25)
        X["target_year_cos"] = np.cos(2 * np.pi * valid_times.dayofyear / 365.25)
        available = data.FF.notna() & target.notna() & (data.index >= data.index[72])
        train = available & (valid_times < TRAIN_END)
        # Purge origins before the validation boundary to avoid using validation labels in training.
        val = available & (data.index >= TRAIN_END) & (valid_times < TEST_START)
        assert valid_times[train].max() < data.index[val].min()
        model = HistGradientBoostingRegressor(
            loss="squared_error", max_iter=max(STAGES), max_leaf_nodes=15,
            learning_rate=0.05, l2_regularization=10.0,
            min_samples_leaf=40, early_stopping=False, random_state=42,
        )
        model.fit(X.loc[train], target.loc[train])
        validation = {}
        for stage, pred in enumerate(model.staged_predict(X.loc[val]), start=1):
            if stage in STAGES:
                validation[stage] = score(target.loc[val], np.maximum(pred, 0))
        best = min(STAGES, key=lambda k: validation[k]["rmse_kmh"])
        origins = test_times - pd.Timedelta(hours=horizon)
        assert (test_times - origins == pd.Timedelta(hours=horizon)).all()
        assert origins.min() >= TEST_START
        for stage, pred in enumerate(model.staged_predict(X.loc[origins]), start=1):
            if stage == best:
                pred = np.maximum(pred, 0)
                break
        truth = data.FF.loc[test_times].to_numpy()
        persistence = data.FF.loc[origins].to_numpy()
        climatology = np.array([climate.get((t.month, t.hour), global_mean) for t in test_times])
        values = {"Persistence": persistence, "Seasonal mean": climatology, "Direct boosted trees": pred}
        for name, estimate in values.items():
            rows.append({"horizon_hours": horizon, "method": name, **score(truth, estimate)})
        high = truth >= high_threshold
        details[str(horizon)] = {
            "train_n": int(train.sum()), "validation_n": int(val.sum()),
            "test_n": len(test_times), "selected_iterations": best,
            "validation_candidates": validation,
            "high_wind_threshold_kmh_training_p90": high_threshold,
            "high_wind_scores": {k: score(truth[high], v[high]) for k, v in values.items()},
        }
        if horizon == 24:
            details["24"]["skill_vs_persistence"] = skill_uncertainty(truth, persistence, pred)
            details["24"]["skill_vs_seasonal_mean"] = skill_uncertainty(truth, climatology, pred)
        predictions[horizon] = pd.DataFrame({"issue_time": origins, "valid_time": test_times,
            "observed_kmh": truth, "persistence_kmh": persistence,
            "seasonal_mean_kmh": climatology, "model_kmh": pred})
        print(f"+{horizon}h: {score(truth, pred)}; selected {best} trees", flush=True)
    return pd.DataFrame(rows), details, predictions


def draw(metrics, predictions, details):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.labelcolor": "#334155"})
    colors = {"Persistence": "#94a3b8", "Seasonal mean": "#d18a23", "Direct boosted trees": "#087e8b"}
    fig = plt.figure(figsize=(14, 10), facecolor="#f8fafc")
    grid = fig.add_gridspec(2, 2, top=.85, bottom=.12, left=.08, right=.97, hspace=.52, wspace=.24)
    ax1, ax2 = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    for ax, metric, title in ((ax1, "mae_kmh", "Average absolute error · MAE"),
                               (ax2, "rmse_kmh", "Large errors count more · RMSE")):
        for name, color in colors.items():
            group = metrics.loc[metrics.method == name]
            ax.plot(group.horizon_hours, group[metric], "o-", color=color, label=name, lw=2.5)
        ax.set(xlabel="Forecast lead time (hours)", ylabel="Error (km/h)", title=title, xticks=HORIZONS)
        ax.grid(axis="y", alpha=.18)
    ax1.legend(frameon=False, fontsize=10)
    ax = fig.add_subplot(grid[1, :])
    data = predictions[24].set_index("valid_time")
    # Fixed first seven test days, chosen before inspecting errors.
    sample = data.loc["2025-01-02":"2025-01-08 23:00:00"]
    ax.plot(sample.index, sample.observed_kmh, color="#172554", label="Observed wind", lw=1.8)
    ax.plot(sample.index, sample.model_kmh, color=colors["Direct boosted trees"],
            label="Direct +24h model", lw=1.8)
    ax.plot(sample.index, sample.persistence_kmh, color=colors["Persistence"], alpha=.8,
            label="Persistence +24h", lw=1, linestyle="--")
    ax.set(title="Fixed example: 2–8 January 2025 · every prediction issued 24 hours earlier",
           ylabel="Wind speed (km/h)", xlabel="Verification date (source timestamps)")
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.grid(axis="y", alpha=.18)
    ax.legend(frameon=False, ncol=3, fontsize=10)
    fig.text(.08, .955, "How much harder is tomorrow's wind?", fontsize=24, weight="bold", color="#0f172a")
    fig.text(.08, .915, "SAINT-ETIENNE  |  observed mean wind  |  2025 holdout", fontsize=13, color="#475569")
    fig.text(.08, .885, "Train: 2020–2023  ·  model selection: 2024  ·  identical verifying timestamps for every lead time",
             fontsize=10, color="#475569")
    fig.text(.08, .055, "Measured exploratory results. Boosted trees use only past station data; these are not GRU or AROME scores.", fontsize=10, color="#475569")
    fig.text(.08, .03, "Each point in the example is a separate forecast. Missing targets are excluded. No future observations are inputs.", fontsize=10, color="#475569")
    fig.savefig(OUT / "wind_horizon_results.png", dpi=170)
    fig.savefig(OUT / "wind_horizon_results.svg")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data, audit = load_observations()
    with threadpool_limits(limits=4):
        metrics, details, predictions = benchmark(data, make_features(data))
    summary = {
        "study": "Exploratory direct wind-speed forecasts; not the existing GRU",
        "train_target_years": "2020–2023", "validation_year": 2024, "test_year": 2025,
        "selection": "50/100/200 iterations; choose minimum 2024 RMSE for each lead",
        "fit_policy": "Fit only on 2020–2023; no refit on validation observations",
        "features": "Past station observations up to 72h, causal fill max 3h, future calendar only",
        "model": "HistGradientBoostingRegressor, squared error, 15 leaves, learning rate .05, L2 10, min leaf 40",
        "baseline_definition": "Persistence = observed FF at issue_time; seasonal mean = train-only month/hour mean",
        "limitations": ["Single test year; exploratory, not operational validation",
                        "Historical observation availability assumed immediate at issue_time",
                        "No archived NWP forecast inputs; no GRU retraining",
                        "No quality-code rejection beyond excluding missing/negative FF",
                        "Historical dataset used in earlier repo work; not a never-viewed external test set"],
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "data_audit": audit, "scores": metrics.to_dict(orient="records"), "horizon_details": details,
    }
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    metrics.to_csv(OUT / "horizon_metrics.csv", index=False)
    predictions[24].to_csv(OUT / "predictions_24h.csv", index=False)
    draw(metrics, predictions, details)
    print(metrics.to_string(index=False), flush=True)
    print(f"Artifacts: {OUT}", flush=True)


if __name__ == "__main__":
    main()
