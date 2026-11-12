"""How far ahead can the station's own sensors forecast each variable?

Inputs are only what the planned station measures (SHT30: T, U; BMP280: pressure;
anemometer and vane: FF, DD, gusts; tipping-bucket gauge: rain), taken from the
Météo-France SAINT-ETIENNE record (pressure from ST ETIENNE-BOUTHEON, which the
city station lacks). One direct boosted-tree model per variable and lead time is
compared with persistence, the seasonal mean and, at +24/+48/+72 h, raw
Open-Meteo archived forecasts (mean of six models, aged like the wind benchmarks).

    python analysis/sensor_horizon_study.py
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "sensor_horizons"
SOURCES = [ROOT / "data/meteofrance_history/H_42_2000-2009.csv.gz",
           ROOT / "data/meteofrance_history/H_42_2010-2019.csv.gz",
           ROOT / "data/meteofrance/H_42_previous-2020-2024.csv.gz",
           ROOT / "data/meteofrance/H_42_latest-2025-2026.csv.gz"]
NWP_FILE = ROOT / "data/nwp_multisource/nwp_archive.csv"
TARGET, PRESSURE_STATION = "42218011", "42005001"
SENSORS = ("T", "U", "PSTAT", "FF", "DD", "FXI", "RR1")
HORIZONS = (1, 3, 6, 12, 24, 48, 72)
NWP_HORIZONS = (24, 48, 72)
NWP_AGE_BUFFER_HOURS = 24
NWP_MODELS = ("ecmwf_ifs025", "icon_global", "icon_eu", "gfs_global", "gem_global", "jma_seamless")
RAIN_THRESHOLD_MM = 0.2  # one tip of the Météo-France gauge resolution
# Target: (label, unit, column, kind). "rain" is a probability scored with the Brier score.
TARGETS = {
    "T": ("Température", "°C", "T", "value"),
    "U": ("Humidité relative", "%", "U", "value"),
    "PSTAT": ("Pression station (Bouthéon)", "hPa", "PSTAT", "value"),
    "FF": ("Vent moyen", "km/h", "FF", "value"),
    "FXI": ("Rafale max horaire", "km/h", "FXI", "value"),
    "RAIN": ("Probabilité de pluie dans l'heure", "Brier", "RR1", "rain"),
}
# Open-Meteo counterparts, only where the quantity matches the station's.
NWP_COUNTERPART = {"T": "temperature_2m", "U": "relative_humidity_2m", "FF": "wind_speed_10m",
                   "FXI": "wind_gusts_10m", "RAIN": "precipitation"}
TRAIN_END = pd.Timestamp("2022-01-01")
VALIDATION_END = pd.Timestamp("2024-01-01")
TEST_START = VALIDATION_END + pd.Timedelta(hours=max(HORIZONS))
STAGES = (100, 200, 400)


def load_station():
    columns = ["NUM_POSTE", "AAAAMMJJHH", *SENSORS]
    parts = []
    for path in SOURCES:
        print(f"Reading {path.name}", flush=True)
        for chunk in pd.read_csv(path, sep=";", usecols=columns, chunksize=200_000,
                                 dtype={"NUM_POSTE": str, "AAAAMMJJHH": str}):
            parts.append(chunk.loc[chunk.NUM_POSTE.isin([TARGET, PRESSURE_STATION])])
    raw = pd.concat(parts, ignore_index=True)
    raw["time"] = pd.to_datetime(raw.AAAAMMJJHH, format="%Y%m%d%H")
    frames = {}
    for station, frame in raw.groupby("NUM_POSTE"):
        frame = frame.set_index("time").sort_index()
        if not frame.index.is_unique:
            raise ValueError(f"Duplicate timestamps for {station}")
        frames[station] = frame
    target = frames[TARGET]
    start = target.FF.first_valid_index().floor("D")
    index = pd.date_range(start, target.index.max(), freq="h")
    data = target[list(SENSORS)].reindex(index)
    data["PSTAT"] = frames[PRESSURE_STATION].PSTAT.reindex(index)
    for name in ("FF", "FXI"):
        data[name] *= 3.6
    for name in ("FF", "FXI", "RR1"):
        data.loc[data[name] < 0, name] = np.nan
    return data


def sensor_features(data):
    """Causal history of the station's own sensors (bounded forward fill only)."""
    history = data.ffill(limit=3)
    angle = np.deg2rad(history.pop("DD"))
    history["wind_u"] = -history.FF * np.sin(angle)
    history["wind_v"] = -history.FF * np.cos(angle)
    history["rain"] = (history.RR1 >= RAIN_THRESHOLD_MM).astype(float).where(history.RR1.notna())
    cols = {}
    for name in history:
        for lag in (0, 1, 2, 3, 6, 12, 24, 48):
            cols[f"{name}_lag{lag}"] = history[name].shift(lag)
    for name in ("T", "U", "PSTAT", "FF", "wind_u", "wind_v"):
        for size in (6, 24, 72):
            cols[f"{name}_mean{size}"] = history[name].rolling(size, min_periods=size).mean()
    for name in ("T", "U", "PSTAT", "FF"):
        for step in (1, 3, 6, 12, 24):
            cols[f"{name}_diff{step}"] = history[name] - history[name].shift(step)
    for size in (6, 24, 72):
        cols[f"RR1_sum{size}"] = history.RR1.rolling(size, min_periods=size).sum()
    cols["T_range24"] = history["T"].rolling(24, min_periods=24).max() - history["T"].rolling(24, min_periods=24).min()
    cols["FF_std24"] = history.FF.rolling(24, min_periods=24).std()
    return pd.DataFrame(cols, index=data.index)


def target_series(data, key):
    column, kind = TARGETS[key][2], TARGETS[key][3]
    series = data[column]
    return (series >= RAIN_THRESHOLD_MM).astype(float).where(series.notna()) if kind == "rain" else series


def load_nwp(index):
    nwp = pd.read_csv(NWP_FILE)
    nwp["valid_time"] = pd.to_datetime(nwp.valid_time, utc=True).dt.tz_localize(None)
    for name in ("wind_speed_10m", "wind_gusts_10m"):
        nwp[name] = pd.to_numeric(nwp[name], errors="coerce") * 3.6
    out = {}
    for horizon in NWP_HORIZONS:
        source = nwp.loc[(nwp.horizon_hours == horizon + NWP_AGE_BUFFER_HOURS) & nwp.model.isin(NWP_MODELS)]
        for key, variable in NWP_COUNTERPART.items():
            table = source.pivot(index="valid_time", columns="model", values=variable).reindex(index)
            table = table[list(NWP_MODELS)].apply(pd.to_numeric, errors="coerce")
            # Models that never provide this variable (ECMWF/JMA gusts) are left out, not filled.
            table = table.dropna(axis=1, how="all")
            if TARGETS[key][3] == "rain":
                # Share of models with measurable rain: a crude ensemble probability.
                present = table.notna().all(axis=1)
                out[(key, horizon)] = (table >= RAIN_THRESHOLD_MM).mean(axis=1).where(present)
            else:
                out[(key, horizon)] = table.mean(axis=1, skipna=False)
    return out


def score(y, pred, kind):
    error = np.asarray(pred, float) - np.asarray(y, float)
    if kind == "rain":
        return {"brier": float(np.mean(error ** 2)), "n": int(len(error))}
    return {"mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.mean(error ** 2))),
            "bias": float(error.mean()), "n": int(len(error))}


def headline(metrics, kind):
    return metrics["brier"] if kind == "rain" else metrics["mae"]


def model():
    return HistGradientBoostingRegressor(max_iter=max(STAGES), max_leaf_nodes=31, learning_rate=.05,
                                         l2_regularization=10, min_samples_leaf=80,
                                         early_stopping=False, random_state=42)


def staged(estimator, X):
    return {stage: pred for stage, pred in enumerate(estimator.staged_predict(X), 1) if stage in STAGES}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_station()
    features = sensor_features(data)
    nwp = load_nwp(data.index)
    valid_all = data.index
    rows, details, cases = [], {}, {}
    with threadpool_limits(limits=4):
        for key, (label, unit, _, kind) in TARGETS.items():
            y = target_series(data, key)
            clip = (0, 1) if kind == "rain" else (0, 100) if key == "U" else (0, None) if key in ("FF", "FXI") else (None, None)
            # Same verifying hours for every lead and method of this variable.
            common = y.notna().to_numpy().copy()
            for horizon in HORIZONS:
                common &= y.shift(horizon).notna().to_numpy()
            test_common = common & (valid_all >= TEST_START)
            for horizon in NWP_HORIZONS:
                if key in NWP_COUNTERPART:
                    test_common &= nwp[(key, horizon)].notna().to_numpy()
            history = y.loc[valid_all < VALIDATION_END - pd.Timedelta(hours=max(HORIZONS))].dropna()
            climate = history.groupby([history.index.month, history.index.hour]).mean()
            details[key] = {"label": label, "unit": unit, "test_n": int(test_common.sum()),
                            "test_start": str(valid_all[test_common].min()), "test_end": str(valid_all[test_common].max()),
                            "horizons": {}}
            for horizon in HORIZONS:
                origins = valid_all - pd.Timedelta(hours=horizon)
                X = features.reindex(origins).set_axis(valid_all)
                X["target_hour_sin"] = np.sin(2 * np.pi * valid_all.hour / 24)
                X["target_hour_cos"] = np.cos(2 * np.pi * valid_all.hour / 24)
                X["target_year_sin"] = np.sin(2 * np.pi * valid_all.day_of_year / 365.25)
                X["target_year_cos"] = np.cos(2 * np.pi * valid_all.day_of_year / 365.25)
                X = X.astype(np.float32)
                base = y.notna().to_numpy() & (valid_all >= valid_all[0] + pd.Timedelta(hours=72))
                train = base & (valid_all < TRAIN_END - pd.Timedelta(hours=horizon))
                validation = common & (valid_all >= TRAIN_END) & (valid_all < VALIDATION_END - pd.Timedelta(hours=max(HORIZONS)))
                refit = base & (valid_all < VALIDATION_END - pd.Timedelta(hours=horizon))
                assert valid_all[refit].max() < (valid_all[test_common] - pd.Timedelta(hours=horizon)).min()
                estimator = model().fit(X.loc[train], y.loc[train])
                yv = y.loc[validation].to_numpy()
                choices = {s: headline(score(yv, np.clip(p, *clip), kind), kind)
                           for s, p in staged(estimator, X.loc[validation]).items()}
                iterations = min(choices, key=choices.get)
                print(f"{key} +{horizon}h: train={train.sum()} validation choice {iterations} it", flush=True)
                final = model().set_params(max_iter=iterations).fit(X.loc[refit], y.loc[refit])
                yt = y.loc[test_common].to_numpy()
                index = valid_all[test_common]
                keys = pd.MultiIndex.from_arrays([index.month, index.hour])
                preds = {"Persistance": y.reindex(origins).set_axis(valid_all).loc[test_common].to_numpy(),
                         "Moyenne saisonnière": climate.reindex(keys).to_numpy(),
                         "Capteurs seuls (arbres)": np.clip(final.predict(X.loc[test_common]), *clip)}
                if horizon in NWP_HORIZONS and key in NWP_COUNTERPART:
                    preds["Open-Meteo brut (moy. 6 modèles)"] = nwp[(key, horizon)].loc[test_common].to_numpy()
                details[key]["horizons"][str(horizon)] = {"selected_iterations": iterations,
                                                          "train_n": int(train.sum()), "refit_n": int(refit.sum())}
                for method, pred in preds.items():
                    rows.append({"variable": key, "label": label, "unit": unit, "horizon_hours": horizon,
                                 "method": method, **score(yt, pred, kind)})
                cases[(key, horizon)] = pd.DataFrame({"valid_time": index, "observed": yt,
                                                      **{m: p for m, p in preds.items()}})
    scores = pd.DataFrame(rows)
    scores.to_csv(OUT / "scores.csv", index=False, float_format="%.4f")
    for (key, horizon), frame in cases.items():
        if horizon in (1, 24, 72):
            frame.to_csv(OUT / f"predictions_{key}_{horizon}h.csv", index=False, float_format="%.3f")
    summary = {
        "study": "Sensor-only forecasts of every station variable, +1 to +72 h, against raw Open-Meteo at +24/+48/+72 h",
        "inputs": "SAINT-ETIENNE T, U, FF, DD, FXI, RR1 and BOUTHEON PSTAT history; calendar of the target hour",
        "split": {"train_targets_before": str(TRAIN_END), "validation": "2022-01-01 to 2023-12-28 (iterations only)",
                  "refit_targets_before": str(VALIDATION_END), "test_targets_from": str(TEST_START)},
        "rain_definition": f"RR1 >= {RAIN_THRESHOLD_MM} mm during the target hour; probability scored with the Brier score",
        "nwp": f"Open-Meteo Previous Runs, archive offset = lead + {NWP_AGE_BUFFER_HOURS} h, mean of {len(NWP_MODELS)} models; "
               "rain probability = share of models with >= 0.2 mm",
        "test_hours": "same verifying hours for all methods and leads of a variable; restricted to hours with Open-Meteo "
                      "counterpart available (2024-2026)",
        "limitations": ["2025 and early 2026 were inspected in earlier wind studies; this is not a never-viewed test",
                        "Météo-France sensors and siting, not the future station's: calibration still needed",
                        "Pressure comes from Bouthéon (13 km, 395 m), not the target station",
                        "Open-Meteo values are point forecasts with no local correction here"],
        "variables": details, "scores": rows,
        "inputs_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [*SOURCES, NWP_FILE]},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "sklearn": sklearn.__version__},
    }
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(scores.pivot_table(index=["variable", "method"], columns="horizon_hours",
                             values="mae", aggfunc="first").round(2).to_string(), flush=True)
    print(scores.loc[scores.variable == "RAIN"].pivot_table(index="method", columns="horizon_hours",
                                                            values="brier").round(4).to_string(), flush=True)


if __name__ == "__main__":
    sys.exit(main())
