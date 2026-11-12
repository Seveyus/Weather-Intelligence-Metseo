"""Multi-source wind benchmark for Saint-Étienne at +24/+48/+72 h.

Follows reports/wind_multisource/PROTOCOLE.md. Two phases:
    python analysis/benchmark_wind_multisource.py --phase select    # rolling validation only
    python analysis/benchmark_wind_multisource.py --phase evaluate  # one final test scoring
The select phase never reads test targets; evaluate refuses to run if the
protocol or this script changed after selection.json was written.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wind_horizon_study import make_features, score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "wind_multisource"
OBS_DIR = ROOT / "data" / "meteofrance"
NWP_FILE = ROOT / "data" / "nwp_multisource" / "nwp_archive.csv"
CACHE = ROOT / "data" / "observations_multisource.pkl.gz"
PROTOCOL = OUT / "PROTOCOLE.md"

HORIZONS = (24, 48, 72)
NWP_AGE_BUFFER_HOURS = 24
DEV_START = pd.Timestamp("2024-01-05")
TEST_ISSUE_START = pd.Timestamp("2026-02-05")
TEST_START = TEST_ISSUE_START + pd.Timedelta(hours=max(HORIZONS))
FOLDS = (("2025-01-01", "2025-04-01"), ("2025-04-01", "2025-07-01"), ("2025-07-01", "2025-10-01"),
         ("2025-10-01", "2026-01-01"), ("2026-01-01", "2026-02-05"))
PURGE = pd.Timedelta(hours=max(HORIZONS))
STAGES = (75, 150, 300)

TARGET_STATION = "42218011"
TARGET_VARIABLES = ("FF", "DD", "FXY", "FXI", "T", "TD", "U", "PSTAT", "RR1")
WIND_STATIONS = {  # <= 60 km, FF present >= 99% since 2024 (see protocol)
    "42005001": "bouthéon", "42207005": "st_chamond", "42101001": "pilat",
    "43137003": "monistrol", "69028001": "brindas", "63319002": "st_anthème",
    "43130002": "mazet", "63003004": "ambert", "42039003": "chalmazel",
    "69029001": "lyon_bron", "69174002": "les_sauvages",
}
PRESSURE_STATIONS = {"42005001": "bouthéon", "69029001": "lyon_bron", "69299001": "lyon_st_exupéry",
                     "42253001": "roanne", "63113001": "clermont"}
V1_MODELS = ("ecmwf_ifs025", "icon_global", "icon_eu", "gfs_global")
MODELS = V1_MODELS + ("gem_global", "jma_seamless")
V1_NWP_VARIABLES = ("temperature_2m", "surface_pressure")
NWP_SCALARS = ("temperature_2m", "surface_pressure", "pressure_msl", "relative_humidity_2m",
               "cloud_cover", "precipitation")
FEATURE_SETS = ("A", "B", "C")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_files():
    return sorted(OBS_DIR.glob("H_*_*.csv.gz"))


# ----------------------------------------------------------------------------- data

def read_regional_observations():
    """Hourly frames for the target and the protocol's neighbouring stations (cached)."""
    files = source_files()
    if not files:
        raise FileNotFoundError("Run analysis/fetch_meteofrance.py first")
    fingerprint = {p.name: p.stat().st_size for p in files}
    if CACHE.exists():
        cached = pd.read_pickle(CACHE)
        if cached["fingerprint"] == fingerprint:
            return cached["stations"], cached["audit"]
    wanted = {TARGET_STATION, *WIND_STATIONS, *PRESSURE_STATIONS}
    columns = ["NUM_POSTE", "AAAAMMJJHH", "QFF", "PMER", *TARGET_VARIABLES]
    parts = []
    for path in files:
        print(f"Reading {path.name}", flush=True)
        for chunk in pd.read_csv(path, sep=";", usecols=columns, chunksize=200_000,
                                 dtype={"NUM_POSTE": str, "AAAAMMJJHH": str}):
            parts.append(chunk.loc[chunk.NUM_POSTE.isin(wanted)])
    raw = pd.concat(parts, ignore_index=True)
    raw["time"] = pd.to_datetime(raw.AAAAMMJJHH, format="%Y%m%d%H")
    target = raw.loc[raw.NUM_POSTE == TARGET_STATION].set_index("time").sort_index()
    index = pd.date_range(target.index.min(), target.index.max(), freq="h")
    stations, audit = {}, {}
    for station, frame in raw.groupby("NUM_POSTE"):
        frame = frame.set_index("time").sort_index()
        if not frame.index.is_unique:
            raise ValueError(f"Duplicate timestamps for station {station}")
        data = frame[["QFF", "PMER", *TARGET_VARIABLES]].reindex(index)
        for name in ("FF", "FXY", "FXI"):
            data[name] *= 3.6
        data.loc[data.FF < 0, "FF"] = np.nan
        stations[station] = data
        audit[station] = {"rows": int(len(frame)), "ff_present": float(data.FF.notna().mean()),
                          "pmer_present": float(data.PMER.notna().mean())}
    # Same target definition as the frozen baselines: pressure falls back to Bouthéon.
    stations[TARGET_STATION]["PSTAT"] = stations[TARGET_STATION].PSTAT.fillna(stations["42005001"].PSTAT)
    pd.to_pickle({"fingerprint": fingerprint, "stations": stations, "audit": audit}, CACHE)
    return stations, audit


def neighbour_features(stations):
    """Causal regional features at the issue time (bounded forward fill only)."""
    cols = {}
    for station, name in WIND_STATIONS.items():
        data = stations[station][["FF", "DD", "FXI"]].ffill(limit=3)
        angle = np.deg2rad(data.DD)
        series = {"FF": data.FF, "u": -data.FF * np.sin(angle), "v": -data.FF * np.cos(angle)}
        for var, values in series.items():
            for lag in (0, 3, 6, 24):
                cols[f"st_{name}_{var}_lag{lag}"] = values.shift(lag)
        cols[f"st_{name}_FF_mean24"] = data.FF.rolling(24, min_periods=24).mean()
        cols[f"st_{name}_FXI_lag0"] = data.FXI
    pmer = {name: stations[station].PMER.ffill(limit=3) for station, name in PRESSURE_STATIONS.items()}
    for name, values in pmer.items():
        cols[f"st_{name}_PMER_lag0"] = values
        cols[f"st_{name}_PMER_diff3"] = values - values.shift(3)
        cols[f"st_{name}_PMER_diff24"] = values - values.shift(24)
        if name != "bouthéon":
            cols[f"st_grad_{name}_minus_bouthéon"] = values - pmer["bouthéon"]
    target_t = stations[TARGET_STATION]["T"].ffill(limit=3)
    cols["st_pilat_minus_target_T"] = stations["42101001"]["T"].ffill(limit=3) - target_t
    cols["st_bouthéon_minus_target_T"] = stations["42005001"]["T"].ffill(limit=3) - target_t
    return pd.DataFrame(cols)


def read_nwp():
    if not NWP_FILE.exists():
        raise FileNotFoundError("Run analysis/fetch_nwp_archive.py with the protocol's options first")
    nwp = pd.read_csv(NWP_FILE)
    nwp["valid_time"] = pd.to_datetime(nwp.valid_time, utc=True).dt.tz_localize(None)
    if nwp.duplicated(["valid_time", "model", "horizon_hours"]).any():
        raise ValueError("Duplicate NWP model/horizon/valid times")
    for name in ("wind_speed_10m", "wind_gusts_10m", "wind_speed_100m"):
        nwp[name] = pd.to_numeric(nwp[name], errors="coerce") * 3.6
    return nwp


def nwp_features(nwp, horizon, valid):
    source = nwp.loc[nwp.horizon_hours == horizon + NWP_AGE_BUFFER_HOURS]
    cols, speeds = {}, {}
    for model in MODELS:
        group = source.loc[source.model == model].set_index("valid_time")
        aligned = group.reindex(valid)
        speed = aligned.wind_speed_10m
        angle = np.deg2rad(pd.to_numeric(aligned.wind_direction_10m, errors="coerce"))
        prefix = f"nwp_{model}"
        speeds[model] = f"{prefix}_speed"
        cols[f"{prefix}_speed"] = speed
        cols[f"{prefix}_u"] = -speed * np.sin(angle)
        cols[f"{prefix}_v"] = -speed * np.cos(angle)
        cols[f"{prefix}_gust"] = aligned.wind_gusts_10m
        cols[f"{prefix}_speed100"] = aligned.wind_speed_100m
        cols[f"{prefix}_shear"] = aligned.wind_speed_100m / (speed + 1.0)
        for variable in NWP_SCALARS:
            cols[f"{prefix}_{variable}"] = pd.to_numeric(aligned[variable], errors="coerce")
        for shift in (-3, 3):
            # Same archive offset: the run is still >= 21 h older than the issue time.
            cols[f"{prefix}_speed_t{shift:+d}"] = group.wind_speed_10m.reindex(valid + pd.Timedelta(hours=shift)).to_numpy()
    frame = pd.DataFrame(cols, index=valid)
    return frame, speeds


def build_frames():
    stations, audit = read_regional_observations()
    target = stations[TARGET_STATION][list(TARGET_VARIABLES)]
    past = make_features(target)
    for name in target:
        past[f"{name}_missing"] = target[name].isna().astype(float)
    regional = neighbour_features(stations)
    nwp = read_nwp()
    y = target.FF
    frames = {}
    for horizon in HORIZONS:
        valid = target.index
        origins = valid - pd.Timedelta(hours=horizon)
        base = past.reindex(origins).set_axis(valid)
        base["target_hour_sin"] = np.sin(2 * np.pi * valid.hour / 24)
        base["target_hour_cos"] = np.cos(2 * np.pi * valid.hour / 24)
        base["target_year_sin"] = np.sin(2 * np.pi * valid.day_of_year / 365.25)
        base["target_year_cos"] = np.cos(2 * np.pi * valid.day_of_year / 365.25)
        persistence = y.reindex(origins).set_axis(valid)
        model_frame, speeds = nwp_features(nwp, horizon, valid)
        v1_speeds = [speeds[m] for m in V1_MODELS]
        all_speeds = [speeds[m] for m in MODELS]
        derived = pd.DataFrame({
            "nwp_mean": model_frame[v1_speeds].mean(axis=1),
            "nwp_spread": model_frame[v1_speeds].std(axis=1, ddof=0),
            "nwp_mean6": model_frame[all_speeds].mean(axis=1),
            "nwp_spread6": model_frame[all_speeds].std(axis=1, ddof=0),
            "nwp_gust_mean": model_frame[[f"nwp_{m}_gust" for m in MODELS]].mean(axis=1),
            "nwp_speed100_mean": model_frame[[f"nwp_{m}_speed100" for m in MODELS]].mean(axis=1),
            "nwp_mean6_t-3": model_frame[[f"nwp_{m}_speed_t-3" for m in MODELS]].mean(axis=1),
            "nwp_mean6_t+3": model_frame[[f"nwp_{m}_speed_t+3" for m in MODELS]].mean(axis=1),
        }, index=valid)
        derived["nwp_minus_current"] = derived.nwp_mean - persistence
        derived["nwp_mean6_minus_current"] = derived.nwp_mean6 - persistence
        region = regional.reindex(origins).set_axis(valid)
        X = pd.concat([base, model_frame, derived, region], axis=1).astype(np.float32)

        v1_nwp = [f"nwp_{m}_{v}" for m in V1_MODELS for v in ("speed", "u", "v", *V1_NWP_VARIABLES)]
        set_a = list(base.columns) + v1_nwp + ["nwp_mean", "nwp_spread", "nwp_minus_current"]
        set_b = set_a + [c for c in list(model_frame.columns) + list(derived.columns) if c not in set_a]
        set_c = set_b + list(region.columns)
        common = y.notna() & persistence.notna() & X[all_speeds].notna().all(axis=1)
        common &= valid >= DEV_START
        frames[horizon] = {"X": X, "y": y, "persistence": persistence, "sets": {"A": set_a, "B": set_b, "C": set_c},
                           "residual_base": {"A": "nwp_mean", "B": "nwp_mean6", "C": "nwp_mean6"},
                           "raw": {**{f"NWP {m}": speeds[m] for m in MODELS},
                                   "NWP mean (4 models)": "nwp_mean", "NWP mean (6 models)": "nwp_mean6"},
                           "common": common, "climate_obs": y}
    # Same target hours for every method and every horizon, in validation and test.
    shared = np.logical_and.reduce([frame.pop("common") for frame in frames.values()])
    for horizon, frame in frames.items():
        valid = frame["X"].index
        frame["dev"] = shared & (valid < TEST_ISSUE_START)
        frame["test"] = shared & (valid >= TEST_START)
        assert valid[frame["dev"]].max() < (valid[frame["test"]] - pd.Timedelta(hours=horizon)).min()
    return frames, audit


# ------------------------------------------------------------------------ candidates

def ridge():
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                         StandardScaler(), Ridge(alpha=100.0))


def hgb(leaves):
    return HistGradientBoostingRegressor(max_iter=max(STAGES), max_leaf_nodes=leaves, learning_rate=.04,
                                         l2_regularization=20, min_samples_leaf=40,
                                         early_stopping=False, random_state=42)


def candidate_specs():
    specs = []
    for feature_set in FEATURE_SETS:
        specs.append({"name": f"{feature_set} Ridge residual", "set": feature_set, "kind": "residual",
                      "estimator": ridge()})
        for kind in ("direct", "residual"):
            for leaves in (7, 15):
                specs.append({"name": f"{feature_set} HGB{leaves} {kind}", "set": feature_set, "kind": kind,
                              "estimator": hgb(leaves)})
    return specs


def seasonal_mean(y, before):
    history = y.loc[(y.index < before)].dropna()
    table = history.groupby([history.index.month, history.index.hour]).mean()
    return table, float(history.mean())


def predict_climate(table, fallback, index):
    keys = pd.MultiIndex.from_arrays([index.month, index.hour])
    return table.reindex(keys).fillna(fallback).to_numpy()


def staged_predictions(spec, fitted, frame, mask):
    """Predictions keyed by variant name: one per HGB stage, a single one otherwise."""
    X = frame["X"].loc[mask, fitted.feature_columns_]
    offset = frame["X"].loc[mask, frame["residual_base"][spec["set"]]].to_numpy() if spec["kind"] == "residual" else 0
    if isinstance(fitted, HistGradientBoostingRegressor):
        out = {}
        for stage, pred in enumerate(fitted.staged_predict(X), 1):
            if stage in STAGES:
                out[f"{spec['name']} it{stage}"] = np.maximum(pred + offset, 0)
        return out
    return {spec["name"]: np.maximum(fitted.predict(X) + offset, 0)}


def fit(spec, frame, mask, iterations=None):
    X = frame["X"].loc[mask, frame["sets"][spec["set"]]]
    # Columns empty or constant in this training window carry no information
    # (and break histogram binning); the choice uses training rows only.
    columns = list(X.columns[X.nunique() >= 2])
    X = X[columns]
    y = frame["y"].loc[mask]
    if spec["kind"] == "residual":
        y = y - frame["X"].loc[mask, frame["residual_base"][spec["set"]]]
    estimator = clone(spec["estimator"])
    if iterations is not None:
        estimator.set_params(max_iter=iterations)
    estimator.fit(X, y)
    estimator.feature_columns_ = columns
    return estimator


def selection_score(metrics, reference):
    return .5 * (metrics["mae_kmh"] / reference["mae_kmh"] + metrics["rmse_kmh"] / reference["rmse_kmh"])


def reference_predictions(frame, mask, before):
    table, fallback = seasonal_mean(frame["climate_obs"], before)
    index = frame["X"].index[mask]
    out = {"Persistence": frame["persistence"].loc[mask].to_numpy(),
           "Seasonal mean": predict_climate(table, fallback, index)}
    for name, column in frame["raw"].items():
        out[name] = frame["X"].loc[mask, column].to_numpy(dtype=float)
    return out


# ------------------------------------------------------------------------- phases

def select(frames):
    selection = {"protocol_sha256": sha256(PROTOCOL), "script_sha256": sha256(Path(__file__)),
                 "written_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "horizons": {}}
    specs = candidate_specs()
    for horizon, frame in frames.items():
        valid = frame["X"].index
        fold_scores, oof = {}, {}
        for start, end in FOLDS:
            start, end = pd.Timestamp(start), pd.Timestamp(end)
            train = frame["dev"] & (valid < start - PURGE)
            val = frame["dev"] & (valid >= start) & (valid < end)
            assert valid[train].max() < (valid[val] - pd.Timedelta(hours=horizon)).min()
            y = frame["y"].loc[val].to_numpy()
            preds = reference_predictions(frame, val, start - PURGE)
            reference = score(y, preds["Seasonal mean"])
            for spec in specs:
                print(f"+{horizon}h fold {start.date()} train={train.sum()} val={val.sum()} fit {spec['name']}", flush=True)
                preds.update(staged_predictions(spec, fit(spec, frame, train), frame, val))
            for name, pred in preds.items():
                metrics = score(y, pred)
                fold_scores.setdefault(name, []).append({"fold_start": str(start.date()), **metrics,
                                                          "selection_score": selection_score(metrics, reference)})
                oof.setdefault(name, []).append(y - pred)
        mean_score = {name: float(np.mean([f["selection_score"] for f in folds])) for name, folds in fold_scores.items()}
        finalists = {}
        for feature_set in FEATURE_SETS:
            names = [n for n in mean_score if n.startswith(f"{feature_set} ")]
            finalists[feature_set] = min(names, key=mean_score.get)
        chosen = min(finalists.values(), key=mean_score.get)
        best_raw = min(frame["raw"], key=mean_score.get)
        radii = {name: interval_radius(np.concatenate(oof[name])) for name in [*finalists.values(), best_raw]}
        selection["horizons"][str(horizon)] = {
            "dev_targets": int(frame["dev"].sum()), "finalists": finalists, "selected": chosen,
            "best_raw_nwp": best_raw, "mean_selection_score": dict(sorted(mean_score.items(), key=lambda kv: kv[1])),
            "fold_scores": fold_scores, "interval_radius_80_kmh": radii,
            "feature_counts": {s: len(cols) for s, cols in frame["sets"].items()},
        }
        print(f"+{horizon}h finalists {finalists}; selected {chosen}; best raw {best_raw}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selection.json").write_text(json.dumps(selection, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved {OUT / 'selection.json'} (no test target read)", flush=True)


def interval_radius(residuals, coverage=.8):
    n = len(residuals)
    rank = min(n, int(np.ceil((n + 1) * coverage)))
    return float(np.sort(np.abs(residuals))[rank - 1])


def block_bootstrap_gain(y, baseline, pred, dates, days=7, replicates=2000):
    """Paired MAE gain (percent) with weekly block resampling; descriptive only."""
    a, b = np.abs(y - baseline), np.abs(y - pred)
    week = ((dates - dates.min()) // pd.Timedelta(days=days)).to_numpy()
    blocks = np.unique(week)
    sa = np.array([a[week == k].sum() for k in blocks])
    sb = np.array([b[week == k].sum() for k in blocks])
    draws = np.random.default_rng(42).integers(0, len(blocks), size=(replicates, len(blocks)))
    samples = 100 * (1 - sb[draws].sum(axis=1) / sa[draws].sum(axis=1))
    return {"mae_gain_percent": float(100 * (1 - b.mean() / a.mean())),
            "approx_95pct_interval": np.quantile(samples, [.025, .975]).tolist(), "blocks": int(len(blocks))}


SEASONS = {"Feb–Mar": (2, 3), "Apr–Jun": (4, 5, 6), "Jul–Aug": (7, 8), "Sep–Oct": (9, 10)}


def evaluate(frames, audit):
    selection = json.loads((OUT / "selection.json").read_text())
    if selection["protocol_sha256"] != sha256(PROTOCOL) or selection["script_sha256"] != sha256(Path(__file__)):
        raise RuntimeError("Protocol or script changed after selection; the frozen choice no longer applies")
    specs = {spec["name"]: spec for spec in candidate_specs()}
    scores, details = [], {}
    for horizon, frame in frames.items():
        chosen = selection["horizons"][str(horizon)]
        dev, test = frame["dev"], frame["test"]
        y = frame["y"].loc[test].to_numpy()
        dates = frame["X"].index[test]
        preds = reference_predictions(frame, test, TEST_ISSUE_START)
        for name, spec in specs.items():
            print(f"+{horizon}h refit {name} on {dev.sum()} development targets", flush=True)
            preds.update(staged_predictions(spec, fit(spec, frame, dev), frame, test))
        finalists = chosen["finalists"]
        threshold = float(frame["y"].loc[dev].quantile(.9))
        high = y >= threshold
        month = dates.month
        for name, pred in preds.items():
            row = {"horizon_hours": horizon, "method": name, **score(y, pred),
                   "high_wind_mae_kmh": float(np.abs(pred[high] - y[high]).mean()),
                   "high_wind_bias_kmh": float((pred[high] - y[high]).mean())}
            for season, months in SEASONS.items():
                mask = np.isin(month, months)
                row[f"mae_{season}"] = float(np.abs(pred[mask] - y[mask]).mean()) if mask.any() else None
            scores.append(row)
        selected = preds[chosen["selected"]]
        radius = chosen["interval_radius_80_kmh"][chosen["selected"]]
        lower, upper = np.maximum(selected - radius, 0), selected + radius
        best_raw = preds[chosen["best_raw_nwp"]]
        details[str(horizon)] = {
            "test_n": int(len(y)), "test_start": str(dates.min()), "test_end": str(dates.max()),
            "dev_targets": int(dev.sum()), "finalists": finalists, "selected": chosen["selected"],
            "best_raw_nwp_from_validation": chosen["best_raw_nwp"],
            "gain_selected_vs_raw_nwp": block_bootstrap_gain(y, best_raw, selected, dates),
            "gain_C_vs_A": block_bootstrap_gain(y, preds[finalists["A"]], preds[finalists["C"]], dates),
            "gain_B_vs_A": block_bootstrap_gain(y, preds[finalists["A"]], preds[finalists["B"]], dates),
            "gain_C_vs_B": block_bootstrap_gain(y, preds[finalists["B"]], preds[finalists["C"]], dates),
            "development_p90_wind_kmh": threshold, "high_wind_test_n": int(high.sum()),
            "interval_radius_80_kmh": radius,
            "interval_test_coverage": float(((y >= lower) & (y <= upper)).mean()),
            "interval_mean_width_kmh": float((upper - lower).mean()),
        }
        pd.DataFrame({"valid_time": dates, "issue_time": dates - pd.Timedelta(hours=horizon), "observed_kmh": y,
                      "selected_kmh": selected, "lower80_kmh": lower, "upper80_kmh": upper,
                      "finalist_A_kmh": preds[finalists["A"]], "finalist_B_kmh": preds[finalists["B"]],
                      "finalist_C_kmh": preds[finalists["C"]], "raw_nwp_kmh": best_raw,
                      "persistence_kmh": preds["Persistence"]}).to_csv(OUT / f"predictions_{horizon}h.csv",
                                                                       index=False, float_format="%.3f")
        print(f"+{horizon}h selected {chosen['selected']}: {score(y, selected)}", flush=True)
    summary = {
        "study": "Multi-source local correction of archived NWP for Saint-Étienne, +24/+48/+72 h",
        "protocol": "reports/wind_multisource/PROTOCOLE.md", "protocol_sha256": selection["protocol_sha256"],
        "selection_written_at_utc": selection["written_at_utc"],
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "test_issue_times_from": str(TEST_ISSUE_START), "horizons": details, "scores": scores,
        "observation_audit": audit,
        "inputs": {"nwp_archive_sha256": sha256(NWP_FILE),
                   "observations": {p.name: sha256(p) for p in source_files()}},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "sklearn": sklearn.__version__},
    }
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    pd.DataFrame(scores).to_csv(OUT / "all_scores.csv", index=False, float_format="%.4f")
    print(f"Saved {OUT}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("select", "evaluate"), required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    frames, audit = build_frames()
    with threadpool_limits(limits=args.threads):
        select(frames) if args.phase == "select" else evaluate(frames, audit)
