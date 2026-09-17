"""Direct +24/+48/+72h wind forecasting with archived NWP and local correction.

Run from the repository root after fetch_nwp_archive.py. Model choices are frozen
on 2025 validation before scoring the January–February 2026 winter holdout.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from wind_horizon_study import load_observations, make_features, score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "wind_advanced"
DATA = OUT / "data"
HORIZONS = (24, 48, 72)
NWP_AGE_BUFFER_HOURS = 24
VAL_START = pd.Timestamp("2025-01-01")
CAL_START = pd.Timestamp("2025-10-01")
TEST_START = pd.Timestamp("2026-01-01")
STAGES = (75, 150, 300)


def read_observations():
    cache = DATA / "observations_hourly.csv"
    stamp = DATA / "observations_provenance.json"
    files = [ROOT / name for name in ("H_42_previous-2020-2024.csv", "H_42_latest-2025-2026.csv")]
    fingerprints = [{"name": p.name, "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in files]
    if cache.exists() and stamp.exists():
        provenance = json.loads(stamp.read_text())
        if provenance["sources"] == fingerprints:
            return pd.read_csv(cache, index_col=0, parse_dates=True), provenance["audit"]
    observations, audit = load_observations()
    cache.parent.mkdir(parents=True, exist_ok=True)
    observations.to_csv(cache, index_label="time")
    stamp.write_text(json.dumps({"sources": fingerprints, "audit": audit}, indent=2) + "\n")
    return observations, audit


def load_frames():
    obs, audit = read_observations()
    nwp = pd.concat([pd.read_csv(DATA / "nwp_archive.csv"),
                     pd.read_csv(DATA / "day4" / "nwp_archive.csv")], ignore_index=True)
    nwp["valid_time"] = pd.to_datetime(nwp.valid_time, utc=True).dt.tz_localize(None)
    if nwp.duplicated(["valid_time", "model", "horizon_hours"]).any():
        raise ValueError("Duplicate NWP model/horizon/valid times")
    past = make_features(obs)
    # Missingness is itself useful; filling is bounded and exclusively causal.
    for name in obs:
        past[f"{name}_missing"] = obs[name].isna().astype(float)
    frames, model_columns = {}, {}
    for horizon in HORIZONS:
        valid = obs.index
        origins = valid - pd.Timedelta(hours=horizon)
        X = past.reindex(origins).set_axis(valid)
        X["target_hour_sin"] = np.sin(2 * np.pi * valid.hour / 24)
        X["target_hour_cos"] = np.cos(2 * np.pi * valid.hour / 24)
        X["target_year_sin"] = np.sin(2 * np.pi * valid.dayofyear / 365.25)
        X["target_year_cos"] = np.cos(2 * np.pi * valid.dayofyear / 365.25)
        obs_columns = list(X.columns)
        # Previous Runs chooses by initialization, not actual publication time.
        # An extra day means all input runs were initialized at least 24h before
        # our issue time; this is conservative relative to normal release delays.
        source = nwp.loc[nwp.horizon_hours == horizon + NWP_AGE_BUFFER_HOURS]
        speeds = {}
        for name, group in source.groupby("model"):
            group = group.set_index("valid_time").reindex(valid)
            speed = pd.to_numeric(group.wind_speed_10m, errors="coerce") * 3.6
            # Determine eligibility by coverage only, never by test performance.
            train_n = speed.loc[valid < VAL_START].notna().sum()
            validation_coverage = speed.loc[(valid >= VAL_START) & (valid < CAL_START)].notna().mean()
            if train_n < 1000 or validation_coverage < .80:
                print(f"+{horizon}h exclude {name}: train_n={train_n}, val coverage={validation_coverage:.1%}", flush=True)
                continue
            col = f"nwp_{name}_speed"
            X[col] = speed
            speeds[name] = col
            angle = np.deg2rad(pd.to_numeric(group.wind_direction_10m, errors="coerce"))
            X[f"nwp_{name}_u"] = -speed * np.sin(angle)
            X[f"nwp_{name}_v"] = -speed * np.cos(angle)
            for variable in ("temperature_2m", "surface_pressure"):
                X[f"nwp_{name}_{variable}"] = pd.to_numeric(group[variable], errors="coerce")
        if not speeds:
            raise ValueError(f"No NWP model has adequate archive coverage at +{horizon}h")
        X["nwp_mean"] = X[list(speeds.values())].mean(axis=1)
        X["nwp_spread"] = X[list(speeds.values())].std(axis=1, ddof=0)
        X["nwp_minus_current"] = X.nwp_mean - obs.FF.reindex(origins).to_numpy()
        frame = {"X": X.astype(np.float32), "obs_columns": obs_columns,
                 "y": obs.FF, "origin": origins, "raw": speeds,
                 "persistence": obs.FF.reindex(origins).set_axis(valid)}
        frames[horizon] = frame
        model_columns[horizon] = speeds
    # Fair comparisons: every method and horizon evaluated on the same target hours.
    common = obs.FF.notna()
    for h, frame in frames.items():
        common &= frame["persistence"].notna()
        common &= frame["X"][list(frame["raw"].values())].notna().all(axis=1)
    for h, frame in frames.items():
        valid, origins = obs.index, frame["origin"]
        available = obs.FF.notna() & frame["persistence"].notna()
        available &= origins >= obs.index.min() + pd.Timedelta(hours=72)
        frame["train_obs"] = available & (valid < VAL_START)
        frame["train_nwp"] = frame["train_obs"] & frame["X"][list(frame["raw"].values())].notna().all(axis=1)
        frame["refit_obs"] = available & (valid < CAL_START)
        frame["refit_nwp"] = frame["refit_obs"] & frame["X"][list(frame["raw"].values())].notna().all(axis=1)
        frame["val"] = common & (valid >= VAL_START + pd.Timedelta(hours=max(HORIZONS))) & (valid < CAL_START)
        frame["cal"] = common & (valid >= CAL_START + pd.Timedelta(hours=max(HORIZONS))) & (valid < TEST_START)
        frame["test"] = common & (valid >= TEST_START + pd.Timedelta(hours=max(HORIZONS)))
        assert valid[frame["train_obs"]].max() < origins[frame["val"]].min()
        assert valid[frame["refit_obs"]].max() < origins[frame["cal"]].min()
        assert valid[frame["cal"]].max() < origins[frame["test"]].min()
        for split in ("train_nwp", "val", "cal", "test"):
            if frame[split].sum() < 100:
                raise ValueError(f"Insufficient +{h}h {split} cases: {frame[split].sum()}")
    return obs, audit, frames


def linear_estimator(pca=None):
    steps = [SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True), StandardScaler()]
    if pca is not None:
        steps.append(PCA(n_components=pca, svd_solver="full"))
    steps.append(Ridge(alpha=100.0))
    return make_pipeline(*steps)


def candidates(frame):
    specs = [{"name": "Persistence", "kind": "persistence"},
             {"name": "Seasonal mean", "kind": "climate"}]
    specs += [{"name": f"NWP {model}", "kind": "raw", "column": col} for model, col in frame["raw"].items()]
    specs += [{"name": "NWP ensemble mean", "kind": "raw", "column": "nwp_mean"}]
    for pca in (None, .95, .99):
        label = "Ridge" if pca is None else f"PCA{int(pca*100)} + Ridge"
        specs.append({"name": "Observations " + label, "kind": "obs", "estimator": linear_estimator(pca)})
        specs.append({"name": "NWP residual " + label, "kind": "residual", "estimator": linear_estimator(pca)})
    for kind in ("obs", "direct", "residual"):
        for leaves in ((15,) if kind == "obs" else (7, 15)):
            model = HistGradientBoostingRegressor(max_iter=max(STAGES), max_leaf_nodes=leaves,
                learning_rate=.04, l2_regularization=20, min_samples_leaf=40,
                early_stopping=False, random_state=42)
            specs.append({"name": f"{kind} HGB{leaves}", "kind": kind, "estimator": model})
    return specs


def features_for(spec, frame):
    return frame["X"][frame["obs_columns"]] if spec["kind"] == "obs" else frame["X"]


def selection_score(metrics, reference):
    return .5 * (metrics["mae_kmh"] / reference["mae_kmh"] + metrics["rmse_kmh"] / reference["rmse_kmh"])


def fit_spec(spec, frame, refit=False, reference=None):
    fitted = deepcopy(spec)
    if spec["kind"] == "climate":
        mask = frame["refit_obs" if refit else "train_obs"]
        y = frame["y"].loc[mask]
        fitted["climate"] = y.groupby([y.index.month, y.index.hour]).mean().to_dict()
        fitted["fallback"] = float(y.mean())
    elif "estimator" in spec:
        suffix = "obs" if spec["kind"] == "obs" else "nwp"
        mask = frame[("refit_" if refit else "train_") + suffix]
        X, y = features_for(spec, frame), frame["y"].copy()
        if spec["kind"] == "residual":
            y = y - frame["X"].nwp_mean
        estimator = clone(spec["estimator"])
        estimator.fit(X.loc[mask], y.loc[mask])
        fitted["estimator"] = estimator
        if isinstance(estimator, HistGradientBoostingRegressor) and not refit:
            options = []
            for stage, pred in enumerate(estimator.staged_predict(X.loc[frame["val"]]), 1):
                if stage in STAGES:
                    if spec["kind"] == "residual":
                        pred = pred + frame["X"].loc[frame["val"], "nwp_mean"].to_numpy()
                    metrics = score(frame["y"].loc[frame["val"]], np.maximum(pred, 0))
                    options.append((selection_score(metrics, reference), stage, metrics))
            _, best, _ = min(options)
            fitted["selected_iterations"] = best
            fitted["stage_scores"] = [{"stage": k, **m} for _, k, m in options]
        fitted["fit_samples"] = int(mask.sum())
    return fitted


def predict_spec(spec, frame, mask):
    subset = frame["X"].loc[mask]
    if spec["kind"] == "persistence":
        pred = frame["persistence"].loc[mask].to_numpy()
    elif spec["kind"] == "climate":
        pred = np.array([spec["climate"].get((t.month, t.hour), spec["fallback"]) for t in subset.index])
    elif spec["kind"] == "raw":
        pred = subset[spec["column"]].to_numpy()
    else:
        estimator = spec["estimator"]
        X = features_for(spec, frame).loc[mask]
        if "selected_iterations" in spec:
            for stage, pred in enumerate(estimator.staged_predict(X), 1):
                if stage == spec["selected_iterations"]:
                    break
        else:
            pred = estimator.predict(X)
        if spec["kind"] == "residual":
            pred = pred + subset.nwp_mean.to_numpy()
    return np.maximum(np.asarray(pred), 0)


def freeze_selection(frame):
    fitted, forecasts, metrics = {}, {}, {}
    y = frame["y"].loc[frame["val"]].to_numpy()
    climate = fit_spec({"name": "Seasonal mean", "kind": "climate"}, frame)
    reference = score(y, predict_spec(climate, frame, frame["val"]))
    for spec in candidates(frame):
        print("  Fit", spec["name"], flush=True)
        fitted[spec["name"]] = fit_spec(spec, frame, reference=reference)
        pred = predict_spec(fitted[spec["name"]], frame, frame["val"])
        forecasts[spec["name"]] = pred
        metrics[spec["name"]] = score(y, pred)
    ranking = sorted(metrics, key=lambda k: selection_score(metrics[k], reference))
    best = ranking[0]
    blend = {best: 1.0}
    best_score = selection_score(metrics[best], reference)
    # Small, prespecified convex ensemble search using only validation predictions.
    for left in range(min(3, len(ranking))):
        for right in range(left + 1, min(3, len(ranking))):
            a, b = ranking[left], ranking[right]
            for weight in (.25, .5, .75):
                pred = weight * forecasts[a] + (1 - weight) * forecasts[b]
                candidate_score = selection_score(score(y, pred), reference)
                if candidate_score < best_score:
                    best_score, blend = candidate_score, {a: weight, b: 1 - weight}
    raw = [name for name in metrics if name.startswith("NWP ") and fitted[name]["kind"] == "raw"]
    raw_best = min(raw, key=lambda k: selection_score(metrics[k], reference))
    return fitted, {"blend": blend, "validation_scores": metrics, "best_raw_nwp": raw_best,
                    "selection_score": best_score, "ranking": ranking,
                    "hgb_iterations": {name: spec["selected_iterations"] for name, spec in fitted.items() if "selected_iterations" in spec}}


def refit(fitted, frame):
    models = {}
    for name, spec in fitted.items():
        clean = deepcopy(spec)
        if "selected_iterations" in clean:
            clean["estimator"] = clone(clean["estimator"]).set_params(max_iter=clean.pop("selected_iterations"))
        models[name] = fit_spec(clean, frame, refit=True)
    return models


def blend_predict(models, blend, frame, mask):
    return sum(weight * predict_spec(models[name], frame, mask) for name, weight in blend.items())


def interval_radius(residuals, coverage=.8):
    n = len(residuals)
    rank = min(n, int(np.ceil((n + 1) * coverage)))
    return float(np.sort(np.abs(residuals))[rank - 1])


def paired_skill(y, baseline, pred):
    rng = np.random.default_rng(42)
    blocks = [slice(i, min(i+168, len(y))) for i in range(0, len(y), 168)]
    a, b = np.abs(y-baseline), np.abs(y-pred)
    sa = np.array([a[s].sum() for s in blocks])
    sb = np.array([b[s].sum() for s in blocks])
    draws = rng.integers(0, len(blocks), size=(2000, len(blocks)))
    samples = 100 * (1 - sb[draws].sum(axis=1) / sa[draws].sum(axis=1))
    return {"mae_skill_percent": float(100*(1-b.mean()/a.mean())),
            "approx_95pct_interval": np.quantile(samples,[.025,.975]).tolist(),
            "blocks": len(blocks), "block_scored_observations": 168}


def train():
    OUT.mkdir(parents=True, exist_ok=True)
    obs, audit, frames = load_frames()
    initial, selection = {}, {}
    with threadpool_limits(limits=4):
        for horizon, frame in frames.items():
            print(f"Select +{horizon}h: train NWP={frame['train_nwp'].sum()}, validation={frame['val'].sum()}", flush=True)
            initial[horizon], selection[horizon] = freeze_selection(frame)
            print("Selected:", selection[horizon]["blend"], flush=True)
        # This file is written before reading any final-test error.
        (OUT / "selection.json").write_text(json.dumps(selection, indent=2)+"\n")
        all_scores, details, artifacts = [], {}, {}
        for horizon, frame in frames.items():
            print(f"Refit and final evaluation +{horizon}h", flush=True)
            models = refit(initial[horizon], frame)
            blend = selection[horizon]["blend"]
            cal_pred = blend_predict(models, blend, frame, frame["cal"])
            cal_y = frame["y"].loc[frame["cal"]].to_numpy()
            radius = interval_radius(cal_y-cal_pred)
            pred = blend_predict(models, blend, frame, frame["test"])
            y = frame["y"].loc[frame["test"]].to_numpy()
            dates = frame["y"].index[frame["test"]]
            train_y = frame["y"].loc[frame["refit_obs"]]
            threshold = float(train_y.quantile(.9))
            high = y >= threshold
            case_predictions = {name: predict_spec(spec, frame, frame["test"]) for name,spec in models.items()}
            case_predictions["Selected ensemble"] = pred
            for name, values in case_predictions.items():
                all_scores.append({"horizon_hours": horizon, "method": name, **score(y,values),
                    "high_wind_mae_kmh": float(np.abs(values[high]-y[high]).mean()),
                    "high_wind_bias_kmh": float((values[high]-y[high]).mean())})
            baseline = case_predictions[selection[horizon]["best_raw_nwp"]]
            lower, upper = np.maximum(pred-radius,0), pred+radius
            details[horizon] = {"train_nwp_n": int(frame["train_nwp"].sum()),
                "refit_nwp_n": int(frame["refit_nwp"].sum()), "validation_n": int(frame["val"].sum()),
                "calibration_n": int(frame["cal"].sum()), "test_n": len(y),
                "test_start": str(dates.min()), "test_end": str(dates.max()),
                "selected_blend": blend, "best_raw_nwp_from_validation": selection[horizon]["best_raw_nwp"],
                "skill_against_raw_nwp": paired_skill(y, baseline, pred),
                "training_p90_wind_kmh": threshold, "high_wind_test_n": int(high.sum()),
                "interval_nominal_coverage": .8, "interval_radius_kmh": radius,
                "interval_test_coverage": float(((y>=lower)&(y<=upper)).mean()),
                "interval_mean_width_kmh": float((upper-lower).mean())}
            result = pd.DataFrame({"valid_time": dates,
                "issue_time": dates-pd.Timedelta(hours=horizon), "observed_kmh": y,
                "selected_kmh": pred, "lower80_kmh": lower, "upper80_kmh": upper,
                "raw_nwp_kmh": baseline, "persistence_kmh": case_predictions["Persistence"],
                "seasonal_mean_kmh": case_predictions["Seasonal mean"],
                "observations_hgb_kmh": case_predictions["obs HGB15"]})
            result.to_csv(OUT/f"predictions_{horizon}h.csv", index=False)
            artifacts[horizon] = {"models": {name: models[name] for name in blend}, "blend": blend,
                "interval_radius_kmh": radius, "feature_columns": list(frame["X"].columns),
                "obs_columns": frame["obs_columns"], "nwp_models": list(frame["raw"]),
                "horizon_hours": horizon, "nwp_age_buffer_hours": NWP_AGE_BUFFER_HOURS}
            print(f"+{horizon}h selected {score(y,pred)}; raw {score(y,baseline)}", flush=True)
        joblib.dump(artifacts, OUT / "wind_models.joblib", compress=3)
    summary = {"study": "Local correction for +24/+48/+72h, using conservatively aged archived NWP",
        "nwp_age_buffer_hours": NWP_AGE_BUFFER_HOURS,
        "nwp_input_archive_leads_hours": [h + NWP_AGE_BUFFER_HOURS for h in HORIZONS],
        "selection_rule": "Minimize equal-weight MAE/seasonal-MAE + RMSE/seasonal-RMSE on Jan-Sep2025; optional convex top3 blend",
        "initial_training_targets_before": str(VAL_START), "refit_targets_before": str(CAL_START),
        "validation": "2025-01-04 to 2025-09-30", "calibration": "2025-10-04 to 2025-12-31",
        "test": "2026-01-04 to end of available observations; all issue times >=2026-01-01",
        "interval_method": "80% split-conformal absolute residual radius, Oct-Dec2025; coverage not guaranteed for dependent/drifting weather",
        "limitations": ["Short winter holdout, not annual operational validation",
            "Previous Runs offsets use initialization, not publication: a conservative 24h buffer is imposed, but original publication timestamps/outages are not supplied",
            "Station observations assumed available at timestamp; reporting latency not measured",
            "Original target quality codes not independently audited beyond excluding missing/negative FF"],
        "data_audit": audit, "horizons": details, "scores": all_scores,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/"metrics.json").write_text(json.dumps(summary,indent=2)+"\n")
    pd.DataFrame(all_scores).to_csv(OUT/"all_scores.csv",index=False)
    print(f"Saved {OUT}",flush=True)


def predict_at(issue_time):
    """Reconstruct an archived forecast; requires matching cached NWP and observations."""
    artifacts = joblib.load(OUT / "wind_models.joblib")
    _, _, frames = load_frames()
    issue = pd.Timestamp(issue_time)
    if issue.tzinfo is not None:
        issue = issue.tz_convert("UTC").tz_localize(None)
    if issue < TEST_START:
        raise ValueError("Saved prediction intervals use calibration through December 2025; require an issue time from January 2026")
    result = []
    with threadpool_limits(limits=4):
        for h, artifact in artifacts.items():
            frame = frames[h]
            if (list(frame["X"].columns) != artifact["feature_columns"]
                    or frame["obs_columns"] != artifact["obs_columns"]
                    or list(frame["raw"]) != artifact["nwp_models"]
                    or artifact["nwp_age_buffer_hours"] != NWP_AGE_BUFFER_HOURS):
                raise ValueError("Input feature/model schema differs from the saved artifact; retrain or restore the frozen archive")
            valid = issue+pd.Timedelta(hours=h)
            if valid not in frame["X"].index or not np.isfinite(frame["X"].loc[valid,list(frame["raw"].values())]).all():
                raise ValueError(f"Archived input forecasts unavailable at {valid} (+{h}h)")
            if not np.isfinite(frame["persistence"].loc[valid]):
                raise ValueError(f"Observed wind unavailable at issue time {issue}")
            mask = frame["X"].index == valid
            point = float(blend_predict(artifact["models"],artifact["blend"],frame,mask)[0])
            radius = artifact["interval_radius_kmh"]
            result.append({"issue_time_utc":str(issue),"valid_time_utc":str(valid),"horizon_hours":h,
                           "wind_kmh":point,"lower80_kmh":max(0,point-radius),"upper80_kmh":point+radius})
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predict-at",help="Historical issue time (UTC); uses archived inputs, not a live forecast")
    args = parser.parse_args()
    predict_at(args.predict_at) if args.predict_at else train()
