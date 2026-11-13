"""Pooled gradient-boosted trees for the station-network benchmark (CPU).

Same stations, periods, validation sample and test hours as station_network.py,
so trees and networks are compared on identical cases (PROTOCOLE.md, method 2).
One HistGradientBoosting model per target and lead time, fitted on 1 million (protocol amendment: 3 million initially)
issue times drawn from the training stations; the number of iterations is chosen
on the validation sample, with no refit.

    python analysis/station_trees.py --phase select
    python analysis/station_trees.py --phase evaluate
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import station_network as net  # noqa: E402

TRAIN_SAMPLES = 1_000_000  # protocol amendment of 2026-11-12: 3 M did not fit in RAM and time
STAGES = (100, 200, 400)
GROUP = 60  # stations loaded at once on the CPU, to bound memory
MODELS = net.CHECKPOINTS / "trees.joblib"
SELECTION = net.OUT / "trees_selection.json"
CPU = torch.device("cpu")


def features(network, positions):
    """Physical-unit summary of the 72 h window; NaN where a sensor was missing."""
    x, m, calendar, static, _, _ = network.batch(positions)
    x = x[..., :8].masked_fill(~m[..., :8], float("nan"))
    lags = x[:, [-1, -2, -3, -4, -7, -13, -25, -49]].flatten(1)
    means = torch.cat([x[:, -k:].nanmean(1) for k in (6, 24, 72)], 1)
    diffs = torch.cat([x[:, -1] - x[:, -1 - k] for k in (1, 3, 12, 24)], 1)
    rain = torch.cat([x[:, -k:, net.CHANNEL_INDEX["RR1"]].nansum(1, keepdim=True) for k in (6, 24)], 1)
    return torch.cat([lags, means, diffs, rain, calendar, static], 1).numpy().astype(np.float32)


def targets(network, positions):
    return net.observed(network, positions).numpy()  # (N, horizons, targets), NaN where absent


def groups(ids):
    return [ids[i:i + GROUP] for i in range(0, len(ids), GROUP)]


def training_sample(split, meta):
    """TRAIN_SAMPLES issue times, drawn uniformly over all training-station issue times."""
    rng = np.random.default_rng(42)
    counts, chunks = [], groups(split["train"])
    for ids in chunks:
        network = net.Network(ids, meta, CPU)
        counts.append(len(network.issue_positions(net.EPOCH, net.TRAIN_END)))
    total = sum(counts)
    X, Y = [], []
    for ids, count in zip(chunks, counts):
        network = net.Network(ids, meta, CPU)
        positions = network.issue_positions(net.EPOCH, net.TRAIN_END)
        take = round(TRAIN_SAMPLES * count / total)
        chosen = positions[torch.from_numpy(np.sort(rng.choice(len(positions), take, replace=False)))]
        X.append(features(network, chosen))
        Y.append(targets(network, chosen))
        print(f"  sampled {take:,} issue times from {len(ids)} stations", flush=True)
    return np.concatenate(X), np.concatenate(Y), total


def validation_sample(split, meta):
    """Exactly the validation positions used by station_network.select()."""
    network = net.Network(split["validation"], meta, CPU)
    positions = network.issue_positions(net.TRAIN_END, net.VALIDATION_END)
    generator = torch.Generator(device="cpu").manual_seed(0)
    positions = positions[torch.randperm(len(positions), generator=generator)[:500_000]]
    return network, positions


def tree():
    return HistGradientBoostingRegressor(max_iter=max(STAGES), max_leaf_nodes=63, learning_rate=.08,
                                         l2_regularization=10, min_samples_leaf=200,
                                         early_stopping=False, random_state=42)


def select(threads):
    split, meta = net.load_split()
    started = time.time()
    X, Y, available = training_sample(split, meta)
    val, val_pos = validation_sample(split, meta)
    Xv, Yv = features(val, val_pos), targets(val, val_pos)
    print(f"Train sample {len(X):,} x {X.shape[1]} features ({available:,} available); "
          f"validation {len(Xv):,} ({time.time() - started:.0f} s)", flush=True)
    models, chosen, scores = {}, {}, {}
    with threadpool_limits(limits=threads):
        for j, target in enumerate(net.TARGETS):
            for k, horizon in enumerate(net.HORIZONS):
                rows = np.isfinite(Y[:, k, j])
                start = time.time()
                model = tree().fit(X[rows], Y[rows, k, j])
                ok = np.isfinite(Yv[:, k, j])
                errors = {}
                for stage, pred in enumerate(model.staged_predict(Xv[ok]), 1):
                    if stage in STAGES:
                        pred = np.clip(pred, 0, 1) if target == "rain" else pred
                        e = pred - Yv[ok, k, j]
                        errors[stage] = float(np.mean(e ** 2)) if target == "rain" else float(np.abs(e).mean())
                best = min(errors, key=errors.get)
                models[(target, horizon)], chosen[f"{target}_{horizon}"] = model, best
                scores[f"{target}_{horizon}"] = errors
                print(f"{target} +{horizon}h: {rows.sum():,} rows, {best} it, validation {errors[best]:.4f} "
                      f"({time.time() - start:.0f} s)", flush=True)
    net.CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "iterations": chosen}, MODELS, compress=3)
    # Validation score in the same units as the networks' selection score.
    pred = predict({"models": models, "iterations": chosen}, Xv)
    y = torch.from_numpy(Yv)
    reference = net.metrics(net.references(val, val_pos)["Persistance"], y)
    skill = net.skill_score(net.metrics(torch.from_numpy(pred), y), reference)
    SELECTION.write_text(json.dumps({
        "protocol_sha256": net.sha256(net.PROTOCOL), "script_sha256": net.sha256(Path(__file__)),
        "written_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "validation_score": skill, "train_samples": int(len(X)), "train_issue_times_available": int(available),
        "iterations": chosen, "validation_errors": scores, "features": int(X.shape[1]),
    }, indent=1) + "\n")
    print(f"Trees validation score {skill:.4f} (1 = persistence)", flush=True)


def predict(bundle, X):
    out = np.full((len(X), len(net.HORIZONS), len(net.TARGETS)), np.nan, np.float32)
    for j, target in enumerate(net.TARGETS):
        for k, horizon in enumerate(net.HORIZONS):
            model, stop = bundle["models"][(target, horizon)], bundle["iterations"][f"{target}_{horizon}"]
            for stage, pred in enumerate(model.staged_predict(X), 1):
                if stage == stop:
                    break
            out[:, k, j] = np.clip(pred, 0, 1) if target == "rain" else pred
    return out


def group_features(network, positions, rows=200_000):
    """features() in slices, so the 72 h windows of a whole station group never sit in memory at once."""
    return np.concatenate([features(network, chunk) for chunk in positions.split(rows)])


def evaluate(threads):
    selection = json.loads(SELECTION.read_text())
    selected_commit = net.check_frozen(selection, Path(__file__))
    split, meta = net.load_split()
    bundle = joblib.load(MODELS)
    results = {}
    with threadpool_limits(limits=threads):
        for role, ids in (("test_stations", split["test"]), ("saint_etienne", [net.TARGET_STATION]),
                          ("train_stations", split["train"])):
            sums, issues = {}, 0
            for chunk in groups(ids):
                network = net.Network(chunk, meta, CPU)
                positions = network.issue_positions(net.TEST_START, pd.Timestamp("2100-01-01"))
                pred = torch.from_numpy(predict(bundle, group_features(network, positions)))
                net.accumulate(sums, pred, net.observed(network, positions))
                issues += len(positions)
                del network, positions, pred
                print(f"  {role}: {issues:,} issue times scored", flush=True)
            results[role] = {"stations": len(ids), "issue_times": issues, "scores": {"Arbres groupés": net.finish(sums)}}
            # Saved after each role: the training stations alone take hours on the CPU.
            (net.OUT / "trees_metrics.json").write_text(json.dumps({
                "protocol_sha256": selection["protocol_sha256"], "selected_script_commit": selected_commit,
                "evaluation_script_sha256": net.sha256(Path(__file__)),
                "evaluated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "test_period": f"issue times from {net.TEST_START}", "complete": role == "train_stations",
                "results": results,
            }, indent=1, ensure_ascii=False) + "\n")
            print(role, len(ids), "stations evaluated and saved", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("select", "evaluate"), required=True)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    select(args.threads) if args.phase == "select" else evaluate(args.threads)
