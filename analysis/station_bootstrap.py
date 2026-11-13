"""Block-bootstrap intervals for the station-network test results (PROTOCOLE.md, "Intervalles").

Re-runs the frozen models on the test stations and SAINT-ETIENNE, sums errors per
(station x calendar month) block, then resamples blocks with replacement. Gives 95 %
intervals for each method's MAE (Brier for rain), for the network-minus-trees
difference, and for the protocol's aggregate score (mean ratio to persistence).

    python analysis/station_bootstrap.py
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import station_network as net  # noqa: E402
import station_trees as trees  # noqa: E402

METHODS = ("Réseau MLP (2 graines)", "Arbres groupés", "Persistance")
RESAMPLES = 2000
BLOCKS = net.OUT / "bootstrap_blocks.npz"


def block_sums(role, ids, meta, models, norm, bundle, device, data_device):
    """Per block: count and summed error, shape (blocks, methods, horizons, targets)."""
    counts, errors, labels = [], [], []
    for start in range(0, len(ids), net.EVAL_GROUP):
        began = time.time()
        network = net.Network(ids[start:start + net.EVAL_GROUP], meta, data_device, device)
        positions = network.issue_positions(net.TEST_START, pd.Timestamp("2100-01-01"))
        y = net.observed(network, positions).cpu().numpy()
        preds = [torch.stack([net.predict(m, network, positions, norm) for m, _ in models]).mean(0).cpu().numpy()]
        network.compute = trees.CPU  # tree features are computed on the CPU
        preds.append(trees.predict(bundle, trees.group_features(network, positions)))
        preds.append(net.references(network, positions)["Persistance"].cpu().numpy())
        station = network.station[positions].cpu().numpy().astype(np.int64) + start
        hours = network.time[positions].cpu().numpy().astype("timedelta64[h]") + np.datetime64(net.EPOCH, "h")
        month = hours.astype("datetime64[M]").astype(np.int64)
        keys, block = np.unique(station * 10_000 + month, return_inverse=True)
        count = np.zeros((len(keys), len(METHODS)) + y.shape[1:])
        error = np.zeros_like(count)
        for i, pred in enumerate(preds):
            e = pred - y
            ok = np.isfinite(e)
            e = np.where(ok, e, 0.0)
            e[..., -1] **= 2              # Brier for rain
            e[..., :-1] = np.abs(e[..., :-1])
            for k in range(y.shape[1]):
                for j in range(y.shape[2]):
                    count[:, i, k, j] = np.bincount(block, ok[:, k, j].astype(np.float64), len(keys))
                    error[:, i, k, j] = np.bincount(block, e[:, k, j], len(keys))
        counts.append(count), errors.append(error), labels.append(keys)
        print(f"  {role}: stations {start}-{start + len(network.ids)}, {len(positions):,} issue times, {len(keys)} blocks ({time.time() - began:.0f} s)", flush=True)
        del network
    return np.concatenate(counts), np.concatenate(errors), np.concatenate(labels)


def summarise(count, error, rng):
    """Point values and 95 % intervals from block resampling."""
    n_blocks = len(count)
    weights = rng.multinomial(n_blocks, np.full(n_blocks, 1 / n_blocks), size=RESAMPLES).astype(np.float64)
    flat_c, flat_e = count.reshape(n_blocks, -1), error.reshape(n_blocks, -1)
    with np.errstate(invalid="ignore", divide="ignore"):
        point = (flat_e.sum(0) / flat_c.sum(0)).reshape(count.shape[1:])
        boot = ((weights @ flat_e) / (weights @ flat_c)).reshape((RESAMPLES,) + count.shape[1:])
    ci = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]
    cells = []
    for k, h in enumerate(net.HORIZONS):
        for j, target in enumerate(net.TARGETS):
            if not np.isfinite(point[0, k, j]):
                continue
            row = {"target": target, "horizon_hours": h, "metric": "brier" if target == "rain" else "mae"}
            for i, method in enumerate(METHODS):
                row[method] = {"value": float(point[i, k, j]), "ci95": ci(boot[:, i, k, j])}
            diff = boot[:, 0, k, j] - boot[:, 1, k, j]
            row["réseau_moins_arbres"] = {"value": float(point[0, k, j] - point[1, k, j]), "ci95": ci(diff),
                                         "share_resamples_network_better": float(np.mean(diff < 0))}
            cells.append(row)
    valid = np.isfinite(point[0])
    score = lambda values, i: np.nanmean((values[..., i, :, :] / values[..., 2, :, :])[..., valid], axis=-1)
    aggregate = {}
    for i, method in enumerate(METHODS[:2]):
        aggregate[method] = {"value": float(score(point, i)), "ci95": ci(score(boot, i))}
    diff = score(boot, 0) - score(boot, 1)
    aggregate["réseau_moins_arbres"] = {"value": float(score(point, 0) - score(point, 1)), "ci95": ci(diff),
                                       "share_resamples_network_better": float(np.mean(diff < 0))}
    no_pressure = valid & (np.array(net.TARGETS) != "PSTAT")[None, :]
    score_np = lambda values, i: np.nanmean((values[..., i, :, :] / values[..., 2, :, :])[..., no_pressure], axis=-1)
    diff = score_np(boot, 0) - score_np(boot, 1)
    aggregate["réseau_moins_arbres_sans_pression"] = {
        "value": float(score_np(point, 0) - score_np(point, 1)), "ci95": ci(diff),
        "share_resamples_network_better": float(np.mean(diff < 0))}
    return {"blocks": n_blocks, "aggregate_score": aggregate, "cells": cells}


def main():
    selection = json.loads((net.OUT / "selection.json").read_text())
    tree_selection = json.loads(trees.SELECTION.read_text())
    net.check_frozen(selection)
    net.check_frozen(tree_selection, Path(trees.__file__))
    device, data_device = net.devices()
    split, meta = net.load_split()
    models = [net.load_model(selection["selected_architecture"], seed, device) for seed in (11, 23)]
    norm = net.Normalizer.__new__(net.Normalizer).load(models[0][1]["normalizer"], device)
    bundle = joblib.load(trees.MODELS)
    rng = np.random.default_rng(42)
    results, arrays = {}, {}
    for role, ids in (("test_stations", split["test"]), ("saint_etienne", [net.TARGET_STATION])):
        count, error, keys = block_sums(role, ids, meta, models, norm, bundle, device, data_device)
        arrays.update({f"{role}_count": count, f"{role}_error": error, f"{role}_keys": keys})
        results[role] = summarise(count, error, rng)
        agg = results[role]["aggregate_score"]
        print(f"{role}: network {agg[METHODS[0]]['value']:.4f} {agg[METHODS[0]]['ci95']}, "
              f"trees {agg[METHODS[1]]['value']:.4f} {agg[METHODS[1]]['ci95']}", flush=True)
    np.savez_compressed(BLOCKS, **arrays)
    (net.OUT / "bootstrap.json").write_text(json.dumps({
        "method": f"{RESAMPLES} resamples of (station x calendar month) blocks, 95 % percentile intervals",
        "aggregate_score": "mean over variables and lead times of method error / persistence error (protocol score)",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "results": results,
    }, indent=1, ensure_ascii=False) + "\n")
    print(f"Saved {net.OUT / 'bootstrap.json'}", flush=True)


if __name__ == "__main__":
    main()
