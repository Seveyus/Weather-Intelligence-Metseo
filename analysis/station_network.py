"""Sensor-only multi-variable forecasts learned on the Météo-France station network.

Follows reports/station_network/PROTOCOLE.md. Phases:
    python analysis/station_network.py --phase split      # eligibility + station split (no training)
    python analysis/station_network.py --phase select     # train candidates, choose on validation
    python analysis/station_network.py --phase evaluate   # one scoring of the frozen choice on the test
Station series come from build_station_dataset.py (data/stations).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "stations"
OUT = ROOT / "reports" / "station_network"
CHECKPOINTS = OUT / "checkpoints"
PROTOCOL = OUT / "PROTOCOLE.md"
EPOCH = pd.Timestamp("2000-01-01")

RAW = ("T", "U", "PSTAT", "FF", "DD", "FXI", "RR1")          # column order in the station arrays
REQUIRED = ("T", "U", "FF", "DD", "RR1")
MIN_HOURS = 17_532                                          # two years per required variable
TARGET_STATION = "42218011"
WINDOW = 72
HORIZONS = (1, 2, 3, 6, 12, 24, 48, 72)
TRAIN_END = pd.Timestamp("2022-01-01")
VALIDATION_END = pd.Timestamp("2024-01-01")
TEST_START = pd.Timestamp("2024-01-04")
RAIN_MM = 0.2
# Input channels built from the raw columns, and which presence mask each one uses.
CHANNELS = ("T", "U", "PSTAT", "FF", "FXI", "RR1", "u", "v")
MASKS = ("T", "U", "PSTAT", "FF", "FXI", "RR1")
# Targets per horizon: regression on standardized values, plus the rain probability.
REGRESSION = ("T", "U", "PSTAT", "FF", "FXI", "u", "v")
TARGETS = REGRESSION + ("rain",)
UNITS = {"T": "°C", "U": "%", "PSTAT": "hPa", "FF": "km/h", "FXI": "km/h", "u": "km/h", "v": "km/h", "rain": "Brier"}


def hour_index(stamp):
    return int((pd.Timestamp(stamp) - EPOCH) / pd.Timedelta(hours=1))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------------ split

def make_split():
    stations = pd.read_csv(DATA / "stations.csv", dtype={"station": str})
    eligible = stations.loc[(stations[[f"{v}_present" for v in REQUIRED]] >= MIN_HOURS).all(axis=1)].copy()
    if TARGET_STATION not in set(eligible.station):
        raise ValueError("SAINT-ETIENNE does not meet the eligibility rule")
    rng = np.random.default_rng(42)
    others = eligible.loc[eligible.station != TARGET_STATION, "station"].to_numpy()
    others = others[rng.permutation(len(others))]
    n_val, n_test = round(.10 * len(others)), round(.15 * len(others))
    split = {"validation": sorted(others[:n_val]), "test": sorted([*others[n_val:n_val + n_test], TARGET_STATION]),
             "train": sorted(others[n_val + n_test:])}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "split.json").write_text(json.dumps({
        "rule": f"{', '.join(REQUIRED)} each present >= {MIN_HOURS} h; PSTAT and FXI optional (masked)",
        "seed": 42, "written_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol_sha256": sha256(PROTOCOL), "dataset_readme_sha256": sha256(DATA / "README.json"),
        "counts": {k: len(v) for k, v in split.items()}, "eligible": len(eligible), "all_stations": len(stations),
        "with_pressure": {k: int(eligible.loc[eligible.station.isin(v), "PSTAT_present"].ge(MIN_HOURS).sum())
                          for k, v in split.items()},
        **split}, indent=1) + "\n")
    print({k: len(v) for k, v in split.items()}, flush=True)


# ------------------------------------------------------------------------- data

class Network:
    """All stations of one role concatenated, padded so windows never cross stations."""

    def __init__(self, station_ids, meta, device, compute=None):
        # Preallocate on the target device and fill one station at a time, so the
        # whole network is never held twice in memory (it can reach ~10 GB).
        pad = WINDOW + max(HORIZONS)
        lengths = [int(meta.loc[station, "hours"]) for station in station_ids]
        total = pad + sum(length + pad for length in lengths)
        channels = len(CHANNELS) + 1
        self.values = torch.zeros((total, channels), dtype=torch.float32, device=device)
        self.present = torch.zeros((total, channels), dtype=torch.bool, device=device)
        self.time = torch.full((total,), -10**9, dtype=torch.int32, device=device)
        self.station = torch.full((total,), -1, dtype=torch.int32, device=device)
        position = pad
        for k, (station, length) in enumerate(zip(station_ids, lengths)):
            series = torch.from_numpy(np.load(DATA / "series" / f"{station}.npy"))
            if len(series) != length:
                raise ValueError(f"{station}: series length differs from stations.csv")
            values, present = derive(series)
            block = slice(position, position + length)
            self.values[block], self.present[block] = values.to(device), present.to(device)
            start = hour_index(meta.loc[station, "start"])
            self.time[block] = torch.arange(start, start + length, dtype=torch.int32, device=device)
            self.station[position - pad:position + length] = k
            position += length + pad
        self.static = torch.tensor(meta.loc[list(station_ids), ["lat", "lon", "alt"]].to_numpy(),
                                   dtype=torch.float32, device=device)
        self.ids = list(station_ids)
        self.device = device                 # where the station arrays live
        self.compute = compute or device     # where batches are sent

    def issue_positions(self, start, end):
        """Issue times with every target horizon inside [start, end) and some recent data."""
        t = self.time
        ok = (t >= hour_index(start)) & (t.long() + max(HORIZONS) < hour_index(end))
        ok &= self.present[:, :, ].any(dim=1)  # at least one sensor at the issue hour
        return torch.nonzero(ok).squeeze(1)

    def batch(self, positions):
        offsets = torch.arange(-WINDOW + 1, 1, device=positions.device)
        window = positions[:, None] + offsets[None, :]
        x = self.values[window]
        m = self.present[window]
        future = positions[:, None] + torch.tensor(HORIZONS, device=positions.device)[None, :]
        y = self.values[future]
        ym = self.present[future]
        hours = self.time[positions].double()
        hour = (hours % 24) / 24 * 2 * math.pi
        day = ((hours / 24) % 365.25) / 365.25 * 2 * math.pi
        calendar = torch.stack([hour.sin(), hour.cos(), day.sin(), day.cos()], dim=1).float()
        static = self.static[self.station[positions].long()]
        return tuple(t.to(self.compute, non_blocking=True) for t in (x, m, calendar, static, y, ym))


def derive(raw):
    """Channels used by the models from raw station columns; masks mark real observations."""
    col = {name: raw[:, i] for i, name in enumerate(RAW)}
    angle = torch.deg2rad(col["DD"])
    u, v = -col["FF"] * torch.sin(angle), -col["FF"] * torch.cos(angle)
    rain = (col["RR1"] >= RAIN_MM).float().where(col["RR1"].isfinite(), torch.tensor(float("nan")))
    channels = torch.stack([col["T"], col["U"], col["PSTAT"], col["FF"], col["FXI"], col["RR1"], u, v, rain], dim=1)
    present = channels.isfinite()
    present[:, 6] &= col["DD"].isfinite()
    present[:, 7] &= col["DD"].isfinite()
    return torch.nan_to_num(channels), present


# Indices into the derived channel tensor (last column = rain occurrence target).
CHANNEL_INDEX = {name: i for i, name in enumerate(CHANNELS + ("rain",))}


class Normalizer:
    def __init__(self, network, positions, sample=2_000_000):
        idx = positions[torch.randperm(len(positions), device=positions.device)[:sample]]
        v, p = network.values[idx], network.present[idx]
        self.mean = torch.stack([v[p[:, i], i].mean() for i in range(v.shape[1])])
        self.std = torch.stack([v[p[:, i], i].std() for i in range(v.shape[1])]).clamp_min(1e-3)
        self.mean[-1], self.std[-1] = 0, 1  # rain occurrence stays a 0/1 label
        s = network.static
        self.static_mean, self.static_std = s.mean(0), s.std(0).clamp_min(1e-3)

    def on(self, device):
        copy = Normalizer.__new__(Normalizer)
        return copy.load(self.state(), device)

    def state(self):
        return {k: getattr(self, k).cpu() for k in ("mean", "std", "static_mean", "static_std")}

    def load(self, state, device):
        for k, value in state.items():
            setattr(self, k, value.to(device))
        return self


def model_inputs(batch, norm):
    x, m, calendar, static, _, _ = batch
    channels = [CHANNEL_INDEX[c] for c in CHANNELS]
    masks = [CHANNEL_INDEX[c] for c in MASKS]
    xs = (x[..., channels] - norm.mean[channels]) / norm.std[channels]
    xs = xs * m[..., channels]
    sequence = torch.cat([xs, m[..., masks].float()], dim=2)                   # (B, 72, 14)
    context = torch.cat([calendar, (static - norm.static_mean) / norm.static_std], dim=1)  # (B, 7)
    return sequence, context


def model_targets(batch, norm):
    _, _, _, _, y, ym = batch
    idx = [CHANNEL_INDEX[c] for c in REGRESSION]
    reg = (y[..., idx] - norm.mean[idx]) / norm.std[idx]
    return reg, ym[..., idx], y[..., -1], ym[..., -1]


# ------------------------------------------------------------------------ models

def handcrafted(sequence):
    """Compact summary of the window (lags, means, tendencies) for the MLP candidate."""
    lags = sequence[:, [-1, -2, -3, -4, -7, -13, -25, -49], :8].flatten(1)
    means = torch.cat([sequence[:, -k:, :8].mean(1) for k in (6, 24, 72)], 1)
    diffs = torch.cat([sequence[:, -1, :8] - sequence[:, -1 - k, :8] for k in (3, 12, 24)], 1)
    masks = sequence[:, -1, 8:]
    return torch.cat([lags, means, diffs, masks], 1)


class MultiHead(nn.Module):
    """Shared encoder, one output head per target variable (each predicts all horizons)."""

    def __init__(self, encoder, width):
        super().__init__()
        self.encoder = encoder
        self.heads = nn.ModuleDict({name: nn.Sequential(nn.Linear(width, width // 2), nn.GELU(),
                                                        nn.Linear(width // 2, len(HORIZONS))) for name in TARGETS})

    def forward(self, sequence, context):
        z = self.encoder(sequence, context)
        out = torch.stack([self.heads[name](z) for name in TARGETS], dim=2)  # (B, horizons, targets)
        return out


class MLPEncoder(nn.Module):
    def __init__(self, width=512):
        super().__init__()
        size = 8 * 8 + 8 * 3 + 8 * 3 + 6 + 7
        self.net = nn.Sequential(nn.Linear(size, width), nn.GELU(), nn.LayerNorm(width), nn.Dropout(.1),
                                 nn.Linear(width, width), nn.GELU(), nn.LayerNorm(width))

    def forward(self, sequence, context):
        return self.net(torch.cat([handcrafted(sequence), context], 1))


class GRUEncoder(nn.Module):
    def __init__(self, width=256):
        super().__init__()
        self.gru = nn.GRU(14, width, num_layers=2, batch_first=True, dropout=.1)
        self.mix = nn.Sequential(nn.Linear(width + 7, width), nn.GELU(), nn.LayerNorm(width))

    def forward(self, sequence, context):
        _, h = self.gru(sequence)
        return self.mix(torch.cat([h[-1], context], 1))


class TCNEncoder(nn.Module):
    def __init__(self, width=192):
        super().__init__()
        layers, channels = [], 14
        for dilation in (1, 2, 4, 8, 16, 32):
            layers += [nn.Conv1d(channels, width, 3, padding=2 * dilation, dilation=dilation), nn.GELU()]
            channels = width
        self.conv = nn.ModuleList(layers)
        self.mix = nn.Sequential(nn.Linear(width + 7, width), nn.GELU(), nn.LayerNorm(width))

    def forward(self, sequence, context):
        h = sequence.transpose(1, 2)
        for layer in self.conv:
            h = layer(h)
            if isinstance(layer, nn.Conv1d):
                h = h[..., :sequence.shape[1]]  # causal crop
        return self.mix(torch.cat([h[..., -1], context], 1))


ARCHITECTURES = {"MLP": (MLPEncoder, 512), "GRU": (GRUEncoder, 256), "TCN": (TCNEncoder, 192)}


def build(name):
    encoder, width = ARCHITECTURES[name]
    return MultiHead(encoder(width), width)


def loss_fn(out, batch, norm):
    reg, reg_mask, rain, rain_mask = model_targets(batch, norm)
    huber = nn.functional.huber_loss(out[..., :len(REGRESSION)], reg, reduction="none", delta=1.0)
    per_target = [(huber[..., i] * reg_mask[..., i]).sum() / reg_mask[..., i].sum().clamp_min(1)
                  for i in range(len(REGRESSION))]
    bce = nn.functional.binary_cross_entropy_with_logits(out[..., -1], rain, reduction="none")
    per_target.append((bce * rain_mask).sum() / rain_mask.sum().clamp_min(1))
    return torch.stack(per_target).mean()


# -------------------------------------------------------------------- scoring

def predict(model, network, positions, norm, batch_size=8192):
    model.eval()
    outs = []
    compute = network.compute
    with torch.no_grad(), torch.autocast(compute.type, enabled=compute.type == "cuda"):
        for chunk in positions.split(batch_size):
            sequence, context = model_inputs(network.batch(chunk), norm)
            outs.append(model(sequence, context).float().to(network.device))
    out = torch.cat(outs)
    norm = norm.on(network.device)
    idx = [CHANNEL_INDEX[c] for c in REGRESSION]
    physical = out[..., :len(REGRESSION)] * norm.std[idx] + norm.mean[idx]
    return torch.cat([physical, torch.sigmoid(out[..., -1:])], dim=2)  # (N, horizons, targets)


def references(network, positions):
    """Persistence and, up to +24 h, the value observed 24 h before the target."""
    idx = [CHANNEL_INDEX[c] for c in REGRESSION] + [CHANNEL_INDEX["rain"]]
    now = network.values[positions][:, idx]
    present = network.present[positions][:, idx]
    persistence = now[:, None, :].expand(-1, len(HORIZONS), -1).clone()
    persistence_ok = present[:, None, :].expand(-1, len(HORIZONS), -1)
    diurnal = torch.full_like(persistence, float("nan"))
    for k, h in enumerate(HORIZONS):
        if h <= 24:
            back = positions + h - 24
            diurnal[:, k] = network.values[back][:, idx].where(network.present[back][:, idx],
                                                               torch.tensor(float("nan"), device=positions.device))
    persistence = persistence.where(persistence_ok, torch.tensor(float("nan"), device=positions.device))
    return {"Persistance": persistence, "Cycle diurne (J-1)": diurnal}


def observed(network, positions):
    idx = [CHANNEL_INDEX[c] for c in REGRESSION] + [CHANNEL_INDEX["rain"]]
    future = positions[:, None] + torch.tensor(HORIZONS, device=positions.device)[None, :]
    y = network.values[future][..., idx]
    return y.where(network.present[future][..., idx], torch.tensor(float("nan"), device=positions.device))


def metrics(pred, y):
    """Per (target, horizon): MAE/RMSE/bias (Brier for rain) on hours where both exist."""
    rows = []
    for j, name in enumerate(TARGETS):
        for k, h in enumerate(HORIZONS):
            p, o = pred[:, k, j], y[:, k, j]
            ok = p.isfinite() & o.isfinite()
            if ok.sum() == 0:
                continue
            e = (p[ok] - o[ok]).double()
            row = {"target": name, "horizon_hours": h, "n": int(ok.sum())}
            if name == "rain":
                row["brier"] = float((e ** 2).mean())
            else:
                row.update(mae=float(e.abs().mean()), rmse=float((e ** 2).mean().sqrt()), bias=float(e.mean()))
            rows.append(row)
    return rows


def headline(row):
    return row["brier"] if row["target"] == "rain" else row["mae"]


def skill_score(model_rows, reference_rows):
    """Mean ratio model / persistence over targets and horizons (lower is better)."""
    ref = {(r["target"], r["horizon_hours"]): headline(r) for r in reference_rows}
    ratios = [headline(r) / ref[(r["target"], r["horizon_hours"])] for r in model_rows
              if (r["target"], r["horizon_hours"]) in ref and ref[(r["target"], r["horizon_hours"])] > 0]
    return float(np.mean(ratios))


# ------------------------------------------------------------------- training

def train_candidate(name, seed, train, train_pos, val, val_pos, norm, args):
    torch.manual_seed(seed)
    device = train.compute
    model = build(name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    val_y = observed(val, val_pos)
    reference = metrics(references(val, val_pos)["Persistance"], val_y)
    best, history, stale = math.inf, [], 0
    steps = args.samples_per_epoch // args.batch_size
    for epoch in range(1, args.epochs + 1):
        model.train()
        started, total = time.time(), 0.0
        for _ in range(steps):
            positions = train_pos[torch.randint(len(train_pos), (args.batch_size,), device=train.device)]
            batch = train.batch(positions)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                out = model(*model_inputs(batch, norm))
            loss = loss_fn(out.float(), batch, norm)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += loss.detach().item()
        score = skill_score(metrics(predict(model, val, val_pos, norm), val_y), reference)
        history.append({"epoch": epoch, "train_loss": total / steps, "validation_score": score,
                        "seconds": time.time() - started})
        print(f"{name} seed {seed} epoch {epoch}: loss {total / steps:.4f} validation {score:.4f} "
              f"({time.time() - started:.0f} s)", flush=True)
        if score < best - 1e-4:
            best, stale = score, 0
            torch.save({"architecture": name, "seed": seed, "state": model.state_dict(),
                        "normalizer": norm.state(), "epoch": epoch}, CHECKPOINTS / f"{name}_{seed}.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break
    return {"architecture": name, "seed": seed, "best_validation_score": best, "history": history}


def load_split():
    split = json.loads((OUT / "split.json").read_text())
    meta = pd.read_csv(DATA / "stations.csv", dtype={"station": str}).set_index("station")
    return split, meta


def select(args):
    device, data_device = devices()
    split, meta = load_split()
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    train = Network(split["train"][:args.max_stations], meta, data_device, device)
    val = Network(split["validation"], meta, data_device, device)
    train_pos = train.issue_positions(EPOCH, TRAIN_END)
    val_pos = val.issue_positions(TRAIN_END, VALIDATION_END)
    # Fixed validation sample, identical for every candidate.
    generator = torch.Generator(device="cpu").manual_seed(0)
    val_pos = val_pos[torch.randperm(len(val_pos), generator=generator)[:args.validation_samples].to(data_device)]
    norm = Normalizer(train, train_pos).on(device)
    print(f"Train issue times {len(train_pos):,} on {len(train.ids)} stations; validation sample {len(val_pos):,} "
          f"on {len(val.ids)} stations; device {device}", flush=True)
    runs = [train_candidate(name, seed, train, train_pos, val, val_pos, norm, args)
            for name in args.architectures for seed in (11, 23)]
    by_architecture = {}
    for run in runs:
        by_architecture.setdefault(run["architecture"], []).append(run["best_validation_score"])
    # The ensemble of both seeds is what gets chosen and evaluated.
    val_y = observed(val, val_pos)
    reference = metrics(references(val, val_pos)["Persistance"], val_y)
    ensemble_scores = {}
    for name in by_architecture:
        preds = [predict(load_model(name, seed, device)[0], val, val_pos, norm) for seed in (11, 23)]
        ensemble_scores[name] = skill_score(metrics(torch.stack(preds).mean(0), val_y), reference)
    chosen = min(ensemble_scores, key=ensemble_scores.get)
    (OUT / "selection.json").write_text(json.dumps({
        "protocol_sha256": sha256(PROTOCOL), "script_sha256": sha256(Path(__file__)),
        "written_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selected_architecture": chosen, "ensemble_validation_scores": ensemble_scores,
        "single_run_scores": by_architecture, "runs": runs, "train_issue_times": int(len(train_pos)),
        "validation_sample": int(len(val_pos)), "settings": vars(args), "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }, indent=1) + "\n")
    print(f"Selected {chosen}: {ensemble_scores}", flush=True)


def devices():
    """Compute on the GPU; keep station arrays there too only if it has room (>= 12 GB)."""
    if not torch.cuda.is_available():
        return torch.device("cpu"), torch.device("cpu")
    # With an external GPU next to the laptop one, use the card with the most memory.
    index = max(range(torch.cuda.device_count()), key=lambda i: torch.cuda.get_device_properties(i).total_memory)
    torch.cuda.set_device(index)
    gpu = torch.device("cuda", index)
    big = torch.cuda.get_device_properties(index).total_memory >= 12 * 1024 ** 3
    return gpu, gpu if big else torch.device("cpu")


def load_model(name, seed, device):
    state = torch.load(CHECKPOINTS / f"{name}_{seed}.pt", map_location=device)
    model = build(name).to(device)
    model.load_state_dict(state["state"])
    return model, state


def evaluate(args):
    selection = json.loads((OUT / "selection.json").read_text())
    if selection["protocol_sha256"] != sha256(PROTOCOL) or selection["script_sha256"] != sha256(Path(__file__)):
        raise RuntimeError("Protocol or script changed after selection")
    device, data_device = devices()
    split, meta = load_split()
    name = selection["selected_architecture"]
    models = [load_model(name, seed, device) for seed in (11, 23)]
    norm = Normalizer.__new__(Normalizer).load(models[0][1]["normalizer"], device)
    results = {}
    for role, ids in (("test_stations", split["test"]), ("saint_etienne", [TARGET_STATION]),
                      ("train_stations", split["train"][:args.max_stations])):
        network = Network(ids, meta, data_device, device)
        positions = network.issue_positions(TEST_START, pd.Timestamp("2100-01-01"))
        y = observed(network, positions)
        preds = {f"Réseau {name} (2 graines)": torch.stack([predict(m, network, positions, norm) for m, _ in models]).mean(0),
                 **references(network, positions)}
        results[role] = {"stations": len(ids), "issue_times": int(len(positions)),
                         "scores": {method: metrics(pred, y) for method, pred in preds.items()}}
        if role == "saint_etienne":
            frame = pd.DataFrame({"issue_time": [str(EPOCH + pd.Timedelta(hours=int(t))) for t in network.time[positions].cpu()]})
            pred = preds[f"Réseau {name} (2 graines)"].cpu().numpy()
            obs = y.cpu().numpy()
            for j, target in enumerate(TARGETS):
                for k, h in enumerate(HORIZONS):
                    if h in (1, 3, 6, 12, 24, 72):
                        frame[f"{target}_{h}h_pred"] = pred[:, k, j]
                        frame[f"{target}_{h}h_obs"] = obs[:, k, j]
            frame.to_csv(OUT / "predictions_saint_etienne.csv", index=False, float_format="%.3f")
        print(role, len(ids), "stations evaluated", flush=True)
    (OUT / "metrics.json").write_text(json.dumps({
        "protocol_sha256": selection["protocol_sha256"], "selected_architecture": name,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "test_period": f"issue times from {TEST_START}", "results": results,
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
    }, indent=1, ensure_ascii=False) + "\n")
    print(f"Saved {OUT / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("split", "select", "evaluate"), required=True)
    parser.add_argument("--architectures", nargs="+", default=list(ARCHITECTURES), choices=list(ARCHITECTURES))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--samples-per-epoch", type=int, default=8_000_000)
    parser.add_argument("--validation-samples", type=int, default=500_000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-stations", type=int, default=None, help="Debug only: limit training stations")
    args = parser.parse_args()
    {"split": lambda: make_split(), "select": lambda: select(args), "evaluate": lambda: evaluate(args)}[args.phase]()
