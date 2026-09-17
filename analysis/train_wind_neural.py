"""Train, compare and replay genuine GRU/LSTM wind forecasts.

No tree predictions enter the network. Selection uses 2025 only; 2026 is a
previously viewed retrospective comparison on the exact frozen baseline cases.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import platform

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from neural_models import HybridWindRNN, WindRNNConfig
from neural_data import build_dataset, fit_preprocessors, transform
from train_wind_advanced import load_frames, interval_radius, paired_skill
from wind_horizon_study import score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "wind_neural"
BASE = ROOT / "reports" / "wind_advanced"
SEEDS = (11, 23)
MAX_EPOCHS = 32
PATIENCE = 6
BATCH_SIZE = 256
CANDIDATES = {
    "bigru": {"cell": "gru", "attention": False, "neutralize_history": False},
    "bilstm": {"cell": "lstm", "attention": False, "neutralize_history": False},
    "bigru_attention": {"cell": "gru", "attention": True, "neutralize_history": False},
    "history_ablation": {"cell": "gru", "attention": False, "neutralize_history": True},
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def device():
    torch.set_num_threads(4)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_data():
    obs, audit, frames = load_frames()
    raw = build_dataset(obs, frames, history_hours=72, comparison_directory=BASE)
    return raw, obs, audit


def loader(batch, shuffle=False, seed=11):
    tensors = [torch.from_numpy(np.asarray(x, dtype=np.float32)) for x in
        (batch.history, batch.context, batch.nwp_mean, batch.targets, batch.label_mask & batch.input_mask)]
    return DataLoader(TensorDataset(*tensors), batch_size=BATCH_SIZE, shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed), num_workers=0,
        pin_memory=torch.cuda.is_available())


def forward(model, history, context, baseline, neutralize_history):
    if neutralize_history:
        history = torch.zeros_like(history)
    return model(history, context, baseline)


def predictions(model, data_loader, run_device, neutralize_history=False):
    model.eval()
    results = []
    with torch.no_grad():
        for history, context, baseline, _, _ in data_loader:
            pred = forward(model, history.to(run_device), context.to(run_device),
                           baseline.to(run_device), neutralize_history)
            results.append(pred.cpu().numpy())
    return np.concatenate(results)


def horizon_scores(batch, pred):
    valid = batch.label_mask & batch.input_mask
    return {str(h): score(batch.targets[valid[:,i].astype(bool),i],
                         pred[valid[:,i].astype(bool),i])
            for i,h in enumerate((24,48,72))}


def objective(metrics, reference):
    return float(np.mean([.5*(metrics[h]["mae_kmh"]/reference[h]["mae_kmh"]
        +metrics[h]["rmse_kmh"]/reference[h]["rmse_kmh"]) for h in metrics]))


def reference_metrics(obs, batch):
    y = obs.FF.loc[obs.index < pd.Timestamp("2025-01-01")].dropna()
    climate = y.groupby([y.index.month,y.index.hour]).mean()
    pred = np.empty_like(batch.targets)
    for j,h in enumerate((24,48,72)):
        dates = batch.issue_times + pd.Timedelta(hours=h)
        pred[:,j] = [climate.get((t.month,t.hour),y.mean()) for t in dates]
    return horizon_scores(batch,pred)


def model_config(batch, spec):
    return WindRNNConfig(past_features=batch.history.shape[-1],future_features=batch.context.shape[-1],
        hidden_size=48,num_layers=1,bidirectional=True,dropout=.1,
        cell=spec["cell"],attention=spec["attention"])


def experiment_provenance():
    paths = [Path(__file__), ROOT/"analysis/neural_models.py", ROOT/"analysis/neural_data.py",
        ROOT/"analysis/train_wind_advanced.py", ROOT/"analysis/wind_horizon_study.py",
        OUT/"PROTOCOLE.md", BASE/"data/observations_hourly.csv",
        BASE/"data/nwp_archive.csv", BASE/"data/day4/nwp_archive.csv"]
    paths += [ROOT/name for name in ("H_42_previous-2020-2024.csv", "H_42_latest-2025-2026.csv")]
    records = {}
    for path in paths:
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        records[str(path.relative_to(ROOT))] = {"size":path.stat().st_size,"sha256":digest}
    return records


def checkpoint(path, model, cfg, metadata):
    torch.save({"state_dict": {k:v.detach().cpu() for k,v in model.state_dict().items()},
                "config":cfg.to_dict(), **metadata},path)


def restore(path, run_device):
    payload = torch.load(path,map_location=run_device,weights_only=True)
    model = HybridWindRNN(WindRNNConfig(**payload["config"])).to(run_device)
    model.load_state_dict(payload["state_dict"])
    return model,payload


def fit_one(name, seed, batch, valid, reference, fixed_epochs=None):
    set_seed(seed)
    run_device = device()
    spec = CANDIDATES[name]
    cfg = model_config(batch,spec)
    model = HybridWindRNN(cfg).to(run_device)
    opt = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    train_loader = loader(batch,shuffle=True,seed=seed)
    val_loader = loader(valid) if valid is not None else None
    phase = "selection" if fixed_epochs is None else "refit"
    path = OUT/"checkpoints"/f"{phase}_{name}_{seed}.pt"
    best, best_epoch, stale = float("inf"),0,0
    history = []
    started = time.monotonic()
    for epoch in range(1,(fixed_epochs or MAX_EPOCHS)+1):
        model.train()
        loss_sum, count = 0.,0
        for hist, context, baseline, y, mask in train_loader:
            hist,context,baseline,y,mask = [x.to(run_device,non_blocking=True) for x in (hist,context,baseline,y,mask)]
            opt.zero_grad(set_to_none=True)
            pred = forward(model,hist,context,baseline,spec["neutralize_history"])
            error = (pred-y)/5.0
            loss = ((error.square()+.2*error.abs())*mask).sum()/mask.sum().clamp_min(1)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss: {name}, seed {seed}, epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
            loss_sum += loss.item()*len(y)
            count += len(y)
        row = {"epoch":epoch,"train_loss":loss_sum/count}
        if val_loader is not None:
            pred = predictions(model,val_loader,run_device,spec["neutralize_history"])
            metrics = horizon_scores(valid,pred)
            value = objective(metrics,reference)
            row.update({"validation_objective":value,"validation_scores":metrics})
            if value < best-1e-5:
                best,best_epoch,stale=value,epoch,0
                checkpoint(path,model,cfg,{"name":name,"seed":seed,"epoch":epoch,"phase":phase,
                    "neutralize_history":spec["neutralize_history"]})
            else:
                stale += 1
        history.append(row)
        print(f"{phase} {name} seed={seed} epoch={epoch:02d} loss={row['train_loss']:.4f}"
              + (f" val={row['validation_objective']:.4f} best={best_epoch}" if val_loader is not None else ""),flush=True)
        if stale >= PATIENCE:
            break
    elapsed = time.monotonic()-started
    if fixed_epochs is not None:
        best_epoch=fixed_epochs
        checkpoint(path,model,cfg,{"name":name,"seed":seed,"epoch":best_epoch,"phase":phase,
            "neutralize_history":spec["neutralize_history"]})
    stats = {"candidate":name,"seed":seed,"best_epoch":best_epoch,"epochs_run":len(history),
        "duration_seconds":elapsed,"parameters":sum(p.numel() for p in model.parameters()),
        "config":cfg.to_dict(),"device":str(run_device),"history":history,
        "checkpoint":str(path.relative_to(ROOT))}
    del model,opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return stats


def select():
    raw,obs,audit=load_data()
    prep=fit_preprocessors(raw,raw.splits["train"])
    joblib.dump(prep,OUT/"preprocessing.joblib")
    train=transform(raw,prep,raw.splits["train"])
    val=transform(raw,prep,raw.splits["val"])
    reference=reference_metrics(obs,val)
    run_device=device()
    results={}
    metadata={"candidates":CANDIDATES,"seeds":list(SEEDS),"max_epochs":MAX_EPOCHS,"patience":PATIENCE,
        "batch_size":BATCH_SIZE,"history_hours":72,"training_n":len(train.issue_times),
        "validation_n":len(val.issue_times),"validation_reference":reference,"data":raw.metadata,
        "test_status":"Previously viewed retrospective comparison; no new independent test",
        "torch":str(torch.__version__),"numpy":np.__version__,"python":platform.python_version(),
        "pandas":pd.__version__,"joblib":joblib.__version__,"provenance":experiment_provenance(),
        "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    (OUT/"run_config.json").write_text(json.dumps(metadata,indent=2,default=str)+"\n")
    for name in CANDIDATES:
        pred_all, runs=[],[]
        for seed in SEEDS:
            stats=fit_one(name,seed,train,val,reference)
            runs.append(stats)
            (OUT/f"history_{name}_{seed}.json").write_text(json.dumps(stats,indent=2)+"\n")
            model,_=restore(ROOT/stats["checkpoint"],run_device)
            pred_all.append(predictions(model,loader(val),run_device,CANDIDATES[name]["neutralize_history"]))
            del model
        pred=np.mean(pred_all,axis=0)
        metrics=horizon_scores(val,pred)
        results[name]={"runs":[{k:v for k,v in run.items() if k!="history"} for run in runs],
            "validation_scores":metrics,"objective":objective(metrics,reference)}
        print(f"ENSEMBLE {name}: {metrics}, objective={results[name]['objective']:.4f}",flush=True)
        (OUT/"validation_candidates.json").write_text(json.dumps(results,indent=2)+"\n")
    eligible=[name for name in results if not CANDIDATES[name]["neutralize_history"]]
    selected=min(eligible,key=lambda name:results[name]["objective"])
    frozen={"selected":selected,"seeds":list(SEEDS),"candidates":results,
        "selected_epochs":{str(run["seed"]):run["best_epoch"] for run in results[selected]["runs"]},
        "selection_only":"2025 validation; 2026 predictions not computed in this phase"}
    (OUT/"selection.json").write_text(json.dumps(frozen,indent=2)+"\n")
    print("FROZEN NEURAL SELECTION",selected,frozen["selected_epochs"],flush=True)


def refit():
    selection=json.loads((OUT/"selection.json").read_text())
    raw,_,_=load_data()
    prep=fit_preprocessors(raw,raw.splits["refit"])
    joblib.dump(prep,OUT/"refit_preprocessing.joblib")
    train=transform(raw,prep,raw.splits["refit"])
    logs=[]
    for seed in selection["seeds"]:
        logs.append(fit_one(selection["selected"],seed,train,None,None,
            fixed_epochs=selection["selected_epochs"][str(seed)]))
    (OUT/"refit_history.json").write_text(json.dumps(logs,indent=2)+"\n")


def ensemble(batch,selection):
    run_device=device()
    pred=[]
    for seed in selection["seeds"]:
        model,payload=restore(OUT/"checkpoints"/f"refit_{selection['selected']}_{seed}.pt",run_device)
        pred.append(predictions(model,loader(batch),run_device,payload["neutralize_history"]))
        del model
    return np.mean(pred,axis=0)


def evaluate():
    selection=json.loads((OUT/"selection.json").read_text())
    prep=joblib.load(OUT/"refit_preprocessing.joblib")
    raw,obs,_=load_data()
    cal=transform(raw,prep,raw.splits["cal"])
    compare_mask=raw.comparison_mask.any(axis=1)
    test=transform(raw,prep,compare_mask)
    pred_cal=ensemble(cal,selection)
    pred_test=ensemble(test,selection)
    summary={"selected":selection["selected"],"test_status":"Previously viewed retrospective comparison",
        "scores":[],"horizons":{},"seeds":selection["seeds"],"test_uses_tree_inputs":False,
        "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    threshold=float(obs.FF.loc[obs.index<pd.Timestamp("2025-10-01")].quantile(.9))
    for j,h in enumerate((24,48,72)):
        valid_cal=cal.label_mask[:,j].astype(bool)
        radius=interval_radius(cal.targets[valid_cal,j]-pred_cal[valid_cal,j])
        baseline=pd.read_csv(BASE/f"predictions_{h}h.csv",parse_dates=["issue_time","valid_time"])
        pred_series=pd.Series(pred_test[:,j],index=test.issue_times)
        pred=pred_series.reindex(pd.DatetimeIndex(baseline.issue_time)).to_numpy()
        if not np.isfinite(pred).all():
            raise ValueError(f"Missing neural forecasts for frozen +{h}h baseline cases")
        y=baseline.observed_kmh.to_numpy()
        high=y>=threshold
        for name,values in (("Tree baseline",baseline.selected_kmh.to_numpy()),("Neural ensemble",pred),
                            ("Raw NWP",baseline.raw_nwp_kmh.to_numpy())):
            summary["scores"].append({"horizon_hours":h,"method":name,**score(y,values),
                "high_wind_mae_kmh":float(np.abs(values[high]-y[high]).mean()),
                "high_wind_bias_kmh":float((values[high]-y[high]).mean())})
        lower,upper=np.maximum(pred-radius,0),pred+radius
        summary["horizons"][str(h)]={"skill_vs_trees":paired_skill(y,baseline.selected_kmh.to_numpy(),pred),
            "high_wind_threshold_kmh":threshold,"high_wind_n":int(high.sum()),
            "interval_radius_kmh":radius,"interval_coverage":float(((y>=lower)&(y<=upper)).mean()),
            "interval_mean_width_kmh":float((upper-lower).mean()),"test_n":len(y)}
        baseline["neural_kmh"]=pred
        baseline["neural_lower80_kmh"]=lower
        baseline["neural_upper80_kmh"]=upper
        baseline.to_csv(OUT/f"predictions_{h}h.csv",index=False)
        print(f"TEST +{h}h neural {score(y,pred)} tree {score(y,baseline.selected_kmh)}",flush=True)
    (OUT/"metrics.json").write_text(json.dumps(summary,indent=2)+"\n")


def replay(issue):
    issue=pd.Timestamp(issue)
    if issue.tzinfo is not None:
        issue=issue.tz_convert("UTC").tz_localize(None)
    if issue<pd.Timestamp("2026-01-01"):
        raise ValueError("Prediction intervals require issue time after December2025 calibration")
    raw,_,_=load_data()
    mask=raw.issue_times==issue
    if not mask.any() or not raw.input_mask[mask].all():
        raise ValueError("All three archived input forecasts must be available at this issue time")
    prep=joblib.load(OUT/"refit_preprocessing.joblib")
    batch=transform(raw,prep,mask)
    selection=json.loads((OUT/"selection.json").read_text())
    metrics=json.loads((OUT/"metrics.json").read_text())
    pred=ensemble(batch,selection)[0]
    output=[]
    for j,h in enumerate((24,48,72)):
        r=metrics["horizons"][str(h)]["interval_radius_kmh"]
        output.append({"issue_time":str(issue),"valid_time":str(issue+pd.Timedelta(hours=h)),
            "horizon_hours":h,"neural_kmh":float(pred[j]),"lower80_kmh":max(0,float(pred[j]-r)),
            "upper80_kmh":float(pred[j]+r)})
    print(json.dumps(output,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=["select","refit","evaluate","all"],default="all")
    parser.add_argument("--predict-at")
    args=parser.parse_args()
    (OUT/"checkpoints").mkdir(parents=True,exist_ok=True)
    if args.predict_at:
        replay(args.predict_at)
    else:
        if args.phase in ("select","all"):select()
        if args.phase in ("refit","all"):refit()
        if args.phase in ("evaluate","all"):evaluate()
