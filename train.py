import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dataset import build_datasets
from model import GRURegressor


CONFIG = {
    "data_path": "dataset_clean.csv",
    "window": 24,
    "horizon": 1,
    "batch_size": 2048,
    "epochs": 60,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.2,
    "bidirectional": True,
    "grad_clip": 1.0,
    "patience": 10,
    "num_workers": 4,
    "seed": 42,
    "checkpoint_dir": "checkpoints",
    "model_name": "gru_wind_best.pth",
    "preferred_cuda_index": 1,
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_device(preferred_index=1):
    if not torch.cuda.is_available():
        print("--- CUDA indisponible : utilisation du CPU ---")
        return torch.device("cpu")

    gpu_count = torch.cuda.device_count()
    print(f"--- {gpu_count} GPU CUDA détecté(s) ---")
    for i in range(gpu_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    if gpu_count > preferred_index:
        device = torch.device(f"cuda:{preferred_index}")
        print(f"--- Lancement sur l'eGPU : {torch.cuda.get_device_name(device)} (cuda:{preferred_index}) ---")
        return device

    device = torch.device("cuda:0")
    print(f"--- eGPU index {preferred_index} non dispo, fallback sur : {torch.cuda.get_device_name(device)} (cuda:0) ---")
    return device


def inverse_ff_from_scaled_uv(y_scaled, scaler_y):
    uv = scaler_y.inverse_transform(y_scaled)
    ff = np.sqrt(uv[:, 0] ** 2 + uv[:, 1] ** 2)
    return ff


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def evaluate(model, loader, loss_fn, device, use_amp=False):
    model.eval()
    losses = []
    preds_all = []
    targets_all = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                pred = model(xb)
                loss = loss_fn(pred, yb)

            losses.append(loss.item())
            preds_all.append(pred.detach().cpu().numpy())
            targets_all.append(yb.detach().cpu().numpy())

    preds_all = np.concatenate(preds_all, axis=0)
    targets_all = np.concatenate(targets_all, axis=0)

    return float(np.mean(losses)), preds_all, targets_all


def main():
    set_seed(CONFIG["seed"])
    ensure_dir(CONFIG["checkpoint_dir"])

    device = get_device(CONFIG["preferred_cuda_index"])
    use_amp = device.type == "cuda"

    data = build_datasets(
        path=CONFIG["data_path"],
        window=CONFIG["window"],
        horizon=CONFIG["horizon"],
    )

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    n_features = data["n_features"]
    scaler_y = data["scaler_y"]

    print("Train:", X_train.shape, y_train.shape)
    print("Val  :", X_val.shape, y_val.shape)
    print("Test :", X_test.shape, y_test.shape)
    print("n_features:", n_features)

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        ),
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=(device.type == "cuda"),
        persistent_workers=(CONFIG["num_workers"] > 0),
    )

    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        ),
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=(device.type == "cuda"),
        persistent_workers=(CONFIG["num_workers"] > 0),
    )

    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32),
        ),
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=(device.type == "cuda"),
        persistent_workers=(CONFIG["num_workers"] > 0),
    )

    model = GRURegressor(
        n_features=n_features,
        hidden_size=CONFIG["hidden_size"],
        num_layers=CONFIG["num_layers"],
        dropout=CONFIG["dropout"],
        bidirectional=CONFIG["bidirectional"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
    )

    loss_fn = nn.MSELoss()
    scaler_amp = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_loss = float("inf")
    patience_count = 0

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    ckpt_path = os.path.join(CONFIG["checkpoint_dir"], CONFIG["model_name"])

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                pred = model(xb)
                loss = loss_fn(pred, yb)

            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            scaler_amp.step(optimizer)
            scaler_amp.update()

            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss, _, _ = evaluate(model, val_loader, loss_fn, device, use_amp=use_amp)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | lr={lr_now:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_count += 1
            if patience_count >= CONFIG["patience"]:
                print("Early stopping.")
                break

    print(f"Best model saved to: {ckpt_path}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    test_loss, preds_scaled, targets_scaled = evaluate(
        model, test_loader, loss_fn, device, use_amp=use_amp
    )

    ff_pred = inverse_ff_from_scaled_uv(preds_scaled, scaler_y)
    ff_real = inverse_ff_from_scaled_uv(targets_scaled, scaler_y)

    test_mae = mae(ff_real, ff_pred)
    test_rmse = rmse(ff_real, ff_pred)

    print("\n===== TEST METRICS =====")
    print(f"Test loss (scaled u,v MSE): {test_loss:.4f}")
    print(f"Test MAE FF (km/h): {test_mae:.4f}")
    print(f"Test RMSE FF (km/h): {test_rmse:.4f}")

    ff_baseline = np.roll(ff_real, 1)
    ff_baseline = ff_baseline[1:]
    ff_real_baseline = ff_real[1:]

    baseline_mae = mae(ff_real_baseline, ff_baseline)
    baseline_rmse = rmse(ff_real_baseline, ff_baseline)

    print("\n===== BASELINE =====")
    print(f"Persistence MAE FF (km/h): {baseline_mae:.4f}")
    print(f"Persistence RMSE FF (km/h): {baseline_rmse:.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(history["train_loss"], label="Train loss")
    plt.plot(history["val_loss"], label="Val loss")
    plt.title("Train / Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["checkpoint_dir"], "loss_curve.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(14, 6))
    plt.plot(ff_real[:300], label="Réel")
    plt.plot(ff_pred[:300], label="Prédiction", linestyle="--")
    plt.title("Comparaison vitesse du vent (FF)")
    plt.xlabel("Temps (heures)")
    plt.ylabel("Vitesse du vent (km/h)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["checkpoint_dir"], "pred_vs_real.png"), dpi=150)
    plt.close()

    with open(os.path.join(CONFIG["checkpoint_dir"], "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_val_loss": best_val_loss,
                "test_loss_scaled_uv": test_loss,
                "test_mae_ff_kmh": test_mae,
                "test_rmse_ff_kmh": test_rmse,
                "baseline_mae_ff_kmh": baseline_mae,
                "baseline_rmse_ff_kmh": baseline_rmse,
                "config": CONFIG,
                "device": str(device),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\nArtifacts saved in checkpoints/")


if __name__ == "__main__":
    main()