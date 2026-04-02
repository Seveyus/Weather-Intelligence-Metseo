import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from dataset import build_dataset
from model import GRURegressor

device = "cuda" if torch.cuda.is_available() else "cpu"

# Chargement des données
X, y, n_features = build_dataset(window=24)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
    batch_size=256, shuffle=True
)

test_loader = DataLoader(
    TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)),
    batch_size=256
)

model = GRURegressor(n_features).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

# Amélioration 2 : 50 epochs
epochs = 50

for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    # Optionnel : Calcul de l'erreur sur FF (vitesse) pour monitoring
    if epoch % 5 == 0 or epoch == epochs - 1:
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch:02d} | Loss (u,v): {avg_loss:.4f}")

print("Entraînement terminé.")


model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        preds = model(xb).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(yb.numpy())

# Concaténation
all_preds = np.concatenate(all_preds)
all_targets = np.concatenate(all_targets)

# Calcul de FF pour les prédictions et le réel
ff_pred = np.sqrt(all_preds[:, 0]**2 + all_preds[:, 1]**2)
ff_real = np.sqrt(all_targets[:, 0]**2 + all_targets[:, 1]**2)

# Erreur Absolue Moyenne (MAE)
mae = np.mean(np.abs(ff_pred - ff_real))
print(f"Erreur moyenne sur la vitesse du vent : {mae:.2f} m/s")


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(ff_real[:200], label="Réel", color="blue", alpha=0.7)
plt.plot(ff_pred[:200], label="Prédiction", color="red", linestyle="--")
plt.title("Comparaison Vitesse du Vent : Réel vs Prédiction")
plt.xlabel("Temps (Heures)")
plt.ylabel("Vitesse (FF)")
plt.legend()
plt.show()

torch.save(model.state_dict(), "model_wind_v1.pth")