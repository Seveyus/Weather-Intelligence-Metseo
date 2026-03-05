import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from dataset import build_dataset
from model import GRURegressor

device = "cuda" if torch.cuda.is_available() else "cpu"

X, y, n_features = build_dataset(window=24)

train_size = int(len(X)*0.8)

X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]

train_loader = DataLoader(
TensorDataset(
torch.tensor(X_train, dtype=torch.float32),
torch.tensor(y_train, dtype=torch.float32)
),
batch_size=256,
shuffle=True
)

test_loader = DataLoader(
TensorDataset(
torch.tensor(X_test, dtype=torch.float32),
torch.tensor(y_test, dtype=torch.float32)
),
batch_size=256
)

model = GRURegressor(n_features).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

for epoch in range(10):
    model.train()

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()

        pred = model(xb)

        loss = loss_fn(pred, yb)

        loss.backward()

        optimizer.step()

    print("epoch", epoch, "loss", loss.item())