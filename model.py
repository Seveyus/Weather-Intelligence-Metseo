import torch
import torch.nn as nn

class GRURegressor(nn.Module):
    def __init__(self, n_features, hidden_size=64):
        super().__init__()

        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            batch_first=True
        )

        # Sortie de taille 2 pour (wind_u, wind_v)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out, _ = self.gru(x)
        # On prend le dernier état caché de la séquence
        last = out[:, -1, :]
        out = self.fc(last)
        return out # Shape: [batch, 2]