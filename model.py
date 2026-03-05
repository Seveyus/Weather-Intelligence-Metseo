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

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)

        last = out[:, -1, :]

        out = self.fc(last)

        return out.squeeze(1)

