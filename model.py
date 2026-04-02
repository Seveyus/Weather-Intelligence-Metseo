import torch
import torch.nn as nn


class GRURegressor(nn.Module):
    def __init__(
        self,
        n_features,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        bidirectional=False,
    ):
        super().__init__()

        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.norm = nn.LayerNorm(hidden_size * self.num_directions)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        last = self.norm(last)
        out = self.head(last)
        return out