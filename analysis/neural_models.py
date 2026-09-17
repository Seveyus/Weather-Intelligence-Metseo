"""Compact recurrent networks for local, multi-horizon wind correction.

The history contains observations available at the forecast issue time only.
``future`` contains numerical weather predictions already available at that
same issue time, plus deterministic calendar features; it must never contain
future station observations. Bidirectionality is confined to the past window.

The model returns speed in km/h, with one direct prediction per configured
horizon. It does not recursively feed predictions back as observations.
"""

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class WindRNNConfig:
    past_features: int
    future_features: int
    horizons: tuple[int, ...] = (24, 48, 72)
    cell: str = "gru"
    hidden_size: int = 64
    num_layers: int = 1
    bidirectional: bool = True
    attention: bool = False
    dropout: float = 0.1
    residual_scale: float = 5.0
    horizon_embedding_size: int = 8
    minimum_baseline: float = 0.05

    def __post_init__(self) -> None:
        # JSON checkpoints turn tuples into lists; canonicalize on restore.
        object.__setattr__(self, "horizons", tuple(self.horizons))
        object.__setattr__(self, "cell", self.cell.lower())
        if self.cell not in ("gru", "lstm"):
            raise ValueError("cell must be 'gru' or 'lstm'")
        for name in (
            "past_features", "future_features", "hidden_size", "num_layers",
            "horizon_embedding_size",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not self.horizons or any(h <= 0 for h in self.horizons):
            raise ValueError("horizons must contain positive forecast leads")
        if len(set(self.horizons)) != len(self.horizons):
            raise ValueError("horizons must be unique")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if any(
            not math.isfinite(value) or value <= 0
            for value in (self.residual_scale, self.minimum_baseline)
        ):
            raise ValueError("residual_scale and minimum_baseline must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HybridWindRNN(nn.Module):
    """Historical GRU/LSTM + forecast context + shared horizon decoder.

    Inputs:
        past: standardized observations, ``[batch, history_hours, past_features]``.
        future: standardized NWP/calendar context, ``[batch, horizons, features]``.
        baseline: nonnegative reference NWP speed in km/h, ``[batch, horizons]``.

    The correction head starts at zero, reproducing the reference forecast
    (with a 0.05 km/h floor for exactly calm wind). Rather than hard clipping,
    its output corrects an inverse-softplus transform of that reference. This
    keeps wind nonnegative while allowing gradients below zero. For ordinary
    positive winds, the transform is approximately an additive km/h correction.
    ``residual_scale`` sets the scale of this latent correction, not the units
    of the returned predictions, which are always km/h.
    """

    def __init__(self, config: WindRNNConfig):
        super().__init__()
        self.config = config
        recurrent = nn.GRU if config.cell == "gru" else nn.LSTM
        self.encoder = recurrent(
            input_size=config.past_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=config.bidirectional,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        encoded_size = config.hidden_size * (2 if config.bidirectional else 1)
        self.future_encoder = nn.Sequential(
            nn.Linear(config.future_features, config.hidden_size),
            nn.SiLU(),
            nn.LayerNorm(config.hidden_size),
        )
        self.horizon_embedding = nn.Embedding(
            len(config.horizons), config.horizon_embedding_size
        )
        decoder_size = (
            encoded_size + config.hidden_size + config.horizon_embedding_size
        )
        if config.attention:
            self.attention_key = nn.Linear(encoded_size, config.hidden_size, bias=False)
            self.attention_query = nn.Linear(
                config.hidden_size + config.horizon_embedding_size,
                config.hidden_size,
                bias=False,
            )
            self.attention_score = nn.Linear(config.hidden_size, 1, bias=False)
            decoder_size += encoded_size
        self.decoder = nn.Sequential(
            nn.LayerNorm(decoder_size),
            nn.Linear(decoder_size, config.hidden_size),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size // 2 or 1),
            nn.SiLU(),
            nn.Linear(config.hidden_size // 2 or 1, 1),
        )
        # Start from a meteorologically meaningful forecast rather than from
        # random wind. Encoder gradients begin after the first head update.
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def encode_history(self, past: Tensor) -> tuple[Tensor, Tensor]:
        """Return encoded steps and complete final forward/backward summaries.

        ``output[:, -1]`` would summarize only the last historical hour in its
        backward half. The final hidden states summarize the whole window in
        both directions, including for multilayer LSTMs.
        """
        output, states = self.encoder(past)
        hidden = states[0] if isinstance(states, tuple) else states
        directions = 2 if self.config.bidirectional else 1
        hidden = hidden.reshape(
            self.config.num_layers, directions, past.shape[0], self.config.hidden_size
        )
        summary = hidden[-1].transpose(0, 1).reshape(past.shape[0], -1)
        return output, summary

    def _validate_shapes(self, past: Tensor, future: Tensor, baseline: Tensor) -> None:
        if past.ndim != 3 or future.ndim != 3 or baseline.ndim != 2:
            raise ValueError("expected past[B,T,F], future[B,H,C], baseline[B,H]")
        batch, history, features = past.shape
        horizons = len(self.config.horizons)
        if history < 1 or features != self.config.past_features:
            raise ValueError("history must be nonempty with configured past_features")
        if future.shape != (batch, horizons, self.config.future_features):
            raise ValueError("future shape does not match configured features/horizons")
        if baseline.shape != (batch, horizons):
            raise ValueError("baseline must match batch and configured horizons")
        if not torch.isfinite(baseline).all() or (baseline < 0).any():
            raise ValueError("baseline must contain finite nonnegative wind in km/h")

    def latent_correction(self, past: Tensor, future: Tensor) -> Tensor:
        """Return dimensionless corrections; use forward() for speed in km/h."""
        encoded, summary = self.encode_history(past)
        batch, horizons, _ = future.shape
        context = self.future_encoder(future)
        horizon_ids = torch.arange(horizons, device=past.device)
        horizon_vectors = self.horizon_embedding(horizon_ids).unsqueeze(0)
        horizon_vectors = horizon_vectors.expand(batch, -1, -1)
        repeated_summary = summary.unsqueeze(1).expand(-1, horizons, -1)
        components = [repeated_summary, context, horizon_vectors]
        if self.config.attention:
            keys = self.attention_key(encoded).unsqueeze(1)
            queries = self.attention_query(
                torch.cat([context, horizon_vectors], dim=-1)
            ).unsqueeze(2)
            scores = self.attention_score(torch.tanh(keys + queries)).squeeze(-1)
            weights = torch.softmax(scores, dim=-1)
            components.append(torch.einsum("bht,btd->bhd", weights, encoded))
        return self.decoder(torch.cat(components, dim=-1)).squeeze(-1)

    def forward(self, past: Tensor, future: Tensor, baseline: Tensor) -> Tensor:
        self._validate_shapes(past, future, baseline)
        correction = self.latent_correction(past, future)
        reference = baseline.clamp_min(self.config.minimum_baseline)
        # Stable inverse softplus: avoids exp(reference) overflow at high wind.
        latent_reference = reference + torch.log(-torch.expm1(-reference))
        return F.softplus(latent_reference + self.config.residual_scale * correction)
