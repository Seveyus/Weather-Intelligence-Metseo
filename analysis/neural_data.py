"""Causal, timestamp-aligned arrays for multi-horizon recurrent wind models.

The archived weather inputs come from ``train_wind_advanced.load_frames`` and
retain its conservative 24-hour NWP availability buffer. Targets are never
imputed. A missing label is represented explicitly by ``label_mask``. All array
times are timezone-naive UTC, following the source observations and baseline.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from train_wind_advanced import NWP_AGE_BUFFER_HOURS


HORIZONS = (24, 48, 72)
VAL_START = pd.Timestamp("2025-01-01")
CAL_START = pd.Timestamp("2025-10-01")
TEST_START = pd.Timestamp("2026-01-01")
VARIABLES = ("FF", "DD", "FXY", "FXI", "T", "TD", "U", "PSTAT", "RR1")


@dataclass
class RawDataset:
    """Unexpanded histories keep the multi-year source arrays reasonably small."""

    issue_times: pd.DatetimeIndex
    history_values: np.ndarray
    history_positions: np.ndarray
    context: np.ndarray
    nwp_mean: np.ndarray
    targets: np.ndarray
    label_mask: np.ndarray
    input_mask: np.ndarray
    splits: dict[str, np.ndarray]
    comparison_mask: np.ndarray
    metadata: dict


@dataclass
class NumericPreprocessor:
    median: np.ndarray
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "NumericPreprocessor":
        values = np.asarray(values, dtype=np.float64)
        values = np.where(np.isfinite(values), values, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median = np.nanmedian(values, axis=0)
        median = np.nan_to_num(median, nan=0.0)
        filled = np.where(np.isfinite(values), values, median)
        mean, scale = filled.mean(axis=0), filled.std(axis=0)
        scale = np.where(scale > 1e-6, scale, 1.0)
        return cls(median.astype(np.float32), mean.astype(np.float32), scale.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        filled = np.where(np.isfinite(values), values, self.median)
        return ((filled - self.mean) / self.scale).astype(np.float32)


@dataclass
class Preprocessors:
    history: NumericPreprocessor
    context: NumericPreprocessor
    metadata: dict
    fit_issue_start: str
    fit_issue_end: str
    fit_latest_target: str
    fit_samples: int


@dataclass
class NeuralBatch:
    history: np.ndarray
    context: np.ndarray
    nwp_mean: np.ndarray
    targets: np.ndarray
    label_mask: np.ndarray
    input_mask: np.ndarray
    issue_times: pd.DatetimeIndex
    comparison_mask: np.ndarray

    def subset(self, rows) -> "NeuralBatch":
        """Slice every aligned array together (boolean mask or integer indices)."""
        return NeuralBatch(**{name: getattr(self, name)[rows]
                              for name in self.__dataclass_fields__})


def history_features(observations: pd.DataFrame) -> pd.DataFrame:
    """Features at hour t depend only on measurements at or before hour t."""
    raw = observations.loc[:, list(VARIABLES)].replace([np.inf, -np.inf], np.nan)
    past = raw.ffill(limit=3)
    angle = np.deg2rad(past.DD)
    result = past.drop(columns="DD").copy()
    result["direction_sin"] = np.sin(angle)
    result["direction_cos"] = np.cos(angle)
    result["wind_u"] = -past.FF * np.sin(angle)
    result["wind_v"] = -past.FF * np.cos(angle)
    result["hour_sin"] = np.sin(2 * np.pi * raw.index.hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * raw.index.hour / 24)
    result["year_sin"] = np.sin(2 * np.pi * raw.index.dayofyear / 365.25)
    result["year_cos"] = np.cos(2 * np.pi * raw.index.dayofyear / 365.25)
    for name in VARIABLES:
        result[f"{name}_missing"] = raw[name].isna().astype(np.float32)
    return result.astype(np.float32)


def build_dataset(observations: pd.DataFrame, frames: dict, history_hours: int = 72,
                  comparison_directory: str | Path | None = None,
                  include_recent_errors: bool = True) -> RawDataset:
    """Align each issue time with past observations and three future contexts.

    All baseline predictors are kept. The caller may request the exact union of
    frozen baseline CSV issue times, with separate per-head comparison masks.
    An issue remains usable for inference when another head's verifying target
    or archived forecast is unavailable; only that head is excluded from scoring.
    """
    index = observations.index
    if (not isinstance(index, pd.DatetimeIndex) or not index.is_unique
            or not index.is_monotonic_increasing or index.tz is not None):
        raise ValueError("Observations need a unique sorted timezone-naive DatetimeIndex")
    if len(index) < history_hours or history_hours < 1:
        raise ValueError("Insufficient observations for the requested history")
    if not np.all(np.diff(index.to_numpy()) == np.timedelta64(1, "h")):
        raise ValueError("Observations must use an exact hourly grid, including missing rows")
    hist = history_features(observations)
    error_columns = []
    if include_recent_errors:
        # These are past verification errors: FF(s) is already observed, and
        # each archived NWP forecast verifying at s originated before s. No
        # forecast error at any s > the issue time enters the history window.
        for horizon in HORIZONS:
            for model, column in frames[horizon]["raw"].items():
                name = f"past_nwp_error_h{horizon}_{model}"
                values = observations.FF - frames[horizon]["X"][column].reindex(index)
                hist[name] = values.ffill(limit=3).astype(np.float32)
                hist[f"{name}_missing"] = values.isna().astype(np.float32)
                error_columns.extend([name, f"{name}_missing"])
    positions = np.arange(history_hours - 1, len(index))
    issues = index[positions]
    columns = list(frames[HORIZONS[0]]["X"].columns)
    contexts, raw_means, targets, inputs = [], [], [], []
    raw_columns = {}
    for horizon in HORIZONS:
        frame = frames[horizon]
        if list(frame["X"].columns) != columns:
            raise ValueError("All horizons must share the same ordered context feature schema")
        valid = issues + pd.Timedelta(hours=horizon)
        context = frame["X"].reindex(valid)
        raw_columns[horizon] = dict(frame["raw"])
        available = np.isfinite(context[list(frame["raw"].values())].to_numpy()).all(axis=1)
        available &= np.isfinite(observations.FF.reindex(issues).to_numpy())
        contexts.append(context.to_numpy(dtype=np.float32))
        raw_means.append(context.nwp_mean.to_numpy(dtype=np.float32))
        targets.append(frame["y"].reindex(valid).to_numpy(dtype=np.float32))
        inputs.append(available)
    context = np.stack(contexts, axis=1)
    means, y = np.stack(raw_means, axis=1), np.stack(targets, axis=1)
    input_mask = np.stack(inputs, axis=1)
    label_mask = np.isfinite(y) & (y >= 0)
    complete_inputs = input_mask.all(axis=1)
    eligible = complete_inputs & label_mask.any(axis=1)
    latest = issues + pd.Timedelta(hours=max(HORIZONS))
    splits = {
        "train": eligible & (latest < VAL_START),
        "val": eligible & (issues >= VAL_START) & (latest < CAL_START),
        "refit": eligible & (latest < CAL_START),
        "cal": eligible & (issues >= CAL_START) & (latest < TEST_START),
        "test": input_mask.any(axis=1) & (issues >= TEST_START),
    }
    comparison = np.zeros_like(label_mask)
    if comparison_directory is not None:
        for head, horizon in enumerate(HORIZONS):
            saved = pd.read_csv(Path(comparison_directory) / f"predictions_{horizon}h.csv",
                                parse_dates=["issue_time", "valid_time"])
            expected_valid = saved.issue_time + pd.Timedelta(hours=horizon)
            if not saved.valid_time.equals(expected_valid.rename("valid_time")):
                raise ValueError(f"Baseline +{horizon}h CSV has incorrect lead times")
            if saved.issue_time.duplicated().any():
                raise ValueError("Baseline contains duplicate issue times")
            rows = issues.get_indexer(saved.issue_time)
            if (rows < 0).any():
                raise ValueError("A baseline issue time is absent from the neural dataset")
            if not (input_mask[rows, head] & label_mask[rows, head]).all():
                raise ValueError("A frozen baseline case has unavailable neural input or target")
            if not np.allclose(y[rows, head], saved.observed_kmh, atol=1e-5, rtol=1e-5):
                raise ValueError("Frozen baseline and neural target observations differ")
            if not (issues[rows] >= TEST_START).all():
                raise ValueError("Frozen baseline comparison contains a pre-test issue time")
            comparison[rows, head] = True
        splits["test"] = comparison.any(axis=1)
    metadata = {
        "history_hours": history_hours,
        "horizons": list(HORIZONS),
        "history_columns": list(hist.columns),
        "recent_error_columns": error_columns,
        "recent_error_indices": [hist.columns.get_loc(c) for c in error_columns],
        "context_columns": columns,
        "transformed_context_columns": columns + [f"{c}__missing" for c in columns],
        "nwp_models": raw_columns,
        "nwp_age_buffer_hours": NWP_AGE_BUFFER_HOURS,
        "history_policy": "t-history_hours+1 through t inclusive; forward fill at most 3 hours",
        "past_error_policy": "Observed FF(s) minus archived NWP verifying at s, exclusively s<=issue",
        "target_policy": "Never impute targets; missing/nonnegative validity mask",
    }
    return RawDataset(issues, hist.to_numpy(), positions, context, means, y,
                      label_mask, input_mask, splits, comparison, metadata)


def _indices(dataset: RawDataset, mask) -> np.ndarray:
    if mask is None:
        return np.arange(len(dataset.issue_times))
    rows = np.asarray(mask)
    if rows.dtype == bool:
        if rows.shape != (len(dataset.issue_times),):
            raise ValueError("Selection mask has an incorrect shape")
        return np.flatnonzero(rows)
    if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer):
        raise ValueError("Select rows with a one-dimensional mask or integer indices")
    if (rows < 0).any() or (rows >= len(dataset.issue_times)).any():
        raise ValueError("Selection row outside dataset")
    return rows


def fit_preprocessors(dataset: RawDataset, mask) -> Preprocessors:
    """Fit on selected training/refit rows only; never calibration or final test."""
    rows = _indices(dataset, mask)
    if not len(rows):
        raise ValueError("Cannot fit preprocessing on an empty selection")
    if not dataset.splits["refit"][rows].all():
        raise ValueError("Preprocessors may only fit eligible pre-calibration training/refit rows")
    positions = dataset.history_positions[rows]
    length = dataset.metadata["history_hours"]
    # Each observed hour contributes once, even when shared by many windows.
    changes = np.zeros(len(dataset.history_values) + 1, dtype=np.int64)
    np.add.at(changes, positions - length + 1, 1)
    np.add.at(changes, positions + 1, -1)
    used = np.cumsum(changes[:-1]) > 0
    history = NumericPreprocessor.fit(dataset.history_values[used])
    context = NumericPreprocessor.fit(dataset.context[rows].reshape(-1, dataset.context.shape[-1]))
    issues = dataset.issue_times[rows]
    return Preprocessors(history, context, dataset.metadata.copy(), str(issues.min()),
                         str(issues.max()), str(issues.max() + pd.Timedelta(hours=max(HORIZONS))),
                         len(rows))


def transform(dataset: RawDataset, preprocessor: Preprocessors, mask=None) -> NeuralBatch:
    """Transform selected issues with a frozen preprocessing artifact.

    Missing targets become zero only in the numeric training tensor; callers
    must use ``label_mask & input_mask`` in every loss and metric. No target is
    required for inference. Context missing indicators are appended explicitly.
    """
    if dataset.metadata != preprocessor.metadata:
        raise ValueError("Dataset feature schema differs from the frozen preprocessing artifact")
    rows = _indices(dataset, mask)
    length = dataset.metadata["history_hours"]
    positions = dataset.history_positions[rows, None] - np.arange(length - 1, -1, -1)
    history = preprocessor.history.transform(dataset.history_values[positions])
    raw_context = dataset.context[rows]
    missing = (~np.isfinite(raw_context)).astype(np.float32)
    context = np.concatenate([preprocessor.context.transform(raw_context), missing], axis=-1)
    return NeuralBatch(history, context, np.nan_to_num(dataset.nwp_mean[rows], nan=0.0,
                                                     posinf=0.0, neginf=0.0),
                       np.where(dataset.label_mask[rows], dataset.targets[rows], 0).astype(np.float32),
                       dataset.label_mask[rows].copy(), dataset.input_mask[rows].copy(),
                       dataset.issue_times[rows], dataset.comparison_mask[rows].copy())
