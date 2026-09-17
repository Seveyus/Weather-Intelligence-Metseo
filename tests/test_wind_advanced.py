"""Regression checks for forecast timing, causal inputs and saved-model replay.

Run with ``.venv/bin/python -m unittest discover -s tests -v``. All weather
observations and forecast archives used here are synthetic; no network or model
training is required.
"""

from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(ANALYSIS))
import train_wind_advanced as training  # noqa: E402


class ConstantRegressor:
    def __init__(self, value):
        self.value = value

    def predict(self, features):
        return np.full(len(features), self.value, dtype=float)


class WindAdvancedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="wind-advanced-test-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.data_directory = Path(cls.temporary.name)
        (cls.data_directory / "day4").mkdir()
        (cls.data_directory / "nwp_archive.csv").touch()
        (cls.data_directory / "day4" / "nwp_archive.csv").touch()

        index = pd.date_range("2024-01-01", "2026-01-20 23:00", freq="h")
        hour = np.arange(len(index), dtype=float)
        cls.observations = pd.DataFrame(
            {
                "FF": 8 + 2 * np.sin(hour / 17),
                "DD": hour % 360,
                "FXY": 10 + np.cos(hour / 13),
                "FXI": 12 + np.cos(hour / 11),
                "T": 10 + 3 * np.sin(hour / 24),
                "TD": 5 + np.sin(hour / 24),
                "U": 70 + 5 * np.sin(hour / 24),
                "PSTAT": 950 + 3 * np.sin(hour / 72),
                "RR1": np.zeros(len(index)),
            },
            index=index,
        )
        # Include a real gap rather than testing only perfectly complete histories.
        cls.observations.loc["2024-06-01 12:00", "FF"] = np.nan
        cls.archive = pd.concat(
            [
                pd.DataFrame(
                    {
                        "valid_time": index,
                        "forecast_origin": index - pd.Timedelta(hours=lead),
                        "model": "synthetic",
                        "horizon_hours": lead,
                        "wind_speed_10m": 1 + lead / 100,
                        "wind_direction_10m": 180.0,
                        "temperature_2m": 10.0,
                        "surface_pressure": 950.0,
                    }
                )
                for lead in (24, 48, 72, 96)
            ],
            ignore_index=True,
        )
        cls.frames = cls.assemble(cls.observations)

    @classmethod
    def assemble(cls, observations):
        def read_archive(path, *args, **kwargs):
            day4 = "day4" in Path(path).parts
            mask = cls.archive.horizon_hours.eq(96)
            return cls.archive.loc[mask if day4 else ~mask].copy()

        with ExitStack() as stack:
            stack.enter_context(patch.object(training, "DATA", cls.data_directory))
            stack.enter_context(
                patch.object(training, "read_observations", return_value=(observations.copy(), {}))
            )
            stack.enter_context(patch.object(training.pd, "read_csv", side_effect=read_archive))
            _, _, frames = training.load_frames()
        return frames

    def test_exact_horizon_alignment_and_available_nwp_lead(self):
        issue = pd.Timestamp("2025-03-20 12:00")
        buffer_hours = getattr(training, "NWP_AGE_BUFFER_HOURS", 0)
        for horizon, frame in self.frames.items():
            with self.subTest(horizon=horizon):
                valid = issue + pd.Timedelta(hours=horizon)
                position = frame["X"].index.get_loc(valid)
                self.assertEqual(frame["origin"][position], issue)
                self.assertEqual(frame["persistence"].loc[valid], self.observations.FF.loc[issue])
                self.assertAlmostEqual(
                    frame["X"].loc[valid, "FF_lag0"], self.observations.FF.loc[issue], places=5
                )
                self.assertEqual(frame["y"].loc[valid], self.observations.FF.loc[valid])
                self.assertAlmostEqual(
                    frame["X"].loc[valid, "nwp_synthetic_speed"],
                    (1 + (horizon + buffer_hours) / 100) * 3.6,
                    places=5,
                )

    def test_future_observations_cannot_change_forecast_inputs(self):
        issue = pd.Timestamp("2025-03-20 12:00")
        changed = self.observations.copy()
        changed.loc[changed.index > issue] += 1000
        changed_frames = self.assemble(changed)
        for horizon, frame in self.frames.items():
            with self.subTest(horizon=horizon):
                valid = issue + pd.Timedelta(hours=horizon)
                pd.testing.assert_series_equal(
                    frame["X"].loc[valid], changed_frames[horizon]["X"].loc[valid]
                )
                self.assertNotEqual(frame["y"].loc[valid], changed_frames[horizon]["y"].loc[valid])

    def test_missing_observation_is_not_replaced_in_target(self):
        missing_time = pd.Timestamp("2024-06-01 12:00")
        for horizon, frame in self.frames.items():
            with self.subTest(horizon=horizon):
                self.assertTrue(pd.isna(frame["y"].loc[missing_time]))
                self.assertFalse(frame["train_obs"].loc[missing_time])
                valid = missing_time + pd.Timedelta(hours=horizon)
                self.assertEqual(frame["X"].loc[valid, "FF_missing"], 1)
                self.assertAlmostEqual(
                    frame["X"].loc[valid, "FF_lag0"],
                    self.observations.FF.loc[missing_time - pd.Timedelta(hours=1)],
                    places=5,
                )

    def test_splits_purge_future_labels_and_share_test_timestamps(self):
        shared_test = None
        for horizon, frame in self.frames.items():
            with self.subTest(horizon=horizon):
                valid = frame["X"].index
                origins = frame["origin"]
                for earlier, later in (("train_obs", "val"), ("refit_obs", "cal"), ("cal", "test")):
                    self.assertLess(valid[frame[earlier]].max(), origins[frame[later]].min())
                self.assertTrue((valid[frame["train_nwp"]] < training.VAL_START).all())
                self.assertTrue((valid[frame["refit_nwp"]] < training.CAL_START).all())
                self.assertTrue((valid[frame["cal"]] < training.TEST_START).all())
                self.assertTrue((origins[frame["test"]] >= training.TEST_START).all())
                test_times = valid[frame["test"]]
                if shared_test is None:
                    shared_test = test_times
                else:
                    pd.testing.assert_index_equal(test_times, shared_test)

    def test_residual_is_added_back_before_nonnegative_clipping(self):
        index = pd.date_range("2026-01-05", periods=2, freq="h")
        frame = {"X": pd.DataFrame({"nwp_mean": [2.0, 10.0]}, index=index)}
        spec = {"kind": "residual", "estimator": ConstantRegressor(-3)}
        actual = training.predict_spec(spec, frame, np.array([True, True]))
        np.testing.assert_array_equal(actual, [0, 7])

    def test_interval_radius_uses_absolute_residuals_and_finite_sample_rank(self):
        self.assertEqual(training.interval_radius(np.arange(10), coverage=.8), 8)
        self.assertEqual(training.interval_radius(np.array([-5, 1, -2, 4]), coverage=.8), 5)

    def test_replay_rejects_issue_before_interval_calibration_is_available(self):
        with patch.object(training.joblib, "load", return_value={}):
            with patch.object(training, "load_frames", return_value=(None, {}, self.frames)):
                with self.assertRaises(ValueError):
                    training.predict_at("2025-12-31T23:00:00Z")

    def replay_artifact(self, horizon=24):
        frame = self.frames[horizon]
        return {
            "models": {"constant": {"kind": "direct", "estimator": ConstantRegressor(7)}},
            "blend": {"constant": 1.0},
            "interval_radius_kmh": 3.0,
            "feature_columns": list(frame["X"].columns),
            "obs_columns": list(frame["obs_columns"]),
            "nwp_models": list(frame["raw"]),
            "horizon_hours": horizon,
            "nwp_age_buffer_hours": getattr(training, "NWP_AGE_BUFFER_HOURS", 0),
        }

    def test_replay_rejects_feature_or_nwp_model_schema_drift(self):
        for key in ("feature_columns", "obs_columns", "nwp_models"):
            with self.subTest(schema=key):
                artifact = self.replay_artifact()
                artifact[key].append("unexpected")
                with patch.object(training.joblib, "load", return_value={24: artifact}):
                    with patch.object(training, "load_frames", return_value=(None, {}, self.frames)):
                        with self.assertRaisesRegex(ValueError, "schema"):
                            training.predict_at("2026-01-05T00:00:00Z")

    def test_replay_normalizes_issue_timezone_and_uses_saved_interval(self):
        output = io.StringIO()
        artifacts = {horizon: self.replay_artifact(horizon) for horizon in training.HORIZONS}
        with patch.object(training.joblib, "load", return_value=artifacts):
            with patch.object(training, "load_frames", return_value=(None, {}, self.frames)):
                with redirect_stdout(output):
                    training.predict_at("2026-01-05T01:00:00+01:00")
        rows = json.loads(output.getvalue())
        self.assertEqual(len(rows), 3)
        for row in rows:
            issue = pd.Timestamp(row["issue_time_utc"])
            self.assertEqual(issue, pd.Timestamp("2026-01-05T00:00:00"))
            self.assertEqual(pd.Timestamp(row["valid_time_utc"]) - issue, pd.Timedelta(hours=row["horizon_hours"]))
            self.assertEqual((row["wind_kmh"], row["lower80_kmh"], row["upper80_kmh"]), (7, 4, 10))

    def test_replay_rejects_nwp_availability_policy_drift(self):
        artifact = self.replay_artifact()
        artifact["nwp_age_buffer_hours"] += 24
        with patch.object(training.joblib, "load", return_value={24: artifact}):
            with patch.object(training, "load_frames", return_value=(None, {}, self.frames)):
                with self.assertRaisesRegex(ValueError, "schema"):
                    training.predict_at("2026-01-05T00:00:00Z")


if __name__ == "__main__":
    unittest.main()
