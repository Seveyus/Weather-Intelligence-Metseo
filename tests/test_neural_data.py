"""Fast synthetic checks of recurrent inputs, preprocessing and label alignment."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import neural_data as data


def synthetic_source():
    index = pd.date_range("2024-12-01", "2026-01-10", freq="h")
    hours = np.arange(len(index), dtype=float)
    observations = pd.DataFrame({name: 5 + np.sin(hours / 7) for name in data.VARIABLES}, index=index)
    observations["DD"] = hours % 360
    return observations, synthetic_frames(observations)


def synthetic_frames(observations):
    frames = {}
    for horizon in data.HORIZONS:
        valid = observations.index
        issue = valid - pd.Timedelta(hours=horizon)
        speed = pd.Series(6 + np.cos(np.arange(len(valid)) / 18), index=valid)
        context = pd.DataFrame({"FF_lag0": observations.FF.reindex(issue).to_numpy(),
                                "nwp_synthetic_speed": speed, "nwp_mean": speed,
                                "target_hour_sin": np.sin(2 * np.pi * valid.hour / 24)}, index=valid)
        frames[horizon] = {"X": context.astype(np.float32), "y": observations.FF,
                           "raw": {"synthetic": "nwp_synthetic_speed"}}
    return frames


class NeuralDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.observations, cls.frames = synthetic_source()
        cls.raw = data.build_dataset(cls.observations, cls.frames)
        cls.preprocessor = data.fit_preprocessors(cls.raw, cls.raw.splits["train"])

    def test_horizons_history_endpoint_and_units(self):
        issue = pd.Timestamp("2025-03-01 12:00")
        row = self.raw.issue_times.get_loc(issue)
        transformed = data.transform(self.raw, self.preprocessor, [row])
        self.assertEqual(transformed.history.shape[:2], (1, 72))
        self.assertEqual(transformed.context.shape, (1, 3, 8))
        ff = self.raw.metadata["history_columns"].index("FF")
        expected = self.observations.FF.loc[issue - pd.Timedelta(hours=71):issue].to_numpy()
        recovered = transformed.history[0, :, ff] * self.preprocessor.history.scale[ff] + self.preprocessor.history.mean[ff]
        np.testing.assert_allclose(recovered, expected, atol=1e-5)
        for head, horizon in enumerate(data.HORIZONS):
            self.assertAlmostEqual(transformed.targets[0, head],
                                   self.observations.FF.loc[issue + pd.Timedelta(hours=horizon)], places=5)

    def test_future_observations_do_not_change_any_input(self):
        issue = pd.Timestamp("2025-03-01 12:00")
        modified = self.observations.copy()
        modified.loc[modified.index > issue] += 10000
        changed = data.build_dataset(modified, synthetic_frames(modified))
        row = self.raw.issue_times.get_loc(issue)
        original = data.transform(self.raw, self.preprocessor, [row])
        altered = data.transform(changed, self.preprocessor, [row])
        np.testing.assert_array_equal(original.history, altered.history)
        np.testing.assert_array_equal(original.context, altered.context)
        self.assertFalse(np.array_equal(original.targets, altered.targets))

    def test_missing_labels_masked_and_do_not_block_inference(self):
        issue = pd.Timestamp("2025-03-01 12:00")
        modified = self.observations.copy()
        modified.loc[issue + pd.Timedelta(hours=48), "FF"] = np.nan
        changed = data.build_dataset(modified, synthetic_frames(modified))
        row = changed.issue_times.get_loc(issue)
        self.assertTrue(np.isnan(changed.targets[row, 1]))
        batch = data.transform(changed, self.preprocessor, [row])
        np.testing.assert_array_equal(batch.label_mask, [[True, False, True]])
        self.assertEqual(batch.targets[0, 1], 0)
        self.assertTrue(np.isfinite(batch.history).all())
        self.assertTrue(np.isfinite(batch.context).all())
        modified.loc[issue + pd.Timedelta(hours=24):, "FF"] = np.nan
        unlabeled = data.build_dataset(modified, synthetic_frames(modified))
        batch = data.transform(unlabeled, self.preprocessor, [row])
        self.assertFalse(batch.label_mask.any())
        self.assertTrue(batch.input_mask.all())

    def test_history_fill_limited_to_three_hours_and_missingness_preserved(self):
        modified = self.observations.iloc[:100].copy()
        modified.iloc[10:15, modified.columns.get_loc("T")] = np.nan
        features = data.history_features(modified)
        np.testing.assert_array_equal(features.T_missing.iloc[10:15], np.ones(5))
        np.testing.assert_allclose(features["T"].iloc[10:13], modified["T"].iloc[9])
        self.assertTrue(features["T"].iloc[13:15].isna().all())

    def test_preprocessors_exclude_validation_and_later_data(self):
        modified = self.observations.copy()
        modified.loc[modified.index >= data.VAL_START] += 1_000_000
        frames = synthetic_frames(modified)
        for frame in frames.values():
            frame["X"].loc[frame["X"].index >= data.VAL_START] += 1_000_000
        changed = data.build_dataset(modified, frames)
        prep = data.fit_preprocessors(changed, changed.splits["train"])
        for group in ("history", "context"):
            for statistic in ("median", "mean", "scale"):
                np.testing.assert_array_equal(getattr(getattr(prep, group), statistic),
                                              getattr(getattr(self.preprocessor, group), statistic))
        with self.assertRaisesRegex(ValueError, "pre-calibration"):
            data.fit_preprocessors(self.raw, self.raw.splits["cal"])

    def test_splits_purge_all_future_labels(self):
        issues = self.raw.issue_times
        latest = issues + pd.Timedelta(hours=72)
        for split, end in (("train", data.VAL_START), ("refit", data.CAL_START),
                           ("val", data.CAL_START), ("cal", data.TEST_START)):
            self.assertTrue((latest[self.raw.splits[split]] < end).all())
        for split, start in (("val", data.VAL_START), ("cal", data.CAL_START), ("test", data.TEST_START)):
            self.assertTrue((issues[self.raw.splits[split]] >= start).all())

    def test_past_forecast_errors_are_known_at_history_timestamp(self):
        issue = pd.Timestamp("2025-03-01 12:00")
        history_row = self.observations.index.get_loc(issue)
        for horizon in data.HORIZONS:
            name = f"past_nwp_error_h{horizon}_synthetic"
            column = self.raw.metadata["history_columns"].index(name)
            actual = self.raw.history_values[history_row, column]
            expected = self.observations.FF.loc[issue] - self.frames[horizon]["X"].loc[issue, "nwp_synthetic_speed"]
            self.assertAlmostEqual(actual, expected, places=5)
            self.assertIn(column, self.raw.metadata["recent_error_indices"])
        ablated = data.build_dataset(self.observations, self.frames, include_recent_errors=False)
        self.assertEqual(ablated.metadata["recent_error_indices"], [])
        self.assertEqual(ablated.history_values.shape[1], self.raw.history_values.shape[1] - 6)

    def test_exact_baseline_rows_preserved_with_unavailable_other_heads(self):
        with tempfile.TemporaryDirectory() as directory:
            valid = self.observations.index[-24:]
            for horizon in data.HORIZONS:
                pd.DataFrame({"valid_time": valid, "issue_time": valid - pd.Timedelta(hours=horizon),
                              "observed_kmh": self.observations.FF.loc[valid].to_numpy()}).to_csv(
                    Path(directory) / f"predictions_{horizon}h.csv", index=False)
            raw = data.build_dataset(self.observations, self.frames, comparison_directory=directory)
            self.assertEqual(raw.splits["test"].sum(), 72)
            np.testing.assert_array_equal(raw.comparison_mask.sum(axis=0), [24, 24, 24])
            last_issue = raw.issue_times[raw.splits["test"]].max()
            row = raw.issue_times.get_loc(last_issue)
            self.assertTrue(raw.comparison_mask[row, 0])
            self.assertFalse(raw.input_mask[row, 2])
            batch = data.transform(raw, self.preprocessor, raw.splits["test"])
            self.assertTrue(np.isfinite(batch.context).all())

    def test_hour_grid_and_frozen_schema_reject_drift(self):
        irregular = self.observations.drop(self.observations.index[100])
        with self.assertRaisesRegex(ValueError, "hourly grid"):
            data.build_dataset(irregular, synthetic_frames(irregular))
        changed = data.build_dataset(self.observations, self.frames, history_hours=48)
        with self.assertRaisesRegex(ValueError, "schema"):
            data.transform(changed, self.preprocessor, [0])


if __name__ == "__main__":
    unittest.main()
