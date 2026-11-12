"""Causality and split checks for the multi-source benchmark.

Run with ``.venv/bin/python -m unittest discover -s tests -v``. Station records
and forecast archives are synthetic; no download or model training is required.
"""

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
import benchmark_wind_multisource as bench  # noqa: E402


INDEX = pd.date_range("2023-12-01", "2026-03-15 23:00", freq="h")


def synthetic_stations():
    rng = np.random.default_rng(0)
    stations = {}
    for number, station in enumerate({bench.TARGET_STATION, *bench.WIND_STATIONS, *bench.PRESSURE_STATIONS}):
        hour = np.arange(len(INDEX), dtype=float)
        stations[station] = pd.DataFrame({
            "QFF": 1.0, "PMER": 1015 + 5 * np.sin(hour / 90 + number),
            "FF": 8 + 3 * np.sin(hour / 17 + number) + rng.normal(0, .5, len(INDEX)),
            "DD": (hour * 7 + number * 30) % 360, "FXY": 12.0, "FXI": 15.0,
            "T": 10 + 4 * np.sin(hour / 24) - number * .2, "TD": 5.0, "U": 70.0,
            "PSTAT": 950 + 3 * np.sin(hour / 72), "RR1": 0.0,
        }, index=INDEX)
    return stations


def synthetic_nwp():
    rows = []
    for model in bench.MODELS:
        for offset in (48, 72, 96):
            frame = pd.DataFrame({"valid_time": INDEX, "model": model, "horizon_hours": offset})
            # Encode the archive offset so tests can see which one was read.
            frame["wind_speed_10m"] = offset / 10 + np.arange(len(INDEX)) % 24 / 100
            frame["wind_direction_10m"] = 180.0
            for name in ("wind_gusts_10m", "wind_speed_100m"):
                frame[name] = frame.wind_speed_10m * 1.5
            for name in bench.NWP_SCALARS:
                frame[name] = 1.0
            rows.append(frame)
    return pd.concat(rows, ignore_index=True)


class MultisourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stations = synthetic_stations()
        cls.nwp = synthetic_nwp()
        with patch.object(bench, "read_regional_observations", return_value=(cls.stations, {})), \
                patch.object(bench, "read_nwp", return_value=cls.nwp):
            cls.frames, _ = bench.build_frames()

    def test_neighbour_features_ignore_the_future(self):
        cut = pd.Timestamp("2025-06-01 12:00")
        before = bench.neighbour_features(self.stations)
        changed = {k: v.copy() for k, v in self.stations.items()}
        for frame in changed.values():
            frame.loc[frame.index > cut, ["FF", "DD", "FXI", "PMER", "T"]] = 999.0
        after = bench.neighbour_features(changed)
        pd.testing.assert_frame_equal(before.loc[:cut], after.loc[:cut])

    def test_target_features_come_from_issue_time(self):
        for horizon, frame in self.frames.items():
            valid = pd.Timestamp("2025-06-10 15:00")
            issue = valid - pd.Timedelta(hours=horizon)
            expected = self.stations[bench.TARGET_STATION].FF.loc[issue]
            self.assertAlmostEqual(frame["persistence"].loc[valid], expected, places=5)
            self.assertAlmostEqual(frame["X"].loc[valid, "FF_lag0"], expected, places=4)
            pilat = self.stations["42101001"].FF.loc[issue]
            self.assertAlmostEqual(frame["X"].loc[valid, "st_pilat_FF_lag0"], pilat, places=4)

    def test_nwp_uses_aged_archive_offset(self):
        for horizon, frame in self.frames.items():
            speed = frame["X"]["nwp_ecmwf_ifs025_speed"].dropna()
            offset = horizon + bench.NWP_AGE_BUFFER_HOURS
            # read_nwp (km/h conversion) is patched out: values are the raw encoded offset.
            self.assertAlmostEqual(float(speed.min()), offset / 10, places=4)

    def test_same_target_hours_for_all_horizons(self):
        masks = [frame["test"] for frame in self.frames.values()]
        dev = [frame["dev"] for frame in self.frames.values()]
        for mask in masks[1:]:
            np.testing.assert_array_equal(mask, masks[0])
        for mask in dev[1:]:
            np.testing.assert_array_equal(mask, dev[0])

    def test_splits_are_chronological_and_purged(self):
        for horizon, frame in self.frames.items():
            valid = frame["X"].index
            test_issues = valid[frame["test"]] - pd.Timedelta(hours=horizon)
            self.assertGreaterEqual(test_issues.min(), bench.TEST_ISSUE_START)
            self.assertLess(valid[frame["dev"]].max(), bench.TEST_ISSUE_START)
            for start, _ in bench.FOLDS:
                start = pd.Timestamp(start)
                train = frame["dev"] & (valid < start - bench.PURGE)
                self.assertLess(valid[train].max(), start - pd.Timedelta(hours=horizon))

    def test_feature_sets_are_nested(self):
        sets = self.frames[24]["sets"]
        self.assertTrue(set(sets["A"]) < set(sets["B"]) < set(sets["C"]))
        self.assertFalse(any(c.startswith("st_") for c in sets["B"]))
        self.assertFalse(any(c.startswith(("nwp_gem", "nwp_jma")) for c in sets["A"]))

    def test_evaluate_refuses_changed_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "selection.json").write_text(json.dumps({"protocol_sha256": "0", "script_sha256": "0"}))
            with patch.object(bench, "OUT", out), self.assertRaises(RuntimeError):
                bench.evaluate(self.frames, {})


if __name__ == "__main__":
    unittest.main()
