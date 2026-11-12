"""Window, causality and masking checks for the station-network model.

Run with ``.venv/bin/python -m unittest discover -s tests -v``. Station series are
synthetic; no download, GPU or real training is required.
"""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch


ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(ANALYSIS))
import station_network as net  # noqa: E402


def write_station(directory, station, start, hours, offset):
    t = np.arange(hours, dtype=np.float32)
    series = np.stack([10 + offset + np.sin(t / 24), 60 + 0 * t, 950 + offset + 0 * t, 5 + offset + 0 * t,
                       (t * 10) % 360, 8 + offset + 0 * t, (t % 10 == 0) * 0.4], axis=1).astype(np.float32)
    series[5, 2] = np.nan  # one missing pressure value
    np.save(directory / "series" / f"{station}.npy", series)
    return {"station": station, "name": station, "department": "00", "lat": 45.0, "lon": 4.0, "alt": 500.0,
            "start": str(start), "end": str(start + pd.Timedelta(hours=hours - 1)), "hours": hours}


class StationNetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        data = Path(cls.temporary.name)
        (data / "series").mkdir()
        rows = [write_station(data, "A", pd.Timestamp("2021-12-20"), 24 * 30, 0),
                write_station(data, "B", pd.Timestamp("2021-12-25"), 24 * 30, 100)]
        cls.meta = pd.DataFrame(rows).set_index("station")
        cls.patch = patch.object(net, "DATA", data)
        cls.patch.start()
        cls.addClassCleanup(cls.patch.stop)
        cls.network = net.Network(["A", "B"], cls.meta, torch.device("cpu"))

    def test_windows_never_cross_stations(self):
        positions = self.network.issue_positions(net.EPOCH, pd.Timestamp("2100-01-01"))
        x, m, *_ = self.network.batch(positions)
        temperature = x[..., net.CHANNEL_INDEX["T"]]
        present = m[..., net.CHANNEL_INDEX["T"]]
        owner = self.network.station[positions]
        # Station B is offset by +100 °C: any window mixing stations would show both ranges.
        for k in range(len(positions)):
            values = temperature[k][present[k]]
            if owner[k] == 0:
                self.assertTrue((values < 50).all())
            else:
                self.assertTrue((values > 50).all())

    def test_issue_positions_respect_period(self):
        positions = self.network.issue_positions(net.EPOCH, net.TRAIN_END)
        hours = self.network.time[positions].long()
        self.assertTrue(((hours + max(net.HORIZONS)) < net.hour_index(net.TRAIN_END)).all())
        later = self.network.issue_positions(net.TRAIN_END, pd.Timestamp("2100-01-01"))
        self.assertTrue((self.network.time[later].long() >= net.hour_index(net.TRAIN_END)).all())

    def test_references_use_only_the_past(self):
        positions = self.network.issue_positions(net.EPOCH, pd.Timestamp("2100-01-01"))[:50]
        refs = net.references(self.network, positions)
        t = net.CHANNEL_INDEX["T"]
        now = self.network.values[positions][:, t]
        torch.testing.assert_close(refs["Persistance"][:, 0, 0], now)
        k = net.HORIZONS.index(24)
        torch.testing.assert_close(refs["Cycle diurne (J-1)"][:, k, 0], now)  # J-1 of a +24 h target is now
        k = net.HORIZONS.index(48)
        self.assertTrue(refs["Cycle diurne (J-1)"][:, k].isnan().all())  # would need the future: not defined

    def test_missing_targets_do_not_change_the_loss(self):
        positions = self.network.issue_positions(net.EPOCH, pd.Timestamp("2100-01-01"))[:64]
        norm = net.Normalizer(self.network, positions)
        batch = list(self.network.batch(positions))
        torch.manual_seed(0)
        model = net.build("MLP")
        out = model(*net.model_inputs(batch, norm))
        reference = net.loss_fn(out, batch, norm)
        changed = list(batch)
        y, ym = changed[4].clone(), changed[5]
        y[~ym] = 1e6  # garbage where targets are absent
        changed[4] = y
        torch.testing.assert_close(net.loss_fn(out, changed, norm), reference)

    def test_every_architecture_outputs_all_targets(self):
        positions = self.network.issue_positions(net.EPOCH, pd.Timestamp("2100-01-01"))[:8]
        norm = net.Normalizer(self.network, positions)
        sequence, context = net.model_inputs(self.network.batch(positions), norm)
        for name in net.ARCHITECTURES:
            out = net.build(name)(sequence, context)
            self.assertEqual(tuple(out.shape), (8, len(net.HORIZONS), len(net.TARGETS)))

    def test_tcn_is_causal(self):
        torch.manual_seed(0)
        encoder = net.TCNEncoder(16).eval()

        def features(sequence):
            h = sequence.transpose(1, 2)
            for layer in encoder.conv:
                h = layer(h)
                if isinstance(layer, torch.nn.Conv1d):
                    h = h[..., :sequence.shape[1]]
            return h

        sequence = torch.randn(1, net.WINDOW, 14)
        changed = sequence.clone()
        changed[:, 40] += 5
        with torch.no_grad():
            before, after = features(sequence), features(changed)
        # Step 40 may only influence steps 40 and later.
        torch.testing.assert_close(before[..., :40], after[..., :40])
        self.assertFalse(torch.allclose(before[..., 40:], after[..., 40:]))

if __name__ == "__main__":
    unittest.main()
