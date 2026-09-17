"""CPU-only regression checks for masked losses and frozen neural checkpoints.

All observations are synthetic. No archive reads or expensive training runs.
"""
from contextlib import ExitStack, redirect_stdout
from dataclasses import replace
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from neural_data import NeuralBatch
import train_wind_neural as training


def synthetic_batch():
    rng = np.random.default_rng(2026)
    n = 12
    baseline = rng.uniform(3, 12, size=(n, 3)).astype(np.float32)
    labels = np.ones((n, 3), dtype=bool)
    inputs = labels.copy()
    labels[0, 0] = False
    labels[1, 1] = False
    inputs[2, 2] = False
    return NeuralBatch(
        history=rng.normal(size=(n, 8, 4)).astype(np.float32),
        context=rng.normal(size=(n, 3, 5)).astype(np.float32),
        nwp_mean=baseline,
        targets=baseline + np.array([1, -1, 2], dtype=np.float32),
        label_mask=labels,
        input_mask=inputs,
        issue_times=pd.date_range("2024-06-01", periods=n, freq="h"),
        comparison_mask=np.zeros((n, 3), dtype=bool),
    )


class NeuralTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.batch = synthetic_batch()

    def test_missing_labels_and_unavailable_inputs_are_excluded_from_scores(self):
        actual = training.horizon_scores(self.batch, self.batch.nwp_mean)
        modified_y = self.batch.targets.copy()
        modified_prediction = self.batch.nwp_mean.copy()
        excluded = ~(self.batch.label_mask & self.batch.input_mask)
        modified_y[excluded] = 1_000_000
        modified_prediction[excluded] = -1_000_000
        altered = training.horizon_scores(replace(self.batch, targets=modified_y), modified_prediction)
        self.assertEqual(actual, altered)
        for horizon in ("24", "48", "72"):
            self.assertEqual(actual[horizon]["n"], 11)
        self.assertAlmostEqual(actual["24"]["mae_kmh"], 1)
        self.assertAlmostEqual(actual["48"]["rmse_kmh"], 1)
        self.assertAlmostEqual(actual["72"]["mae_kmh"], 2)

    def test_loader_preserves_issue_order_and_intersects_both_masks(self):
        with patch.object(training, "BATCH_SIZE", 5), patch.object(torch.cuda, "is_available", return_value=False):
            batches = list(training.loader(self.batch))
        np.testing.assert_array_equal(torch.cat([row[3] for row in batches]), self.batch.targets)
        np.testing.assert_array_equal(torch.cat([row[4] for row in batches]),
                                      self.batch.label_mask & self.batch.input_mask)

    def test_selection_objective_weights_horizons_mae_and_rmse_equally(self):
        reference = {str(h): {"mae_kmh": 2, "rmse_kmh": 4} for h in (24, 48, 72)}
        metrics = {"24": {"mae_kmh": 1, "rmse_kmh": 4},
                   "48": {"mae_kmh": 2, "rmse_kmh": 2},
                   "72": {"mae_kmh": 4, "rmse_kmh": 8}}
        expected = (.75 + .75 + 2) / 3
        self.assertAlmostEqual(training.objective(metrics, reference), expected)
        self.assertAlmostEqual(training.objective(reference, reference), 1)

    def train_in_temporary_directory(self, directory, batch):
        root = Path(directory)
        out = root / "output"
        (out / "checkpoints").mkdir(parents=True, exist_ok=True)
        with ExitStack() as stack:
            stack.enter_context(patch.object(training, "ROOT", root))
            stack.enter_context(patch.object(training, "OUT", out))
            stack.enter_context(patch.object(training, "BATCH_SIZE", 4))
            stack.enter_context(patch.object(training, "device", return_value=torch.device("cpu")))
            stack.enter_context(patch.object(torch.cuda, "is_available", return_value=False))
            stack.enter_context(redirect_stdout(io.StringIO()))
            stats = training.fit_one("bigru", 37, batch, None, None, fixed_epochs=1)
        path = root / stats["checkpoint"]
        self.assertTrue(path.is_file())
        model, payload = training.restore(path, torch.device("cpu"))
        return stats, model, payload

    def test_training_loss_ignores_masked_targets_and_restores_checkpoint_on_cpu(self):
        altered_targets = self.batch.targets.copy()
        altered_targets[~(self.batch.label_mask & self.batch.input_mask)] = 10_000
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            stats, model, payload = self.train_in_temporary_directory(first, self.batch)
            changed_stats, changed_model, changed_payload = self.train_in_temporary_directory(
                second, replace(self.batch, targets=altered_targets))
        self.assertEqual(stats["best_epoch"], 1)
        self.assertEqual(stats["epochs_run"], 1)
        self.assertEqual((payload["seed"], payload["phase"], payload["epoch"]), (37, "refit", 1))
        self.assertEqual(stats["history"], changed_stats["history"])
        self.assertTrue(np.isfinite(stats["history"][0]["train_loss"]))
        for name, values in payload["state_dict"].items():
            self.assertEqual(values.device.type, "cpu")
            torch.testing.assert_close(values, changed_payload["state_dict"][name], rtol=0, atol=0)
        # Learning really happened: the residual head no longer reproduces NWP.
        with patch.object(torch.cuda, "is_available", return_value=False):
            predictions = training.predictions(model, training.loader(self.batch), torch.device("cpu"))
            restored_predictions = training.predictions(changed_model, training.loader(self.batch), torch.device("cpu"))
            unlabeled = replace(self.batch, targets=np.zeros_like(self.batch.targets),
                                label_mask=np.zeros_like(self.batch.label_mask))
            inference = training.predictions(model, training.loader(unlabeled), torch.device("cpu"))
        np.testing.assert_array_equal(predictions, restored_predictions)
        np.testing.assert_array_equal(predictions, inference)
        self.assertEqual(predictions.shape, (12, 3))
        self.assertTrue(np.isfinite(predictions).all())
        self.assertTrue((predictions >= 0).all())
        self.assertFalse(np.allclose(predictions, self.batch.nwp_mean))

    def test_ensemble_uses_frozen_seed_list_even_if_global_defaults_change(self):
        selected = {"selected": "bigru", "seeds": [17, 29]}
        restored = []

        def restore(path, run_device):
            restored.append(Path(path).name)
            seed = int(Path(path).stem.rsplit("_", 1)[1])
            return seed, {"neutralize_history": False}

        def predict(model, data_loader, run_device, neutralize_history):
            return np.full((12, 3), model, dtype=np.float32)

        with ExitStack() as stack:
            stack.enter_context(patch.object(training, "SEEDS", (999,)))
            stack.enter_context(patch.object(training, "device", return_value=torch.device("cpu")))
            stack.enter_context(patch.object(training, "restore", side_effect=restore))
            stack.enter_context(patch.object(training, "predictions", side_effect=predict))
            stack.enter_context(patch.object(training, "loader", return_value=None))
            actual = training.ensemble(self.batch, selected)
        self.assertEqual(restored, ["refit_bigru_17.pt", "refit_bigru_29.pt"])
        np.testing.assert_array_equal(actual, np.full((12, 3), 23, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
