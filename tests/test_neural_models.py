"""Synthetic contract and learning checks for recurrent wind architectures."""

import io
from pathlib import Path
import sys
import unittest

import torch


ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(ANALYSIS))
from neural_models import HybridWindRNN, WindRNNConfig  # noqa: E402


class NeuralModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(42)
        self.past = torch.randn(4, 12, 5)
        self.future = torch.randn(4, 3, 7)
        self.baseline = torch.tensor([[0.0, 8.0, 20.0]]).repeat(4, 1)

    def model(self, **kwargs):
        return HybridWindRNN(
            WindRNNConfig(past_features=5, future_features=7, hidden_size=8, **kwargs)
        )

    def test_all_architectures_start_from_reference_and_preserve_shape(self):
        for cell in ("gru", "lstm"):
            for bidirectional in (False, True):
                for attention in (False, True):
                    with self.subTest(cell=cell, bidirectional=bidirectional, attention=attention):
                        model = self.model(
                            cell=cell, bidirectional=bidirectional, attention=attention
                        )
                        prediction = model(self.past, self.future, self.baseline)
                        self.assertEqual(prediction.shape, (4, 3))
                        self.assertTrue(torch.isfinite(prediction).all())
                        torch.testing.assert_close(
                            prediction, self.baseline.clamp_min(model.config.minimum_baseline)
                        )

    def test_negative_correction_is_smooth_and_cannot_create_negative_wind(self):
        model = self.model(dropout=0.0)
        with torch.no_grad():
            model.decoder[-1].bias.fill_(-1.0)
        prediction = model(self.past, self.future, self.baseline)
        self.assertTrue((prediction >= 0).all())
        self.assertTrue((prediction < self.baseline.clamp_min(.05)).all())
        prediction.sum().backward()
        self.assertGreater(model.decoder[-1].bias.grad.abs().item(), 0)

    def test_gru_and_lstm_receive_gradients_from_all_forecast_horizons(self):
        for cell in ("gru", "lstm"):
            with self.subTest(cell=cell):
                model = self.model(cell=cell, attention=True, dropout=0.0)
                optimizer = torch.optim.Adam(model.parameters(), lr=.01)
                target = self.baseline + 2
                # A zero head intentionally shields the encoder only on the
                # first update, keeping the initial prediction equal to NWP.
                optimizer.zero_grad()
                (model(self.past, self.future, self.baseline) - target).square().mean().backward()
                optimizer.step()
                optimizer.zero_grad()
                prediction = model(self.past, self.future, self.baseline)
                (prediction - target).square().mean().backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    self.assertGreater(parameter.grad.abs().sum().item(), 0, name)

    def test_bidirectional_summary_uses_complete_final_states(self):
        for cell in ("gru", "lstm"):
            with self.subTest(cell=cell):
                model = self.model(cell=cell, num_layers=2, bidirectional=True, dropout=0)
                output, summary = model.encode_history(self.past)
                hidden = model.config.hidden_size
                torch.testing.assert_close(summary[:, :hidden], output[:, -1, :hidden])
                torch.testing.assert_close(summary[:, hidden:], output[:, 0, hidden:])
                self.assertFalse(torch.allclose(summary[:, hidden:], output[:, -1, hidden:]))

    def test_replay_is_deterministic_with_config_and_state_dict(self):
        model = self.model(cell="lstm", num_layers=2, attention=True, dropout=.3)
        with torch.no_grad():
            model.decoder[-1].weight.normal_(std=.05)
        model.eval()
        with torch.no_grad():
            expected = model(self.past, self.future, self.baseline)
            torch.testing.assert_close(expected, model(self.past, self.future, self.baseline))
        buffer = io.BytesIO()
        torch.save({"config": model.config.to_dict(), "state_dict": model.state_dict()}, buffer)
        buffer.seek(0)
        artifact = torch.load(buffer, weights_only=True)
        restored = HybridWindRNN(WindRNNConfig(**artifact["config"]))
        restored.load_state_dict(artifact["state_dict"])
        restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(expected, restored(self.past, self.future, self.baseline))

    def test_batch_forecasts_do_not_depend_on_other_samples(self):
        # Hidden states must not carry information across unrelated samples or
        # forecast origins. Dataset timestamp causality is tested separately.
        model = self.model(attention=True, dropout=0).eval()
        with torch.no_grad():
            model.decoder[-1].weight.normal_(std=.1)
            expected = model(self.past[:1], self.future[:1], self.baseline[:1])
            changed_past = self.past.clone()
            changed_future = self.future.clone()
            changed_past[1:] += 100
            changed_future[1:] -= 100
            actual = model(changed_past, changed_future, self.baseline)[:1]
        torch.testing.assert_close(expected, actual)

    def test_hourly_forecast_context_is_separate_between_horizons(self):
        model = self.model(attention=True, dropout=0).eval()
        with torch.no_grad():
            model.decoder[-1].weight.normal_(std=.1)
            expected = model(self.past, self.future, self.baseline)
            changed_future = self.future.clone()
            changed_future[:, 2] += 20
            actual = model(self.past, changed_future, self.baseline)
        torch.testing.assert_close(expected[:, :2], actual[:, :2])
        self.assertFalse(torch.allclose(expected[:, 2], actual[:, 2]))

    def test_rejects_incompatible_shapes_and_invalid_baseline(self):
        model = self.model()
        for past, future, baseline in (
            (self.past[:, :0], self.future, self.baseline),
            (self.past, self.future[:, :2], self.baseline),
            (self.past, self.future, self.baseline[:, :2]),
            (self.past, self.future, -self.baseline),
            (self.past, self.future, self.baseline * float("nan")),
        ):
            with self.subTest(shapes=(past.shape, future.shape, baseline.shape)):
                with self.assertRaises(ValueError):
                    model(past, future, baseline)


if __name__ == "__main__":
    unittest.main()
