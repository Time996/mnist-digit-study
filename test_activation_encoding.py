"""Synthetic checks for the classroom activation/encoding experiment.

Run: python -B -m unittest -v test_activation_encoding
No MNIST loading, official test-array access, or formal experiment is performed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

import activation_encoding_core as ann
import activation_encoding_readout as readout


CLASSROOM9 = ("binary_step", "linear", "sigmoid", "tanh", "relu",
              "leaky_relu", "prelu", "elu", "swish")


def parameter_items(parameters):
    items = {"W1": parameters.weights[0], "b1": parameters.biases[0],
             "W2": parameters.weights[1], "b2": parameters.biases[1]}
    if parameters.alpha is not None:
        items["alpha"] = parameters.alpha
    return items


def snapshot(parameters):
    return {key: value.copy() for key, value in parameter_items(parameters).items()}


def stable_loss_from_logits(logits, labels):
    """Independent float64 log-sum-exp reference; no probability clipping."""
    logits = np.asarray(logits, dtype=np.float64)
    maximum = logits.max(axis=0)
    log_normalizer = maximum + np.log(np.exp(logits - maximum).sum(axis=0))
    return float(np.mean(log_normalizer - logits[labels, np.arange(len(labels))]))


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.x = np.asarray([
            [-0.8, 0.2, 0.3, 0.6], [0.4, -0.7, 0.1, -0.3],
            [0.7, 0.5, -0.4, 0.2], [-0.2, -0.5, 0.8, -0.6],
            [0.3, 0.8, 0.5, -0.7], [-0.6, 0.4, -0.2, 0.9],
        ], dtype=np.float64)
        self.y = np.asarray([0, 2, 1, 1, 2, 0], dtype=np.int64)

    def make_parameters(self, activation, dtype=np.float64):
        p = ann.initialize_parameters([4, 5, 3], seed=17, activation=activation, dtype=dtype)
        p.biases[0][...] = np.asarray([-0.21, 0.11, -0.13, 0.19, 0.07], dtype=dtype).reshape(5, 1)
        if activation == "prelu":
            self.assertEqual(p.alpha.shape, (5, 1))
            p.alpha[...] = np.asarray([0.01, 0.04, -0.08, 0.3, 1.2], dtype=dtype).reshape(5, 1)
        return p

    def assert_snapshot_equal(self, before, p):
        self.assertEqual(set(before), set(parameter_items(p)))
        for key, array in parameter_items(p).items():
            np.testing.assert_array_equal(array, before[key], err_msg=key)

    def check_complete_finite_difference(self, activation):
        p = self.make_parameters(activation)
        cache = ann.forward(self.x, p)
        self.assertGreater(float(np.min(np.abs(cache["z"]))), 1e-3)
        self.assertEqual(cache["probabilities"].shape, (3, 6))
        np.testing.assert_allclose(cache["probabilities"].sum(axis=0), 1, atol=1e-14, rtol=0)
        gradients = ann.backward(cache, self.y, p)
        analytic, numeric = [], []
        epsilon = 1e-6
        for key, parameter in parameter_items(p).items():
            self.assertEqual(parameter.dtype, np.float64)
            self.assertEqual(gradients[key].shape, parameter.shape)
            for index in np.ndindex(parameter.shape):
                original = float(parameter[index])
                try:
                    parameter[index] = original + epsilon
                    plus = stable_loss_from_logits(ann.forward(self.x, p)["logits"], self.y)
                    parameter[index] = original - epsilon
                    minus = stable_loss_from_logits(ann.forward(self.x, p)["logits"], self.y)
                finally:
                    parameter[index] = original
                numeric.append((plus - minus) / (2 * epsilon))
                analytic.append(float(gradients[key][index]))
        numeric, analytic = np.asarray(numeric), np.asarray(analytic)
        np.testing.assert_allclose(analytic, numeric, atol=3e-8, rtol=3e-5)
        relative = np.linalg.norm(analytic - numeric) / max(1e-15, np.linalg.norm(analytic) + np.linalg.norm(numeric))
        self.assertLess(relative, 3e-7)

    def test_step_complete_finite_differences_away_from_zero(self):
        self.check_complete_finite_difference("binary_step")

    def test_linear_complete_finite_differences(self):
        self.check_complete_finite_difference("linear")

    def test_sigmoid_complete_finite_differences(self):
        self.check_complete_finite_difference("sigmoid")

    def test_tanh_complete_finite_differences(self):
        self.check_complete_finite_difference("tanh")

    def test_relu_complete_finite_differences_away_from_zero(self):
        self.check_complete_finite_difference("relu")

    def test_leaky_relu_complete_finite_differences(self):
        self.check_complete_finite_difference("leaky_relu")

    def test_prelu_complete_finite_differences_including_negative_alpha(self):
        self.check_complete_finite_difference("prelu")

    def test_elu_complete_finite_differences(self):
        self.check_complete_finite_difference("elu")

    def test_swish_complete_finite_differences(self):
        self.check_complete_finite_difference("swish")

    def test_all_activations_share_exact_initial_weights_and_biases(self):
        reference = ann.initialize_parameters([4, 5, 3], 3790, "relu", dtype=np.float64)
        for name in CLASSROOM9:
            with self.subTest(activation=name):
                p = ann.initialize_parameters([4, 5, 3], 3790, name, dtype=np.float64)
                for actual, expected in zip(p.weights + p.biases, reference.weights + reference.biases):
                    np.testing.assert_array_equal(actual, expected)
        alpha = ann.initialize_parameters([4, 5, 3], 3790, "prelu", dtype=np.float64).alpha
        np.testing.assert_array_equal(alpha, np.full((5, 1), 0.01))

    def test_prelu_leaky_initial_forward_and_zero_one_alpha_equivalences(self):
        prelu = ann.initialize_parameters([4, 5, 3], 17, "prelu", dtype=np.float64)
        leaky = ann.initialize_parameters([4, 5, 3], 17, "leaky_relu", dtype=np.float64)
        np.testing.assert_array_equal(ann.forward(self.x, prelu)["h"], ann.forward(self.x, leaky)["h"])
        for alpha, equivalent in ((0.0, "relu"), (1.0, "linear")):
            with self.subTest(alpha=alpha):
                prelu.alpha[...] = alpha
                other = ann.initialize_parameters([4, 5, 3], 17, equivalent, dtype=np.float64)
                for key in ("z", "h", "logits", "probabilities"):
                    np.testing.assert_array_equal(ann.forward(self.x, prelu)[key], ann.forward(self.x, other)[key])
                g_prelu = ann.backward(ann.forward(self.x, prelu), self.y, prelu)
                g_other = ann.backward(ann.forward(self.x, other), self.y, other)
                for key in ("W1", "b1", "W2", "b2"):
                    np.testing.assert_array_equal(g_prelu[key], g_other[key])

    def test_backpropagation_has_unaveraged_deltas_and_single_mean_gradient(self):
        for name in CLASSROOM9:
            with self.subTest(activation=name):
                p = self.make_parameters(name)
                cache = ann.forward(self.x, p)
                gradients = ann.backward(cache, self.y, p)
                targets = np.zeros_like(cache["probabilities"])
                targets[self.y, np.arange(len(self.y))] = 1
                expected_delta2 = cache["probabilities"] - targets
                np.testing.assert_allclose(gradients["delta2"], expected_delta2, atol=1e-14, rtol=0)
                np.testing.assert_allclose(gradients["U"], p.weights[1].T @ expected_delta2, atol=1e-14, rtol=0)
                np.testing.assert_allclose(gradients["delta1"], gradients["U"] * gradients["derivative"], atol=1e-14, rtol=0)
                np.testing.assert_allclose(gradients["W2"], expected_delta2 @ cache["h"].T / len(self.y), atol=1e-14, rtol=0)
                np.testing.assert_allclose(gradients["W1"], gradients["delta1"] @ cache["a0"].T / len(self.y), atol=1e-14, rtol=0)

    def test_freeze_encoder_keeps_W1_b1_alpha_exactly_and_updates_output(self):
        for name in CLASSROOM9:
            with self.subTest(activation=name):
                p = self.make_parameters(name)
                initial = snapshot(p)
                for _ in range(3):
                    g = ann.backward(ann.forward(self.x, p), self.y, p)
                    applied = ann.update_parameters(p, g, 0.03, train_encoder=False)
                    self.assertEqual(applied["W1"], 0)
                    self.assertEqual(applied["b1"], 0)
                    self.assertEqual(applied["alpha"], 0)
                np.testing.assert_array_equal(p.weights[0], initial["W1"])
                np.testing.assert_array_equal(p.biases[0], initial["b1"])
                if p.alpha is not None:
                    np.testing.assert_array_equal(p.alpha, initial["alpha"])
                self.assertGreater(np.linalg.norm(p.weights[1] - initial["W2"]), 0)

    def test_true_step_gradient_is_zero_and_encoder_stays_frozen(self):
        p = self.make_parameters("binary_step")
        initial = snapshot(p)
        for _ in range(4):
            g = ann.backward(ann.forward(self.x, p), self.y, p)
            for key in ("W1", "b1", "U", "delta1", "derivative"):
                self.assertTrue(np.all(np.isfinite(g[key])))
            for key in ("W1", "b1", "delta1", "derivative"):
                np.testing.assert_array_equal(g[key], np.zeros_like(g[key]))
            applied = ann.update_parameters(p, g, 0.03, train_encoder=True)
            self.assertEqual(applied["W1"], 0)
            self.assertEqual(applied["b1"], 0)
        np.testing.assert_array_equal(p.weights[0], initial["W1"])
        np.testing.assert_array_equal(p.biases[0], initial["b1"])
        self.assertGreater(np.linalg.norm(p.weights[1] - initial["W2"]), 0)

    def test_step_alias_and_zero_point_convention(self):
        p = ann.initialize_parameters([4, 5, 3], 17, "step", dtype=np.float64)
        self.assertEqual(p.activation, "binary_step")
        p.weights[0][...] = 0
        p.biases[0][...] = 0
        cache = ann.forward(self.x, p)
        np.testing.assert_array_equal(cache["h"], np.zeros((5, 6)))
        gradients = ann.backward(cache, self.y, p)
        np.testing.assert_array_equal(gradients["derivative"], np.zeros((5, 6)))

    def test_reported_actual_update_norm_matches_float32_after_minus_before(self):
        p = self.make_parameters("prelu", dtype=np.float32)
        before = snapshot(p)
        g = ann.backward(ann.forward(self.x.astype(np.float32), p), self.y, p)
        norms = ann.update_parameters(p, g, 0.037, train_encoder=True)
        self.assertGreater(norms["alpha"], 0)
        for key, after in parameter_items(p).items():
            expected = float(np.linalg.norm(after.astype(np.float64) - before[key].astype(np.float64)))
            self.assertAlmostEqual(norms[key], expected, places=14)

    def test_rejected_nonfinite_update_does_not_partially_mutate_parameters(self):
        p = self.make_parameters("prelu")
        before = snapshot(p)
        g = ann.backward(ann.forward(self.x, p), self.y, p)
        g["W2"] = g["W2"].copy()
        g["W2"][0, 0] = np.inf
        with self.assertRaises((FloatingPointError, ValueError)):
            ann.update_parameters(p, g, 0.03, train_encoder=True)
        self.assert_snapshot_equal(before, p)

    def test_probe_is_diagnostic_nonmutating_and_preserves_inputs(self):
        for name in CLASSROOM9:
            with self.subTest(activation=name):
                p = self.make_parameters(name)
                before = snapshot(p)
                x_before, y_before = self.x.copy(), self.y.copy()
                scalars, units, raw = ann.probe_diagnostics(self.x, self.y, p)
                self.assertIsInstance(scalars, dict)
                self.assertIsInstance(units, dict)
                self.assertIsInstance(raw, dict)
                self.assertGreater(len(scalars), 0)
                self.assertGreater(len(units), 0)
                self.assertGreater(len(raw), 0)
                self.assert_snapshot_equal(before, p)
                np.testing.assert_array_equal(self.x, x_before)
                np.testing.assert_array_equal(self.y, y_before)

    def test_evaluation_loss_matches_logits_and_is_batch_partition_invariant(self):
        p = self.make_parameters("swish")
        expected = stable_loss_from_logits(ann.forward(self.x, p)["logits"], self.y)
        first = ann.evaluate_dataset(self.x, self.y, p, batch_size=2)
        second = ann.evaluate_dataset(self.x, self.y, p, batch_size=5)
        self.assertAlmostEqual(first["loss"], expected, places=13)
        self.assertAlmostEqual(second["loss"], expected, places=13)
        self.assertEqual(first["accuracy"], second["accuracy"])
        np.testing.assert_allclose(first["probabilities"], second["probabilities"], atol=1e-14, rtol=0)
        self.assertEqual(first["probabilities"].shape, (6, 3))

    def test_extreme_logits_use_stable_logsumexp_not_probability_clipped_CE(self):
        p = ann.Parameters("linear", [np.eye(3), 500 * np.eye(3)], [np.zeros((3, 1)), np.zeros((3, 1))], None)
        x = np.asarray([[1000., -1000., 0.], [-1000., 1000., 0.]])
        y = np.asarray([1, 0], dtype=np.int64)
        cache = ann.forward(x, p)
        self.assertTrue(np.all(np.isfinite(cache["probabilities"])))
        np.testing.assert_allclose(cache["probabilities"].sum(axis=0), 1, atol=1e-14, rtol=0)
        expected = stable_loss_from_logits(cache["logits"], y)
        evaluated = ann.evaluate_dataset(x, y, p)
        loss, accuracy, probabilities = evaluated["loss"], evaluated["accuracy"], evaluated["probabilities"]
        self.assertEqual(expected, 1000000.)
        self.assertEqual(loss, expected)
        self.assertEqual(accuracy, 0)
        self.assertTrue(np.all(np.isfinite(probabilities)))

    def test_softmax_explicit_jacobian_vector_product_matches_finite_difference(self):
        logits = np.asarray([[0.7, -1.4], [-0.2, 0.3], [1.1, 0.8]], dtype=np.float64)
        direction = np.asarray([[0.4, -0.6], [-0.8, 0.2], [0.3, 0.5]], dtype=np.float64)
        probabilities = ann.softmax(logits)
        epsilon = 1e-6
        numerical = (ann.softmax(logits + epsilon * direction)
                     - ann.softmax(logits - epsilon * direction)) / (2 * epsilon)
        for column in range(logits.shape[1]):
            p = probabilities[:, column]
            jacobian = np.diag(p) - np.outer(p, p)
            np.testing.assert_allclose(jacobian @ direction[:, column], numerical[:, column],
                                       atol=3e-10, rtol=3e-8)
            np.testing.assert_allclose(jacobian.sum(axis=0), 0, atol=1e-14, rtol=0)

    def test_probe_gain_factorization_zero_denominators_and_unconstrained_alpha(self):
        for name in CLASSROOM9:
            with self.subTest(activation=name):
                p = self.make_parameters(name)
                scalars, _, _ = ann.probe_diagnostics(self.x, self.y, p)
                self.assertIsNotNone(scalars["probe_upstream_weight_gain"])
                self.assertIsNotNone(scalars["probe_local_jacobian_gain"])
                self.assertAlmostEqual(scalars["probe_delta1_delta2_ratio"],
                                       scalars["probe_upstream_weight_gain"]
                                       * scalars["probe_local_jacobian_gain"], places=14)
                if name == "prelu":
                    self.assertEqual(scalars["probe_alpha_negative_fraction"], 0.2)
                    self.assertEqual(scalars["probe_alpha_gt_one_fraction"], 0.2)
        p = self.make_parameters("linear")
        p.weights[1][...] = 0
        scalars, _, _ = ann.probe_diagnostics(self.x, self.y, p)
        self.assertEqual(scalars["probe_upstream_weight_gain"], 0)
        self.assertIsNone(scalars["probe_local_jacobian_gain"])
        self.assertEqual(scalars["probe_delta1_delta2_ratio"], 0)
        # Finite logits whose probabilities underflow to an exact one-hot give
        # zero output delta; undefined 0/0 diagnostics must be None, not NaN.
        p.biases[1][...] = np.asarray([1000., -1000., -1000.]).reshape(3, 1)
        scalars, _, _ = ann.probe_diagnostics(self.x, np.zeros(len(self.y), dtype=np.int64), p)
        for key in ("probe_upstream_weight_gain", "probe_local_jacobian_gain",
                    "probe_delta1_delta2_ratio"):
            self.assertIsNone(scalars[key])

    def test_copy_and_checkpoint_roundtrip_are_independent_and_exact(self):
        with TemporaryDirectory(prefix="activation_encoding_unit_") as temp:
            for name in CLASSROOM9:
                with self.subTest(activation=name):
                    p = self.make_parameters(name)
                    before = snapshot(p)
                    clone = ann.copy_parameters(p)
                    for key, original in parameter_items(p).items():
                        self.assertFalse(np.shares_memory(original, parameter_items(clone)[key]))
                    clone.weights[0][0, 0] += 1
                    self.assert_snapshot_equal(before, p)
                    path = Path(temp) / f"{name}.npz"
                    ann.save_parameters(path, p)
                    loaded = ann.load_parameters(path)
                    self.assertEqual(loaded.activation, p.activation)
                    self.assert_snapshot_equal(before, loaded)
                    np.testing.assert_array_equal(ann.forward(self.x, loaded)["probabilities"], ann.forward(self.x, p)["probabilities"])


class ReadoutTests(unittest.TestCase):
    def setUp(self):
        # Last feature is constant, and validation distribution is deliberately
        # shifted to make accidental train+validation fitting easy to detect.
        self.train = np.asarray([
            [1., 2., -1., 3., 4.], [2., 4., 0., 2., 4.],
            [3., 1., 1., 1., 4.], [4., 3., 2., 0., 4.],
            [5., 5., -2., -1., 4.], [6., 6., 3., -2., 4.],
        ])
        self.y_train = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)
        self.val = np.asarray([[101., 12., 9., 13., 4.], [102., 14., 10., 12., 4.], [103., 11., 11., 11., 4.]])
        self.y_val = np.asarray([0, 1, 2], dtype=np.int64)
        p = ann.initialize_parameters([4, 5, 3], 3790, "prelu", dtype=np.float64)
        self.initial_w = p.weights[1].copy()
        self.initial_b = p.biases[1].copy()

    def fit(self, output, standardize):
        return readout.fit_output_head(
            self.train, self.y_train, self.val, self.y_val,
            self.initial_w, self.initial_b, epochs=2, learning_rate=0.025,
            batch_size=4, seed=3790, standardize=standardize, output=output,
        )

    def test_readout_head_complete_finite_differences(self):
        w, b = self.initial_w.copy(), self.initial_b.copy()
        gradients = readout.head_gradients(self.train, self.y_train, w, b)
        epsilon = 1e-6
        for key, parameter in (("gradient_W2", w), ("gradient_b2", b)):
            numeric = np.empty_like(parameter)
            for index in np.ndindex(parameter.shape):
                original = float(parameter[index])
                try:
                    parameter[index] = original + epsilon
                    plus = stable_loss_from_logits(readout.head_logits(self.train, w, b), self.y_train)
                    parameter[index] = original - epsilon
                    minus = stable_loss_from_logits(readout.head_logits(self.train, w, b), self.y_train)
                finally:
                    parameter[index] = original
                numeric[index] = (plus - minus) / (2 * epsilon)
            np.testing.assert_allclose(gradients[key], numeric, atol=3e-8, rtol=3e-5)

    def test_complete_train_gradient_sums_before_rms_with_short_last_batch(self):
        expected = readout.head_gradients(self.train, self.y_train, self.initial_w, self.initial_b)
        evaluated = readout.evaluate_head(self.train, self.y_train, self.initial_w, self.initial_b,
                                         batch_size=4, compute_gradients=True)
        np.testing.assert_allclose(evaluated["gradient_W2"], expected["gradient_W2"], atol=1e-14, rtol=0)
        np.testing.assert_allclose(evaluated["gradient_b2"], expected["gradient_b2"], atol=1e-14, rtol=0)
        dw, db = expected["gradient_W2"], expected["gradient_b2"]
        expected_rms = float(np.sqrt((np.sum(dw ** 2) + np.sum(db ** 2)) / (dw.size + db.size)))
        self.assertAlmostEqual(evaluated["gradient_rms"], expected_rms, places=14)

    def test_standardization_uses_only_training_population_moments_and_zero_std_scale_one(self):
        train_before, val_before = self.train.copy(), self.val.copy()
        weight_before, bias_before = self.initial_w.copy(), self.initial_b.copy()
        with TemporaryDirectory(prefix="activation_readout_unit_") as temp:
            target = Path(temp) / "standardized"
            summary = self.fit(target, True)
            self.assertEqual(summary["epochs_completed"], 2)
            self.assertTrue(summary["standardize"])
            with np.load(target / "standardization.npz", allow_pickle=False) as values:
                np.testing.assert_allclose(values["mean"], self.train.mean(axis=0), atol=1e-14, rtol=0)
                np.testing.assert_allclose(values["std"], self.train.std(axis=0, ddof=0), atol=1e-14, rtol=0)
                expected_scale = self.train.std(axis=0, ddof=0)
                expected_scale[expected_scale == 0] = 1
                np.testing.assert_allclose(values["scale"], expected_scale, atol=1e-14, rtol=0)
                self.assertEqual(values["scale"][-1], 1)
                self.assertTrue(np.all(np.isfinite(values["scale"])))
            with np.load(target / "final_head.npz", allow_pickle=False) as values:
                for key in ("W2", "b2", "mean", "scale"):
                    self.assertTrue(np.all(np.isfinite(values[key])))
        np.testing.assert_array_equal(self.train, train_before)
        np.testing.assert_array_equal(self.val, val_before)
        np.testing.assert_array_equal(self.initial_w, weight_before)
        np.testing.assert_array_equal(self.initial_b, bias_before)

    def test_raw_readout_does_not_center_or_rescale_features(self):
        with TemporaryDirectory(prefix="activation_readout_unit_") as temp:
            target = Path(temp) / "raw"
            summary = self.fit(target, False)
            self.assertFalse(summary["standardize"])
            with np.load(target / "final_head.npz", allow_pickle=False) as values:
                np.testing.assert_array_equal(values["mean"], np.zeros(5))
                np.testing.assert_array_equal(values["scale"], np.ones(5))
            with np.load(target / "initial_head.npz", allow_pickle=False) as values:
                np.testing.assert_array_equal(values["W2"], self.initial_w)
                np.testing.assert_array_equal(values["b2"], self.initial_b)

    def test_extract_features_does_not_mutate_encoder_including_alpha(self):
        p = ann.initialize_parameters([4, 5, 3], 3790, "prelu", dtype=np.float64)
        before = snapshot(p)
        x = np.arange(24, dtype=np.float64).reshape(6, 4) / 24
        expected = ann.forward(x, p)["h"].T
        actual = readout.extract_features(x, p, batch_size=2)
        np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0)
        self.assertEqual(actual.shape, (6, 5))
        for key, value in parameter_items(p).items():
            np.testing.assert_array_equal(value, before[key])

    def test_existing_readout_directory_is_rejected_and_not_overwritten(self):
        with TemporaryDirectory(prefix="activation_readout_unit_") as temp:
            target = Path(temp) / "raw"
            self.fit(target, False)
            hashes = {str(path.relative_to(target)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in target.rglob("*") if path.is_file()}
            with self.assertRaises((FileExistsError, ValueError)):
                self.fit(target, False)
            after = {str(path.relative_to(target)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in target.rglob("*") if path.is_file()}
            self.assertEqual(hashes, after)


class RunnerFilesystemTests(unittest.TestCase):
    def test_timestamped_run_directories_are_unique_and_preserve_previous_run(self):
        from mnist_activation_encoding_experiments import new_run_directory

        with TemporaryDirectory(prefix="activation_runner_unit_") as temp:
            base = Path(temp)
            first = new_run_directory(base, "synthetic_unit")
            marker = first / "preserved.npy"
            np.save(marker, np.asarray([3790], dtype=np.int64))
            before = hashlib.sha256(marker.read_bytes()).hexdigest()
            second = new_run_directory(base, "synthetic_unit")
            self.assertNotEqual(first.resolve(), second.resolve())
            self.assertTrue(first.is_dir())
            self.assertTrue(second.is_dir())
            self.assertEqual(first.resolve().parent, base.resolve())
            self.assertEqual(second.resolve().parent, base.resolve())
            self.assertEqual(before, hashlib.sha256(marker.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()

