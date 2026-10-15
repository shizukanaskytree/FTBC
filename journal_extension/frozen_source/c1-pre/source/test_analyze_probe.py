"""Small CPU tests for diagnostic semantics, including invalid-output reporting."""
import copy
import contextlib
import io
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import analyze_probe as probe


def capture(family, inputs, teacher, bits, proxy=None, bias=None):
    inputs = np.asarray(inputs, dtype=np.float32)
    T, n = inputs.shape
    teacher = np.asarray(teacher, dtype=np.float32)
    return {
        "metadata": {"schema_version": 1, "interface": family, "T": T,
                     "branch": "plain", "spike_encoding": "amplitude",
                     "image_ids": list(range(n)), "sample_axes": "image,position"},
        "layers": {"activation": {
            "teacher": teacher[:, None],
            "proxy": np.asarray(teacher if proxy is None else proxy, dtype=np.float32)[:, None],
            "spikes": np.asarray(bits, dtype=np.float32)[:, :, None],
            "snn_input": inputs[:, :, None],
            "alpha": np.array([1.0]), "theta": np.array([1.0]),
            "initial_mem": np.array([0.5]),
            "bias": np.asarray(bias if bias is not None else np.zeros(T))[:, None],
            "channels": [7], "spatial_positions": [[0, 0]],
        }},
    }


class ProbeTests(unittest.TestCase):
    def test_crossing_serial_path_is_not_a_final_mean_order(self):
        # Common subtracted beta=[0,2], psi=.5, theta=1.
        cap = capture("serial", [[2, 0], [-1, 1.5]], [1, 0], [[1, 0], [0, 0]], bias=[0, 2])
        result = probe.analyze_capture(cap)["layers"]["activation"]
        self.assertEqual(result["comparable_pair_fraction"], 0)
        self.assertEqual(result["greedy_disjoint_pair_excess_lower_bound"], 0)
        self.assertEqual(result["observed_order_violation_count"], 0)
        self.assertEqual(result["channels"][0]["floating_point_audit"]["fp64_threshold_reconstruction_mismatch_count"], 0)
        inputs = cap["layers"]["activation"]["snn_input"][:, :, 0]
        parallel_order, _ = probe.order_relation(inputs, "parallel")
        self.assertTrue(parallel_order[0, 1])
        bound = probe.pair_bound(np.array([1.0, 0.0]), 2, parallel_order)
        self.assertAlmostEqual(bound["greedy_disjoint_pair_excess_lower_bound"], 0.25)

    def test_overlapping_pair_sums_are_not_used(self):
        # Two reversed pairs share observation zero. Summing both is invalid.
        order = np.triu(np.ones((3, 3), dtype=bool), k=1)
        result = probe.pair_bound(np.array([1., 0., 0.]), 1, order)
        self.assertEqual(result["positive_grid_penalty_directed_pairs"], 2)
        self.assertEqual(len(result["selected_disjoint_pairs"]), 1)
        self.assertAlmostEqual(result["greedy_disjoint_pair_excess_lower_bound"], 1 / 3)
        optimum = min(np.mean((np.array(k) - [1, 0, 0]) ** 2)
                      for k in itertools.product((0, 1), repeat=3) if list(k) == sorted(k))
        self.assertLessEqual(result["grid_plus_pair_lower_bound"], optimum)
        self.assertGreater(2 / 3, optimum)

    def test_grid_relevant_penalty_can_be_zero_for_teacher_inversion(self):
        order = np.array([[False, True], [False, False]])
        result = probe.pair_bound(np.array([0.30, 0.28]), 2, order)
        self.assertEqual(result["teacher_order_conflict_directed_pairs"], 1)
        self.assertEqual(result["positive_grid_penalty_directed_pairs"], 0)
        self.assertEqual(result["greedy_disjoint_pair_excess_lower_bound"], 0)

    def test_pair_bound_against_all_ordered_grid_outputs(self):
        checked = 0
        for T in (1, 2, 3):
            order = np.triu(np.ones((4, 4), dtype=bool), k=1)
            for target in itertools.product((-0.2, 0.0, 0.45, 1.2), repeat=4):
                result = probe.pair_bound(np.array(target), T, order)
                for counts in itertools.combinations_with_replacement(range(T + 1), 4):
                    actual = np.mean((np.array(counts) / T - target) ** 2)
                    self.assertGreaterEqual(actual + 1e-12, result["grid_plus_pair_lower_bound"])
                    checked += 1
        self.assertEqual(checked, 14080)

    def test_partial_order_bound_on_all_feasible_small_outputs(self):
        inputs = np.array([[0.0, 0.5, 1.0], [1.0, -0.1, -1.0]])
        order, _ = probe.order_relation(inputs, "serial")
        for target in itertools.product((0., 0.5, 1.), repeat=3):
            result = probe.pair_bound(np.array(target), 2, order)
            for counts in itertools.product(range(3), repeat=3):
                counts = np.array(counts)
                if np.any(order & (counts[:, None] > counts[None, :])):
                    continue
                actual = np.mean((counts / 2 - target) ** 2)
                self.assertGreaterEqual(actual + 1e-12, result["grid_plus_pair_lower_bound"])

    def test_equal_drive_incompatibility_is_reported_separately(self):
        relation = np.array([[False, True], [True, False]])
        result = probe.pair_bound(np.array([1., 0.]), 2, relation)
        self.assertEqual(result["equal_coordinate_unordered_pairs"], 1)
        self.assertEqual(result["positive_grid_penalty_equal_coordinate_pairs"], 1)
        self.assertAlmostEqual(result["grid_plus_pair_lower_bound"], .25)

    def test_mse_identity_retains_negative_cross_term(self):
        cap = capture("parallel", [[-2, 2], [-2, 2]], [0, 1], [[0, 1], [0, 1]], proxy=[1, 0])
        result = probe.analyze_capture(cap)["layers"]["activation"]
        self.assertEqual(result["normalized_local_mse"], 0)
        self.assertEqual(result["normalized_proxy_teacher_mse"], 1)
        self.assertEqual(result["normalized_spike_proxy_mse"], 1)
        self.assertEqual(result["normalized_twice_cross_term"], -2)
        self.assertEqual(result["mse_identity_residual"], 0)

    def test_observed_order_violation_is_visible(self):
        cap = capture("parallel", [[0.0, 1.0], [0.0, 1.0]], [1, 0], [[1, 0], [1, 0]])
        result = probe.analyze_capture(cap)["layers"]["activation"]
        self.assertEqual(result["observed_order_violation_count"], 1)
        ch = result["channels"][0]
        self.assertFalse(ch["lower_bound_applies_to_this_observed_count_vector"])
        self.assertLess(ch["observed_mse_minus_grid_and_pair_bound"], 0)
        self.assertTrue(ch["floating_point_audit"]["observed_order_violation_witnesses"])

    def test_invalid_spike_amplitude_is_not_coerced(self):
        cap = capture("parallel", [[0], [0]], [0], [[0.4], [0]])
        result = probe.analyze_capture(cap)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["layers"]["activation"]["status"], "invalid_capture")

    def test_comparison_requires_fixed_teacher(self):
        plain = capture("parallel", [[0, 1], [0, 1]], [0, 1], [[0, 1], [0, 1]])
        spatial = copy.deepcopy(plain)
        spatial["layers"]["activation"]["teacher"][0, 0] += 0.25
        result = probe.paired_comparison(plain, spatial, probe.analyze_capture(plain), probe.analyze_capture(spatial))
        self.assertEqual(result["status"], "unpaired_or_invalid")
        self.assertIn("activation.teacher differs", result["errors"])

    def test_valid_paired_delta_is_spatial_minus_plain(self):
        plain = capture("parallel", [[0, 1], [0, 1]], [0, 1], [[0, 1], [0, 1]])
        spatial = copy.deepcopy(plain)
        spatial["layers"]["activation"]["proxy"][0, 0] = .5
        result = probe.paired_comparison(plain, spatial, probe.analyze_capture(plain), probe.analyze_capture(spatial))
        self.assertEqual(result["status"], "paired")
        self.assertEqual(result["aggregate"]["normalized_proxy_teacher_mse"], .125)

    def test_runner_tensor_schema_cli_and_no_overwrite(self):
        plain = capture("parallel", [[0, 1], [0, 1]], [0, 1], [[0, 1], [0, 1]])
        # Match the real runner: tensor/primitive dictionaries, not pickled numpy objects.
        for layer in plain["layers"].values():
            for key, value in list(layer.items()):
                if isinstance(value, np.ndarray):
                    layer[key] = torch.from_numpy(value)
        with tempfile.TemporaryDirectory(prefix="ftbc-order-analysis-test-") as tmp:
            tmp = Path(tmp)
            torch.save(plain, tmp / "plain.pt")
            torch.save(plain, tmp / "spatial.pt")
            argv = ["analyze_probe", "--plain", str(tmp / "plain.pt"),
                    "--spatial", str(tmp / "spatial.pt"), "--output", str(tmp / "result.json")]
            with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(probe.main(), 0)
            result = json.loads((tmp / "result.json").read_text())
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["paired_comparison"]["status"], "paired")
            with patch("sys.argv", argv):
                with self.assertRaises(FileExistsError):
                    probe.main()


if __name__ == "__main__":
    unittest.main(verbosity=2)
