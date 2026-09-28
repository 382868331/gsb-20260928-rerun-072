"""定点化研究系统首轮端到端测试（纯标准库 unittest）。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge_deploy.converter import (
    SUPPORTED_LAYERS,
    convert,
)
from edge_deploy.evaluate import evaluate_split
from edge_deploy.float_runner import run as float_run
from edge_deploy.numeric import round_half_away
from edge_deploy.quant_runner import run as quant_run
from edge_deploy.sample_data import build_demo_datasets, build_demo_model


MODEL = build_demo_model()
DATASETS = build_demo_datasets(MODEL)


def _calibration_payload():
    return {
        "split": "calibration",
        "samples": [{"id": str(i), "input": x} for i, x in enumerate(DATASETS["calibration"])],
    }


class FixedPointTests(unittest.TestCase):
    def setUp(self):
        self.artifact = convert(MODEL, _calibration_payload())

    def test_rounding_rule_is_half_away_from_zero(self):
        self.assertEqual(round_half_away(0.5), 1)
        self.assertEqual(round_half_away(1.5), 2)
        self.assertEqual(round_half_away(-0.5), -1)
        self.assertEqual(round_half_away(-1.5), -2)
        self.assertEqual(round_half_away(0.49), 0)

    def test_quantized_outputs_close_in_calibration_range(self):
        for x in DATASETS["test_in_range"]:
            ref = float_run(MODEL, x)
            got = quant_run(self.artifact, x)
            for ref_value, got_value in zip(ref, got):
                self.assertLess(abs(ref_value - got_value), 0.15)

    def test_out_of_range_is_checked_and_reports_saturation(self):
        payload = {
            "split": "test_out_of_range",
            "samples": [
                {"id": f"ood-{i}", "input": x}
                for i, x in enumerate(DATASETS["test_out_of_range"])
            ],
        }
        report = evaluate_split(MODEL, self.artifact, payload)
        self.assertGreater(report["sample_count"], 0)
        self.assertGreater(sum(report["overflow_counts"].values()), 0)
        self.assertTrue(report["any_saturation"])
        self.assertIn("fc3", report["boundary_errors"])

    def test_artifact_executes_standalone_without_model(self):
        x = DATASETS["test_out_of_range"][0]
        output_via_artifact = quant_run(self.artifact, x)
        minimal_artifact = {
            k: v
            for k, v in self.artifact.items()
            if k in {"conversion", "conversion_version", "rules", "layers", "activations"}
        }
        output_minimal = quant_run(minimal_artifact, x)
        self.assertEqual(output_via_artifact, output_minimal)

    def test_unsupported_layer_is_rejected_entirely(self):
        bad_model = {
            "name": "bad",
            "input_dim": 2,
            "layers": [
                {
                    "name": "sigmoid1",
                    "type": "sigmoid",
                    "weight": [[1.0, 0.0]],
                    "bias": [0.0],
                }
            ],
        }
        # sigmoid 在模型结构校验阶段即不合法。
        from edge_deploy.schema import SchemaError, validate_model

        with self.assertRaises(SchemaError):
            validate_model(bad_model)

    def test_whitelist_contents(self):
        self.assertEqual(SUPPORTED_LAYERS, frozenset({"linear", "relu"}))

    def test_unknown_artifact_layer_refused(self):
        from edge_deploy.quant_runner import ArtifactError

        artifact = {
            "format_version": 1,
            "conversion": "fixed_point",
            "conversion_version": "test",
            "rules": {"rounding": "half_away_from_zero"},
            "activations": [
                {"name": "__input__", "scale": 0.01, "qmin": -128, "qmax": 127},
                {"name": "softmax1", "scale": 0.01, "qmin": -128, "qmax": 127},
            ],
            "layers": [{"name": "softmax1", "type": "softmax", "io_spec": {"input": "__input__", "output": "softmax1"}}],
        }
        with self.assertRaises(ArtifactError):
            quant_run(artifact, [1.0])


if __name__ == "__main__":
    unittest.main()
