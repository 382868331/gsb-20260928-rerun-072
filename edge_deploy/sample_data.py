"""构造演示用小型数值模型与三类样本：校准、域内测试、超校准范围（OOD）。

模型为 3-输入、两层 hidden 的 MLP：linear(3x8) -> relu -> linear(8x8) -> relu -> linear(8x2)。
样本由固定伪随机序列生成，保证可复现；OOD 样本幅度与符号组合刻意超出校准范围。
"""
from __future__ import annotations

import math
from typing import Any

from .float_runner import run
from .io_utils import dataset_path, model_path, write_json
from .schema import validate_model, validate_samples


def _lcg(seed: int):
    state = seed & 0xFFFFFFFF

    def draw() -> float:
        nonlocal state
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        return state / 0x7FFFFFFF * 2.0 - 1.0  # [-1, 1)

    return draw


def _linear_layer(name: str, fan_in: int, fan_out: int, draw) -> dict[str, Any]:
    weight = [[0.25 * draw() for _ in range(fan_in)] for _ in range(fan_out)]
    bias = [0.1 * draw() for _ in range(fan_out)]
    return {"name": name, "type": "linear", "weight": weight, "bias": bias}


def build_demo_model() -> dict[str, Any]:
    draw = _lcg(20260929)
    model = {
        "name": "mlp_demo",
        "input_dim": 3,
        "description": "3->8->8->2 的小型数值 MLP，用于定点化首轮研究",
        "layers": [
            _linear_layer("fc1", 3, 8, draw),
            {"name": "act1", "type": "relu"},
            _linear_layer("fc2", 8, 8, draw),
            {"name": "act2", "type": "relu"},
            _linear_layer("fc3", 8, 2, draw),
        ],
    }
    validate_model(model)
    return model


def _samples_with_labels(
    model: dict[str, Any], inputs: list[list[float]], tag: str
) -> dict[str, Any]:
    records = []
    for index, x in enumerate(inputs):
        records.append(
            {
                "id": f"{tag}-{index:03d}",
                "input": x,
                "expected_output": run(model, x),
            }
        )
    return {"model": model["name"], "split": tag, "samples": records}


def build_demo_datasets(model: dict[str, Any]) -> dict[str, list[list[float]]]:
    draw_x = _lcg(77)
    calibration = [[draw_x() for _ in range(3)] for _ in range(32)]
    in_domain = [[0.8 * draw_x() for _ in range(3)] for _ in range(16)]
    # 超校准范围：幅度 3x~6x，含同号叠加，专门制造饱和压力。
    out_of_range = [
        [4.0, -4.0, 4.0],
        [-6.0, -6.0, -6.0],
        [3.5, 5.2, -4.8],
        [0.2, 6.0, 0.2],
        [-5.5, 3.3, 5.9],
        [6.0, 6.0, -6.0],
    ]
    return {
        "calibration": calibration,
        "test_in_range": in_domain,
        "test_out_of_range": out_of_range,
    }


def materialize_demo(root: str) -> None:
    model = build_demo_model()
    write_json(model_path(root, model["name"]), model)
    for split, inputs in build_demo_datasets(model).items():
        payload = _samples_with_labels(model, inputs, split)
        validate_samples(model, payload)
        write_json(dataset_path(root, model["name"], split), payload)
