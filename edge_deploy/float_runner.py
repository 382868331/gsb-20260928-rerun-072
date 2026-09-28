"""浮点参考模型执行器（计算口径基准）。"""
from __future__ import annotations

from typing import Any


def run_layer(layer: dict[str, Any], x: list[float]) -> list[float]:
    ltype = layer["type"]
    if ltype == "linear":
        return [
            sum(weight_value * xi for weight_value, xi in zip(row, x)) + bias_value
            for row, bias_value in zip(layer["weight"], layer["bias"])
        ]
    if ltype == "relu":
        return [max(0.0, value) for value in x]
    raise ValueError(f"不支持的层类型: {ltype}")


def run_trace(model: dict[str, Any], x: list[float]) -> dict[str, list[float]]:
    """返回输入与每一层输出（key 为层名），用于逐层误差核对。"""
    trace: dict[str, list[float]] = {"__input__": [float(value) for value in x]}
    current = [float(value) for value in x]
    for layer in model["layers"]:
        current = run_layer(layer, current)
        trace[layer["name"]] = current
    return trace


def run(model: dict[str, Any], x: list[float]) -> list[float]:
    return run_trace(model, x)[model["layers"][-1]["name"]]
