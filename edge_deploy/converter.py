"""定点化转换器（首轮唯一支持的压缩/转换方法：W8A8 对称定点）。

固定口径（不允许逐样本或逐层更改）：
  - 权重 int8 对称 per-tensor，scale_w = max|w|/127，舍入 half-away-from-zero
  - 激活 int8 对称 per-tensor，scale 由校准集逐边界 absmax 标定
  - 偏置 int32，scale_b = scale_w * scale_x
  - 线性层：acc(int32) = sum(q_w*q_x) + q_b；q_y = sat_int8(round_half_away(M*acc))，
    M = scale_w*scale_x/scale_y
  - ReLU 在整数域执行：q_y = max(0, q_x)，输入输出共享同一刻度
仅支持层白名单 {linear, relu}；出现其它层时整单拒绝，不产出半成品产物。
"""
from __future__ import annotations

from typing import Any

from .float_runner import run_trace
from .numeric import quantize_scalar, round_half_away, symmetric_scale
from .schema import validate_model, validate_samples

CONVERSION_NAME = "fixed_point"
CONVERSION_VERSION = "w8a8-i8sym-v1"
SUPPORTED_LAYERS = frozenset({"linear", "relu"})

ACT_BITS = 8
ACT_QMIN, ACT_QMAX = -128, 127
W_QMIN, W_QMAX = -128, 127
ACC_MIN, ACC_MAX = -(1 << 31), (1 << 31) - 1


class UnsupportedLayerError(ValueError):
    """白名单之外的层：拒绝整单转换。"""


def _check_whitelist(model: dict[str, Any]) -> None:
    for layer in model["layers"]:
        if layer["type"] not in SUPPORTED_LAYERS:
            raise UnsupportedLayerError(
                f"层 {layer.get('name')!r} 类型 {layer['type']!r} 不在定点化白名单 "
                f"{sorted(SUPPORTED_LAYERS)} 内，已拒绝转换（不产出部分产物）"
            )


def _calibration_boundaries(
    model: dict[str, Any], calibration_samples: list[dict[str, Any]]
) -> list[tuple[str, float]]:
    """对网络每个张量边界（输入 + 各层输出）统计校准 absmax。"""
    boundary_names = ["__input__"] + [layer["name"] for layer in model["layers"]]
    absmax = {name: 0.0 for name in boundary_names}
    for record in calibration_samples:
        trace = run_trace(model, record["input"])
        for name in boundary_names:
            for value in trace[name]:
                absmax[name] = max(absmax[name], abs(value))
    # 零范围边界（例如恰好恒零的 ReLU 输出）给一个保守的最小刻度。
    return [(name, max(value, 1e-8)) for name, value in absmax.items()]


def convert(
    model: dict[str, Any], calibration: dict[str, Any]
) -> dict[str, Any]:
    validate_model(model)
    validate_samples(model, calibration)
    _check_whitelist(model)

    boundaries = _calibration_boundaries(model, calibration["samples"])
    act_scale = {name: symmetric_scale(absmax, ACT_QMAX) for name, absmax in boundaries}

    artifact_layers: list[dict[str, Any]] = []
    previous_name = "__input__"
    for layer in model["layers"]:
        name = layer["name"]
        io_spec = {"input": previous_name, "output": name}
        if layer["type"] == "relu":
            # ReLU 不改变刻度；输入输出必须使用同一激活刻度。
            scale = act_scale[previous_name]
            act_scale[name] = scale
            artifact_layers.append(
                {"name": name, "type": "relu", "io_spec": io_spec,
                 "scale": scale}
            )
        else:
            scale_x = act_scale[previous_name]
            scale_y = act_scale[name]
            weights = layer["weight"]
            w_absmax = max(abs(value) for row in weights for value in row)
            scale_w = symmetric_scale(w_absmax, W_QMAX)
            weight_int: list[list[int]] = []
            weight_clamps = 0
            for row in weights:
                int_row = []
                for value in row:
                    q, clamped = quantize_scalar(value, scale_w, W_QMIN, W_QMAX)
                    weight_clamps += int(clamped)
                    int_row.append(q)
                weight_int.append(int_row)
            scale_b = scale_w * scale_x
            bias_int: list[int] = []
            bias_clamps = 0
            for value in layer["bias"]:
                q, clamped = quantize_scalar(value, scale_b, ACC_MIN, ACC_MAX)
                bias_clamps += int(clamped)
                bias_int.append(q)
            artifact_layers.append(
                {
                    "name": name,
                    "type": "linear",
                    "io_spec": io_spec,
                    "weight_int": weight_int,
                    "weight_scale": scale_w,
                    "bias_int": bias_int,
                    "bias_scale": scale_b,
                    "input_scale": scale_x,
                    "output_scale": scale_y,
                    "requant_scale": scale_w * scale_x / scale_y,
                    "weight_quant_clamps": weight_clamps,
                    "bias_quant_clamps": bias_clamps,
                }
            )
        previous_name = name

    return {
        "format_version": 1,
        "source_model": model["name"],
        "conversion": CONVERSION_NAME,
        "conversion_version": CONVERSION_VERSION,
        "rules": {
            "rounding": "half_away_from_zero",
            "saturation": "clamp_to_range",
            "weight": "int8 symmetric per-tensor",
            "activation": "int8 symmetric per-tensor, calibrated absmax",
            "bias": "int32, scale=weight_scale*input_scale",
            "accumulator": "int32",
        },
        "activations": [
            {"name": name, "bits": ACT_BITS, "signed": True,
             "qmin": ACT_QMIN, "qmax": ACT_QMAX, "scale": act_scale[name],
             "calibration_absmax": next(v for n, v in boundaries if n == name)}
            for name in ["__input__"] + [layer["name"] for layer in model["layers"]]
        ],
        "layers": artifact_layers,
    }
