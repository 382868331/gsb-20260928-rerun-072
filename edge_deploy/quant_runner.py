"""定点产物的独立整数运行器。

只读取 artifact.json：不再需要原模型与校准集。
每一步均按产物中记录的固定规则做 int32 累加与 int8 饱和，
同时统计输入量化、累加器、反量化饱和与 ReLU 负向截断次数。
"""
from __future__ import annotations

from typing import Any

from .numeric import round_half_away, saturate

ACT_QMIN, ACT_QMAX = -128, 127
ACC_MIN, ACC_MAX = -(1 << 31), (1 << 31) - 1


class ArtifactError(ValueError):
    pass


def _expect_artifact(artifact: dict[str, Any]) -> None:
    if artifact.get("conversion") != "fixed_point":
        raise ArtifactError("产物不是 fixed_point 转换结果，定点运行器拒绝执行")
    if artifact.get("rules", {}).get("rounding") != "half_away_from_zero":
        raise ArtifactError("产物舍入规则与运行器不一致，拒绝执行")


def _new_layer_stats() -> dict[str, int]:
    return {
        "input_quant_clamps": 0,
        "accumulator_clamps": 0,
        "requant_clamps": 0,
        "relu_negatives_cut": 0,
    }


def run_trace(
    artifact: dict[str, Any], x: list[float]
) -> tuple[dict[str, list[int]], dict[str, dict[str, int]]]:
    """返回每层整数输出与逐层饱和/截断计数。"""
    _expect_artifact(artifact)
    first_linear = next(
        (layer for layer in artifact["layers"] if layer["type"] == "linear"), None
    )
    expected_dim = len(first_linear["weight_int"][0]) if first_linear else len(x)
    if len(x) != expected_dim:
        raise ArtifactError(
            f"输入维度 {len(x)} 与产物声明 {expected_dim} 不一致，拒绝执行"
        )
    scale_by_name = {item["name"]: item["scale"] for item in artifact["activations"]}
    qmin_by_name = {item["name"]: item["qmin"] for item in artifact["activations"]}
    qmax_by_name = {item["name"]: item["qmax"] for item in artifact["activations"]}

    input_scale = scale_by_name["__input__"]
    current: list[int] = []
    input_clamps = 0
    for value in x:
        q, clamped = saturate(round_half_away(value / input_scale), ACT_QMIN, ACT_QMAX)
        input_clamps += int(clamped)
        current.append(q)

    trace: dict[str, list[int]] = {"__input__": current}
    stats: dict[str, dict[str, int]] = {}
    pending_input_clamps = input_clamps
    for layer in artifact["layers"]:
        layer_stats = _new_layer_stats()
        layer_stats["input_quant_clamps"] = pending_input_clamps
        if layer["type"] == "relu":
            out = []
            cut = 0
            for q in current:
                if q < 0:
                    cut += 1
                out.append(max(0, q))
            layer_stats["relu_negatives_cut"] = cut
            current = out
        elif layer["type"] == "linear":
            m = layer["requant_scale"]
            out: list[int] = []
            for row, bias_q in zip(layer["weight_int"], layer["bias_int"]):
                acc = bias_q
                for w_q, x_q in zip(row, current):
                    acc += w_q * x_q
                acc, acc_clamped = saturate(acc, ACC_MIN, ACC_MAX)
                layer_stats["accumulator_clamps"] += int(acc_clamped)
                q, out_clamped = saturate(round_half_away(m * acc), ACT_QMIN, ACT_QMAX)
                layer_stats["requant_clamps"] += int(out_clamped)
                out.append(q)
            current = out
        else:
            raise ArtifactError(f"产物包含未知层类型 {layer['type']!r}，拒绝执行")

        io_name = layer["io_spec"]["output"]
        if layer["type"] == "linear":
            assert layer["output_scale"] == scale_by_name[io_name]
        # 中间激活在产出时已按 int8 饱和，后续层不存在额外的输入量化截断。
        pending_input_clamps = 0
        trace[layer["name"]] = current
        stats[layer["name"]] = layer_stats
        _ = (qmin_by_name, qmax_by_name)
    return trace, stats


def dequantize_trace(
    artifact: dict[str, Any], trace: dict[str, list[int]]
) -> dict[str, list[float]]:
    scale_by_name = {item["name"]: item["scale"] for item in artifact["activations"]}
    return {
        name: [q * scale_by_name[name] for q in values]
        for name, values in trace.items()
    }


def run(artifact: dict[str, Any], x: list[float]) -> list[float]:
    trace, _ = run_trace(artifact, x)
    out_name = artifact["layers"][-1]["name"]
    scale = next(a["scale"] for a in artifact["activations"] if a["name"] == out_name)
    return [q * scale for q in trace[out_name]]
