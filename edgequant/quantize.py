"""校准 + 逐层混合精度转换。

口径：激活统一对称 int8；权重逐层在 4bit / 8bit 中选择。
在用户给出的允许误差（逐层 max_abs_err 上限）与存储预算（权重+偏置
总比特数）下做贪心选择；预算装不下或最高精度仍超误差时显式失败，
绝不悄悄降低标准。不支持的层直接拒绝。
"""
import numpy as np
from .layers import (FP_LAYERS, CONVERTIBLE, quantize_tensor,
                     WEIGHT_BIT_CHOICES, qmin_qmax)


class UnsupportedLayerError(Exception):
    pass


class PrecisionTargetError(Exception):
    """无法在给定误差/预算下完成转换。"""


def run_fp_trace(layers, samples):
    """逐层前向，记录每层的输入/输出激活，用于校准与误差对比。"""
    acts = []
    x = samples
    for l in layers:
        y = FP_LAYERS[l["type"]](x, l["w"], l["b"]) if l["type"] == "dense" \
            else FP_LAYERS[l["type"]](x)
        acts.append({"name": l["name"], "type": l["type"],
                     "input": x, "output": y})
        x = y
    return acts


def calibrate(layers, samples):
    """按校准样本的最大绝对值确定每个激活张量的 scale（对称）。"""
    acts = run_fp_trace(layers, samples)
    scales = {}
    in_max = float(np.max(np.abs(samples)))
    scales["__input__"] = in_max / 127.0 if in_max > 0 else 1.0
    for a in acts:
        m = float(np.max(np.abs(a["output"])))
        scales[a["name"]] = m / 127.0 if m > 0 else 1.0
    return scales


def _quantize_dense_layer(layer, in_scale, out_scale, bits):
    w = layer["w"]
    _, w_hi = qmin_qmax(bits)
    w_max = float(np.max(np.abs(w)))
    w_scale = w_max / w_hi if w_max > 0 else 1.0
    wq, w_clipped = quantize_tensor(w, w_scale, bits)
    b_scale = in_scale * w_scale
    bq = np.rint(layer["b"] / b_scale).astype(np.int32)
    return {"w": wq, "b": bq, "w_scale": w_scale,
            "w_bits": bits, "w_clipped": w_clipped}


def _assemble_qmodel(scales, layers, bits_for):
    qlayers = []
    prev_scale = scales["__input__"]
    for l in layers:
        e = {"type": l["type"], "name": l["name"],
             "in_scale": prev_scale, "out_scale": scales[l["name"]]}
        if l["type"] == "dense":
            qd = _quantize_dense_layer(l, prev_scale,
                                       scales[l["name"]],
                                       bits_for[l["name"]])
            e.update({"w": qd["w"].tolist(), "b": qd["b"].tolist(),
                      "w_scale": qd["w_scale"], "w_bits": qd["w_bits"]})
        else:
            e["w_bits"] = None
        qlayers.append(e)
        prev_scale = scales[l["name"]]
    return {"quant_scheme": "mixed-weight [4,8]bit + act int8, "
                            "round-half-to-even, saturate",
            "in_scale": scales["__input__"], "layers": qlayers}


def _storage_cost(layers, bits_for):
    """权重 + 偏置总比特数。"""
    total = 0
    per = {}
    for l in layers:
        if l["type"] != "dense":
            per[l["name"]] = 0
            continue
        c = int(l["w"].size) * bits_for[l["name"]] + 32 * int(l["b"].size)
        per[l["name"]] = c
        total += c
    return total, per


def _layer_max_errs(layers, qmodel, samples):
    """完整前向，逐层 max_abs_err（含整网误差传播）。"""
    from .runner import run_quant
    fp_acts = run_fp_trace(layers, samples)
    _, trace = run_quant(qmodel, samples)
    errs = {}
    for fa, qt in zip(fp_acts, trace):
        ref = fa["output"]
        got = qt["q_out"].astype(np.float64) * qt["out_scale"]
        errs[fa["name"]] = float(np.max(np.abs(got - ref)))
    return errs


def _select_precision(layers, samples, scales, tol_max_abs, storage_budget):
    """贪心：全部从最低 4bit 起步，对超误差层按误差收益/额外比特升级。"""
    dense_names = [l["name"] for l in layers if l["type"] == "dense"]
    bits_for = {l["name"]: WEIGHT_BIT_CHOICES[0] for l in layers}
    all8 = {l["name"]: 8 for l in layers}

    # 只有带权重的 dense 层可选择精度；relu 等无权重层的误差来自激活 int8
    # 粒度，固定不可降，仍在报告中披露但不作为权重选择的达标条件。
    errs8 = _layer_max_errs(layers, _assemble_qmodel(scales, layers, all8),
                            samples)
    bad8 = {n: errs8[n] for n in dense_names if errs8[n] > tol_max_abs}
    if bad8:
        raise PrecisionTargetError(
            f"误差要求无法达到：允许各 dense 层 max_abs_err <= {tol_max_abs}，"
            f"全部 8bit 后仍超差的层 {bad8}")

    errs = _layer_max_errs(layers,
                           _assemble_qmodel(scales, layers, bits_for), samples)
    while any(errs[n] > tol_max_abs for n in dense_names):
        best = None
        cost_now, _ = _storage_cost(layers, bits_for)
        for n in dense_names:
            if bits_for[n] >= 8:
                continue
            trial = dict(bits_for)
            trial[n] = 8
            cost_new, _ = _storage_cost(layers, trial)
            extra = cost_new - cost_now
            if storage_budget is not None and cost_new > storage_budget:
                continue
            trial_errs = _layer_max_errs(
                layers, _assemble_qmodel(scales, layers, trial), samples)
            gain = max(errs[n2] for n2 in dense_names) - \
                max(trial_errs[n2] for n2 in dense_names)
            score = gain / extra if extra > 0 else float("inf")
            if best is None or score > best[0]:
                best = (score, n, trial, trial_errs)
        if best is None:
            raise PrecisionTargetError(
                "无法在存储预算内满足误差要求：没有可升级的层能继续降低误差，"
                f"当前逐层误差 {errs}，预算 {storage_budget} bits")
        _, _, bits_for, errs = best

    qmodel = _assemble_qmodel(scales, layers, bits_for)
    total_cost, per_cost = _storage_cost(layers, bits_for)
    return qmodel, {"per_layer_bits": per_cost, "total_bits": total_cost,
                    "budget_bits": storage_budget,
                    "tol_max_abs": tol_max_abs,
                    "final_layer_max_abs": errs}


def convert(layers, calib_samples, tol_max_abs=1e-3,
            storage_budget=None, weight_bits=None):
    """返回可独立序列化的定点模型描述（dict，含逐层选择报告）。

    - weight_bits=None：在 tol_max_abs / storage_budget 约束下自动混合精度；
    - weight_bits=4|8：强制所有 dense 层同一精度（仍校验误差与预算）。
    """
    bad = [l["name"] for l in layers if l["type"] not in CONVERTIBLE]
    if bad:
        raise UnsupportedLayerError(
            f"拒绝转换：不支持的层 {bad}，当前仅支持 {sorted(CONVERTIBLE)}")

    scales = calibrate(layers, calib_samples)
    if weight_bits is not None:
        if weight_bits not in WEIGHT_BIT_CHOICES:
            raise ValueError(f"weight_bits 仅支持 {WEIGHT_BIT_CHOICES}")
        bits_for = {l["name"]: weight_bits for l in layers}
        qmodel = _assemble_qmodel(scales, layers, bits_for)
        errs = _layer_max_errs(layers, qmodel, calib_samples)
        total_cost, per_cost = _storage_cost(layers, bits_for)
        bad = {n: errs[n] for n in (x["name"] for x in layers
                                    if x["type"] == "dense")
               if errs[n] > tol_max_abs}
        if bad:
            raise PrecisionTargetError(
                f"强制 {weight_bits}bit 不满足误差要求 {tol_max_abs}：{bad}")
        if storage_budget is not None and total_cost > storage_budget:
            raise PrecisionTargetError(
                f"强制 {weight_bits}bit 代价 {total_cost} bits 超过预算 "
                f"{storage_budget} bits")
        report = {"per_layer_bits": per_cost, "total_bits": total_cost,
                  "budget_bits": storage_budget,
                  "tol_max_abs": tol_max_abs,
                  "final_layer_max_abs": errs}
    else:
        qmodel, report = _select_precision(
            layers, calib_samples, scales, tol_max_abs, storage_budget)
    qmodel["mixed_precision_report"] = report
    return qmodel
