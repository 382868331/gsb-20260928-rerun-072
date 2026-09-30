"""校准 + 定点转换。不支持的层直接拒绝。

convert 接受可选的逐层位宽 bits_map（8/16/32），缺省全部 int8，
与首轮口径一致；32 表示该层保留浮点。
"""
import numpy as np
from .layers import (FP_LAYERS, CONVERTIBLE, quantize_tensor, qmax_for,
                     PRECISIONS, QMIN, QMAX)


class UnsupportedLayerError(Exception):
    pass


def run_fp_trace(layers, samples):
    """逐层前向，记录每层的输入/输出激活，用于校准与误差对比。"""
    acts = []
    x = samples
    for l in layers:
        y = FP_LAYERS[l["type"]](x, l["w"], l["b"]) if l["type"] == "dense" \
            else FP_LAYERS[l["type"]](x)
        acts.append({"name": l["name"], "type": l["type"], "input": x, "output": y})
        x = y
    return acts


def calibrate(layers, samples):
    """按校准样本记录每个激活张量的最大绝对值（对称校准的统计量）。
    具体 scale 在转换时按目标层位宽由该统计量推出，
    因此浮点层下游的定点层仍能拿到正确的输入 scale。"""
    acts = run_fp_trace(layers, samples)
    amax = {"__input__": float(np.max(np.abs(samples)))}
    for a in acts:
        amax[a["name"]] = float(np.max(np.abs(a["output"])))
    return amax


def _scale_for(amax_value, bits):
    """由校准最大值与位宽推对称 scale；浮点层 scale 固定 1.0。"""
    if bits == 32:
        return 1.0
    return amax_value / qmax_for(bits) if amax_value > 0 else 1.0


def convert(layers, calib_samples, bits_map=None):
    """返回可独立序列化的定点模型描述（dict）。"""
    bad = [l["name"] for l in layers if l["type"] not in CONVERTIBLE]
    if bad:
        raise UnsupportedLayerError(
            f"拒绝转换：不支持的层 {bad}，当前仅支持 {sorted(CONVERTIBLE)}")
    bits_map = bits_map or {}
    for n, b in bits_map.items():
        if b not in PRECISIONS:
            raise ValueError(f"层 {n} 位宽 {b} 非法，仅支持 {PRECISIONS}")

    amax = calibrate(layers, calib_samples)
    qlayers = []
    prev_name = "__input__"
    for l in layers:
        bits = bits_map.get(l["name"], 8)
        in_scale = _scale_for(amax[prev_name], bits)
        out_scale = _scale_for(amax[l["name"]], bits)
        e = {"type": l["type"], "name": l["name"],
             "bits": bits,
             "in_scale": in_scale, "out_scale": out_scale}
        if l["type"] == "dense" and bits != 32:
            w_max = float(np.max(np.abs(l["w"])))
            w_scale = w_max / qmax_for(bits) if w_max > 0 else 1.0
            wq = quantize_tensor(l["w"], w_scale, bits)
            b_scale = in_scale * w_scale  # 与累加器同 scale
            bq = np.rint(l["b"] / b_scale).astype(np.int32)
            e.update(w=wq.tolist(), b=bq.tolist(), w_scale=w_scale)
        elif l["type"] == "dense":  # 浮点层原样保留权重
            e.update(w=np.asarray(l["w"], dtype=np.float64).tolist(),
                     b=np.asarray(l["b"], dtype=np.float64).tolist(),
                     w_scale=1.0)
        qlayers.append(e)
        prev_name = l["name"]
    return {"quant_scheme": "symmetric int8/int16 per-layer (32=fp), "
                            "round-half-to-even, saturate, int32 accum",
            "in_scale": _scale_for(amax["__input__"],
                                   bits_map.get(layers[0]["name"], 8)
                                   if layers else 8),
            "in_bits": bits_map.get(layers[0]["name"], 8) if layers else 8,
            "layers": qlayers}
