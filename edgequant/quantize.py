"""校准 + 定点转换。不支持的层直接拒绝。"""
import numpy as np
from .layers import FP_LAYERS, CONVERTIBLE, quantize_tensor, QMIN, QMAX


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
    """按校准样本的最大绝对值确定每个激活张量的 scale（对称）。"""
    acts = run_fp_trace(layers, samples)
    scales = {}
    in_max = float(np.max(np.abs(samples)))
    scales["__input__"] = in_max / 127.0 if in_max > 0 else 1.0
    for a in acts:
        m = float(np.max(np.abs(a["output"])))
        scales[a["name"]] = m / 127.0 if m > 0 else 1.0
    return scales


def convert(layers, calib_samples):
    """返回可独立序列化的定点模型描述（dict）。"""
    bad = [l["name"] for l in layers if l["type"] not in CONVERTIBLE]
    if bad:
        raise UnsupportedLayerError(
            f"拒绝转换：不支持的层 {bad}，当前仅支持 {sorted(CONVERTIBLE)}")

    scales = calibrate(layers, calib_samples)
    qlayers = []
    prev_scale = scales["__input__"]
    for l in layers:
        e = {"type": l["type"], "name": l["name"],
             "in_scale": prev_scale, "out_scale": scales[l["name"]]}
        if l["type"] == "dense":
            w_max = float(np.max(np.abs(l["w"])))
            w_scale = w_max / 127.0 if w_max > 0 else 1.0
            wq = quantize_tensor(l["w"], w_scale)
            b_scale = prev_scale * w_scale  # 与累加器同 scale
            bq = np.rint(l["b"] / b_scale).astype(np.int32)
            e.update(w=wq.tolist(), b=bq.tolist(), w_scale=w_scale)
        qlayers.append(e)
        prev_scale = scales[l["name"]]
    return {"quant_scheme": "symmetric-int8, round-half-to-even, saturate",
            "in_scale": scales["__input__"], "layers": qlayers}

