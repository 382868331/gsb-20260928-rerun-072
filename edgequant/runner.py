"""独立运行器：只依赖转换产物 JSON，纯整数推理。"""
import numpy as np
from .layers import Q_LAYERS, quantize_tensor, dequantize_tensor


def run_quant(qmodel, x):
    """x: 浮点输入。返回 (浮点输出, 逐层轨迹)。"""
    xq = quantize_tensor(np.asarray(x, dtype=np.float64), qmodel["in_scale"])
    trace = []
    for l in qmodel["layers"]:
        if l["type"] == "dense":
            out = Q_LAYERS["dense"](
                xq, np.asarray(l["w"], dtype=np.int8),
                np.asarray(l["b"], dtype=np.int32),
                l["in_scale"], l["w_scale"], l["out_scale"])
        else:
            out = Q_LAYERS[l["type"]](xq)
        trace.append({"name": l["name"], "type": l["type"],
                      "q_in": xq, "q_out": out, "out_scale": l["out_scale"]})
        xq = out
    return dequantize_tensor(xq, qmodel["layers"][-1]["out_scale"]), trace

