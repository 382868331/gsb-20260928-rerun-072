"""独立运行器：只依赖转换产物 JSON，定点层纯整数推理，
bits=32 的层按浮点执行（混合精度）。"""
import numpy as np
from .layers import Q_LAYERS, FP_LAYERS, quantize_tensor, dequantize_tensor


def run_quant(qmodel, x):
    """x: 浮点输入。返回 (浮点输出, 逐层轨迹)。"""
    in_bits = qmodel.get("in_bits", 8)
    if in_bits == 32:
        xq = np.asarray(x, dtype=np.float64)  # 首层浮点：输入不定点化
        cur_scale = 1.0
    else:
        xq = quantize_tensor(np.asarray(x, dtype=np.float64),
                             qmodel["in_scale"], in_bits)
        cur_scale = qmodel["in_scale"]
    trace = []
    for l in qmodel["layers"]:
        bits = l.get("bits", 8)
        if bits == 32:
            # 浮点层：先反量化输入，按浮点语义执行
            xf = dequantize_tensor(xq, cur_scale) \
                if not np.issubdtype(np.asarray(xq).dtype, np.floating) \
                else np.asarray(xq, dtype=np.float64)
            if l["type"] == "dense":
                out = FP_LAYERS["dense"](xf, np.asarray(l["w"], dtype=np.float64),
                                         np.asarray(l["b"], dtype=np.float64))
            else:
                out = FP_LAYERS[l["type"]](xf)
            trace.append({"name": l["name"], "type": l["type"], "bits": bits,
                          "in_bits": in_bits,
                          "q_in": xq, "q_out": out, "out_scale": 1.0})
            xq = out
            in_bits = 32
            continue
        if not np.issubdtype(np.asarray(xq).dtype, np.integer):
            xq = quantize_tensor(xq, l["in_scale"], bits)
            in_bits = bits  # 浮点层输出在进入定点层前被重新量化
        elif cur_scale != l["in_scale"]:
            # 相邻层位宽不同：按 scale 比重量化到本层输入精度
            xq = quantize_tensor(
                xq.astype(np.float64) * cur_scale, l["in_scale"], bits)
            in_bits = bits
        if l["type"] == "dense":
            out = Q_LAYERS["dense"](
                xq, np.asarray(l["w"]),
                np.asarray(l["b"], dtype=np.int32),
                l["in_scale"], l["w_scale"], l["out_scale"], bits)
        else:
            out = Q_LAYERS[l["type"]](xq)
        trace.append({"name": l["name"], "type": l["type"], "bits": bits,
                      "in_bits": in_bits,
                      "q_in": xq, "q_out": out, "out_scale": l["out_scale"]})
        xq = out
        cur_scale = l["out_scale"]
        in_bits = bits
    last = qmodel["layers"][-1]
    out_scale = 1.0 if last.get("bits", 8) == 32 else last["out_scale"]
    return dequantize_tensor(xq, out_scale), trace
