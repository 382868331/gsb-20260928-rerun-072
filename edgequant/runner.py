"""独立运行器：只依赖转换产物 JSON，纯整数推理。

关键口径：对校准范围之外的输入，不能只返回一个看似正常的预测。
输入量化截断、激活饱和（重定标后被裁剪）、int32 累加溢出都计入逐层
风险计数，并汇总到 risk 字段随结果一起返回。恰好落在量化边界上不算饱和。
"""
import numpy as np
from .layers import (Q_LAYERS, quantize_tensor, dequantize_tensor,
                     qmin_qmax, INT32_MIN, INT32_MAX)


def run_quant(qmodel, x, return_risk=False):
    """x: 浮点输入。

    return_risk=False（默认，向后兼容）：返回 (浮点输出, 逐层轨迹)。
    return_risk=True：额外返回风险汇总 dict。
    """
    x = np.asarray(x, dtype=np.float64)
    in_scale = qmodel["in_scale"]
    xq, in_clipped = quantize_tensor(x, in_scale)
    in_max = float(np.max(np.abs(x)))
    in_limit = in_scale * 127.0
    trace = []
    acc_sat_total = 0
    for l in qmodel["layers"]:
        sat_in = in_clipped if not trace else trace[-1]["saturated_out"]
        acc_sat = 0
        if l["type"] == "dense":
            w_bits = l.get("w_bits", 8)
            w_dtype = np.int8 if w_bits == 8 else np.int32
            wq = np.asarray(l["w"], dtype=w_dtype)
            bq = np.asarray(l["b"], dtype=np.int32)
            # 用 int64 求真实累加，先检测 int32 累加溢出，再显式饱和
            acc64 = xq.astype(np.int64) @ wq.astype(np.int64) + bq.astype(np.int64)
            acc_sat = int(np.sum((acc64 < INT32_MIN) | (acc64 > INT32_MAX)))
            acc_sat_total += acc_sat
            out, _ = Q_LAYERS["dense"](
                xq, wq, bq, l["in_scale"], l["w_scale"],
                l["out_scale"], w_bits)
            # 重定标前的真实值是否越过 int8 输出边界
            rescale = (l["in_scale"] * l["w_scale"]) / l["out_scale"]
            rescaled = np.clip(acc64, INT32_MIN, INT32_MAX).astype(np.float64) \
                * rescale
            _, hi = qmin_qmax(8)
            sat_out = int(np.sum(np.abs(rescaled) > hi))
        else:
            out = Q_LAYERS[l["type"]](xq)
            _, hi = qmin_qmax(8)
            sat_out = int(np.sum(np.abs(xq.astype(np.int64)) > hi))
        trace.append({
            "name": l["name"], "type": l["type"],
            "q_in": xq, "q_out": out, "out_scale": l["out_scale"],
            "w_bits": l.get("w_bits"),
            "saturated_in": int(sat_in), "saturated_out": int(sat_out),
            "acc_int32_saturated": int(acc_sat),
        })
        xq = out
    out_scale = qmodel["layers"][-1]["out_scale"]
    y = dequantize_tensor(xq, out_scale)
    if not return_risk:
        return y, trace
    act_sat = sum(t["saturated_out"] for t in trace)
    risk = {
        "input_clipped": int(in_clipped),
        "input_out_of_calib": bool(in_max > in_limit + 1e-12),
        "input_absmax": in_max,
        "input_calib_limit": in_limit,
        "activation_saturated": int(act_sat),
        "acc_int32_saturated": int(acc_sat_total),
        "per_layer": [
            {"name": t["name"], "w_bits": t["w_bits"],
             "saturated_in": t["saturated_in"],
             "saturated_out": t["saturated_out"],
             "acc_int32_saturated": t["acc_int32_saturated"]}
            for t in trace],
    }
    risk["has_risk"] = bool(in_clipped or act_sat or acc_sat_total)
    return y, trace, risk
