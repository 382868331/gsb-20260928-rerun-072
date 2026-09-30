"""逐层误差与溢出检查 + 超校准范围边界验证。
校准范围外的输入绝不只给"正常预测"：报告保留逐层饱和计数与风险标记。"""
import numpy as np
from .layers import FP_LAYERS, qmax_for
from .quantize import run_fp_trace
from .runner import run_quant


def _sat_count(q, bits):
    if bits == 32:
        return 0  # 浮点层无定点饱和
    q = np.asarray(q)
    m = qmax_for(bits)
    return int(np.sum((q == -m) | (q == m)))


def evaluate(layers, qmodel, samples, label="calib-range"):
    fp_acts = run_fp_trace(layers, samples)
    _, q_trace = run_quant(qmodel, samples)
    report = {"label": label, "n_samples": int(len(samples)), "layers": []}
    for fa, qt in zip(fp_acts, q_trace):
        ref = fa["output"]
        got = qt["q_out"].astype(np.float64) * qt["out_scale"]
        err = got - ref
        sig = float(np.sum(ref ** 2))
        noise = float(np.sum(err ** 2))
        report["layers"].append({
            "name": fa["name"], "type": fa["type"],
            "bits": qt["bits"],
            "max_abs_err": float(np.max(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "sqnr_db": float(10 * np.log10(sig / noise)) if noise > 0 else float("inf"),
            "saturated_out": _sat_count(qt["q_out"], qt["bits"]),
            "saturated_in": _sat_count(qt["q_in"], qt.get("in_bits", qt["bits"])),
            "out_size": int(ref.size),
        })
    sat_in = sum(l["saturated_in"] for l in report["layers"])
    sat_out = sum(l["saturated_out"] for l in report["layers"])
    report["risk"] = {
        "saturated_in_total": sat_in,
        "saturated_out_total": sat_out,
        "overflow": bool(sat_in or sat_out),
        "note": "输入/输出触及定点饱和截断，预测可信度下降" if (sat_in or sat_out)
                else "未见饱和",
    }
    return report


def boundary_check(layers, qmodel, calib_samples, factor=2.0):
    """用超出校准范围的输入验证：必须给出逐层饱和/误差报告，
    而不是只看文件变小。返回 (报告, 是否出现饱和)。"""
    m = float(np.max(np.abs(calib_samples)))
    rng = np.random.default_rng(0)
    oob = rng.uniform(-factor * m, factor * m, size=calib_samples.shape)
    rep = evaluate(layers, qmodel, oob, label=f"out-of-calib x{factor}")
    overflow = rep["risk"]["overflow"]
    return rep, overflow
