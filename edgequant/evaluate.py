"""逐层误差与溢出检查 + 超校准范围边界验证。"""
import numpy as np
from .layers import FP_LAYERS, QMIN, QMAX
from .quantize import run_fp_trace
from .runner import run_quant


def _sat_count(q):
    q = np.asarray(q)
    return int(np.sum((q == QMIN) | (q == QMAX)))


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
            "max_abs_err": float(np.max(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "sqnr_db": float(10 * np.log10(sig / noise)) if noise > 0 else float("inf"),
            "saturated_out": _sat_count(qt["q_out"]),
            "saturated_in": _sat_count(qt["q_in"]),
            "out_size": int(ref.size),
        })
    return report


def boundary_check(layers, qmodel, calib_samples, factor=2.0):
    """用超出校准范围的输入验证：必须给出逐层饱和/误差报告，
    而不是只看文件变小。返回 (报告, 是否出现饱和)。"""
    m = float(np.max(np.abs(calib_samples)))
    rng = np.random.default_rng(0)
    oob = rng.uniform(-factor * m, factor * m, size=calib_samples.shape)
    rep = evaluate(layers, qmodel, oob, label=f"out-of-calib x{factor}")
    overflow = any(l["saturated_in"] or l["saturated_out"] for l in rep["layers"])
    return rep, overflow

