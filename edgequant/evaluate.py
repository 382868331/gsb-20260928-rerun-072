"""逐层误差与饱和/风险检查 + 超校准范围边界验证 + 压力对比。"""
import numpy as np
from .layers import FP_LAYERS, QMIN, QMAX
from .quantize import run_fp_trace
from .runner import run_quant


def _sat_count(q):
    q = np.asarray(q)
    return int(np.sum((q == QMIN) | (q == QMAX)))


def evaluate(layers, qmodel, samples, label="calib-range"):
    """逐层误差报告，并附带每层饱和/累加溢出与整批输入风险汇总。"""
    fp_acts = run_fp_trace(layers, samples)
    _, q_trace, risk = run_quant(qmodel, samples, return_risk=True)
    report = {"label": label, "n_samples": int(len(samples)),
              "risk": risk, "layers": []}
    risk_by = {r["name"]: r for r in risk["per_layer"]}
    for fa, qt in zip(fp_acts, q_trace):
        ref = fa["output"]
        got = qt["q_out"].astype(np.float64) * qt["out_scale"]
        err = got - ref
        sig = float(np.sum(ref ** 2))
        noise = float(np.sum(err ** 2))
        rk = risk_by[fa["name"]]
        report["layers"].append({
            "name": fa["name"], "type": fa["type"],
            "w_bits": qt.get("w_bits"),
            "max_abs_err": float(np.max(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "sqnr_db": float(10 * np.log10(sig / noise))
                      if noise > 0 else float("inf"),
            "saturated_out": rk["saturated_out"],
            "saturated_in": rk["saturated_in"],
            "acc_int32_saturated": rk["acc_int32_saturated"],
            "out_size": int(ref.size),
        })
    return report


def boundary_check(layers, qmodel, calib_samples, factor=2.0):
    """用超出校准范围的输入验证：必须给出逐层饱和/误差与风险标记，
    而不是只看文件变小或拿到一个看似正常的预测。
    返回 (报告, 是否出现风险)。"""
    m = float(np.max(np.abs(calib_samples)))
    rng = np.random.default_rng(0)
    oob = rng.uniform(-factor * m, factor * m, size=calib_samples.shape)
    rep = evaluate(layers, qmodel, oob, label=f"out-of-calib x{factor}")
    has_risk = rep["risk"]["has_risk"] or rep["risk"]["input_out_of_calib"]
    return rep, has_risk


def fp_forward(layers, x):
    """纯浮点参考前向，供压力对比使用。"""
    h = np.asarray(x, dtype=np.float64)
    for l in layers:
        h = FP_LAYERS[l["type"]](h, l["w"], l["b"]) if l["type"] == "dense" \
            else FP_LAYERS[l["type"]](h)
    return h


def stress_compare(layers, qmodel, stress_inputs, label="stress"):
    """对包含大累加/不同尺度通道/异常值的输入，比较转换前后输出，
    并保留量化侧饱和与风险信息。"""
    x = np.asarray(stress_inputs, dtype=np.float64)
    fp_out = fp_forward(layers, x)
    q_out, _, risk = run_quant(qmodel, x, return_risk=True)
    diff = q_out - fp_out
    return {
        "label": label, "n_samples": int(len(x)),
        "fp_output": fp_out.tolist(),
        "quant_output": q_out.tolist(),
        "max_abs_err": float(np.max(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "risk": risk,
    }
