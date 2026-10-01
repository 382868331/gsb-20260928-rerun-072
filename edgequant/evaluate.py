"""逐层误差与饱和/风险检查 + 超校准范围边界验证 + 压力对比 + 复核。"""
import numpy as np
from .layers import FP_LAYERS, QMIN, QMAX
from .quantize import run_fp_trace, _group_layer_errs, _violations
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


def reverify(layers, qmodel, groups, boundary_factor=2.0, seed=0):
    """产物重载后的重复核对：

    - 逐输入组重新预测，逐层 max_abs_err 与产物中记录逐元素重算核对；
    - 每组独立判定是否满足其原始容差，不以平均掩盖失效；
    - 超校准范围输入单独跑边界验证，必须报告裁剪/饱和与风险；
    - 校准来源（provenance）原样回显，保持可追溯。
    返回 (核对报告, 全部通过标志)。
    """
    from .model import normalize_groups
    groups = normalize_groups(groups)
    dense_names = [l["name"] for l in layers if l["type"] == "dense"]
    recorded = {g["name"]: {p["layer"]: p
                            for p in g["per_layer"]}
                for g in
                qmodel["mixed_precision_report"]["groups"]}
    group_checks = []
    all_ok = True
    for name, samples, tol_default in groups:
        rep = evaluate(layers, qmodel, samples, label=f"reverify:{name}")
        # 重新预测并逐层核对误差
        ge = _group_layer_errs(layers, qmodel,
                               [(name, samples, tol_default)], dense_names)
        tol_for = {name: {n: (tol_default if tol_default is not None
                              else recorded.get(name, {})
                              .get(n, {}).get("tol_max_abs"))
                          for n in dense_names}}
        for n in dense_names:
            if tol_for[name][n] is None:
                tol_for[name][n] = recorded[name][n]["tol_max_abs"]
        violations = _violations(ge, tol_for)
        matches_record = True
        mismatch = []
        if name in recorded:
            for n in dense_names:
                old = recorded[name][n]["max_abs_err"]
                new = ge[name][n]
                if abs(old - new) > 1e-12:
                    matches_record = False
                    mismatch.append({"layer": n, "recorded": old,
                                     "recomputed": new})
        else:
            matches_record = False
        # 组复核只判定：原始容差是否仍满足、重载后预测是否与产物记录一致。
        # 饱和/裁剪属于风险披露，组内风险计数照常附带，不作为达标否决项；
        # 超校准范围的风险由专门的 out_of_calib 段强制要求报告。
        ok = (not violations) and matches_record
        all_ok = all_ok and ok
        group_checks.append({
            "name": name, "n_samples": int(len(samples)),
            "tol_max_abs": {n: tol_for[name][n] for n in dense_names},
            "per_layer_max_abs": ge[name],
            "violations": violations,
            "matches_recorded_prediction": matches_record,
            "mismatch": mismatch,
            "risk": rep["risk"],
            "satisfied": bool(ok),
        })

    # 超校准范围：基于全部组的并集范围构造越界输入
    m = max(float(np.max(np.abs(s))) for _, s, _ in groups)
    width = groups[0][1].shape[1]
    rng = np.random.default_rng(seed)
    oob = rng.uniform(-boundary_factor * m, boundary_factor * m,
                      (32, width))
    boundary = evaluate(layers, qmodel, oob,
                        label=f"reverify:out-of-calib x{boundary_factor}")
    oob_risk = boundary["risk"]
    oob_flag = oob_risk["has_risk"] and \
        oob_risk["input_out_of_calib"] and \
        oob_risk["input_clipped"] > 0

    report = {
        "groups": group_checks,
        "all_group_constraints_satisfied":
            all(g["satisfied"] for g in group_checks),
        "out_of_calib": {
            "factor": boundary_factor,
            "input_clipped": oob_risk["input_clipped"],
            "activation_saturated": oob_risk["activation_saturated"],
            "acc_int32_saturated": oob_risk["acc_int32_saturated"],
            "input_out_of_calib": oob_risk["input_out_of_calib"],
            "risk_reported": bool(oob_risk["has_risk"]),
            "layers": boundary["layers"],
        },
        "out_of_calib_risk_reported": bool(oob_flag),
        "provenance": qmodel.get("provenance"),
    }
    report["passed"] = bool(report["all_group_constraints_satisfied"]
                            and report["out_of_calib_risk_reported"])
    return report, report["passed"]


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
