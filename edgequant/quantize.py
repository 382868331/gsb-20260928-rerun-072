"""多输入组校准 + 逐层混合精度转换 + 方案对比。

口径：激活统一对称 int8；权重逐层在 4bit / 8bit 中选择。
多个代表性输入组各自带逐层 max_abs_err 上限，共享一份存储预算；
选择必须让每一组的每一层都达标，不允许用组间平均掩盖某一组失效。
预算装不下或最高精度仍有组超差时显式失败，绝不悄悄放宽标准。
不支持的层直接拒绝。
"""
import hashlib
import json
import time
import numpy as np
from .layers import (FP_LAYERS, CONVERTIBLE,
                     WEIGHT_BIT_CHOICES, qmin_qmax, quantize_tensor)


class UnsupportedLayerError(Exception):
    pass


class PrecisionTargetError(Exception):
    """无法在给定误差/预算下完成转换。"""


def _fingerprint(groups):
    """对校准输入组内容取指纹，保证旧方案可追溯到校准来源。"""
    h = hashlib.sha256()
    for name, s, tol in groups:
        h.update(name.encode())
        h.update(np.ascontiguousarray(s, dtype=np.float64).tobytes())
        h.update(json.dumps(tol, sort_keys=True).encode())
    return h.hexdigest()[:16]


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


def calibrate(layers, groups):
    """所有输入组共同决定 scale（取各组逐层激活最大绝对值的并集上界）。

    一份定点产物只能有一组 scale，因此多组共享由全部组覆盖范围确定的
    对称 scale；逐组差异通过逐组误差约束体现，而不是各自定标。
    """
    if isinstance(groups, np.ndarray) or (
            isinstance(groups, (list, tuple)) and groups and
            not isinstance(groups[0], tuple)):
        from .model import normalize_groups
        groups = normalize_groups(groups)
    in_max = max(float(np.max(np.abs(s))) for _, s, _ in groups)
    out_max = {l["name"]: 0.0 for l in layers}
    for _, samples, _ in groups:
        for a in run_fp_trace(layers, samples):
            m = float(np.max(np.abs(a["output"])))
            out_max[a["name"]] = max(out_max[a["name"]], m)
    scales = {}
    scales["__input__"] = in_max / 127.0 if in_max > 0 else 1.0
    for name, m in out_max.items():
        scales[name] = m / 127.0 if m > 0 else 1.0
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


def _group_layer_errs(layers, qmodel, groups, dense_names):
    """逐输入组 × 逐层（dense）的 max_abs_err，不取任何平均。"""
    out = {}
    for name, samples, _ in groups:
        errs = _layer_max_errs(layers, qmodel, samples)
        out[name] = {n: errs[n] for n in dense_names}
    return out


def _violations(group_errs, tol_for):
    """返回未达标清单 [{group, layer, err, tol}]（逐组披露，不平均）。"""
    v = []
    for gname, errs in group_errs.items():
        for lname, e in errs.items():
            if e > tol_for[gname][lname] + 1e-15:
                v.append({"group": gname, "layer": lname,
                          "max_abs_err": e, "tol": tol_for[gname][lname]})
    return v


def _worst_margin(group_errs, tol_for):
    """所有 (组,层) 中 (err - tol) 的最大值；>0 表示仍有约束未满足。"""
    return max(e - tol_for[g][l]
               for g, errs in group_errs.items()
               for l, e in errs.items())


def _select_precision(layers, groups, scales, tol_for, storage_budget):
    """贪心：全部从最低 4bit 起步，对超差（组,层）按最坏裕度收益/额外
    比特升级。任一组任一层不达标都视为失败，不做组间平均。"""
    dense_names = [l["name"] for l in layers if l["type"] == "dense"]
    bits_for = {l["name"]: WEIGHT_BIT_CHOICES[0] for l in layers}
    all8 = {l["name"]: 8 for l in layers}

    # relu 等无权重层的误差来自激活 int8 粒度，固定不可降，照常逐组披露，
    # 但不作为权重精度选择的达标条件。
    q8 = _assemble_qmodel(scales, layers, all8)
    ge8 = _group_layer_errs(layers, q8, groups, dense_names)
    bad8 = _violations(ge8, tol_for)
    if bad8:
        raise PrecisionTargetError(
            "误差要求无法达到：全部权重 8bit 后仍有输入组/层超差（不自动"
            "放宽目标）：%s" % bad8)

    q_now = _assemble_qmodel(scales, layers, bits_for)
    group_errs = _group_layer_errs(layers, q_now, groups, dense_names)
    while _violations(group_errs, tol_for):
        best = None
        cost_now, _ = _storage_cost(layers, bits_for)
        margin_now = _worst_margin(group_errs, tol_for)
        for n in dense_names:
            if bits_for[n] >= 8:
                continue
            trial = dict(bits_for)
            trial[n] = 8
            cost_new, _ = _storage_cost(layers, trial)
            extra = cost_new - cost_now
            if storage_budget is not None and cost_new > storage_budget:
                continue
            trial_errs = _group_layer_errs(
                layers, _assemble_qmodel(scales, layers, trial),
                groups, dense_names)
            # 收益：最坏（组,层）超差裕度下降了多少；只有真正降低最坏
            # 裕度的升级才会被考虑（先把最严重的违约压下去）。
            gain = margin_now - _worst_margin(trial_errs, tol_for)
            if gain <= 0:
                continue
            score = gain / extra if extra > 0 else float("inf")
            if best is None or score > best[0]:
                best = (score, n, trial, trial_errs, cost_new)
        if best is None:
            raise PrecisionTargetError(
                "无法在共同存储预算内满足全部输入组的误差要求（不自动放宽"
                "目标）：预算 %s bits，未达标项 %s"
                % (storage_budget, _violations(group_errs, tol_for)))
        _, _, bits_for, group_errs, _ = best

    qmodel = _assemble_qmodel(scales, layers, bits_for)
    total_cost, per_cost = _storage_cost(layers, bits_for)
    return qmodel, bits_for, group_errs, total_cost, per_cost


def _build_report(layers, groups, bits_for, group_errs, total_cost,
                  per_cost, tol_for, storage_budget, strategy, scales):
    """组装逐组达标报告：每一组独立说明，绝不以平均掩盖失效。"""
    group_reports = []
    for name, samples, tol_default in groups:
        tol = tol_for[name]
        per_layer = []
        all_ok = True
        for l in layers:
            if l["type"] != "dense":
                continue
            e = group_errs[name][l["name"]]
            ok = e <= tol[l["name"]] + 1e-15
            all_ok = all_ok and ok
            per_layer.append({"layer": l["name"],
                              "w_bits": bits_for[l["name"]],
                              "max_abs_err": e,
                              "tol_max_abs": tol[l["name"]],
                              "satisfied": bool(ok)})
        group_reports.append({
            "name": name, "n_samples": int(len(samples)),
            "tol_max_abs": tol_default,
            "satisfied": bool(all_ok and all(p["satisfied"]
                                             for p in per_layer)),
            "per_layer": per_layer,
            "max_abs_err_overall":
                max(p["max_abs_err"] for p in per_layer),
        })
    all_group_names = [g[0] for g in groups]
    errs_flat = {n: max(group_errs[g][n] for g in all_group_names)
                 for n in (l["name"] for l in layers if l["type"] == "dense")}
    return {
        "strategy": strategy,
        "per_layer_bits": per_cost,
        "per_layer_weight_bits":
            {l["name"]: bits_for[l["name"]] for l in layers},
        "total_bits": total_cost,
        "budget_bits": storage_budget,
        "budget_satisfied": (storage_budget is None
                             or total_cost <= storage_budget),
        "group_tol_max_abs":
            {n: tol_for[n] for n in all_group_names},
        "groups": group_reports,
        "all_groups_satisfied": all(g["satisfied"] for g in group_reports),
        # 向后兼容字段：所有组上的逐层最坏误差（非平均）
        "tol_max_abs": "per-group, see group_tol_max_abs",
        "final_layer_max_abs": errs_flat,
    }


def convert(layers, calib, tol_max_abs=1e-3, storage_budget=None,
            weight_bits=None, calib_source=None, created_at=None):
    """返回可独立序列化的定点模型描述（dict，含逐组选择报告）。

    calib 可以是单 ndarray（单组 default，容差 tol_max_abs）或多输入组
    （见 model.normalize_groups）。多组时每组可带各自的逐层/统一容差；
    缺省回退到 tol_max_abs。
    - weight_bits=None：在逐组容差 / 共同存储预算下自动混合精度；
    - weight_bits=4|8：强制所有 dense 层同精度（仍逐组校验误差与预算）。
    任一组任一层超差或预算不足时抛 PrecisionTargetError，不放宽目标。
    """
    bad = [l["name"] for l in layers if l["type"] not in CONVERTIBLE]
    if bad:
        raise UnsupportedLayerError(
            f"拒绝转换：不支持的层 {bad}，当前仅支持 {sorted(CONVERTIBLE)}")

    from .model import normalize_groups
    groups = normalize_groups(calib)
    groups = [(n, s, tol_max_abs if t is None else t) for n, s, t in groups]
    dense_names = [l["name"] for l in layers if l["type"] == "dense"]
    tol_for = {}
    for name, _, t in groups:
        tol_for[name] = {n: float(t) for n in dense_names}

    scales = calibrate(layers, groups)
    if weight_bits is not None:
        if weight_bits not in WEIGHT_BIT_CHOICES:
            raise ValueError(f"weight_bits 仅支持 {WEIGHT_BIT_CHOICES}")
        bits_for = {l["name"]: weight_bits for l in layers}
        qmodel = _assemble_qmodel(scales, layers, bits_for)
        group_errs = _group_layer_errs(layers, qmodel, groups, dense_names)
        total_cost, per_cost = _storage_cost(layers, bits_for)
        badv = _violations(group_errs, tol_for)
        if badv:
            raise PrecisionTargetError(
                f"强制 {weight_bits}bit 不满足逐输入组误差要求（不自动放宽"
                f"目标）：{badv}")
        if storage_budget is not None and total_cost > storage_budget:
            raise PrecisionTargetError(
                f"强制 {weight_bits}bit 代价 {total_cost} bits 超过预算 "
                f"{storage_budget} bits")
        strategy = f"forced-all-{weight_bits}bit"
        report = _build_report(layers, groups, bits_for, group_errs,
                               total_cost, per_cost, tol_for,
                               storage_budget, strategy, scales)
    else:
        qmodel, bits_for, group_errs, total_cost, per_cost = \
            _select_precision(layers, groups, scales, tol_for,
                              storage_budget)
        report = _build_report(layers, groups, bits_for, group_errs,
                               total_cost, per_cost, tol_for,
                               storage_budget,
                               "greedy-mixed-[4,8]bit-per-group", scales)
    qmodel["mixed_precision_report"] = report
    qmodel["provenance"] = {
        "calib_source": calib_source,
        "calib_groups": [{"name": n, "n_samples": int(len(s)),
                          "tol_max_abs": t} for n, s, t in groups],
        "calib_sha256_16": _fingerprint(groups),
        "storage_budget_bits": storage_budget,
        "created_at": created_at or time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    return qmodel


def candidate_plans(layers, calib, tol_max_abs=1e-3, storage_budget=None):
    """构建候选部署方案集合（不做静默淘汰）。

    返回 [{"plan", "qmodel", "error"?, "satisfied", "report"?}]：
    all-4bit / greedy-mixed / all-8bit。即使方案不满足约束也会组装出
    qmodel 用于逐组测量改善/退化（只是不标记为可行）；若连量化模型都
    无法组装（如全部 8bit 仍有组超差），qmodel 为 None 并保留错误。
    """
    from .model import normalize_groups
    groups = [(n, s, tol_max_abs if t is None else t)
              for n, s, t in normalize_groups(calib)]
    dense_names = [l["name"] for l in layers if l["type"] == "dense"]
    tol_for = {n: {d: float(t) for d in dense_names}
               for n, _, t in groups}
    # 仅用于“测量用”组装：抹掉组自带容差，让全局 tol_max_abs 生效
    calib_no_tol = [{"name": n, "samples": s, "tol_max_abs": None}
                    for n, s, _ in groups]

    plans = []
    for label, wb in (("all-4bit", 4), ("greedy-mixed", None),
                      ("all-8bit", 8)):
        q, err = None, None
        if wb is None:
            # 贪心：先按真实约束；预算挡住则去掉预算做最优努力；
            # 容差本身不可达则最后退到无容差组装（等价全 4bit）。
            for use_budget, use_tol, use_calib in (
                    (storage_budget, tol_max_abs, calib),
                    (None, tol_max_abs, calib),
                    (None, float("inf"), calib_no_tol)):
                try:
                    q = convert(layers, use_calib, tol_max_abs=use_tol,
                                storage_budget=use_budget, weight_bits=None,
                                calib_source=f"plan:{label}")
                    break
                except PrecisionTargetError as e:
                    err = str(e)
        else:
            # 固定精度：无约束组装仅用于测量各组误差，可行性另判
            q = convert(layers, calib_no_tol, tol_max_abs=float("inf"),
                        storage_budget=None, weight_bits=wb,
                        calib_source=f"plan:{label}")
        ge = _group_layer_errs(layers, q, groups, dense_names)
        badv = _violations(ge, tol_for)
        total_cost, per_cost = _storage_cost(
            layers, q["mixed_precision_report"]["per_layer_weight_bits"])
        over_budget = (storage_budget is not None
                       and total_cost > storage_budget)
        bits_for = q["mixed_precision_report"]["per_layer_weight_bits"]
        rep = _build_report(layers, groups, bits_for, ge, total_cost,
                            per_cost, tol_for, storage_budget,
                            f"{label}:measured", None)
        satisfied = not badv and not over_budget
        if satisfied:
            err = None
        elif err is None:
            err = ("未满足逐输入组误差：%s" % badv if badv
                   else "存储 %s 超过共同预算 %s"
                        % (total_cost, storage_budget))
        plans.append({"plan": label, "qmodel": q, "report": rep,
                      "satisfied": satisfied, "error": err})
    return plans


def compare_plans(layers, calib, tol_max_abs=1e-3, storage_budget=None):
    """对比候选方案：逐层精度变化如何影响每个输入组。

    返回 {"plans", "selected", "feasible", "pairwise", "all_satisfied"}。
    没有任何方案满足全部组约束 + 预算时 feasible=False 并明确列出失败，
    绝不自动放宽目标。
    """
    plans = candidate_plans(layers, calib, tol_max_abs, storage_budget)
    plan_map = {p["plan"]: p for p in plans}
    dense_names = [l["name"] for l in layers if l["type"] == "dense"]

    summary = []
    for p in plans:
        if p["qmodel"] is None:
            summary.append({"plan": p["plan"], "feasible": False,
                            "error": p["error"]})
            continue
        r = p["report"]
        s = {
            "plan": p["plan"], "feasible": p["satisfied"],
            "total_bits": r["total_bits"],
            "budget_bits": r["budget_bits"],
            "weight_bits": {n: r["per_layer_weight_bits"][n]
                            for n in dense_names},
            "groups": [
                {"name": g["name"], "satisfied": g["satisfied"],
                 "max_abs_err_overall": g["max_abs_err_overall"]}
                for g in r["groups"]],
        }
        if not p["satisfied"]:
            s["error"] = p["error"]
        summary.append(s)

    def glines(plan_name):
        p = plan_map[plan_name]
        if p["qmodel"] is None:
            return None
        return {g["name"]: {x["layer"]: x for x in g["per_layer"]}
                for g in p["report"]["groups"]}

    pairwise = []
    for a, b in (("all-4bit", "greedy-mixed"),
                 ("greedy-mixed", "all-8bit"),
                 ("all-4bit", "all-8bit")):
        ga, gb = glines(a), glines(b)
        if ga is None or gb is None:
            pairwise.append({"from": a, "to": b, "available": False})
            continue
        changes = []
        for gname in ga:
            for lname in dense_names:
                ea = ga[gname][lname]["max_abs_err"]
                eb = gb[gname][lname]["max_abs_err"]
                delta = eb - ea
                if abs(delta) <= 1e-15:
                    effect = "unchanged"
                elif delta < 0:
                    effect = "improved"
                else:
                    effect = "degraded"
                changes.append({
                    "group": gname, "layer": lname,
                    "bits": {"from": ga[gname][lname]["w_bits"],
                             "to": gb[gname][lname]["w_bits"]},
                    "max_abs_err": {"from": ea, "to": eb,
                                    "delta": delta},
                    "effect": effect,
                    "to_satisfied": gb[gname][lname]["satisfied"],
                })
        pairwise.append({
            "from": a, "to": b, "available": True,
            "storage_delta_bits":
                plan_map[b]["report"]["total_bits"]
                - plan_map[a]["report"]["total_bits"],
            "per_layer_effects": changes,
        })

    feasible_plans = [s["plan"] for s in summary if s.get("feasible")]
    selected = None
    # 选可行方案中存储最小的；相同存储选标签优先级（mixed > 4bit > 8bit）
    if feasible_plans:
        def key(n):
            r = plan_map[n]["report"]
            rank = {"greedy-mixed": 0, "all-4bit": 1, "all-8bit": 2}[n]
            return (r["total_bits"], rank)
        selected = min(feasible_plans, key=key)
    return {
        "plans": summary, "pairwise": pairwise,
        "selected_plan": selected,
        "feasible": bool(feasible_plans),
        "all_constraints": {
            "tol_max_abs_default": tol_max_abs,
            "storage_budget_bits": storage_budget},
        "failures": [{"plan": p["plan"], "error": p["error"]}
                     for p in plans if p["qmodel"] is None]
                     + [{"plan": p["plan"],
                         "unsatisfied_groups":
                             [g["name"] for g in p["report"]["groups"]
                              if not g["satisfied"]],
                         "budget_satisfied":
                             p["report"]["budget_satisfied"]}
                        for p in plans
                        if p["qmodel"] is not None and not p["satisfied"]],
    }
