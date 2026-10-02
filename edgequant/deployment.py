"""统一方案与按组方案的部署产物构建、执行和比较。"""
import copy
import time
import numpy as np

from .quantize import (convert, _fingerprint, _group_layer_errs,
                       _storage_cost, PrecisionTargetError)
from .routing import (normalize_conditions, route_inputs,
                      policy_fingerprint)


DEPLOYMENT_FORMAT = "edgequant.deployment.v2"


def _report_bits(qmodel, layers):
    bits_for = qmodel["mixed_precision_report"]["per_layer_weight_bits"]
    total, _ = _storage_cost(layers, bits_for)
    return total, bits_for


def _group_worst(qmodel, layers, groups, dense_names):
    errs = _group_layer_errs(layers, qmodel, groups, dense_names)
    return {name: max(per.values()) for name, per in errs.items()}, errs


def _strip_for_plan(qmodel, plan, group=None):
    q = copy.deepcopy(qmodel)
    q["plan_name"] = plan
    q["calibration_group"] = group
    q.pop("provenance", None)
    return q


def _route_report(groups, rules, fallback_plan):
    rows = []
    for name, samples, _ in groups:
        assignments = route_inputs(samples, rules, fallback_plan)
        counts = {}
        for a in assignments:
            key = (a["status"], a["plan"], a["group"] or "")
            counts[key] = counts.get(key, 0) + 1
        rows.append({
            "group": name, "n_samples": len(samples),
            "assignments": assignments,
            "summary": [{"status": s, "plan": p, "group": g or None,
                         "count": c} for (s, p, g), c in counts.items()]})
    return rows


def build_grouped_deployment(layers, calib, conditions=None, tol_max_abs=1e-3,
                             storage_budget=None, calib_source=None,
                             fallback_plan="unified", created_at=None):
    """构建含确定性路由的部署产物。

    选择在转换前完成：统一方案作为公共回退，每个输入组各自校准一个候选。
    预算不足时按“相对统一方案的误差改善/额外存储”选择部分候选；未入选
    组和未覆盖输入都回到统一方案，不允许按事后误差改路。
    """
    from .model import normalize_groups
    groups = [(n, s, tol_max_abs if t is None else t)
              for n, s, t in normalize_groups(calib)]
    dense_names = [l["name"] for l in layers if l["type"] == "dense"]
    conditions = normalize_conditions(groups, conditions)
    condition_by_name = {c["name"]: {k: v for k, v in c.items()
                                     if k != "name"}
                         for c in conditions}

    unified = convert(layers, groups, tol_max_abs=tol_max_abs,
                      storage_budget=storage_budget,
                      calib_source=calib_source)
    unified_cost, unified_bits = _report_bits(unified, layers)
    unified_worst, _ = _group_worst(unified, layers, groups, dense_names)

    candidates = []
    for name, samples, tol in groups:
        q = None
        infeasible = None
        try:
            q = convert(layers, [(name, samples, tol)], tol_max_abs=tol,
                        storage_budget=None,
                        calib_source="%s#group=%s" % (calib_source, name))
        except PrecisionTargetError as e:
            infeasible = str(e)
        if q is None:
            cost, bits = 0, {}
            grouped_worst_value = float("inf")
            grouped_layer_errs = {n: None for n in dense_names}
            improvement = -float("inf")
        else:
            cost, bits = _report_bits(q, layers)
            group_input = [(name, samples, tol)]
            grouped_worst_map, layer_errs = _group_worst(
                q, layers, group_input, dense_names)
            grouped_worst_value = grouped_worst_map[name]
            grouped_layer_errs = layer_errs[name]
            improvement = unified_worst[name] - grouped_worst_value
        candidates.append({
            "group": name, "qmodel": q, "cost_bits": cost,
            "extra_bits": max(0, cost), "bits_for": bits,
            "unified_worst": unified_worst[name],
            "grouped_worst": grouped_worst_value,
            "improvement_bits": improvement,
            "group_layer_errors": grouped_layer_errs,
            "condition": condition_by_name[name],
            "infeasible_reason": infeasible})

    ranked = sorted([c for c in candidates if c["improvement_bits"] > 1e-15],
                    key=lambda c: (c["improvement_bits"] / c["extra_bits"],
                                   c["improvement_bits"]),
                    reverse=True)
    selected_groups = set()
    total_cost = unified_cost
    for c in ranked:
        if storage_budget is not None and \
                total_cost + c["extra_bits"] > storage_budget:
            continue
        selected_groups.add(c["group"])
        total_cost += c["extra_bits"]

    if storage_budget is not None and total_cost > storage_budget:
        raise RuntimeError("内部错误：部署存储超出预算")

    plans = {"unified": _strip_for_plan(unified, "unified")}
    rules = []
    for c in candidates:
        plan = "group:%s" % c["group"] if c["group"] in selected_groups \
            else "unified"
        if c["group"] in selected_groups:
            plans[plan] = _strip_for_plan(c["qmodel"], plan, c["group"])
        rules.append({
            "order": len(rules), "group": c["group"],
            "plan": plan, "condition": c["condition"],
            "selection": "selected-grouped" if plan != "unified"
                         else "budget-or-no-improvement:unified-fallback"})

    route_rows = _route_report(groups, rules, fallback_plan)
    deployment = {
        "format": DEPLOYMENT_FORMAT,
        "fallback_plan": fallback_plan,
        "plans": plans,
        "route_policy": {
            "decision_time": "before-prediction-and-frozen",
            "overlap_policy": "first-rule-in-order; mark ambiguous",
            "uncovered_policy": fallback_plan,
            "rules": rules,
            "calibration": {
                "policy_sha256_16": policy_fingerprint(groups, conditions),
                "group_fingerprints": {
                    n: _fingerprint([(n, s, t)]) for n, s, t in groups}}},
        "storage": {
            "unified_bits": unified_cost,
            "selected_group_plan_bits": {
                c["group"]: c["cost_bits"] for c in candidates
                if c["group"] in selected_groups},
            "shared_unique_total_bits": total_cost,
            "budget_bits": storage_budget,
            "budget_satisfied": storage_budget is None
                                or total_cost <= storage_budget,
            "accounting": "one unified fallback + selected complete group "
                          "plans; scales/weights are not deduplicated"},
        "deployment_report": {
            "unified_group_worst_max_abs": unified_worst,
            "group_candidates": [{
                "group": c["group"], "selected": c["group"]
                in selected_groups,
                "grouped_plan_feasible": c["qmodel"] is not None,
                "infeasible_reason": c["infeasible_reason"],
                "total_bits": c["cost_bits"],
                "extra_bits_over_unified": c["extra_bits"],
                "unified_worst_max_abs": c["unified_worst"],
                "grouped_worst_max_abs": c["grouped_worst"],
                "improvement_max_abs": c["improvement_bits"],
                "per_layer_max_abs": c["group_layer_errors"]}
                for c in candidates],
            "calibration_routes": route_rows},
        "provenance": {
            "calib_source": calib_source,
            "calib_groups": [{"name": n, "n_samples": len(s),
                              "tol_max_abs": t} for n, s, t in groups],
            "calib_sha256_16": _fingerprint(groups),
            "storage_budget_bits": storage_budget,
            "created_at": created_at or
            time.strftime("%Y-%m-%dT%H:%M:%S%z")}}
    return deployment


def run_deployment(deployment, x, return_risk=False):
    """按固化规则选择方案并执行；返回拼接输出、路径和风险。"""
    from .runner import run_quant
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    rules = deployment["route_policy"]["rules"]
    fallback = deployment.get("fallback_plan", "unified")
    assignments = route_inputs(x, rules, fallback)
    by_plan = {}
    for i, a in enumerate(assignments):
        by_plan.setdefault(a["plan"], []).append(i)
    outputs = [None] * len(x)
    risks = {}
    for plan, indices in by_plan.items():
        qmodel = deployment["plans"][plan]
        out, _, risk = run_quant(qmodel, x[indices], return_risk=True)
        risks[plan] = risk
        for local, global_index in enumerate(indices):
            outputs[global_index] = out[local]
    merged_risk = {
        "route_assignments": assignments,
        "plans_executed": sorted(by_plan),
        "per_plan": risks,
        "uncovered_count": sum(a["status"] == "uncovered"
                               for a in assignments),
        "ambiguous_count": sum(a["status"] == "ambiguous"
                               for a in assignments),
        "fallback_count": sum(a["fallback_used"] for a in assignments)}
    merged_risk["resolved_by"] = [a["resolved_by"] for a in assignments]
    y = np.asarray(outputs, dtype=np.float64)
    if not return_risk:
        return y, [], assignments
    return y, [], merged_risk, assignments


def compare_unified_and_grouped(layers, calib, conditions=None,
                                tol_max_abs=1e-3, storage_budget=None,
                                calib_source=None):
    deployment = build_grouped_deployment(
        layers, calib, conditions, tol_max_abs, storage_budget,
        calib_source)
    return {
        "deployment": deployment,
        "unified_storage_bits":
            deployment["storage"]["unified_bits"],
        "grouped_storage_bits":
            deployment["storage"]["shared_unique_total_bits"],
        "additional_storage_bits":
            deployment["storage"]["shared_unique_total_bits"]
            - deployment["storage"]["unified_bits"],
        "selected_groups": [c["group"] for c in
                            deployment["deployment_report"]
                            ["group_candidates"] if c["selected"]],
        "improvements": [
            {"group": c["group"], "selected": c["selected"],
             "added_storage_bits": c["extra_bits_over_unified"],
             "worst_max_abs_delta": -c["improvement_max_abs"],
             "unified_worst_max_abs": c["unified_worst_max_abs"],
             "grouped_worst_max_abs": c["grouped_worst_max_abs"]}
            for c in deployment["deployment_report"]["group_candidates"]]}


def verify_deployment(layers, deployment, calib, conditions=None,
                      boundary_factor=2.0, seed=0):
    """重载后复核路径、预测误差、回退结果和共同存储。"""
    from .model import normalize_groups
    from .routing import validate_policy_fingerprint
    from .quantize import run_fp_trace
    from .runner import run_quant

    validate_policy_fingerprint(deployment, calib, conditions)
    groups = [(n, s, tol_max_abs if t is None else t)
              for n, s, t in normalize_groups(calib)]
    dense_names = [l["name"] for l in layers if l["type"] == "dense"]
    stacked = np.vstack([s for _, s, _ in groups])
    y, _, route_risk, assignments = run_deployment(
        deployment, stacked, return_risk=True)
    fp_acts = run_fp_trace(layers, stacked)
    final = fp_acts[-1]["output"]
    path_rows = []
    offset = 0
    for name, samples, tol in groups:
        n = len(samples)
        local_assign = assignments[offset:offset + n]
        local_err = np.max(np.abs(y[offset:offset + n] - final[
            offset:offset + n]), axis=1)
        layer_errs = _group_layer_errs(
            layers, deployment["plans"]["unified"],
            [(name, samples, tol)], dense_names)[name]
        selected_plan = "group:%s" % name
        selected = selected_plan in deployment["plans"]
        if selected:
            selected_errs = _group_layer_errs(
                layers, deployment["plans"][selected_plan],
                [(name, samples, tol)], dense_names)[name]
        else:
            selected_errs = None
        path_rows.append({
            "group": name, "n_samples": n,
            "route_status_counts": _count_assign(local_assign),
            "actual_output_max_abs_err": float(np.max(local_err)),
            "fallback_plan": deployment["fallback_plan"],
            "selected_group_plan": selected,
            "selected_plan_calibration_domain_errors": selected_errs,
            "unified_plan_calibration_domain_errors": layer_errs,
            "tol_max_abs": float(tol),
            "assignments": local_assign})
        offset += n

    m = float(np.max(np.abs(stacked)))
    width = stacked.shape[1]
    sign_rng = np.random.default_rng(seed)
    signs = sign_rng.choice([-1.0, 1.0], size=32)
    boundary = np.full((32, width), boundary_factor * m)
    boundary *= signs[:, None]
    by, _, brisk, bassign = run_deployment(
        deployment, boundary, return_risk=True)
    bfp = run_fp_trace(layers, boundary)[-1]["output"]
    uncovered = all(a["status"] == "uncovered" and
                    a["plan"] == deployment["fallback_plan"]
                    for a in bassign)
    report = {
        "policy_fingerprint_valid": True,
        "groups": path_rows,
        "storage": deployment["storage"],
        "boundary": {
            "factor": boundary_factor,
            "all_uncovered_and_fallback": uncovered,
            "assignments": bassign,
            "plans_executed": brisk["plans_executed"],
            "fallback_output_max_abs_err":
                float(np.max(np.abs(by - bfp))),
            "per_plan_risk": brisk["per_plan"]},
        "route_risk": route_risk}
    report["passed"] = bool(
        uncovered and
        deployment["storage"]["budget_satisfied"])
    return report, report["passed"]


def _count_assign(assignments):
    out = {}
    for a in assignments:
        key = "%s:%s" % (a["status"], a["plan"])
        out[key] = out.get(key, 0) + 1
    return out
