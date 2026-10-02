"""按输入组选择精度方案的部署产物：路由条件 + 多方案 + 显式回退。

口径：
- 每个输入组可带路由条件 condition={"abs_max_le": b}：max|x| <= b 命中；
  缺省取该组校准样本的最大绝对值（即校准覆盖范围）。条件随产物保存，
  是部署产物的一部分。
- 交叠：按组声明顺序定优先级，先命中者生效；全部命中组都记入路径，
  绝不事后按预测误差挑选方案。
- 未覆盖：显式走回退方案（统一方案 unified），路径标记 fallback；
  未覆盖范围与回退结果在产物重载后与保存前一致。
- 共同存储 = 回退方案 + 各专用方案占用之和；预算装不下全部候选时按
  “误差改善 / 额外比特”贪心选子集，其余组显式记为由 unified 覆盖，
  不静默丢弃。
- 分组条件指纹进 provenance：改条件后旧校准证明不自动沿用到新范围。
"""
import hashlib
import json
import time
import numpy as np

from .quantize import (convert, run_fp_trace, _group_layer_errs,
                       _fingerprint, PrecisionTargetError)
from .runner import run_quant
from .evaluate import fp_forward

DEPLOY_FORMAT = "edgequant.deploy.v1"
FALLBACK_SCHEME = "unified"


def normalize_deploy_groups(groups):
    """统一成 [(name, samples, tol, condition)]，condition 缺省取校准范围。"""
    from .model import normalize_groups
    raw_conditions = {}
    if (isinstance(groups, (list, tuple)) and groups and
            isinstance(groups[0], dict)):
        for g in groups:
            c = g.get("condition")
            if c is not None:
                raw_conditions[str(g["name"])] = dict(c)
    out = []
    for name, samples, tol in normalize_groups(groups):
        cond = raw_conditions.get(name) or {
            "abs_max_le": float(np.max(np.abs(samples)))}
        if "abs_max_le" not in cond:
            raise ValueError("输入组 %s 的路由条件缺少 abs_max_le" % name)
        out.append((name, samples, tol,
                    {"abs_max_le": float(cond["abs_max_le"])}))
    return out


def _conditions_fingerprint(entries):
    """对 (组名, 路由条件) 取指纹：条件变了，旧校准证明不能自动沿用。"""
    h = hashlib.sha256()
    for name, cond in entries:
        h.update(name.encode())
        h.update(json.dumps(cond, sort_keys=True).encode())
    return h.hexdigest()[:16]


def _scheme_name(group):
    return "group:%s" % group


def _oob_probe(dg, factor, seed):
    """确定性越界探针：每个样本都保证超出所有组条件上界。"""
    max_bound = max(c["abs_max_le"] for _, _, _, c in dg)
    width = dg[0][1].shape[1]
    rng = np.random.default_rng(seed)
    mag = rng.uniform(max_bound * (1.0 + 1e-6), factor * max_bound,
                      (32, width))
    oob = mag * rng.choice([-1.0, 1.0], (32, width))
    return oob, max_bound


def route_samples(deploy, x):
    """逐样本路由：只依据产物中保存的条件与优先级，不看预测误差。

    返回与输入行一一对应的路径记录：生效方案、命中组（含交叠时的
    全部命中）、原因与是否落入回退。
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    conds = deploy["routing"]["conditions"]
    priority = deploy["routing"]["priority"]
    paths = []
    for row in x:
        am = float(np.max(np.abs(row)))
        matched = [n for n in priority
                   if am <= conds[n]["abs_max_le"] + 1e-12]
        if matched:
            primary = matched[0]
            paths.append({
                "scheme": deploy["assignment"][primary],
                "group": primary,
                "matched_groups": matched,
                "reason": ("unique-match" if len(matched) == 1
                           else "overlap:first-match-by-priority"),
                "input_abs_max": am,
                "uncovered": False,
            })
        else:
            paths.append({
                "scheme": deploy["routing"]["fallback_scheme"],
                "group": None,
                "matched_groups": [],
                "reason": "no-group-match:fallback",
                "input_abs_max": am,
                "uncovered": True,
            })
    return paths


def predict(layers, deploy, x):
    """按路由分批推理：返回逐样本输出、实际走过路径、逐样本误差与风险。

    同一路由决策在产物重载后重复执行结果一致（纯查表 + 整数推理）。
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    paths = route_samples(deploy, x)
    fp = fp_forward(layers, x)
    y = np.zeros_like(fp)
    risk_by_scheme = {}
    for scheme in sorted({p["scheme"] for p in paths}):
        idx = [i for i, p in enumerate(paths) if p["scheme"] == scheme]
        out, _, risk = run_quant(deploy["schemes"][scheme], x[idx],
                                 return_risk=True)
        y[idx] = out
        risk_by_scheme[scheme] = risk
    per_sample_err = [float(np.max(np.abs(y[i] - fp[i])))
                      for i in range(len(x))]
    return {"outputs": y, "paths": paths,
            "per_sample_max_abs_err": per_sample_err,
            "risk_by_scheme": risk_by_scheme,
            "storage": deploy["storage"]}


def build_deployment(layers, groups, tol_max_abs=1e-3, storage_budget=None,
                     calib_source=None, created_at=None):
    """构建按组选择方案的部署产物（dict，可独立序列化）。

    - 回退/统一方案：全部组共同校准，必须满足所有组容差，否则显式失败；
    - 每组专用候选：只用该组样本校准；预算不足时按改善/比特贪心选子集；
    - 预算连回退方案都装不下时抛 PrecisionTargetError，不放宽目标。
    """
    dg = normalize_deploy_groups(groups)
    plain = [(n, s, (tol_max_abs if t is None else t)) for n, s, t, _ in dg]
    unified = convert(layers, plain, tol_max_abs=tol_max_abs,
                      storage_budget=None, calib_source=calib_source)
    dense_names = [l["name"] for l in layers if l["type"] == "dense"]
    u_rep = unified["mixed_precision_report"]
    u_bits = u_rep["total_bits"]
    u_err = {g["name"]: g["max_abs_err_overall"] for g in u_rep["groups"]}
    if storage_budget is not None and u_bits > storage_budget:
        raise PrecisionTargetError(
            "共同存储预算 %s bits 连回退方案（%s bits）都装不下，"
            "无法部署（不自动放宽目标）" % (storage_budget, u_bits))

    candidates = []
    for name, samples, tol, _ in dg:
        t = tol_max_abs if tol is None else tol
        try:
            q = convert(layers, [(name, samples, t)], tol_max_abs=t,
                        storage_budget=None, calib_source=calib_source)
        except PrecisionTargetError as e:
            candidates.append({"group": name, "qmodel": None,
                               "error": str(e)})
            continue
        bits = q["mixed_precision_report"]["total_bits"]
        err = q["mixed_precision_report"]["groups"][0]["max_abs_err_overall"]
        candidates.append({"group": name, "qmodel": q, "bits": bits,
                           "err": err,
                           "improvement": u_err[name] - err})

    ok_cands = [c for c in candidates if c["qmodel"] is not None]
    chosen = set()
    used = u_bits
    if storage_budget is None:
        # 只保留真正带来改善的专用方案：统一方案本就要存（回退），
        # 不改善误差的专用方案不增加存储
        beneficial = [c for c in ok_cands if c["improvement"] > 0]
        chosen = {c["group"] for c in beneficial}
        used += sum(c["bits"] for c in beneficial)
    else:
        order = sorted(ok_cands,
                       key=lambda c: -(c["improvement"] / c["bits"]
                                       if c["bits"] else float("inf")))
        for c in order:
            if c["improvement"] <= 0:
                continue
            if used + c["bits"] <= storage_budget:
                chosen.add(c["group"])
                used += c["bits"]

    schemes = {FALLBACK_SCHEME: unified}
    assignment = {}
    dedicated_bits = {}
    for c in candidates:
        name = c["group"]
        if c["qmodel"] is not None and name in chosen:
            schemes[_scheme_name(name)] = c["qmodel"]
            assignment[name] = _scheme_name(name)
            dedicated_bits[name] = c["bits"]
        else:
            assignment[name] = FALLBACK_SCHEME

    group_reports = []
    for name, samples, tol, cond in dg:
        t = tol_max_abs if tol is None else tol
        scheme = assignment[name]
        errs = _group_layer_errs(layers, schemes[scheme],
                                 [(name, samples, t)], dense_names)[name]
        worst = max(errs.values())
        group_reports.append({
            "name": name, "scheme": scheme, "condition": cond,
            "tol_max_abs": t,
            "per_layer_max_abs": errs,
            "max_abs_err_overall": worst,
            "unified_max_abs_err_overall": u_err[name],
            "effect": ("improved" if worst < u_err[name] - 1e-15
                       else "unchanged"),
            "satisfied": bool(worst <= t + 1e-15),
        })

    return {
        "routing": {
            "priority": [n for n, _, _, _ in dg],
            "conditions": {n: c for n, _, _, c in dg},
            "fallback_scheme": FALLBACK_SCHEME,
            "overlap_policy": "first-match-by-priority",
            "no_match_policy": "fallback",
        },
        "schemes": schemes,
        "assignment": assignment,
        "storage": {
            "fallback_bits": u_bits,
            "dedicated_bits": dedicated_bits,
            "total_bits": used,
            "budget_bits": storage_budget,
            "budget_satisfied": (storage_budget is None
                                 or used <= storage_budget),
        },
        "deploy_report": {
            "groups": group_reports,
            "rejected_candidates": [
                {"group": c["group"], "error": c["error"]}
                for c in candidates if c["qmodel"] is None],
            "all_groups_satisfied":
                all(g["satisfied"] for g in group_reports),
        },
        "provenance": {
            "calib_source": calib_source,
            "calib_groups": [{"name": n, "n_samples": int(len(s)),
                              "tol_max_abs": t}
                             for n, s, t in plain],
            "calib_sha256_16": _fingerprint(plain),
            "conditions_sha256_16": _conditions_fingerprint(
                [(n, c) for n, _, _, c in dg]),
            "storage_budget_bits": storage_budget,
            "created_at": created_at
                          or time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    }


def compare_with_unified(layers, deploy, groups, boundary_factor=2.0,
                         seed=0):
    """统一方案 vs 按组选择方案：新增存储换来了哪些组的改善，以及哪些
    边界输入仍然超出校准依据（只能走回退并带风险披露）。"""
    dg = normalize_deploy_groups(groups)
    per_group = []
    for g in deploy["deploy_report"]["groups"]:
        per_group.append({
            "group": g["name"], "scheme": g["scheme"],
            "unified_err": g["unified_max_abs_err_overall"],
            "deploy_err": g["max_abs_err_overall"],
            "effect": g["effect"],
        })
    improved = [g["group"] for g in per_group if g["effect"] == "improved"]

    # 边界输入：超出所有组条件上界，仍超出校准依据
    oob, max_bound = _oob_probe(dg, boundary_factor, seed)
    paths = route_samples(deploy, oob)
    uncovered = [p for p in paths if p["uncovered"]]
    fb = deploy["routing"]["fallback_scheme"]
    _, _, risk = run_quant(deploy["schemes"][fb], oob, return_risk=True)
    return {
        "unified_bits": deploy["storage"]["fallback_bits"],
        "deploy_total_bits": deploy["storage"]["total_bits"],
        "extra_storage_bits": (deploy["storage"]["total_bits"]
                               - deploy["storage"]["fallback_bits"]),
        "per_group": per_group,
        "improved_groups": improved,
        "boundary": {
            "factor": boundary_factor,
            "max_condition_abs_max_le": max_bound,
            "n_samples": int(len(oob)),
            "n_uncovered_fallback": len(uncovered),
            "all_uncovered_use_fallback": all(
                p["scheme"] == fb for p in uncovered),
            "risk": risk,
        },
    }


def reverify_deploy(layers, deploy, groups, boundary_factor=2.0, seed=0):
    """重载完整产物后的复核：路由、预测、回退与保存前一致；
    分组条件被修改时，旧校准证明不自动沿用到新范围。"""
    dg = normalize_deploy_groups(groups)
    cond_fp = _conditions_fingerprint([(n, c) for n, _, _, c in dg])
    recorded_fp = deploy["provenance"].get("conditions_sha256_16")
    conditions_match = (cond_fp == recorded_fp)

    dense_names = [l["name"] for l in layers if l["type"] == "dense"]
    recorded = {g["name"]: g
                for g in deploy["deploy_report"]["groups"]}
    group_checks = []
    all_ok = conditions_match
    for name, samples, tol, cond in dg:
        t = tol if tol is not None else recorded[name]["tol_max_abs"]
        paths = route_samples(deploy, samples)
        expected = deploy["assignment"][name]
        # 交叠条件下，本组样本可能被更高优先级的组捕获；只要求
        # “首要命中本组”的样本落到本组分配的方案，捕获情况如实披露
        primary = [p for p in paths if p["group"] == name]
        captured = len(paths) - len(primary)
        path_consistent = all(p["scheme"] == expected for p in primary)
        errs = _group_layer_errs(layers, deploy["schemes"][expected],
                                 [(name, samples, t)], dense_names)[name]
        rec = recorded[name]["per_layer_max_abs"]
        mismatch = [{"layer": n, "recorded": rec[n], "recomputed": errs[n]}
                    for n in dense_names
                    if abs(rec[n] - errs[n]) > 1e-12]
        satisfied = max(errs.values()) <= t + 1e-15
        ok = path_consistent and not mismatch and satisfied
        all_ok = all_ok and ok
        group_checks.append({
            "name": name, "scheme": expected,
            "n_samples": int(len(samples)),
            "n_primary_match": len(primary),
            "n_captured_by_higher_priority": captured,
            "path_consistent": path_consistent,
            "matches_recorded_prediction": not mismatch,
            "mismatch": mismatch,
            "per_layer_max_abs": errs,
            "tol_max_abs": t,
            "satisfied": bool(satisfied),
        })

    # 未覆盖范围：确定性越界探针必须全部走回退，且风险如实披露
    oob, max_bound = _oob_probe(dg, boundary_factor, seed)
    paths = route_samples(deploy, oob)
    fb = deploy["routing"]["fallback_scheme"]
    fallback_ok = all(p["uncovered"] and p["scheme"] == fb for p in paths)
    _, _, risk = run_quant(deploy["schemes"][fb], oob, return_risk=True)
    risk_reported = bool(risk["has_risk"] or risk["input_out_of_calib"])

    report = {
        "conditions_match_recorded": conditions_match,
        "calib_proof_valid_for_new_conditions": conditions_match,
        "conditions_sha256_16": {"recorded": recorded_fp,
                                 "recomputed": cond_fp},
        "groups": group_checks,
        "uncovered_fallback": {
            "factor": boundary_factor,
            "n_samples": int(len(oob)),
            "all_route_to_fallback": bool(fallback_ok),
            "risk_reported": risk_reported,
            "risk": risk,
        },
        "provenance": deploy.get("provenance"),
    }
    report["passed"] = bool(all_ok and fallback_ok and risk_reported)
    return report, report["passed"]
