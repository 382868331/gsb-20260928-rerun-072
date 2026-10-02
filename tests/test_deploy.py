"""按组选择方案的部署产物：交叠路由、显式回退、部分预算、
条件变更失效与重载一致性。"""
import numpy as np
import pytest
from edgequant import model as M
from edgequant.deploy import (build_deployment, route_samples, predict,
                              compare_with_unified, reverify_deploy,
                              normalize_deploy_groups)
from edgequant.quantize import PrecisionTargetError
from edgequant.stress import build_stress_model, build_stress_groups


def _groups_with_conditions():
    """三组路由条件互相交叠（上界都覆盖各自校准范围）。"""
    groups = build_stress_groups()
    conds = {"nominal": 1.0, "wide": 1.0, "edge": 1.0}
    for g in groups:
        g["condition"] = {"abs_max_le": conds[g["name"]]}
    return groups


def _build(budget=None):
    layers, _ = build_stress_model()
    dep = build_deployment(layers, _groups_with_conditions(),
                           tol_max_abs=0.05, storage_budget=budget,
                           calib_source="test:deploy")
    return layers, dep


def test_overlapping_groups_resolve_by_priority_not_by_error():
    """交叠输入按声明顺序优先级定去向，绝不事后按误差挑方案。"""
    layers, dep = _build()
    # 同时命中 nominal/wide/edge 的输入
    x = np.array([[0.3, -0.2, 0.1, -0.4]])
    (p,) = route_samples(dep, x)
    assert p["matched_groups"] == ["nominal", "wide", "edge"]
    assert p["group"] == "nominal"
    assert p["reason"] == "overlap:first-match-by-priority"
    assert p["scheme"] == dep["assignment"]["nominal"]
    assert not p["uncovered"]


def test_uncovered_input_goes_to_fallback_explicitly():
    """不符合任何组的输入显式走回退方案并标记未覆盖。"""
    layers, dep = _build()
    x = np.array([[5.0, -5.0, 5.0, -5.0]])
    (p,) = route_samples(dep, x)
    assert p["uncovered"]
    assert p["scheme"] == "unified"
    assert p["reason"] == "no-group-match:fallback"
    out = predict(layers, dep, x)
    assert out["paths"][0]["uncovered"]
    assert "unified" in out["risk_by_scheme"]


def test_audit_path_error_and_shared_storage():
    """用户可核对：实际走过路径、逐样本预测误差、各方案共同占用存储。"""
    layers, dep = _build()
    x = np.array([[0.3, -0.2, 0.1, -0.4],
                  [0.9, 0.9, 0.9, 0.9],
                  [3.0, -3.0, 3.0, -3.0]])
    out = predict(layers, dep, x)
    assert len(out["paths"]) == 3
    assert len(out["per_sample_max_abs_err"]) == 3
    st = out["storage"]
    assert st["total_bits"] == (st["fallback_bits"]
                                + sum(st["dedicated_bits"].values()))
    assert st["budget_satisfied"]


def test_budget_fits_only_partial_candidates():
    """预算只容纳部分候选：显式分配，其余组记录为由回退覆盖。"""
    layers, _ = build_stress_model()
    groups = _groups_with_conditions()
    full = build_deployment(layers, groups, tol_max_abs=0.05)
    u = full["storage"]["fallback_bits"]
    ded = sorted(full["storage"]["dedicated_bits"].values())
    budget = u + ded[0]  # 只装得下回退 + 最小的一个专用方案
    dep = build_deployment(layers, groups, tol_max_abs=0.05,
                           storage_budget=budget)
    st = dep["storage"]
    assert st["total_bits"] <= budget
    assert len(st["dedicated_bits"]) == 1
    served = set(st["dedicated_bits"])
    for g in dep["deploy_report"]["groups"]:
        if g["name"] in served:
            assert g["scheme"] == "group:" + g["name"]
        else:
            # 未获得专用方案的组显式记录为回退覆盖，不是静默丢弃
            assert g["scheme"] == "unified"
            assert dep["assignment"][g["name"]] == "unified"


def test_budget_below_fallback_fails_explicitly():
    layers, _ = build_stress_model()
    with pytest.raises(PrecisionTargetError):
        build_deployment(layers, _groups_with_conditions(),
                         tol_max_abs=0.05, storage_budget=10)


def test_compare_shows_improvement_and_boundary_beyond_calib():
    """统一 vs 按组：新增存储换来哪些组的改善；边界输入仍超出校准依据。"""
    layers, dep = _build()
    rep = compare_with_unified(layers, dep, _groups_with_conditions())
    assert rep["extra_storage_bits"] == (
        rep["deploy_total_bits"] - rep["unified_bits"])
    assert rep["extra_storage_bits"] > 0
    assert rep["improved_groups"]  # 有组因专用方案改善
    for g in rep["per_group"]:
        assert g["deploy_err"] <= g["unified_err"] + 1e-12
    b = rep["boundary"]
    assert b["n_uncovered_fallback"] > 0
    assert b["all_uncovered_use_fallback"]
    assert b["risk"]["input_out_of_calib"]


def test_reload_reproduces_routing_prediction_and_fallback(tmp_path):
    """重载完整产物后重复选择与预测：未覆盖范围与回退结果与保存前一致。"""
    layers, dep = _build()
    p = tmp_path / "m.deploy.json"
    M.save_deploy(p, dep)
    loaded = M.load_deploy(p)
    x = np.array([[0.3, -0.2, 0.1, -0.4],
                  [0.9, 0.9, 0.9, 0.9],
                  [4.0, -4.0, 4.0, -4.0]])
    r1 = predict(layers, dep, x)
    r2 = predict(layers, loaded, x)
    assert r1["paths"] == r2["paths"]
    assert np.array_equal(r1["outputs"], r2["outputs"])
    assert r1["per_sample_max_abs_err"] == r2["per_sample_max_abs_err"]
    assert r1["risk_by_scheme"] == r2["risk_by_scheme"]
    rep, ok = reverify_deploy(layers, loaded, _groups_with_conditions())
    assert ok
    assert rep["uncovered_fallback"]["all_route_to_fallback"]
    for g in rep["groups"]:
        assert g["path_consistent"]
        assert g["matches_recorded_prediction"]


def test_changed_conditions_do_not_inherit_old_calib_proof(tmp_path):
    """修改分组条件后，原校准证明不能自动沿用到新范围。"""
    layers, dep = _build()
    p = tmp_path / "m.deploy.json"
    M.save_deploy(p, dep)
    loaded = M.load_deploy(p)
    changed = _groups_with_conditions()
    changed[0]["condition"] = {"abs_max_le": 0.6}  # 改 nominal 范围
    rep, ok = reverify_deploy(layers, loaded, changed)
    assert not ok
    assert not rep["conditions_match_recorded"]
    assert not rep["calib_proof_valid_for_new_conditions"]
    assert (rep["conditions_sha256_16"]["recorded"]
            != rep["conditions_sha256_16"]["recomputed"])


def test_default_condition_is_calib_range():
    groups = normalize_deploy_groups(build_stress_groups())
    for name, samples, _, cond in groups:
        assert cond["abs_max_le"] == float(np.max(np.abs(samples)))
