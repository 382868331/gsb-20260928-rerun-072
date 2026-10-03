import copy
import json
import numpy as np
import pytest
from edgequant import model as M
from edgequant.quantize import (convert, compare_plans, UnsupportedLayerError,
                                PrecisionTargetError)
from edgequant.runner import run_quant
from edgequant.evaluate import (evaluate, boundary_check, stress_compare,
                                reverify)
from edgequant.stress import (build_stress_model, build_stress_inputs,
                              build_stress_groups)
from edgequant.deployment import (build_grouped_deployment,
                                  compare_unified_and_grouped,
                                  verify_deployment)
from edgequant.routing import validate_policy_fingerprint, PrecisionRouteError
from edgequant.sessions import (SessionDeploymentRuntime, preview_switch,
                                compare_session_selection_strategies)


def _setup():
    rng = np.random.default_rng(1)
    layers = [
        {"type": "dense", "name": "fc1", "w": rng.normal(0, 0.5, (4, 6)),
         "b": rng.normal(0, 0.1, 6)},
        {"type": "relu", "name": "relu1"},
        {"type": "dense", "name": "fc2", "w": rng.normal(0, 0.5, (6, 2)),
         "b": rng.normal(0, 0.1, 2)},
    ]
    calib = rng.uniform(-1, 1, (32, 4))
    return layers, calib


def test_roundtrip_and_error(tmp_path):
    layers, calib = _setup()
    q = convert(layers, calib, tol_max_abs=0.2, weight_bits=8)
    p = tmp_path / "q.json"
    M.save_quant(p, q)
    q2 = M.load_quant(p)
    rep = evaluate(layers, q2, calib)
    for l in rep["layers"]:
        assert l["sqnr_db"] > 20
        assert l["max_abs_err"] < 0.2


def test_reject_unsupported():
    with pytest.raises(UnsupportedLayerError):
        convert([{"type": "sigmoid", "name": "s1"}], np.zeros((4, 4)))


def test_boundary_detects_overflow():
    layers, calib = _setup()
    q = convert(layers, calib, tol_max_abs=0.2)
    rep, has_risk = boundary_check(layers, q, calib, factor=4.0)
    assert has_risk
    assert rep["risk"]["input_clipped"] > 0


def test_quantized_output_close_to_fp():
    layers, calib = _setup()
    q = convert(layers, calib, tol_max_abs=0.2)
    x = np.linspace(-0.9, 0.9, 4).reshape(1, 4)
    fp = x
    for l in layers:
        fp = fp @ l["w"] + l["b"] if l["type"] == "dense" else np.maximum(fp, 0)
    got, _ = run_quant(q, x)
    assert np.max(np.abs(got - fp)) < 0.1


def test_mixed_precision_keeps_sensitive_layer_higher():
    """紧预算 + 严误差：敏感层必须升到 8bit，其余可保留 4bit。"""
    layers, calib = build_stress_model()
    # 4bit 全量 = 400 bits；只给 fc2 升 8bit 的空间（+48），不够 fc1 再升（+96）
    budget = 448
    q = convert(layers, calib, tol_max_abs=0.025, storage_budget=budget)
    bits = {l["name"]: l.get("w_bits") for l in q["layers"]
            if l["type"] == "dense"}
    assert bits == {"fc1": 4, "fc2": 8}
    assert q["mixed_precision_report"]["total_bits"] <= budget
    for n in ("fc1", "fc2"):
        assert q["mixed_precision_report"]["final_layer_max_abs"][n] <= 0.025


def test_mixed_precision_reports_per_layer_cost_and_error():
    layers, calib = build_stress_model()
    q = convert(layers, calib, tol_max_abs=0.03)
    rep = q["mixed_precision_report"]
    assert set(rep["per_layer_bits"]) == {"fc1", "relu1", "fc2"}
    assert rep["per_layer_bits"]["fc1"] > 0
    assert rep["per_layer_bits"]["relu1"] == 0
    assert rep["final_layer_max_abs"]["fc1"] <= 0.03
    assert rep["final_layer_max_abs"]["fc2"] <= 0.03


def test_tight_budget_fails_explicitly():
    """预算连全 8bit 都装不下时必须明确失败，不得降低标准。"""
    layers, calib = build_stress_model()
    with pytest.raises(PrecisionTargetError):
        convert(layers, calib, tol_max_abs=0.025, storage_budget=10)


def test_unreachable_tolerance_fails_explicitly():
    """即便全 8bit 也达不到的误差要求必须明确失败。"""
    layers, calib = build_stress_model()
    with pytest.raises(PrecisionTargetError):
        convert(layers, calib, tol_max_abs=1e-12)


def test_loaded_artifact_matches_conversion(tmp_path):
    """独立载入产物后，压力输入上的输出与风险必须与转换阶段一致。"""
    layers, calib = build_stress_model()
    q = convert(layers, calib, tol_max_abs=0.03, storage_budget=None)
    p = tmp_path / "stress.quant.json"
    M.save_quant(p, q)
    loaded = M.load_quant(p)
    x = build_stress_inputs()
    y1, t1, r1 = run_quant(q, x, return_risk=True)
    y2, t2, r2 = run_quant(loaded, x, return_risk=True)
    assert np.array_equal(y1, y2)
    assert r1 == r2
    rep1 = stress_compare(layers, q, x)
    rep2 = stress_compare(layers, loaded, x)
    assert rep1["quant_output"] == rep2["quant_output"]
    assert rep1["risk"] == rep2["risk"]


def test_stress_inputs_preserve_saturation_and_risk():
    """校准范围外/大累加输入：不能只给看似正常的预测，要保留风险。"""
    layers, calib = build_stress_model()
    q = convert(layers, calib, tol_max_abs=0.03)
    rep = stress_compare(layers, q, build_stress_inputs())
    risk = rep["risk"]
    assert risk["input_out_of_calib"]
    assert risk["input_clipped"] > 0
    # 大累加权重 × 极端输入必须暴露激活饱和或 int32 累加饱和
    assert risk["activation_saturated"] > 0 or \
           risk["acc_int32_saturated"] > 0
    assert risk["has_risk"]


def test_in_range_inputs_have_no_risk():
    """校准范围内正常输入不应误报风险。"""
    layers, calib = build_stress_model()
    q = convert(layers, calib, tol_max_abs=0.03)
    rep = stress_compare(layers, q, calib[:4])
    assert rep["risk"]["input_out_of_calib"] is False
    assert rep["risk"]["acc_int32_saturated"] == 0


# ---------- 多输入组：各自容差 + 共同预算，逐组披露，不平均 ----------

def test_per_group_tolerances_all_reported_independently():
    layers, _ = build_stress_model()
    groups = build_stress_groups()
    q = convert(layers, groups, tol_max_abs=0.05)
    rep = q["mixed_precision_report"]
    names = {g["name"] for g in rep["groups"]}
    assert names == {"nominal", "wide", "edge"}
    for g in rep["groups"]:
        assert "satisfied" in g and "per_layer" in g
        for p in g["per_layer"]:
            # 必须逐组逐层与各自容差比较，而不是拿一个全局平均
            assert p["max_abs_err"] <= p["tol_max_abs"] + 1e-12
            assert p["satisfied"] == (
                p["max_abs_err"] <= p["tol_max_abs"] + 1e-12)
    assert rep["all_groups_satisfied"]


def test_one_group_failure_is_not_hidden_by_average():
    """把一个组的容差压到不可达：即使其余组很宽松也必须整体失败，
    不能用平均误差掩盖该组失效。"""
    layers, _ = build_stress_model()
    groups = build_stress_groups()
    groups[0] = {"name": "nominal", "samples": groups[0]["samples"],
                 "tol_max_abs": 1e-9}
    with pytest.raises(PrecisionTargetError):
        convert(layers, groups, tol_max_abs=1.0)


def test_shared_budget_blocks_otherwise_good_precision():
    """共同预算不够把所有层升到 8bit 时，明确失败、不放宽目标。"""
    layers, _ = build_stress_model()
    groups = build_stress_groups()
    # 收紧各组容差，使 4bit 不达标、必须升级；但共同预算不足以完成升级
    for g in groups:
        g["tol_max_abs"] = 0.02
    with pytest.raises(PrecisionTargetError):
        convert(layers, groups, tol_max_abs=0.02, storage_budget=420)


def test_compare_plans_shows_per_group_improvement_and_degradation():
    layers, _ = build_stress_model()
    groups = build_stress_groups()
    for g in groups:
        g["tol_max_abs"] = 0.02
    cmp = compare_plans(layers, groups, tol_max_abs=0.02,
                        storage_budget=100000)
    assert cmp["feasible"]
    assert cmp["selected_plan"] in {"greedy-mixed", "all-8bit"}
    plans = {p["plan"]: p for p in cmp["plans"]}
    # all-4bit 对这些组不可行，必须保留失败原因而不是静默丢弃
    assert plans["all-4bit"]["feasible"] is False
    pair = next(p for p in cmp["pairwise"]
                if p["from"] == "all-4bit" and p["to"] == "all-8bit")
    effects = {(c["group"], c["layer"]): c["effect"]
               for c in pair["per_layer_effects"]}
    assert any(v == "improved" for v in effects.values())
    # 每条对比都必须标注涉及的输入组、层、精度变化与误差 delta
    for c in pair["per_layer_effects"]:
        assert set(c["bits"]) == {"from", "to"}
        assert c["bits"]["from"] == 4 and c["bits"]["to"] == 8


def test_compare_plans_explicit_failure_when_none_feasible():
    layers, _ = build_stress_model()
    groups = build_stress_groups()
    cmp = compare_plans(layers, groups, tol_max_abs=0.02,
                        storage_budget=300)
    assert cmp["feasible"] is False
    assert cmp["selected_plan"] is None
    assert cmp["failures"]


def test_reverify_after_reload_matches_and_reports_oob(tmp_path):
    layers, _ = build_stress_model()
    groups = build_stress_groups()
    q = convert(layers, groups, tol_max_abs=0.05,
                calib_source="test:stress-groups")
    p = tmp_path / "g.quant.json"
    M.save_quant(p, q)
    loaded = M.load_quant(p)
    rep, ok = reverify(layers, loaded, groups)
    assert ok
    for g in rep["groups"]:
        assert g["matches_recorded_prediction"]
        assert not g["violations"]
    oob = rep["out_of_calib"]
    assert oob["input_out_of_calib"]
    assert oob["input_clipped"] > 0
    assert oob["risk_reported"]
    # 旧方案限制与校准来源保持可追溯
    prov = rep["provenance"]
    assert prov["calib_source"] == "test:stress-groups"
    assert prov["calib_sha256_16"]
    got = {x["name"] for x in prov["calib_groups"]}
    assert got == {"nominal", "wide", "edge"}


def test_calib_groups_roundtrip(tmp_path):
    groups = build_stress_groups()
    p = tmp_path / "g.json"
    M.save_calib_groups(p, M.normalize_groups(groups), source="src-x")
    back = M.load_calib_groups(p)
    assert [n for n, _, _ in back] == ["nominal", "wide", "edge"]
    assert back[0][2] == 0.03
    # 旧的单组校准文件仍可统一读取
    p1 = tmp_path / "c.json"
    import numpy as _np
    M.save_calib(p1, _np.zeros((4, 4)))
    one = M.load_calib_groups(p1)
    assert len(one) == 1 and one[0][0] == "default"


# ---------- 按组精度方案：条件冻结、交叠/未覆盖回退、共同存储、重载复核 ----------

def _routed_setup():
    rng = np.random.default_rng(22)
    layers, _ = build_stress_model()
    n = 16
    nominal = rng.uniform(-0.15, 0.15, (n, 4))
    wide = np.full((n, 4), 0.8) * rng.choice([-1.0, 1.0], (n, 4))
    edge = np.full((n, 4), 0.95) * rng.choice([-1.0, 1.0], (n, 4))
    groups = [
        {"name": "nominal", "samples": nominal, "tol_max_abs": 0.01},
        {"name": "wide", "samples": wide, "tol_max_abs": 0.02},
        {"name": "edge", "samples": edge, "tol_max_abs": 0.03},
    ]
    conditions = [
        {"name": "nominal",
         "all": [{"feature": "__absmax__", "max": 0.5}]},
        {"name": "wide",
         "all": [{"feature": "__absmax__", "min": 0.5,
                  "max": 0.9}]},
        {"name": "edge",
         "all": [{"feature": "__absmax__", "min": 0.9,
                  "max": 1.0}]},
    ]
    return layers, groups, conditions


def _add_edge_plan(dep, state_bits=None):
    dep["storage"]["plan_bits"]["group:wide"] = 448
    dep["plans"]["group:edge"] = copy.deepcopy(dep["plans"]["unified"])
    dep["plans"]["group:edge"]["plan_name"] = "group:edge"
    dep["storage"]["plan_bits"]["group:edge"] = 448
    edge_rule = next(r for r in dep["route_policy"]["rules"]
                     if r["group"] == "edge")
    edge_rule["plan"] = "group:edge"
    if state_bits is not None:
        dep["plans"]["group:edge"]["session_state"] = \
            {"state_bits": state_bits}
    return dep


def test_grouped_plan_compares_storage_and_group_improvements():
    layers, groups, conditions = _routed_setup()
    cmp = compare_unified_and_grouped(
        layers, groups, conditions, tol_max_abs=0.02)
    dep = cmp["deployment"]
    assert cmp["selected_groups"] == ["wide"]
    assert cmp["additional_storage_bits"] == 448
    wide = next(c for c in cmp["improvements"] if c["group"] == "wide")
    assert wide["selected"]
    assert wide["grouped_worst_max_abs"] < wide["unified_worst_max_abs"]
    assert dep["storage"]["shared_unique_total_bits"] == 848
    assert dep["route_policy"]["decision_time"] == \
        "before-prediction-and-frozen"


def test_budget_selects_only_affordable_group_candidates():
    layers, groups, conditions = _routed_setup()
    dep = build_grouped_deployment(
        layers, groups, conditions, tol_max_abs=0.02, storage_budget=800)
    assert dep["storage"]["unified_bits"] == 400
    assert dep["storage"]["shared_unique_total_bits"] == 400
    assert dep["plans"].keys() == {"unified"}
    assert all(r["plan"] == "unified"
               for r in dep["route_policy"]["rules"])


def test_overlap_and_uncovered_have_deterministic_fallback_paths():
    layers, groups, conditions = _routed_setup()
    dep = build_grouped_deployment(
        layers, groups, conditions, tol_max_abs=0.02)
    x = np.array([
        [0.2, 0.0, 0.0, 0.0],
        [0.6, 0.6, 0.0, 0.0],
        [0.95, 0.95, 0.95, 0.95],
        [2.0, 0.0, 0.0, 0.0],
    ])
    # 制造规则交叠：wide 与 edge 都能命中，但必须按固定顺序只走第一条
    dep["route_policy"]["rules"][1]["condition"] = {
        "all": [{"feature": "__absmax__", "min": 0.5, "max": 1.0}]}
    y, _, risk, assignments = __import__(
        "edgequant.deployment", fromlist=["run_deployment"]
    ).run_deployment(dep, x, return_risk=True)
    assert y.shape == (4, 2)
    assert assignments[0]["status"] == "matched"
    assert assignments[2]["status"] == "ambiguous"
    assert assignments[2]["plan"] == "group:wide"
    assert assignments[2]["matching_groups"] == ["wide", "edge"]
    assert assignments[3]["status"] == "uncovered"
    assert assignments[3]["plan"] == "unified"
    assert assignments[3]["fallback_used"]
    assert risk["uncovered_count"] == 1
    assert risk["ambiguous_count"] == 1


def test_grouped_artifact_roundtrip_verifies_paths_errors_and_storage(tmp_path):
    layers, groups, conditions = _routed_setup()
    dep = build_grouped_deployment(
        layers, groups, conditions, tol_max_abs=0.02,
        calib_source="test:routed-groups")
    p = tmp_path / "routed.json"
    M.save_deployment(p, dep)
    loaded = M.load_deployment(p)
    rep, ok = verify_deployment(layers, loaded, groups, conditions)
    assert ok
    wide = next(g for g in rep["groups"] if g["group"] == "wide")
    assert wide["selected_group_plan"]
    assert wide["selected_plan_calibration_domain_errors"]["fc2"] < +        wide["unified_plan_calibration_domain_errors"]["fc2"]
    assert rep["storage"]["shared_unique_total_bits"] == 848
    assert rep["boundary"]["all_uncovered_and_fallback"]
    assert rep["boundary"]["plans_executed"] == ["unified"]


def test_changing_group_conditions_invalidates_saved_calibration(tmp_path):
    layers, groups, conditions = _routed_setup()
    dep = build_grouped_deployment(
        layers, groups, conditions, tol_max_abs=0.02)
    changed = copy.deepcopy(conditions)
    changed[1]["all"][0]["max"] = 0.95
    with pytest.raises(PrecisionRouteError):
        validate_policy_fingerprint(dep, groups, changed)
    with pytest.raises(PrecisionRouteError):
        verify_deployment(layers, dep, groups, changed)


# ---------- 有状态会话：最短保持、显式重启/迁移、活跃状态预算与重开 ----------

def test_session_switch_uses_only_current_rule_and_records_restart():
    layers, groups, conditions = _routed_setup()
    dep = build_grouped_deployment(
        layers, groups, conditions, tol_max_abs=0.02,
        session_policy={"min_hold_inputs": 2, "state_bits_per_session": 16},
        runtime_storage_budget=100000)
    _add_edge_plan(dep)
    rt = SessionDeploymentRuntime(dep)
    stream = np.array([[0.2, 0, 0, 0], [0.8, 0.8, 0, 0],
                       [0.8, 0.8, 0, 0], [0.95, 0.95, 0.95, 0.95],
                       [0.95, 0.95, 0.95, 0.95]])
    decisions = []
    for i, row in enumerate(stream):
        decisions.append(rt.process(row, session_id="s1", position=i))

    assert [d["plan"] for d in decisions] == [
        "unified", "unified", "group:wide", "group:wide", "group:edge"]
    assert decisions[1]["assignment"]["status"] == "retained-min-hold"
    switch = decisions[2]["switch"]
    assert switch["action"] == "restart-at-boundary"
    assert switch["compatible"] is False
    assert switch["restart_position"] == 2
    assert switch["previous_output_positions"] == [0, 1]
    assert decisions[3]["assignment"]["status"] == "retained-min-hold"
    session = rt.sessions["s1"]
    assert session["state_lineage"] == [
        {"plan": "unified", "position": 0, "action": "initialize"},
        {"plan": "group:wide", "position": 2,
         "action": "restart-at-boundary"},
        {"plan": "group:edge", "position": 4,
         "action": "restart-at-boundary"}]


def test_preview_does_not_mutate_and_explicit_migration_continues():
    layers, groups, conditions = _routed_setup()
    dep = build_grouped_deployment(
        layers, groups, conditions, tol_max_abs=0.02,
        session_policy={"min_hold_inputs": 2},
        runtime_storage_budget=100000)
    rt = SessionDeploymentRuntime(dep, migrations={
        "unified->group:wide": lambda state, old, new, pos: {
            **state, "plan": new, "migrated_at": pos,
            "samples_seen": state["samples_seen"]}})
    rt.process([0.2, 0, 0, 0], session_id="s", position=0)
    before = copy.deepcopy(rt.sessions["s"])
    preview = preview_switch(before, "group:wide", 1, rt.policy,
                             rt.migrations)
    assert preview["action"] == "retain-min-hold"
    assert preview["previous_output_positions"] == [0]
    assert rt.sessions["s"] == before
    result = rt.process([0.8, 0.8, 0, 0], session_id="s", position=2)
    assert result["switch"]["action"] == "migrate"
    assert result["switch"]["compatible"] is True
    assert rt.sessions["s"]["state"]["migrated_at"] == 2


def test_active_sessions_count_loaded_plans_and_idle_state_storage():
    layers, groups, conditions = _routed_setup()
    dep = build_grouped_deployment(
        layers, groups, conditions, tol_max_abs=0.02,
        session_policy={"state_bits_per_session": 64},
        runtime_storage_budget=1400)
    _add_edge_plan(dep)
    rt = SessionDeploymentRuntime(dep)
    r1 = rt.process([0.8, 0.8, 0, 0], session_id="wide-session", position=0)
    assert r1["storage"]["total_bits"] == 848 + 64
    r2 = rt.process([0.8, 0.8, 0, 0], session_id="same-plan", position=3)
    assert r2["storage"]["loaded_shared_plan_bits"] == 848
    assert r2["storage"]["active_session_state_bits"] == 128
    assert r2["storage"]["budget_satisfied"]
    blocked = rt.process([0.95, 0.95, 0.95, 0.95],
                         session_id="edge-session", position=9)
    assert blocked["plan"] == "unified"
    assert blocked["assignment"]["status"] == "storage-blocked-fallback"
    assert blocked["assignment"]["requested_plan"] == "group:edge"


def test_short_stream_storage_blocked_switch_is_retained_not_rewritten():
    layers, groups, conditions = _routed_setup()
    dep = build_grouped_deployment(
        layers, groups, conditions, tol_max_abs=0.02,
        session_policy={"state_bits_per_session": 100},
        runtime_storage_budget=1000)
    _add_edge_plan(dep, state_bits=200)
    rt = SessionDeploymentRuntime(dep)
    rt.process([0.8, 0.8, 0, 0], session_id="s", position=0)
    blocked = rt.process([0.95, 0.95, 0.95, 0.95], session_id="s",
                         position=1)
    assert blocked["plan"] == "group:wide"
    assert blocked["assignment"]["status"] == "storage-blocked-retain"
    assert blocked["switch"]["reason"] == "storage-budget-exceeded"
    assert rt.sessions["s"]["switches"] == []


def test_session_artifact_roundtrip_continues_same_path(tmp_path):
    layers, groups, conditions = _routed_setup()
    dep = build_grouped_deployment(
        layers, groups, conditions, tol_max_abs=0.02,
        session_policy={"min_hold_inputs": 2},
        runtime_storage_budget=100000)
    _add_edge_plan(dep)
    rt = SessionDeploymentRuntime(dep)
    first = rt.process([0.8, 0.8, 0, 0], session_id="s", position=0)
    M.save_deployment(tmp_path / "sessions.json", rt.to_dict())
    loaded_doc = M.load_deployment(tmp_path / "sessions.json")
    rt2 = SessionDeploymentRuntime.from_dict(loaded_doc)
    retained = rt2.process([0.8, 0.8, 0, 0], session_id="s", position=1)
    held_until_after_retained = rt2.sessions["s"]["held_until"]
    switched = rt2.process([0.2, 0, 0, 0], session_id="s", position=3)
    assert retained["plan"] == "group:wide"
    assert retained["assignment"]["status"] == "matched"
    assert held_until_after_retained == 2
    assert switched["plan"] == "unified"
    assert switched["switch"]["restart_position"] == 3
    assert np.array_equal(switched["output"], first["output"]) is False


def test_strategy_comparison_reports_uncovered_independent_and_stateful():
    layers, groups, conditions = _routed_setup()
    dep = build_grouped_deployment(
        layers, groups, conditions, tol_max_abs=0.02,
        session_policy={"min_hold_inputs": 1},
        runtime_storage_budget=100000)
    _add_edge_plan(dep)
    stream = np.array([[0.2, 0, 0, 0], [0.8, 0.8, 0, 0],
                       [0.2, 0, 0, 0], [2.0, 0, 0, 0]])
    rep = compare_session_selection_strategies(
        dep, {"a": stream})
    assert rep["unified"]["a"]["plan"] == "unified"
    assert [a["plan"] for a in rep["independent"]["a"]["assignments"]] \
        == ["unified", "group:wide", "unified", "unified"]
    assert [d["plan"] for d in rep["stateful"]["a"]["decisions"]] == [
        "unified", "group:wide", "unified", "unified"]
    assert rep["independent"]["a"]["assignments"][3]["status"] == \
        "uncovered"


def test_changing_hold_rule_invalidates_saved_session_policy():
    layers, groups, conditions = _routed_setup()
    dep = build_grouped_deployment(
        layers, groups, conditions, tol_max_abs=0.02,
        session_policy={"min_hold_inputs": 2})
    assert validate_policy_fingerprint(
        dep, groups, conditions, {"min_hold_inputs": 2})
    with pytest.raises(PrecisionRouteError):
        validate_policy_fingerprint(
            dep, groups, conditions, {"min_hold_inputs": 3})
