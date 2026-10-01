"""第二轮：多输入组约束、方案比较、显式失败、重载复核、溯源。"""
import json
import numpy as np
import pytest
from edgequant import model as M
from edgequant.quantize import convert_groups, PrecisionTargetError
from edgequant.groups import InputGroup, save_groups, load_groups
from edgequant import plans as P
from edgequant.compare import compare_qmodels
from edgequant.verify import reload_recheck
from edgequant.stress import build_stress_model, build_stress_inputs


def _groups(calib, tol=0.03):
    rng = np.random.default_rng(11)
    return [
        InputGroup("nominal", calib, tol, in_calib=True),
        InputGroup("tilted", rng.uniform(-0.8, 0.2, (24, 4)),
                   tol, in_calib=True),
        InputGroup("over-range", rng.uniform(-4.0, 4.0, (8, 4)),
                   {"fc1": 5.0, "fc2": 5.0}, in_calib=False),
    ]


def test_each_group_reported_independently():
    layers, calib = build_stress_model()
    groups = _groups(calib)
    q = convert_groups(layers, groups, storage_budget=600)
    reps = {r["name"]: r for r in q["mixed_precision_report"]["group_reports"]}
    assert set(reps) == {"nominal", "tilted", "over-range"}
    for name, rep in reps.items():
        assert rep["tol_satisfied"] is True
        for lname, cell in rep["per_layer_max_abs"].items():
            assert cell["max_abs_err"] <= cell["tol_max_abs"] + 1e-12
            assert cell["satisfied"] is True
    # 越界组必须报告裁剪与饱和风险，不只是给出看似正常的预测
    oob = reps["over-range"]
    assert oob["in_calib_range"] is False
    assert oob["input_clipped"] > 0
    assert oob["has_risk"]


def test_single_group_failure_not_masked_by_average():
    """一组严限制、另一组松限制：严组超差时整体必须失败，
    不允许用两组平均误差掩盖。"""
    layers, calib = build_stress_model()
    rng = np.random.default_rng(5)
    groups = [
        InputGroup("easy", rng.uniform(-0.2, 0.2, (16, 4)),
                   10.0, in_calib=True),
        InputGroup("strict", rng.uniform(-1.0, 1.0, (32, 4)),
                   1e-9, in_calib=True),
    ]
    with pytest.raises(PrecisionTargetError) as ei:
        convert_groups(layers, groups)
    assert "strict" in str(ei.value)


def test_tight_shared_budget_fails_explicitly():
    layers, calib = build_stress_model()
    groups = _groups(calib, tol=0.001)
    with pytest.raises(PrecisionTargetError):
        convert_groups(layers, groups, storage_budget=10)


def test_calibration_source_and_constraints_traceable(tmp_path):
    layers, calib = build_stress_model()
    groups = _groups(calib)
    q = convert_groups(layers, groups, storage_budget=600, plan_id="p1")
    prov = q["provenance"]
    assert prov["plan_id"] == "p1"
    assert prov["original_constraints"]["storage_budget_bits"] == 600
    src = prov["calibration_source"]
    assert src["source_groups"] == ["nominal", "tilted"]
    assert "__input__" in src["activation_scales"]
    group_names = {g["name"] for g in q["mixed_precision_report"]["input_groups"]}
    assert group_names == {"nominal", "tilted", "over-range"}


def test_reload_recheck_matches_predictions_and_risk(tmp_path):
    layers, calib = build_stress_model()
    groups = _groups(calib)
    q = convert_groups(layers, groups, storage_budget=600)
    rep = reload_recheck(q, groups, str(tmp_path / "recheck.json"))
    assert rep["reload_prediction_match"]
    names = {d["group"] for d in rep["per_group"]}
    assert names == {"nominal", "tilted", "over-range"}
    oob = [d for d in rep["per_group"] if d["group"] == "over-range"][0]
    assert oob["input_out_of_calib"]
    assert oob["input_clipped"] > 0


def test_plan_archive_append_only_and_comparison(tmp_path):
    layers, calib = build_stress_model()
    groups = _groups(calib)
    store_path = tmp_path / "plans.json"

    qa = convert_groups(layers, groups, storage_budget=600, plan_id="mix")
    pa = tmp_path / "mix.json"
    M.save_quant(pa, qa)
    store = P.load_store(store_path)
    P.add_plan(store, "mix", str(pa), qa, [g.as_meta() for g in groups])
    P.save_store(store_path, store)

    qb = convert_groups(layers, groups, weight_bits=8, plan_id="full8")
    pb = tmp_path / "full8.json"
    M.save_quant(pb, qb)
    store = P.load_store(store_path)
    P.add_plan(store, "full8", str(pb), qb, [g.as_meta() for g in groups])
    P.save_store(store_path, store)

    # 旧方案不允许覆盖
    with pytest.raises(ValueError):
        P.add_plan(store, "mix", str(pa), qa, [])

    store = P.load_store(store_path)
    assert [p["plan_id"] for p in store["plans"]] == ["mix", "full8"]
    cmp = compare_qmodels(
        layers, groups,
        ("mix", M.load_quant(pa)), ("full8", M.load_quant(pb)))
    changes = {c["layer"]: c["change"]
               for c in cmp["layer_precision_changes"]}
    assert any(v != "unchanged" for v in changes.values())
    total_better = sum(gc["layers_improved"]
                       for gc in cmp["group_comparison"])
    assert total_better >= 1
    for gc in cmp["group_comparison"]:
        # 两个方案都满足约束；逐组逐层的改善/退化必须被独立记录，
        # 不允许把退化组用平均误差藏掉
        assert len(gc["per_layer"]) >= 1
        assert gc["a_all_satisfied"] and gc["b_all_satisfied"]
        recorded = gc["layers_improved"] + gc["layers_degraded"]
        n_changed = sum(p["verdict"] != "unchanged"
                        for p in gc["per_layer"])
        assert recorded == n_changed


def test_comparison_detects_degradation():
    """混合 -> 全 8bit 时个别组/层可能轻微退化（尺度相互作用），
    比较必须指出具体组、具体层的退化，不能只报总平均。"""
    layers, calib = build_stress_model()
    groups = _groups(calib)
    qa = convert_groups(layers, groups, storage_budget=600)
    qb = convert_groups(layers, groups, weight_bits=8)
    cmp = compare_qmodels(layers, groups, ("mix", qa), ("full8", qb))
    found = []
    for gc in cmp["group_comparison"]:
        for p in gc["per_layer"]:
            if p["verdict"] == "degraded":
                assert p["delta_b_minus_a"] > 0
                found.append((gc["group"], p["layer"]))
    assert found  # 至少一处退化被逐组逐层点名


def test_groups_roundtrip(tmp_path):
    gpath = tmp_path / "g.json"
    groups = [InputGroup("a", np.zeros((2, 3)), 0.1, True),
              InputGroup("b", np.ones((1, 3)), {"fc1": 0.2}, False)]
    save_groups(gpath, groups)
    loaded = load_groups(gpath)
    assert [g.name for g in loaded] == ["a", "b"]
    assert loaded[1].in_calib is False
    assert loaded[1].layer_tol("fc1") == 0.2
    assert loaded[1].layer_tol("fc2") is None
