import json
import numpy as np
import pytest
from edgequant import model as M
from edgequant.quantize import (convert, UnsupportedLayerError,
                                PrecisionTargetError)
from edgequant.runner import run_quant
from edgequant.evaluate import evaluate, boundary_check, stress_compare
from edgequant.stress import build_stress_model, build_stress_inputs


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
