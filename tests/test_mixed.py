"""第二轮测试：逐层混合精度 + 超校准范围风险保留。

小模型刻意包含：大累加（128 维点积）、量级悬殊的通道、异常输入。
"""
import numpy as np
import pytest
from edgequant import model as M
from edgequant.mixed import convert_mixed, PrecisionBudgetError
from edgequant.runner import run_quant
from edgequant.evaluate import evaluate, boundary_check
from edgequant.layers import FP_LAYERS


def _hard_model():
    rng = np.random.default_rng(7)
    w1 = rng.normal(0, 0.3, (128, 16))          # 大累加：128 维点积
    w1[:, 0] *= 50.0                             # 量级悬殊的通道
    w2 = rng.normal(0, 0.3, (16, 4))
    w2[3, :] *= 0.01
    layers = [
        {"type": "dense", "name": "fc1", "w": w1,
         "b": rng.normal(0, 0.1, 16)},
        {"type": "relu", "name": "relu1"},
        {"type": "dense", "name": "fc2", "w": w2,
         "b": rng.normal(0, 0.1, 4)},
    ]
    calib = rng.uniform(-1, 1, (64, 128))
    return layers, calib


def _fp_run(layers, x):
    for l in layers:
        x = FP_LAYERS[l["type"]](x, l["w"], l["b"]) if l["type"] == "dense" \
            else FP_LAYERS[l["type"]](x)
    return x


def test_mixed_selects_precision_within_budget():
    layers, calib = _hard_model()
    q, rep = convert_mixed(layers, calib, max_err=0.05,
                           storage_budget=100_000)
    assert rep["storage_bytes"] <= 100_000
    for l in rep["layers"]:
        assert l["bits"] in (8, 16, 32)
        assert l["max_abs_err"] <= 0.05  # 达到误差要求才算成功
        assert l["bytes"] > 0
    # 难模型下至少有一层被升级到更高精度（敏感层保留）
    assert any(l["bits"] > 8 for l in rep["layers"])


def test_mixed_report_matches_artifact():
    layers, calib = _hard_model()
    q, rep = convert_mixed(layers, calib, max_err=0.05)
    for ql, rl in zip(q["layers"], rep["layers"]):
        assert ql["bits"] == rl["bits"]


def test_independent_load_matches_conversion(tmp_path):
    """独立载入产物执行，结果必须与转换阶段完全一致。"""
    layers, calib = _hard_model()
    q, _ = convert_mixed(layers, calib, max_err=0.05)
    p = tmp_path / "q.json"
    M.save_quant(p, q)
    q2 = M.load_quant(p)
    x = np.linspace(-0.8, 0.8, 128).reshape(1, 128)
    out1, _ = run_quant(q, x)
    out2, _ = run_quant(q2, x)
    np.testing.assert_array_equal(out1, out2)


def test_impossible_error_fails_loudly():
    """误差要求无法满足 + 预算卡死时，必须明确失败而非降低标准。"""
    layers, calib = _hard_model()
    with pytest.raises(PrecisionBudgetError):
        convert_mixed(layers, calib, max_err=1e-9, storage_budget=200)


def test_anomalous_input_keeps_saturation_and_risk():
    """异常（超校准范围）输入：报告保留饱和计数与风险标记。"""
    layers, calib = _hard_model()
    q, rep = convert_mixed(layers, calib, max_err=2.0)
    assert any(l["bits"] < 32 for l in rep["layers"])  # 仍有定点层
    rep, overflow = boundary_check(layers, q, calib, factor=8.0)
    assert overflow
    assert rep["risk"]["overflow"]
    assert rep["risk"]["saturated_in_total"] > 0
    assert "layers" in rep and len(rep["layers"]) == len(layers)


def test_mixed_output_close_to_fp():
    layers, calib = _hard_model()
    q, _ = convert_mixed(layers, calib, max_err=0.05)
    x = np.linspace(-0.5, 0.5, 128).reshape(1, 128)
    got, _ = run_quant(q, x)
    ref = _fp_run(layers, x)
    assert np.max(np.abs(got - ref)) < 0.05
