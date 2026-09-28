import numpy as np
import pytest
from edgequant import model as M
from edgequant.quantize import convert, UnsupportedLayerError
from edgequant.runner import run_quant
from edgequant.evaluate import evaluate, boundary_check


def _setup(tmp_path):
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
    layers, calib = _setup(tmp_path)
    q = convert(layers, calib)
    p = tmp_path / "q.json"
    M.save_quant(p, q)
    q2 = M.load_quant(p)
    rep = evaluate(layers, q2, calib)
    for l in rep["layers"]:
        assert l["sqnr_db"] > 20  # int8 至少应有可接受的信噪比
        assert l["max_abs_err"] < 0.2


def test_reject_unsupported():
    layers = [{"type": "sigmoid", "name": "s1"}]
    with pytest.raises(UnsupportedLayerError):
        convert(layers, np.zeros((4, 4)))


def test_boundary_detects_overflow():
    layers, calib = _setup(None)
    q = convert(layers, calib)
    rep, overflow = boundary_check(layers, q, calib, factor=4.0)
    assert overflow  # 4 倍超范围输入必须被检出饱和


def test_quantized_output_close_to_fp():
    layers, calib = _setup(None)
    q = convert(layers, calib)
    x = np.linspace(-0.9, 0.9, 4).reshape(1, 4)
    fp = x
    for l in layers:
        fp = fp @ l["w"] + l["b"] if l["type"] == "dense" else np.maximum(fp, 0)
    got, _ = run_quant(q, x)
    assert np.max(np.abs(got - fp)) < 0.1

