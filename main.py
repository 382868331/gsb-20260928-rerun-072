"""CLI：make-model / convert / evaluate / boundary / stress。"""
import argparse
import json
import numpy as np
from edgequant import model as M
from edgequant.quantize import convert, UnsupportedLayerError, PrecisionTargetError
from edgequant.evaluate import evaluate, boundary_check, stress_compare
from edgequant.stress import build_stress_model, build_stress_inputs


def make_model(path):
    rng = np.random.default_rng(42)
    layers = [
        {"type": "dense", "name": "fc1",
         "w": rng.normal(0, 0.5, (4, 8)), "b": rng.normal(0, 0.1, 8)},
        {"type": "relu", "name": "relu1"},
        {"type": "dense", "name": "fc2",
         "w": rng.normal(0, 0.5, (8, 3)), "b": rng.normal(0, 0.1, 3)},
    ]
    M.save_model(path, layers)
    calib = rng.uniform(-1.0, 1.0, (64, 4))
    M.save_calib(path + ".calib.json", calib)
    print(f"model -> {path}; calib -> {path}.calib.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make-model", "convert", "evaluate",
                                    "boundary", "stress"])
    ap.add_argument("--model", default="artifacts/model.json")
    ap.add_argument("--calib", default="artifacts/model.json.calib.json")
    ap.add_argument("--quant", default="artifacts/model.quant.json")
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="允许的逐层 max_abs_err")
    ap.add_argument("--budget", type=int, default=None,
                    help="权重+偏置存储预算（bits）")
    ap.add_argument("--weight-bits", type=int, default=None, choices=[4, 8],
                    help="强制权重精度；缺省为自动混合精度")
    args = ap.parse_args()

    if args.cmd == "make-model":
        make_model(args.model)
        return

    if args.cmd == "stress":
        layers, calib = build_stress_model()
        try:
            q = convert(layers, calib, tol_max_abs=args.tol,
                        storage_budget=args.budget,
                        weight_bits=args.weight_bits)
        except PrecisionTargetError as e:
            print(f"CONVERT FAILED (目标不可达): {e}")
            raise SystemExit(3)
        rep = stress_compare(layers, q, build_stress_inputs())
        print(json.dumps({
            "mixed_precision_report": q["mixed_precision_report"],
            "stress": rep}, indent=2, ensure_ascii=False))
        return

    layers = M.load_model(args.model)
    calib = M.load_calib(args.calib)
    if args.cmd == "convert":
        try:
            q = convert(layers, calib, tol_max_abs=args.tol,
                        storage_budget=args.budget,
                        weight_bits=args.weight_bits)
        except UnsupportedLayerError as e:
            print(f"CONVERT REJECTED: {e}")
            raise SystemExit(2)
        except PrecisionTargetError as e:
            print(f"CONVERT FAILED (目标不可达): {e}")
            raise SystemExit(3)
        M.save_quant(args.quant, q)
        rep = q["mixed_precision_report"]
        print(f"quant model -> {args.quant}")
        print(f"逐层选择: {rep['per_layer_bits']} bits, 合计 {rep['total_bits']}"
              f" / 预算 {rep['budget_bits']}")
        print(f"逐层 max_abs_err: {rep['final_layer_max_abs']}")
        return

    q = M.load_quant(args.quant)
    if args.cmd == "evaluate":
        rep = evaluate(layers, q, calib)
    else:
        rep, has_risk = boundary_check(layers, q, calib)
        print(f"out-of-calib risk detected: {has_risk}")
    print(json.dumps(rep, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
