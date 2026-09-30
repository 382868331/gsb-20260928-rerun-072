"""CLI：make-model / convert / evaluate / boundary。

convert 支持 --max-err 与 --storage-budget：给定逐层允许误差与存储预算后，
自动做逐层混合精度选择（敏感层升 16 位或保留浮点）；约束无法满足时
以退出码 3 明确失败，不降低误差标准。
"""
import argparse
import json
import numpy as np
from edgequant import model as M
from edgequant.quantize import convert, UnsupportedLayerError
from edgequant.evaluate import evaluate, boundary_check
from edgequant.mixed import convert_mixed, PrecisionBudgetError


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
    ap.add_argument("cmd", choices=["make-model", "convert", "evaluate", "boundary"])
    ap.add_argument("--model", default="artifacts/model.json")
    ap.add_argument("--calib", default="artifacts/model.json.calib.json")
    ap.add_argument("--quant", default="artifacts/model.quant.json")
    ap.add_argument("--max-err", type=float, default=None,
                    help="逐层允许最大绝对误差；给出后启用混合精度选择")
    ap.add_argument("--storage-budget", type=int, default=None,
                    help="转换产物存储预算（字节）")
    args = ap.parse_args()

    if args.cmd == "make-model":
        make_model(args.model)
        return
    layers = M.load_model(args.model)
    calib = M.load_calib(args.calib)
    if args.cmd == "convert":
        try:
            if args.max_err is not None:
                q, rep = convert_mixed(layers, calib, args.max_err,
                                       args.storage_budget)
                print(json.dumps({"mixed_precision": rep}, indent=2,
                                 ensure_ascii=False))
            else:
                q = convert(layers, calib)
        except UnsupportedLayerError as e:
            print(f"CONVERT REJECTED: {e}")
            raise SystemExit(2)
        except PrecisionBudgetError as e:
            print(f"CONVERT FAILED: {e}")
            raise SystemExit(3)
        M.save_quant(args.quant, q)
        print(f"quant model -> {args.quant}")
        return
    q = M.load_quant(args.quant)
    if args.cmd == "evaluate":
        rep = evaluate(layers, q, calib)
    else:
        rep, overflow = boundary_check(layers, q, calib)
        print(f"overflow(饱和) detected: {overflow}")
    print(json.dumps(rep, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
