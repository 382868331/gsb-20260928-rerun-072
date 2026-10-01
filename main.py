"""CLI：make-model / convert / compare / evaluate / boundary / stress /
reverify。支持多输入组（各自容差 + 共同存储预算）。"""
import argparse
import json
import numpy as np
from edgequant import model as M
from edgequant.quantize import (convert, compare_plans, UnsupportedLayerError,
                                PrecisionTargetError)
from edgequant.evaluate import (evaluate, boundary_check, stress_compare,
                                reverify)
from edgequant.stress import (build_stress_model, build_stress_inputs,
                              build_stress_groups)


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
    # 三个代表性输入组：同分布窄幅 / 宽幅 / 贴边，各自容差
    groups = [
        {"name": "nominal",
         "samples": rng.uniform(-0.8, 0.8, (48, 4)),
         "tol_max_abs": 0.05},
        {"name": "wide",
         "samples": rng.uniform(-1.0, 1.0, (48, 4)),
         "tol_max_abs": 0.10},
        {"name": "edge",
         "samples": rng.uniform(0.85, 1.0, (32, 4))
                    * rng.choice([-1.0, 1.0], (32, 4)),
         "tol_max_abs": 0.15},
    ]
    gpath = path + ".calib-groups.json"
    M.save_calib_groups(gpath, M.normalize_groups(groups))
    print(f"model -> {path}; calib -> {path}.calib.json; groups -> {gpath}")


def _load_groups(calib_path):
    """校准文件可能是单组 v1 或多组 v1-groups。"""
    return M.load_calib_groups(calib_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make-model", "convert", "compare",
                                    "evaluate", "boundary", "stress",
                                    "reverify"])
    ap.add_argument("--model", default="artifacts/model.json")
    ap.add_argument("--calib",
                    default="artifacts/model.json.calib-groups.json")
    ap.add_argument("--quant", default="artifacts/model.quant.json")
    ap.add_argument("--comparison",
                    default="artifacts/plan-comparison.json")
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
        layers, _ = build_stress_model()
        calib = build_stress_groups()
        try:
            q = convert(layers, calib, tol_max_abs=args.tol,
                        storage_budget=args.budget,
                        weight_bits=args.weight_bits,
                        calib_source="stress:build_stress_groups")
        except PrecisionTargetError as e:
            print(f"CONVERT FAILED (目标不可达): {e}")
            raise SystemExit(3)
        rep = stress_compare(layers, q, build_stress_inputs())
        vrep, vok = reverify(layers, q, calib)
        print(json.dumps({
            "mixed_precision_report": q["mixed_precision_report"],
            "stress": rep,
            "reverify_passed": vok,
            "reverify": vrep}, indent=2, ensure_ascii=False))
        return

    layers = M.load_model(args.model)
    groups = _load_groups(args.calib)
    if args.cmd == "convert":
        try:
            q = convert(layers, groups, tol_max_abs=args.tol,
                        storage_budget=args.budget,
                        weight_bits=args.weight_bits,
                        calib_source=args.calib)
        except UnsupportedLayerError as e:
            print(f"CONVERT REJECTED: {e}")
            raise SystemExit(2)
        except PrecisionTargetError as e:
            print(f"CONVERT FAILED (目标不可达): {e}")
            raise SystemExit(3)
        M.save_quant(args.quant, q)
        rep = q["mixed_precision_report"]
        print(f"quant model -> {args.quant}")
        print(f"逐层权重精度: {rep['per_layer_weight_bits']}, "
              f"合计 {rep['total_bits']} / 预算 {rep['budget_bits']} bits")
        for g in rep["groups"]:
            flag = "PASS" if g["satisfied"] else "FAIL"
            print(f"  输入组 {g['name']}: {flag} "
                  f"(tol={g['tol_max_abs']}, "
                  f"worst={g['max_abs_err_overall']:.6g})")
        if not rep["all_groups_satisfied"]:
            print("CONVERT FAILED: 存在输入组未满足约束（不自动放宽目标）")
            raise SystemExit(3)
        return

    if args.cmd == "compare":
        cmp = compare_plans(layers, groups, tol_max_abs=args.tol,
                            storage_budget=args.budget)
        with open(args.comparison, "w") as f:
            json.dump(cmp, f, indent=2, ensure_ascii=False)
        print(f"plan comparison -> {args.comparison}")
        for p in cmp["plans"]:
            if not p.get("feasible", True) and "error" in p:
                print(f"  {p['plan']}: 不可行 ({p['error']})")
                continue
            status = "可行" if p["feasible"] else "有组未达标"
            print(f"  {p['plan']}: {status} ({p['total_bits']} bits)")
            for g in p["groups"]:
                print(f"      组 {g['name']}: "
                      f"{'PASS' if g['satisfied'] else 'FAIL'} "
                      f"worst={g['max_abs_err_overall']:.6g}")
        print(f"选中方案: {cmp['selected_plan']}")
        if not cmp["feasible"]:
            print("COMPARE FAILED: 没有方案满足全部输入组约束与共同预算"
                  "（不自动放宽目标）")
            raise SystemExit(3)
        return

    q = M.load_quant(args.quant)
    if args.cmd == "evaluate":
        name0, samples0, _ = groups[0]
        rep = evaluate(layers, q, samples0, label=name0)
        print(json.dumps(rep, indent=2, ensure_ascii=False))
        return
    if args.cmd == "boundary":
        rep, has_risk = boundary_check(layers, q,
                                       np.vstack([s for _, s, _ in groups]))
        print(f"out-of-calib risk detected: {has_risk}")
        print(json.dumps(rep, indent=2, ensure_ascii=False))
        return
    if args.cmd == "reverify":
        rep, ok = reverify(layers, q, groups)
        print(f"reverify passed: {ok}")
        print(json.dumps(rep, indent=2, ensure_ascii=False))
        if not ok:
            raise SystemExit(3)


if __name__ == "__main__":
    main()
