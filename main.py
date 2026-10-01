"""CLI：make-model / convert / convert-groups / evaluate / boundary /
stress / compare-plans / list-plans。

第二轮口径：支持多个代表性输入组，各组独立误差限制、共同存储预算；
转换成功后做产物重载复核并归档方案（旧方案不可覆盖）；方案之间可
逐层、逐组比较改善与退化；没有方案满足全部约束时退出码 3，不放宽目标。
"""
import argparse
import json
import time
import numpy as np
from edgequant import model as M
from edgequant.quantize import (convert, convert_groups,
                                UnsupportedLayerError, PrecisionTargetError)
from edgequant.evaluate import evaluate, boundary_check, stress_compare
from edgequant.stress import build_stress_model, build_stress_inputs
from edgequant.groups import (InputGroup, save_groups, load_groups,
                              groups_from_calib)
from edgequant import plans as P
from edgequant.compare import compare_qmodels
from edgequant.verify import reload_recheck


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

    # 多代表性输入组：常规小输入、幅度较大的工作区输入（仍在各自范围内）、
    # 以及明确超校准范围的越界组（必须报告裁剪与饱和风险）。
    rng2 = np.random.default_rng(43)
    groups = [
        InputGroup("typical", rng2.uniform(-1.0, 1.0, (64, 4)),
                   0.1, in_calib=True),
        InputGroup("wide-range", rng2.uniform(-0.6, 0.6, (32, 4)),
                   0.1, in_calib=True),
        InputGroup("out-of-calib", rng2.uniform(-2.0, 2.0, (16, 4)),
                   {"fc1": 1.5, "fc2": 2.0}, in_calib=False),
    ]
    gpath = path + ".groups.json"
    save_groups(gpath, groups)
    print(f"model -> {path}; calib -> {path}.calib.json; groups -> {gpath}")


def _print_group_summary(report):
    for gr in report["group_reports"]:
        scope = "校准内" if gr["in_calib_range"] else "超校准范围"
        flag = "满足" if gr["tol_satisfied"] else "超差"
        per = ", ".join(
            f"{k}:{v['max_abs_err']:.4g}<={v['tol_max_abs']}"
            if v["satisfied"]
            else f"{k}:{v['max_abs_err']:.4g}>{v['tol_max_abs']}"
            for k, v in gr["per_layer_max_abs"].items())
        risk = (f" 风险(输入裁剪 {gr['input_clipped']}, 激活饱和 "
                f"{gr['activation_saturated']}, int32饱和 "
                f"{gr['acc_int32_saturated']})")
        print(f"  组[{gr['name']}|{scope}] {flag} | {per}{risk}")


def _convert_groups_cmd(args):
    layers = M.load_model(args.model)
    groups = load_groups(args.groups)
    plan_id = args.plan_id or f"plan-{int(time.time())}"
    try:
        q = convert_groups(layers, groups,
                           storage_budget=args.budget,
                           weight_bits=args.weight_bits,
                           plan_id=plan_id)
    except UnsupportedLayerError as e:
        print(f"CONVERT REJECTED: {e}")
        raise SystemExit(2)
    except PrecisionTargetError as e:
        print(f"CONVERT FAILED (全部方案均不满足约束，不自动放宽): {e}")
        raise SystemExit(3)

    # 产物重载后在所有输入组上重复核对预测与风险
    recheck = reload_recheck(q, groups, args.quant + ".recheck.tmp")
    M.save_quant(args.quant, q)

    store = P.load_store(args.store)
    P.add_plan(store, plan_id, args.quant, q,
               groups_meta=[g.as_meta() for g in groups],
               note=args.note)
    store.setdefault("reload_rechecks", {})[plan_id] = recheck
    P.save_store(args.store, store)

    rep = q["mixed_precision_report"]
    print(f"quant model -> {args.quant}; 方案 {plan_id} 归档 -> {args.store}")
    print(f"逐层权重精度: {rep['w_bits_for']}; 合计 {rep['total_bits']} "
          f"/ 共同预算 {rep['budget_bits']}")
    _print_group_summary(rep)
    print(f"重载复核: 预测/风险逐元素一致 ({len(groups)} 组)")


def _compare_cmd(args):
    layers = M.load_model(args.model)
    groups = load_groups(args.groups)
    store = P.load_store(args.store)
    pa, pb = P.get_plan(store, args.plan_a), P.get_plan(store, args.plan_b)
    def _qm(p):
        return p.get("quant_snapshot") or M.load_quant(p["quant_artifact"])
    cmp = compare_qmodels(
        layers, groups,
        (pa["plan_id"], _qm(pa)),
        (pb["plan_id"], _qm(pb)))
    print(f"方案 {args.plan_a} -> {args.plan_b} "
          f"(存储 {cmp['storage_bits_a']} -> {cmp['storage_bits_b']} bits)")
    for ch in cmp["layer_precision_changes"]:
        print(f"  层 {ch['layer']}: {ch['w_bits_a']} -> {ch['w_bits_b']} "
              f"bit ({ch['change']})")
    for gc in cmp["group_comparison"]:
        print(f"  组[{gc['group']}] 改善层 {gc['layers_improved']} / "
              f"退化层 {gc['layers_degraded']} | "
              f"A全满足={gc['a_all_satisfied']} B全满足={gc['b_all_satisfied']}")
        for p in gc["per_layer"]:
            if p["verdict"] != "unchanged":
                print(f"    {p['layer']}: {p['max_abs_err_a']:.4g} -> "
                      f"{p['max_abs_err_b']:.4g} ({p['verdict']})")
    if args.json:
        print(json.dumps(cmp, indent=2, ensure_ascii=False, default=str))


def _list_plans_cmd(args):
    store = P.load_store(args.store)
    for p in store["plans"]:
        rep = p["mixed_precision_report"]
        ok = rep["all_groups_satisfied"]
        print(f"{p['plan_id']}: 存储 {rep['total_bits']}/"
              f"{rep['budget_bits']} bits, 全组满足={ok}, "
              f"产物 {p['quant_artifact']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make-model", "convert", "convert-groups",
                                    "evaluate", "boundary", "stress",
                                    "compare-plans", "list-plans"])
    ap.add_argument("--model", default="artifacts/model.json")
    ap.add_argument("--calib", default="artifacts/model.json.calib.json")
    ap.add_argument("--groups", default="artifacts/model.json.groups.json")
    ap.add_argument("--quant", default="artifacts/model.quant.json")
    ap.add_argument("--store", default="artifacts/plans.json")
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="允许的逐层 max_abs_err（单校准集旧口径）")
    ap.add_argument("--budget", type=int, default=None,
                    help="权重+偏置共同存储预算（bits）")
    ap.add_argument("--weight-bits", type=int, default=None, choices=[4, 8],
                    help="强制权重精度；缺省为自动混合精度")
    ap.add_argument("--plan-id", default=None)
    ap.add_argument("--note", default="")
    ap.add_argument("--plan-a", default=None)
    ap.add_argument("--plan-b", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.cmd == "make-model":
        make_model(args.model)
        return
    if args.cmd == "convert-groups":
        _convert_groups_cmd(args)
        return
    if args.cmd == "compare-plans":
        _compare_cmd(args)
        return
    if args.cmd == "list-plans":
        _list_plans_cmd(args)
        return

    if args.cmd == "stress":
        layers, calib = build_stress_model()
        # 压力场景同样按多输入组处理：校准内 + 异常越界组
        groups = [
            InputGroup("stress-in-calib", calib, 0.025, in_calib=True),
            InputGroup("stress-overrange", build_stress_inputs(),
                       {"fc1": 1e9, "fc2": 1e9}, in_calib=False),
        ]
        try:
            q = convert_groups(layers, groups,
                               storage_budget=args.budget,
                               weight_bits=args.weight_bits,
                               plan_id=args.plan_id)
        except PrecisionTargetError as e:
            print(f"CONVERT FAILED (目标不可达): {e}")
            raise SystemExit(3)
        rep = stress_compare(layers, q, build_stress_inputs())
        print(json.dumps({
            "mixed_precision_report": q["mixed_precision_report"],
            "stress": rep}, indent=2, ensure_ascii=False, default=str))
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
