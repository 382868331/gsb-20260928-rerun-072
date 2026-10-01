"""部署方案之间的比较。

展示哪些层的精度（w_bits）发生变化，以及这些变化在每个输入组上
带来的逐层改善或退化（不靠平均值下结论）。比较直接重放两个产物，
而不是只抄报告里的数字。
"""
from .quantize import group_layer_errs


def _bits_map(qmodel):
    return {l["name"]: l.get("w_bits") for l in qmodel["layers"]}


def compare_qmodels(layers, groups, plan_a, plan_b):
    """plan_a/plan_b: (plan_id, qmodel)。"""
    id_a, qa = plan_a
    id_b, qb = plan_b
    errs_a = group_layer_errs(layers, qa, groups)
    errs_b = group_layer_errs(layers, qb, groups)
    bits_a, bits_b = _bits_map(qa), _bits_map(qb)
    dense_names = [l["name"] for l in qa["layers"]
                   if l.get("w_bits") is not None]

    layer_changes = []
    for name in bits_a:
        ba, bb = bits_a[name], bits_b[name]
        if ba is None:
            continue
        layer_changes.append({
            "layer": name,
            "w_bits_a": ba, "w_bits_b": bb,
            "change": ("upgraded" if bb > ba else
                       "downgraded" if bb < ba else "unchanged"),
        })

    group_cmp = []
    for g in groups:
        per_layer = []
        n_better = n_worse = 0
        for name in errs_a[g.name]["errs"]:
            ea = errs_a[g.name]["errs"][name]
            eb = errs_b[g.name]["errs"][name]
            delta = eb - ea
            if abs(delta) <= 1e-15:
                verdict = "unchanged"
            elif delta < 0:
                verdict = "improved"
                n_better += 1
            else:
                verdict = "degraded"
                n_worse += 1
            lim = g.layer_tol(name)
            per_layer.append({
                "layer": name,
                "max_abs_err_a": ea, "max_abs_err_b": eb,
                "delta_b_minus_a": delta, "verdict": verdict,
                "tol_max_abs": lim,
                "a_satisfies": (lim is None) or ea <= lim,
                "b_satisfies": (lim is None) or eb <= lim,
            })
        group_cmp.append({
            "group": g.name,
            "in_calib_range": g.in_calib,
            "layers_improved": n_better,
            "layers_degraded": n_worse,
            "a_all_satisfied": all(
                p["a_satisfies"] for p in per_layer
                if p["layer"] in dense_names),
            "b_all_satisfied": all(
                p["b_satisfies"] for p in per_layer
                if p["layer"] in dense_names),
            "per_layer": per_layer,
        })

    cost_a = qa["mixed_precision_report"]["total_bits"]
    cost_b = qb["mixed_precision_report"]["total_bits"]
    return {
        "plan_a": id_a, "plan_b": id_b,
        "storage_bits_a": cost_a, "storage_bits_b": cost_b,
        "storage_delta_b_minus_a": cost_b - cost_a,
        "layer_precision_changes": layer_changes,
        "group_comparison": group_cmp,
    }
