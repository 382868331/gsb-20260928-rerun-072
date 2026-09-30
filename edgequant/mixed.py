"""逐层混合精度选择。

用户给出允许误差（逐层 max_abs_err 上限）与可选存储预算（字节）：
从全 int8 出发，贪心地把当前误差最差的层升一档（8 -> 16 -> 32 浮点）；
升级后若总存储超出预算，或层已到最高精度仍超差，明确抛
PrecisionBudgetError，绝不暗中放宽误差标准。
返回 (转换产物, 逐层报告)，报告含每层选定精度、存储代价与实测误差。
"""
from .layers import PRECISIONS
from .quantize import convert
from .evaluate import evaluate


class PrecisionBudgetError(Exception):
    """误差要求无法在精度/存储约束内满足，明确失败。"""
    pass


def layer_storage_bytes(layer, bits):
    """该层在给定位宽下的存储代价（字节）。浮点层按 float32 计。"""
    eff = 4 if bits == 32 else bits // 8
    n = 0
    if layer["type"] == "dense":
        n += layer["w"].size * eff + layer["b"].size * 4  # bias 固定 4 字节
    return n + 16  # scale/元信息常数开销


def total_storage_bytes(layers, bits_map):
    return sum(layer_storage_bytes(l, bits_map.get(l["name"], 8))
               for l in layers)


def _upgrade(bits):
    i = PRECISIONS.index(bits)
    return PRECISIONS[i + 1] if i + 1 < len(PRECISIONS) else None


def convert_mixed(layers, calib_samples, max_err, storage_budget=None):
    """返回 (qmodel, report)。无法满足约束时抛 PrecisionBudgetError。"""
    bits_map = {l["name"]: 8 for l in layers}
    while True:
        qmodel = convert(layers, calib_samples, bits_map)
        rep = evaluate(layers, qmodel, calib_samples)
        errs = {l["name"]: l["max_abs_err"] for l in rep["layers"]}
        over = [n for n, e in errs.items() if e > max_err]
        if not over:
            break
        worst = max(over, key=lambda n: errs[n] / max_err)
        nxt = _upgrade(bits_map[worst])
        if nxt is None:
            # 该层已到最高精度仍超差：误差多半来自上游，
            # 改升其前面最近的可升级层；没有可升级的才算真正失败。
            names = [l["name"] for l in layers]
            idx = names.index(worst)
            cand = [n for n in names[:idx]
                    if _upgrade(bits_map[n]) is not None]
            if not cand:
                raise PrecisionBudgetError(
                    f"层 {worst} 已达最高精度仍误差 {errs[worst]:.6g} > "
                    f"允许 {max_err:.6g}，转换失败（不降低误差标准）")
            worst = cand[-1]
            nxt = _upgrade(bits_map[worst])
        trial = dict(bits_map, **{worst: nxt})
        if storage_budget is not None and \
                total_storage_bytes(layers, trial) > storage_budget:
            raise PrecisionBudgetError(
                f"为满足误差需将层 {worst} 升至 {nxt} 位，"
                f"但存储将超出预算 {storage_budget} 字节，转换失败"
                f"（不降低误差标准）")
        bits_map = trial

    total = total_storage_bytes(layers, bits_map)
    if storage_budget is not None and total > storage_budget:
        raise PrecisionBudgetError(
            f"全 int8 已需 {total} 字节 > 预算 {storage_budget}，转换失败")
    report = {
        "max_err_allowed": max_err,
        "storage_budget": storage_budget,
        "storage_bytes": total,
        "layers": [{
            "name": l["name"],
            "bits": bits_map[l["name"]],
            "bytes": layer_storage_bytes(l, bits_map[l["name"]]),
            "max_abs_err": errs[l["name"]],
        } for l in layers],
    }
    return qmodel, report
