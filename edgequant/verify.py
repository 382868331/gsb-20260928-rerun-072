"""产物重载复核：保存 -> 重新载入 -> 在全部输入组上重放，
预测、逐层风险必须与转换阶段逐元素一致（防止序列化丢信息）。
"""
import os
import numpy as np
from . import model as M
from .runner import run_quant


def reload_recheck(qmodel, groups, tmp_path):
    """把 qmodel 写入 tmp_path 再 load_quant，逐组核对。

    返回复核报告；任何不一致直接抛 AssertionError。
    """
    M.save_quant(tmp_path, qmodel)
    loaded = M.load_quant(tmp_path)
    details = []
    for g in groups:
        y1, t1, r1 = run_quant(qmodel, g.samples, return_risk=True)
        y2, t2, r2 = run_quant(loaded, g.samples, return_risk=True)
        same_y = bool(np.array_equal(y1, y2))
        same_trace = all(
            np.array_equal(a["q_out"], b["q_out"])
            for a, b in zip(t1, t2))
        same_risk = r1 == r2
        assert same_y and same_trace and same_risk, (
            f"重载复核失败：组 {g.name} 预测/轨迹/风险不一致")
        details.append({
            "group": g.name,
            "in_calib_range": g.in_calib,
            "prediction_match": same_y,
            "trace_match": same_trace,
            "risk_match": same_risk,
            "input_clipped": r2["input_clipped"],
            "input_out_of_calib": r2["input_out_of_calib"],
            "activation_saturated": r2["activation_saturated"],
            "acc_int32_saturated": r2["acc_int32_saturated"],
        })
    os.remove(tmp_path)
    return {"reload_prediction_match": True,
            "reload_risk_match": True,
            "groups_checked": [g.name for g in groups],
            "per_group": details}
