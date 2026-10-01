"""部署方案归档：旧方案、原限制、校准来源长期可追溯。

方案库为独立 JSON（edgequant.plans.v1），每个方案记录产物路径、
完整 mixed_precision_report 与 provenance；永不覆盖已归档方案。
"""
import json
import os

PLANS_FORMAT = "edgequant.plans.v1"


def load_store(path):
    if not os.path.exists(path):
        return {"format": PLANS_FORMAT, "plans": []}
    with open(path) as f:
        doc = json.load(f)
    assert doc["format"] == PLANS_FORMAT, doc["format"]
    return doc


def save_store(path, store):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(store, f, indent=2, ensure_ascii=False)
    os.replace(tmp, str(path))


def get_plan(store, plan_id):
    for p in store["plans"]:
        if p["plan_id"] == plan_id:
            return p
    raise KeyError(f"方案 {plan_id} 不在归档中")


def add_plan(store, plan_id, quant_path, qmodel, groups_meta, note=""):
    if any(p["plan_id"] == plan_id for p in store["plans"]):
        raise ValueError(f"方案 {plan_id} 已归档，不允许覆盖旧方案")
    entry = {
        "plan_id": plan_id,
        "quant_artifact": str(quant_path),
        "note": note,
        "input_groups": groups_meta,
        "mixed_precision_report": qmodel["mixed_precision_report"],
        "provenance": qmodel.get("provenance", {}),
        # 归档时冻结产物快照：即使磁盘上的同名文件后来被旧口径命令覆盖，
        # 旧方案仍可按快照完整重放，保证方案/限制/校准来源可追溯。
        "quant_snapshot": qmodel,
    }
    store["plans"].append(entry)
    return entry
