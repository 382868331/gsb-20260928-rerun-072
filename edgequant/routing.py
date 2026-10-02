"""部署前冻结的输入条件与确定性路由。"""
import hashlib
import json
import numpy as np


class PrecisionRouteError(Exception):
    pass


def canonical_condition(condition):
    return json.loads(json.dumps(condition, sort_keys=True,
                                 ensure_ascii=False))


def condition_matches(samples, condition):
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    clauses = condition.get("all", [condition]) if condition else []
    mask = np.ones(len(x), dtype=bool)
    width = x.shape[1]
    for c in clauses:
        feature = c["feature"]
        if feature == "__absmax__":
            values = np.max(np.abs(x), axis=1)
        else:
            feature = int(feature)
            if feature < 0 or feature >= width:
                raise ValueError("条件特征下标越界: %s" % feature)
            values = x[:, feature]
        lo = c.get("min", -np.inf)
        hi = c.get("max", np.inf)
        if lo is not None and not np.isneginf(np.asarray(lo, float)):
            mask &= values >= float(lo) - 1e-12
        if hi is not None and not np.isposinf(np.asarray(hi, float)):
            mask &= values <= float(hi) + 1e-12
    return mask


def infer_box_conditions(groups, pad=0.0):
    out = []
    for name, samples, _ in groups:
        x = np.asarray(samples, dtype=np.float64)
        lo = np.min(x, axis=0) - pad
        hi = np.max(x, axis=0) + pad
        out.append({"name": name,
                    "all": [{"feature": i, "min": float(lo[i]),
                             "max": float(hi[i])}
                            for i in range(x.shape[1])]})
    return out


def normalize_conditions(groups, conditions=None, pad=0.0):
    by_name = {} if conditions is None else {
        c["name"]: canonical_condition(c) for c in conditions}
    out = []
    for name, samples, _ in groups:
        c = by_name.pop(name, None)
        if c is None:
            inferred = infer_box_conditions(
                [(name, samples, None)], pad)[0]
            c = {k: v for k, v in inferred.items() if k != "name"}
        out.append({"name": name, **c})
    if by_name:
        raise ValueError("存在未对应输入组的条件: %s" % sorted(by_name))
    return out


def route_inputs(samples, rules, fallback_plan="unified"):
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    assignments = []
    for i in range(len(x)):
        matched = [r["group"] for r in rules
                   if condition_matches(x[i:i+1], r["condition"])[0]]
        if matched:
            first = next(ri for ri, r in enumerate(rules)
                         if r["group"] == matched[0])
            chosen = rules[first]
            status = "matched" if len(matched) == 1 else "ambiguous"
            assignments.append({
                "index": i, "group": chosen["group"],
                "plan": chosen["plan"], "status": status,
                "matching_groups": matched,
                "fallback_used": False,
                "resolved_by": "rule-order[%d]" % first})
        else:
            assignments.append({
                "index": i, "group": None, "plan": fallback_plan,
                "status": "uncovered", "matching_groups": [],
                "fallback_used": True,
                "resolved_by": "fallback-plan:%s" % fallback_plan})
    return assignments


def policy_fingerprint(groups, conditions):
    payload = {
        "groups": [{"name": n,
                    "tol_max_abs": None if t is None else float(t),
                    "n_samples": int(len(s)),
                    "sample_sha": hashlib.sha256(
                        np.ascontiguousarray(s, np.float64).tobytes()
                    ).hexdigest()[:16]}
                   for n, s, t in groups],
        "conditions": [canonical_condition(c) for c in conditions]}
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def validate_policy_fingerprint(deployment, groups, conditions=None):
    from .model import normalize_groups
    groups = normalize_groups(groups)
    rules = deployment["route_policy"]["rules"]
    if conditions is None:
        conditions = [{"name": r["group"], **r["condition"]}
                      for r in rules]
    current = policy_fingerprint(
        groups, normalize_conditions(groups, conditions))
    saved = deployment["route_policy"]["calibration"]["policy_sha256_16"]
    if current != saved:
        raise PrecisionRouteError(
            "分组样本、容差或条件已改变；原校准证明不能自动沿用到新范围 "
            "(saved=%s, current=%s)" % (saved, current))
    return True
