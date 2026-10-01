"""多个代表性输入组（input groups）的定义与加载。

研究部署不能只拿一份校准资料决定所有层精度：每个输入组有自己的样本、
逐（dense）层 max_abs_err 上限，并标记是否属于校准范围。
所有组共享同一个存储预算；任何一组超差都必须如实报告，不允许用
跨组平均误差掩盖某一组失效。
"""
import json
import hashlib
import numpy as np

GROUP_FORMAT = "edgequant.groups.v1"


class InputGroup:
    """一个代表性输入组。

    name:      组名
    samples:   ndarray (n, in_features)
    tol:       float（所有 dense 层统一上限）或 {层名: 上限}
    in_calib:  该组样本是否属于校准范围（False 表示超校准范围组）
    """

    def __init__(self, name, samples, tol, in_calib=True):
        self.name = str(name)
        self.samples = np.asarray(samples, dtype=np.float64)
        if self.samples.ndim != 2:
            raise ValueError(f"组 {name} 样本必须是二维数组")
        if isinstance(tol, (int, float)):
            self.tol = {"__all__": float(tol)}
        else:
            self.tol = {k: float(v) for k, v in dict(tol).items()}
        self.in_calib = bool(in_calib)

    def layer_tol(self, layer_name):
        if layer_name in self.tol:
            return self.tol[layer_name]
        if "__all__" in self.tol:
            return self.tol["__all__"]
        return None

    def as_meta(self):
        a = self.samples
        return {
            "name": self.name,
            "n_samples": int(a.shape[0]),
            "in_calib_range": self.in_calib,
            "tol_max_abs": (self.tol["__all__"] if "__all__" in self.tol
                            else self.tol),
            "samples_absmax": float(np.max(np.abs(a))),
            "samples_sha256": samples_hash(self.samples),
        }

    def to_json_obj(self):
        return {**self.as_meta(), "samples": a_tolist(self.samples)}


def a_tolist(a):
    return np.asarray(a, dtype=np.float64).tolist()


def samples_hash(a):
    """样本内容指纹，校准来源可追溯。"""
    b = np.asarray(a, dtype=np.float64).tobytes()
    return hashlib.sha256(b).hexdigest()[:16]


def save_groups(path, groups):
    doc = {"format": GROUP_FORMAT,
           "groups": [g.to_json_obj() for g in groups]}
    with open(path, "w") as f:
        json.dump(doc, f)


def load_groups(path):
    with open(path) as f:
        doc = json.load(f)
    assert doc["format"] == GROUP_FORMAT, doc["format"]
    groups = []
    for g in doc["groups"]:
        groups.append(InputGroup(
            g["name"], np.asarray(g["samples"], dtype=np.float64),
            g["tol_max_abs"], g.get("in_calib_range", True)))
    return groups


def groups_from_calib(calib_samples, tol_max_abs,
                      name="calib-default"):
    """旧的单校准集口径 -> 单个校准范围内输入组（向后兼容）。"""
    return [InputGroup(name, calib_samples, tol_max_abs, in_calib=True)]
