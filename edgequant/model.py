"""模型结构、校准样本、转换产物各自独立序列化（JSON）。"""
import json
import numpy as np

MODEL_FORMAT = "edgequant.model.v1"
CALIB_FORMAT = "edgequant.calib.v1"
CALIB_GROUPS_FORMAT = "edgequant.calib-groups.v1"
QUANT_FORMAT = "edgequant.quant.v1"


def _enc(a):
    return np.asarray(a, dtype=np.float64).tolist()


def save_model(path, layers):
    doc = {"format": MODEL_FORMAT, "layers": []}
    for l in layers:
        e = {"type": l["type"], "name": l["name"]}
        for k in ("w", "b"):
            if k in l:
                e[k] = _enc(l[k])
        doc["layers"].append(e)
    _dump(path, doc)


def load_model(path):
    doc = json.load(open(path))
    assert doc["format"] == MODEL_FORMAT, doc["format"]
    for l in doc["layers"]:
        for k in ("w", "b"):
            if k in l:
                l[k] = np.asarray(l[k], dtype=np.float64)
    return doc["layers"]


def save_calib(path, samples):
    _dump(path, {"format": CALIB_FORMAT, "samples": _enc(samples)})


def load_calib(path):
    doc = json.load(open(path))
    assert doc["format"] == CALIB_FORMAT
    return np.asarray(doc["samples"], dtype=np.float64)


def normalize_groups(groups):
    """把多种输入组写法统一成 [(name, samples_ndarray, tol), ...]。

    接受 ndarray 单组；{"name": samples} 或 {"name": (samples, tol)}；
    [{"name", "samples", "tol_max_abs"}, ...] 列表。tol 缺省为 None，
    由调用方用全局容差补齐。
    """
    out = []
    if isinstance(groups, np.ndarray):
        out.append(("default", np.asarray(groups, dtype=np.float64), None))
    elif isinstance(groups, (list, tuple)) and groups and +            isinstance(groups[0], (tuple, list)) and groups[0] and +            isinstance(groups[0][0], str):
        # 已经是 (name, samples, tol) 形式
        for name, s, t in groups:
            out.append((str(name), np.asarray(s, dtype=np.float64),
                        None if t is None else float(t)))
    elif (isinstance(groups, (list, tuple)) and groups and
          not isinstance(groups[0], dict)):
        out.append(("default", np.asarray(groups, dtype=np.float64), None))
    elif isinstance(groups, dict):
        for name, v in groups.items():
            if isinstance(v, (tuple, list)) and len(v) == 2 and +                    hasattr(v[0], "__len__"):
                out.append((str(name), np.asarray(v[0], dtype=np.float64),
                            float(v[1])))
            else:
                out.append((str(name), np.asarray(v, dtype=np.float64), None))
    else:
        for g in groups:
            t = g.get("tol_max_abs")
            out.append((str(g["name"]),
                        np.asarray(g["samples"], dtype=np.float64),
                        None if t is None else float(t)))
    if not out:
        raise ValueError("至少需要一个输入组")
    width = out[0][1].shape[1]
    for name, s, _ in out:
        if s.ndim != 2 or s.shape[1] != width:
            raise ValueError("输入组 %s 特征宽度与其它组不一致" % name)
    return out


def save_calib_groups(path, groups, source=None):
    """groups: normalize_groups 的产物 (name, samples, tol)。"""
    doc = {"format": CALIB_GROUPS_FORMAT, "source": source or str(path),
           "groups": [
               {"name": n, "samples": _enc(s), "tol_max_abs": t}
               for n, s, t in groups]}
    _dump(path, doc)


def load_calib_groups(path):
    """统一读入：v1 单校准文件返回单组，v1-groups 返回多组。"""
    doc = json.load(open(path))
    if doc["format"] == CALIB_FORMAT:
        return normalize_groups(
            np.asarray(doc["samples"], dtype=np.float64))
    assert doc["format"] == CALIB_GROUPS_FORMAT, doc["format"]
    return normalize_groups(doc["groups"])


def save_quant(path, qmodel):
    _dump(path, {"format": QUANT_FORMAT, **qmodel})


def load_quant(path):
    doc = json.load(open(path))
    assert doc["format"] == QUANT_FORMAT
    return doc


def _dump(path, doc):
    with open(path, "w") as f:
        json.dump(doc, f)
