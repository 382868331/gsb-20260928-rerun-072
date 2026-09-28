"""模型结构、校准样本、转换产物各自独立序列化（JSON）。"""
import json
import numpy as np

MODEL_FORMAT = "edgequant.model.v1"
CALIB_FORMAT = "edgequant.calib.v1"
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


def save_quant(path, qmodel):
    _dump(path, {"format": QUANT_FORMAT, **qmodel})


def load_quant(path):
    doc = json.load(open(path))
    assert doc["format"] == QUANT_FORMAT
    return doc


def _dump(path, doc):
    with open(path, "w") as f:
        json.dump(doc, f)

