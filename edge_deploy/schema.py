"""模型、样本、转换产物的结构校验。"""
from __future__ import annotations

from typing import Any


class SchemaError(ValueError):
    pass


def _require(obj: Any, key: str, typ: type | tuple[type, ...], where: str) -> Any:
    if not isinstance(obj, dict) or key not in obj:
        raise SchemaError(f"{where} 缺少字段 {key!r}")
    value = obj[key]
    if not isinstance(value, typ):
        raise SchemaError(f"{where}.{key} 类型错误，期望 {typ}")
    return value


def _is_vector(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(x, (int, float)) for x in value)


def _is_matrix(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(_is_vector(row) for row in value)
        and len({len(row) for row in value}) == 1
    )


def validate_model(model: dict[str, Any]) -> None:
    where = "model"
    name = _require(model, "name", str, where)
    in_dim = _require(model, "input_dim", int, where)
    layers = _require(model, "layers", list, where)
    if in_dim <= 0:
        raise SchemaError("model.input_dim 必须为正整数")
    seen = set()
    current = in_dim
    for index, layer in enumerate(layers):
        prefix = f"model.layers[{index}]"
        if not isinstance(layer, dict):
            raise SchemaError(f"{prefix} 必须是对象")
        ltype = _require(layer, "type", str, prefix)
        lname = _require(layer, "name", str, prefix)
        if lname in seen:
            raise SchemaError(f"{prefix} 层名重复: {lname}")
        seen.add(lname)
        if ltype == "linear":
            weight = _require(layer, "weight", list, prefix)
            bias = _require(layer, "bias", list, prefix)
            if not _is_matrix(weight):
                raise SchemaError(f"{prefix}.weight 必须是二维数值矩阵")
            if not _is_vector(bias):
                raise SchemaError(f"{prefix}.bias 必须是一维数值数组")
            if len(weight) != len(bias):
                raise SchemaError(f"{prefix} weight 行数须与 bias 长度一致")
            if len(weight[0]) != current:
                raise SchemaError(
                    f"{prefix} 输入维度 {len(weight[0])} 与上游输出 {current} 不一致"
                )
            current = len(weight)
        elif ltype == "relu":
            pass
        else:
            raise SchemaError(
                f"{prefix} 未知层类型 {ltype!r}（模型文件仅描述结构，合法值: linear/relu）"
            )
    if not layers:
        raise SchemaError("model.layers 不能为空")
    model.setdefault("description", f"{name} 数值模型")


def validate_samples(model: dict[str, Any], samples: dict[str, Any]) -> None:
    in_dim = model["input_dim"]
    records = _require(samples, "samples", list, "dataset")
    if not records:
        raise SchemaError("dataset.samples 不能为空")
    for index, record in enumerate(records):
        x = _require(record, "input", list, f"dataset.samples[{index}]")
        if not _is_vector(x) or len(x) != in_dim:
            raise SchemaError(
                f"dataset.samples[{index}].input 必须是长度 {in_dim} 的数值向量"
            )


def validate_artifact(artifact: dict[str, Any]) -> None:
    _require(artifact, "format_version", int, "artifact")
    _require(artifact, "source_model", str, "artifact")
    _require(artifact, "conversion", str, "artifact")
    _require(artifact, "conversion_version", str, "artifact")
    layers = _require(artifact, "layers", list, "artifact")
    for index, layer in enumerate(layers):
        prefix = f"artifact.layers[{index}]"
        _require(layer, "name", str, prefix)
        _require(layer, "type", str, prefix)
        _require(layer, "io_spec", dict, prefix)
