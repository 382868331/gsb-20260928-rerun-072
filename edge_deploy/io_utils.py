"""JSON 产物读写与目录约定。

模型结构、校准样本、转换产物分别独立保存：
  models/<name>/model.json
  datasets/<name>/<split>.samples.json
  artifacts/<model>/<version>/artifact.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FORMAT_VERSION = 1


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def read_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def model_path(root: str | Path, model_name: str) -> Path:
    return Path(root) / "models" / model_name / "model.json"


def dataset_path(root: str | Path, dataset_name: str, split: str) -> Path:
    return Path(root) / "datasets" / dataset_name / f"{split}.samples.json"


def artifact_path(root: str | Path, model_name: str, version: str) -> Path:
    return Path(root) / "artifacts" / model_name / version / "artifact.json"
