"""逐层误差与溢出评估：以浮点模型为口径，逐样本、逐边界核对。"""
from __future__ import annotations

import math
from typing import Any

from .float_runner import run_trace as float_trace
from .quant_runner import dequantize_trace, run_trace as quant_trace


def _empty_error() -> dict[str, float]:
    return {
        "max_abs_error": 0.0,
        "mean_abs_error": 0.0,
        "rmse": 0.0,
        "max_rel_error": 0.0,
        "compared_elements": 0,
    }


def _update_error(stat: dict[str, float], ref: float, got: float) -> None:
    abs_err = abs(ref - got)
    stat["max_abs_error"] = max(stat["max_abs_error"], abs_err)
    stat["mean_abs_error"] += abs_err
    stat["rmse"] += (ref - got) ** 2
    denom = max(abs(ref), 1e-12)
    stat["max_rel_error"] = max(stat["max_rel_error"], abs_err / denom)
    stat["compared_elements"] += 1


def _finalize_error(stat: dict[str, float]) -> None:
    n = max(stat["compared_elements"], 1)
    stat["mean_abs_error"] /= n
    stat["rmse"] = math.sqrt(stat["rmse"] / n)


def _empty_counts() -> dict[str, int]:
    return {
        "input_quant_clamps": 0,
        "accumulator_clamps": 0,
        "requant_clamps": 0,
        "relu_negatives_cut": 0,
    }


NUMERIC_CLAMP_KEYS = ("input_quant_clamps", "accumulator_clamps", "requant_clamps")


def evaluate_split(
    model: dict[str, Any], artifact: dict[str, Any], samples: dict[str, Any]
) -> dict[str, Any]:
    layer_names = [layer["name"] for layer in model["layers"]]
    errors = {"__input__": _empty_error()}
    errors.update({name: _empty_error() for name in layer_names})
    totals = _empty_counts()
    per_sample_worst: list[dict[str, Any]] = []

    for record in samples["samples"]:
        ref_trace = float_trace(model, record["input"])
        q_trace, q_stats = quant_trace(artifact, record["input"])
        dq_trace = dequantize_trace(artifact, q_trace)
        sample_worst = 0.0
        for boundary, ref_values in ref_trace.items():
            for ref, got in zip(ref_values, dq_trace[boundary]):
                _update_error(errors[boundary], ref, got)
                sample_worst = max(sample_worst, abs(ref - got))
        for name, layer_counts in q_stats.items():
            for key in totals:
                totals[key] += layer_counts[key]
        per_sample_worst.append({"id": record["id"], "max_abs_error": sample_worst})

    for stat in errors.values():
        _finalize_error(stat)

    output_name = layer_names[-1]
    numeric_clamps = sum(totals[key] for key in NUMERIC_CLAMP_KEYS)
    per_sample_worst.sort(key=lambda item: item["max_abs_error"], reverse=True)
    return {
        "split": samples.get("split", "unknown"),
        "sample_count": len(samples["samples"]),
        "boundary_errors": errors,
        "output_error": errors[output_name],
        "overflow_counts": totals,
        "numeric_clamp_events": numeric_clamps,
        # ReLU 负向截断是算子语义的一部分，不计入数值溢出。
        "relu_negative_cuts": totals["relu_negatives_cut"],
        "any_saturation": numeric_clamps > 0,
        "worst_samples": per_sample_worst[:5],
    }


def compare_reports(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """同口径比较：各版本报告必须覆盖相同 split，且输出误差取相同边界。"""
    splits = {name: report["split"] for name, report in reports.items()}
    table = []
    for version, report in reports.items():
        out = report["output_error"]
        counts = report["overflow_counts"]
        table.append(
            {
                "conversion_version": version,
                "split": report["split"],
                "sample_count": report["sample_count"],
                "output_max_abs_error": out["max_abs_error"],
                "output_rmse": out["rmse"],
                "output_max_rel_error": out["max_rel_error"],
                "total_saturation_events": sum(counts[key] for key in NUMERIC_CLAMP_KEYS),
                "relu_negative_cuts": counts["relu_negatives_cut"],
                "any_saturation": report["any_saturation"],
            }
        )
    table.sort(key=lambda row: row["output_max_abs_error"])
    return {"splits": splits, "ranking": table}
