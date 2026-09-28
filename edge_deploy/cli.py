"""命令行入口：create-demo / convert / evaluate / compare / run-artifact。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .converter import CONVERSION_VERSION, UnsupportedLayerError, convert
from .evaluate import compare_reports, evaluate_split
from .io_utils import (
    artifact_path,
    dataset_path,
    model_path,
    read_json,
    write_json,
)
from .quant_runner import run as run_artifact
from .sample_data import materialize_demo
from .schema import SchemaError, validate_artifact, validate_model, validate_samples


def _load_model(root: Path, name: str) -> dict:
    model = read_json(model_path(root, name))
    validate_model(model)
    return model


def _load_samples(root: Path, name: str, split: str) -> dict:
    samples = read_json(dataset_path(root, name, split))
    validate_samples(_load_model(root, name), samples)
    return samples


def cmd_create_demo(args: argparse.Namespace) -> int:
    materialize_demo(args.root)
    print(f"已生成演示模型与样本到 {args.root}")
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    model = _load_model(args.root, args.model)
    calibration = _load_samples(args.root, args.model, args.calibration_split)
    try:
        artifact = convert(model, calibration)
    except UnsupportedLayerError as exc:
        print(f"[拒绝转换] {exc}", file=sys.stderr)
        return 2
    target = artifact_path(args.root, args.model, args.version)
    write_json(target, artifact)
    validate_artifact(artifact)
    print(f"转换产物已保存: {target}（{len(artifact['layers'])} 层，{args.version}）")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    model = _load_model(args.root, args.model)
    artifact = read_json(artifact_path(args.root, args.model, args.version))
    validate_artifact(artifact)
    reports = {}
    for split in args.splits:
        samples = _load_samples(args.root, args.model, split)
        reports[split] = evaluate_split(model, artifact, samples)
    payload = {
        "model": args.model,
        "conversion": artifact["conversion"],
        "conversion_version": artifact["conversion_version"],
        "reports": reports,
    }
    out_path = (
        Path(args.output)
        if args.output
        else args.root / "artifacts" / args.model / args.version / "evaluation.json"
    )
    write_json(out_path, payload)
    print(f"误差与溢出评估已保存: {out_path}")
    for split, report in reports.items():
        out = report["output_error"]
        sat = report["numeric_clamp_events"]
        relu_cuts = report["relu_negative_cuts"]
        flag = "  ⚠ 饱和!" if sat else ""
        print(
            f"  [{split}] n={report['sample_count']} "
            f"max_abs={out['max_abs_error']:.5f} rmse={out['rmse']:.5f} "
            f"数值饱和={sat} ReLU截断={relu_cuts}{flag}"
        )
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    reports = {}
    for spec in args.versions:
        version, split = spec.split("@", 1)
        path = args.root / "artifacts" / args.model / version / "evaluation.json"
        data = read_json(path)
        if split not in data["reports"]:
            raise SystemExit(f"{version}/evaluation.json 中没有 split {split}")
        reports[version] = data["reports"][split]
    result = compare_reports(reports)
    print(f"同口径比较（split={args.split}）：")
    for row in result["ranking"]:
        flag = "  存在饱和" if row["any_saturation"] else ""
        print(
            f"  {row['conversion_version']:<24} "
            f"max_abs={row['output_max_abs_error']:.5f} "
            f"rmse={row['output_rmse']:.5f} "
            f"饱和={row['total_saturation_events']}{flag}"
        )
    return 0


def cmd_run_artifact(args: argparse.Namespace) -> int:
    artifact = read_json(args.artifact)
    validate_artifact(artifact)
    x = [float(token) for token in args.input.split(",")]
    output = run_artifact(artifact, x)
    print(" ".join(f"{value:.8f}" for value in output))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="edge-deploy", description="边缘模型部署研究系统（首轮：W8A8 定点化）"
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("create-demo", help="生成演示模型与校准/测试样本")
    p.add_argument("--root", default=Path("store"), type=Path)
    p.set_defaults(func=cmd_create_demo)

    p = sub.add_parser("convert", help="用校准集执行定点化转换")
    p.add_argument("--root", default=Path("store"), type=Path)
    p.add_argument("--model", required=True)
    p.add_argument("--version", default=CONVERSION_VERSION)
    p.add_argument("--calibration-split", default="calibration")
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("evaluate", help="逐层误差与溢出评估（必须同时跑超校准范围样本）")
    p.add_argument("--root", default=Path("store"), type=Path)
    p.add_argument("--model", required=True)
    p.add_argument("--version", default=CONVERSION_VERSION)
    p.add_argument(
        "--splits",
        nargs="+",
        default=["calibration", "test_in_range", "test_out_of_range"],
    )
    p.add_argument("--output")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("compare", help="多个转换版本同口径比较")
    p.add_argument("--root", default=Path("store"), type=Path)
    p.add_argument("--model", required=True)
    p.add_argument("--split", required=True, help="所有版本统一使用的 split")
    p.add_argument(
        "--versions",
        required=True,
        nargs="+",
        help="形如 w8a8-i8sym-v1@test_out_of_range，可传多个版本",
    )
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("run-artifact", help="独立运行器：仅凭 artifact.json 执行整数推理")
    p.add_argument("--artifact", required=True, type=Path)
    p.add_argument("--input", required=True, help="逗号分隔的浮点输入，如 4,-4,4")
    p.set_defaults(func=cmd_run_artifact)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except SchemaError as exc:
        print(f"[文件校验失败] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
