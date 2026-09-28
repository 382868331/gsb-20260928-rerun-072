"""定点化规则：舍入与饱和口径全局固定，禁止各层自行选择。

- 舍入：half-away-from-zero（四舍五入，负数 -0.5 -> -1）
- 饱和：超出 [qmin, qmax] 一律夹取，并由调用方计数
- 仿射刻度：x = scale * (q - zero_point)，zero_point=0（对称量化）
"""
from __future__ import annotations

import math


def round_half_away(value: float) -> int:
    if value >= 0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def saturate(q: int, qmin: int, qmax: int) -> tuple[int, bool]:
    """返回（饱和后的整数, 是否发生过饱和）。"""
    if q < qmin:
        return qmin, True
    if q > qmax:
        return qmax, True
    return q, False


def quantize_scalar(
    value: float, scale: float, qmin: int, qmax: int
) -> tuple[int, bool]:
    q, clamped = saturate(round_half_away(value / scale), qmin, qmax)
    return q, clamped


def dequantize_scalar(q: int, scale: float) -> float:
    return q * scale


def symmetric_scale(absmax: float, qmax: int) -> float:
    if absmax <= 0.0 or not math.isfinite(absmax):
        raise ValueError(f"非法的标定绝对值: {absmax!r}")
    return absmax / qmax
