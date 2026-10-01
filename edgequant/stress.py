"""压力小模型：覆盖大累加、不同尺度通道与异常输入场景。

权重列故意设置成跨数量级的尺度，第二层权重较大以便超范围输入触发
int32 累加与激活饱和；校准集保持在正常小范围，压力输入则刻意越界。
"""
import numpy as np


def build_stress_model():
    """返回 (layers, calib)。4 输入 -> 6 隐层 -> 2 输出。"""
    # 四列尺度差异明显：1e-2、1e-1、1e0、5e-1
    w1 = np.zeros((4, 6), dtype=np.float64)
    scales_col = [5e-3, 5e-2, 1.5e-1, 1e-1]
    rng = np.random.default_rng(7)
    for j, s in enumerate(scales_col):
        w1[j, :] = s * rng.choice([-1.0, 1.0], size=6)
    b1 = rng.normal(0, 0.05, 6)
    # 第二层权重适中且符号混合：校准范围内输出温和；
    # 超范围大输入经乘加后产生很大的累加值并触发饱和。
    w2 = 0.3 * np.array([[1.0, -0.8], [1.2, 0.9], [-0.7, 1.1],
                         [0.5, -0.8], [-0.9, 0.7], [0.6, 1.0]],
                        dtype=np.float64)
    b2 = np.array([0.2, -0.1], dtype=np.float64)
    layers = [
        {"type": "dense", "name": "fc1", "w": w1, "b": b1},
        {"type": "relu", "name": "relu1"},
        {"type": "dense", "name": "fc2", "w": w2, "b": b2},
    ]
    # 校准集：小范围正常输入（最大约 1.0）
    calib = rng.uniform(-1.0, 1.0, (32, 4))
    return layers, calib


def build_stress_inputs():
    """异常输入集合：正常、大累加、单通道极端、异尺度混合。"""
    rows = [
        [0.3, -0.2, 0.1, -0.4],       # 正常校准范围内
        [30.0, -40.0, 25.0, -35.0],   # 大累加（远超校准范围）
        [0.0, 0.0, 50.0, 0.0],        # 单通道极端值（高尺度通道）
        [1e-3, 50.0, 1e-3, -45.0],    # 异尺度通道混合
        [-1e4, 1e4, -1e4, 1e4],       # 极端异常值
    ]
    return np.asarray(rows, dtype=np.float64)


def build_stress_groups():
    """三个代表性输入组，各自容差不同，共享存储预算。

    - nominal：校准分布内的正常输入，容差最严；
    - wide：幅度更大但仍覆盖在多组校准并集内的宽幅输入；
    - edge：贴近校准上界的边缘输入，对饱和更敏感，容差放宽但仍独立考核。
    """
    _, calib = build_stress_model()
    rng = np.random.default_rng(11)
    nominal = calib
    wide = rng.uniform(-1.0, 1.0, (24, 4))
    edge = rng.uniform(0.8, 1.0, (16, 4)) * rng.choice([-1.0, 1.0],
                                                       (16, 4))
    return [
        {"name": "nominal", "samples": nominal, "tol_max_abs": 0.03},
        {"name": "wide", "samples": wide, "tol_max_abs": 0.08},
        {"name": "edge", "samples": edge, "tol_max_abs": 0.10},
    ]
