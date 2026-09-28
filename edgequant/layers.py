"""计算层注册表：浮点语义 + 定点语义。

首轮支持的层：dense、relu。饱和与舍入规则固定：
round-half-to-even + 对称 int8 饱和截断 [-127, 127]。
其它层（如 sigmoid/conv）可存在于浮点模型，但定点转换一律拒绝。
"""
import numpy as np

QMIN, QMAX = -127, 127


def dense_fp(x, w, b):
    return x @ w + b


def relu_fp(x):
    return np.maximum(x, 0.0)


FP_LAYERS = {"dense": dense_fp, "relu": relu_fp}


def quantize_tensor(t, scale):
    q = np.rint(t / scale)
    return np.clip(q, QMIN, QMAX).astype(np.int8)


def dequantize_tensor(q, scale):
    return q.astype(np.float64) * scale


def dense_q(xq, wq, bq, x_scale, w_scale, out_scale):
    """int8 乘加，int32 累加，再重量化到输出 scale。"""
    acc = xq.astype(np.int32) @ wq.astype(np.int32) + bq.astype(np.int32)
    rescale = (x_scale * w_scale) / out_scale
    out = np.rint(acc.astype(np.float64) * rescale)
    return np.clip(out, QMIN, QMAX).astype(np.int8)


def relu_q(xq):
    return np.maximum(xq, 0).astype(np.int8)


Q_LAYERS = {"dense": dense_q, "relu": relu_q}

# 转换白名单：不在表内的层一律拒绝转换
CONVERTIBLE = frozenset(Q_LAYERS.keys())

