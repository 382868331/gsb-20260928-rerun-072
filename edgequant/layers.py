"""计算层注册表：浮点语义 + 定点语义。

首轮支持的层：dense、relu。饱和与舍入规则固定：
round-half-to-even + 对称 int8 饱和截断 [-127, 127]。
其它层（如 sigmoid/conv）可存在于浮点模型，但定点转换一律拒绝。

第二轮：逐层混合精度。PRECISIONS 中 8/16 为对称定点（int32 累加），
32 表示该层保留浮点（敏感层不回退标准，只是不参与定点化）。
"""
import numpy as np

QMIN, QMAX = -127, 127

# 允许的逐层精度（位宽）；32 = 保留浮点
PRECISIONS = (8, 16, 32)


def qmax_for(bits):
    return 2 ** (bits - 1) - 1


def int_dtype(bits):
    return np.int8 if bits == 8 else np.int16


def dense_fp(x, w, b):
    return x @ w + b


def relu_fp(x):
    return np.maximum(x, 0.0)


FP_LAYERS = {"dense": dense_fp, "relu": relu_fp}


def quantize_tensor(t, scale, bits=8):
    q = np.rint(t / scale)
    m = qmax_for(bits)
    return np.clip(q, -m, m).astype(int_dtype(bits))


def dequantize_tensor(q, scale):
    return q.astype(np.float64) * scale


def dense_q(xq, wq, bq, x_scale, w_scale, out_scale, bits=8):
    """int8 乘加，int32 累加，再重量化到输出 scale。"""
    acc = xq.astype(np.int32) @ wq.astype(np.int32) + bq.astype(np.int32)
    rescale = (x_scale * w_scale) / out_scale
    out = np.rint(acc.astype(np.float64) * rescale)
    m = qmax_for(bits)
    return np.clip(out, -m, m).astype(int_dtype(bits))


def relu_q(xq):
    return np.maximum(xq, 0).astype(np.asarray(xq).dtype)


Q_LAYERS = {"dense": dense_q, "relu": relu_q}

# 转换白名单：不在表内的层一律拒绝转换
CONVERTIBLE = frozenset(Q_LAYERS.keys())
