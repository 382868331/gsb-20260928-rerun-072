"""计算层注册表：浮点语义 + 定点语义。

支持的层：dense、relu。固定口径：
- 激活：对称 int8，round-half-to-even，饱和截断 [-127, 127]
- 权重：逐层混合精度，可选 4bit / 8bit 对称有符号整数
- 累加：int32；中间用 int64 求值并对 int32 边界显式饱和（报告饱和数，
  绝不静默回绕）
其它层（如 sigmoid/conv）可存在于浮点模型，但转换一律拒绝。
"""
import numpy as np

ACT_BITS = 8
WEIGHT_BIT_CHOICES = (4, 8)
INT32_MIN, INT32_MAX = -(1 << 31), (1 << 31) - 1


def qmin_qmax(bits):
    qmax = (1 << (bits - 1)) - 1
    return -qmax, qmax


QMIN, QMAX = qmin_qmax(ACT_BITS)


def dense_fp(x, w, b):
    return x @ w + b


def relu_fp(x):
    return np.maximum(x, 0.0)


FP_LAYERS = {"dense": dense_fp, "relu": relu_fp}


def quantize_tensor(t, scale, bits=ACT_BITS):
    """对称量化，round-half-to-even + 饱和截断。

    返回 (整数数组, 截断元素个数)。截断计数在量化前统计，
    因此即使真实值恰好落在边界也不会与饱和混淆。
    """
    lo, hi = qmin_qmax(bits)
    scaled = np.asarray(t, dtype=np.float64) / scale
    clipped = int(np.sum(np.abs(scaled) > hi))
    q = np.clip(np.rint(scaled), lo, hi)
    dtype = np.int8 if bits == 8 else np.int32
    return q.astype(dtype), clipped


def dequantize_tensor(q, scale):
    return q.astype(np.float64) * scale


def dense_q(xq, wq, bq, x_scale, w_scale, out_scale, w_bits):
    """整数乘加：int8 激活 × w_bits 权重，int32 累加（显式饱和），
    再重量化到 int8 输出。返回 (out, acc32_saturated_count)。"""
    acc = xq.astype(np.int64) @ wq.astype(np.int64) + bq.astype(np.int64)
    acc_sat = int(np.sum((acc < INT32_MIN) | (acc > INT32_MAX)))
    acc = np.clip(acc, INT32_MIN, INT32_MAX).astype(np.int32)
    rescale = (x_scale * w_scale) / out_scale
    out = np.rint(acc.astype(np.float64) * rescale)
    out = np.clip(out, QMIN, QMAX).astype(np.int8)
    return out, acc_sat


def relu_q(xq):
    return np.maximum(xq, 0).astype(np.int8)


Q_LAYERS = {"dense": dense_q, "relu": relu_q}

# 转换白名单：不在表内的层一律拒绝转换
CONVERTIBLE = frozenset(Q_LAYERS.keys())
