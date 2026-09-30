边缘模型部署研究系统（逐层混合精度 + 误差/预算约束 + 饱和风险保留）

## 结构
- edgequant/layers.py    浮点/定点语义：激活对称 int8；权重逐层 4bit/8bit；
                          round-half-to-even + 饱和截断；乘加 int64 求值后对
                          int32 累加边界显式饱和（不静默回绕）
- edgequant/model.py     模型 / 校准样本 / 转换产物三种 JSON 独立存取
- edgequant/quantize.py  校准、逐层候选精度误差评估与贪心混合精度选择；
                          不支持的层抛 UnsupportedLayerError；误差或预算达不到
                          时抛 PrecisionTargetError，不降低标准
- edgequant/runner.py    独立运行器：只吃转换产物 JSON，纯整数推理；
                          逐层报告输入截断/激活饱和/int32 累加溢出与风险标记
- edgequant/evaluate.py  逐层 max_abs_err/RMSE/SQNR + 风险；boundary_check
                          超校准范围验证；stress_compare 转换前后压力对比
- edgequant/stress.py    压力小模型：大累加、跨数量级尺度通道、异常输入
- main.py                CLI：make-model / convert / evaluate / boundary / stress
- tests/test_quant.py    混合精度选择、预算/误差显式失败、独立载入一致性、
                          压力输入饱和风险等用例

## 固定口径
- 激活：对称 int8，round-half-to-even，饱和截断 [-127,127]
- 权重：逐层混合精度，候选 4bit / 8bit 对称有符号
- 选择策略：全部从 4bit 起步，按“dense 层最大误差下降 / 额外存储比特”
  贪心跳级升级；relu 等无权重层误差来自激活粒度，固定不可降但照常披露
- 存储代价：权重比特 + 32bit 偏置；预算单位为 bit
- 失败语义：最高 8bit 仍超允许误差，或预算内无法达标 -> 明确报错
  （CLI 退出码 3），绝不悄悄放宽要求
- 越界语义：超出校准范围的输入不允许只返回看似正常的预测，输入截断、
  激活饱和、int32 累加溢出全部计入逐层 risk

## 用法
  python main.py make-model                     # 示例模型 + 校准样本
  python main.py convert --tol 0.05             # 自动混合精度（误差约束）
  python main.py convert --tol 0.05 --budget 500  # 再加存储预算(bits)
  python main.py convert --weight-bits 8        # 强制全 8bit（仍校验）
  python main.py evaluate                       # 校准范围内逐层误差/风险
  python main.py boundary                       # 超范围输入边界验证
  python main.py stress --tol 0.025             # 压力小模型转换前后对比
  python -m pytest tests -q

## 产物格式
沿用 edgequant.quant.v1：每层增加 w_bits，顶层增加
mixed_precision_report（逐层/总存储比特、预算、容差、逐层最终误差）。
独立 load_quant 后 run_quant 的输出与风险与转换阶段逐元素一致。
