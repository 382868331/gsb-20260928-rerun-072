边缘模型部署研究系统（首轮：定点化 + 误差评估）

## 结构
- edgequant/layers.py  层注册表：浮点/定点语义，白名单 CONVERTIBLE（首轮 dense、relu）
- edgequant/model.py   模型结构 / 校准样本 / 转换产物三种 JSON 格式独立存取
- edgequant/quantize.py 校准（max-abs 对称 scale）与转换；不支持的层抛 UnsupportedLayerError
- edgequant/runner.py  独立运行器：只吃转换产物 JSON，纯 int8/int32 推理
- edgequant/evaluate.py 逐层误差（max_abs_err/RMSE/SQNR）与饱和计数；boundary_check 用超校准范围输入验证
- main.py              CLI：make-model / convert / evaluate / boundary

## 固定口径（后续压缩方法需遵守同一口径以便同口径比较）
- 定点方案：对称 int8，round-half-to-even，饱和截断 [-127,127]，int32 累加
- 评估指标：逐层 max_abs_err、RMSE、SQNR(dB)、输入/输出饱和计数
- 边界验证：factor 倍超校准范围输入，必须报告饱和与误差，不得只报压缩率

## 用法
  python main.py make-model            # 生成示例模型 + 校准样本
  python main.py convert               # 校准并转换（不支持的层会拒绝）
  python main.py evaluate              # 校准范围内逐层误差/溢出
  python main.py boundary              # 超范围输入边界验证
  python -m pytest tests -q

## 扩展新压缩方法
在 layers.py 注册新语义、在 quantize.py 增加转换路径，产物仍用
edgequant.quant.v1 格式保存，即可复用 runner/evaluate 做同口径比较。

