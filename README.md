# 边缘模型部署研究系统

面向边缘部署的模型压缩研究脚手架。首轮只实现**小型数值模型的 W8A8 对称定点化**，
但存储与评估口径从第一天就与具体压缩方法解耦，后续加入剪枝、蒸馏等方法时可直接做
同口径比较。

## 设计约束

- **三类产物独立保存**：模型结构、校准/测试样本、转换结果互不依赖。
  - 模型：`store/models/<model>/model.json`
  - 样本：`store/datasets/<model>/<split>.samples.json`
  - 产物：`store/artifacts/<model>/<version>/artifact.json`
- **层白名单**：首轮仅支持 `linear`、`relu`。出现其它层时整单拒绝（退出码 2），不落半成品产物。
- **规则固定**：舍入为 half-away-from-zero（`-0.5 -> -1`），超范围一律饱和夹取；
  权重 int8 per-tensor 对称、激活 int8 per-tensor 对称（校准 absmax）、偏置 int32、累加器 int32。
- **独立运行器**：`run-artifact` 只读 `artifact.json` 即可做整数推理，不访问原模型/校准集。
- **核对计算结果**：评估以浮点模型为基准逐层、逐元素比较，且**必须**同时跑超校准范围（OOD）样本；
  不允许只报告体积变化。
- **溢出与语义截断分开**：输入量化、int32 累加、反量化饱和计入数值溢出；
  ReLU 的负向截断属于算子语义，单独计数不报警。

## 快速开始

```bash
python -m edge_deploy create-demo
python -m edge_deploy convert  --model mlp_demo
python -m edge_deploy evaluate --model mlp_demo   # 默认跑校准/域内/超范围三个 split
python -m edge_deploy run-artifact \
  --artifact store/artifacts/mlp_demo/w8a8-i8sym-v1/artifact.json \
  --input 4,-4,4
python -m edge_deploy compare --model mlp_demo --split test_out_of_range \
  --versions w8a8-i8sym-v1@test_out_of_range
```

`evaluate` 输出 `evaluation.json`：每个张量边界的 max/mean abs、RMSE、相对误差、
逐层饱和计数、误差最大的样本。OOD 样本预期出现饱和，用它暴露刻度标定边界而不是回避它。

## 后续接入其它压缩方法

1. 新方法生成自己的 `conversion_version` 产物，保持 `layers/activations` 可执行；
2. 若需要专用运行器，在产物 `conversion` 字段声明并实现独立 runner；
3. 评估统一走 `evaluate.evaluate_split`，`compare` 天然支持跨版本同 split 排名。

## 测试

```bash
python -m unittest discover -s tests
```
