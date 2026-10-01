边缘模型部署研究系统（多输入组 + 逐层混合精度 + 方案比较 + 可追溯归档）

## 第二轮口径（本轮新增）
- 多代表性输入组：精度不由一份校准资料决定。每个输入组有独立样本、
  独立误差限制（统一上限或逐层上限），所有组共享同一个存储预算。
  见 edgequant/groups.py 的 InputGroup（name / samples / tol / in_calib_range）。
- 逐组报告满足情况：每组、每个 dense 层分别给 上限/实测/满足；
  任一组超差即整体失败，绝不以跨组平均误差掩盖某一组失效。
- 方案比较：edgequant/compare.py 重放两个产物，列出哪些层 4/8bit 发生
  变化，以及每个输入组上逐层 improved / degraded / unchanged，
  含存储比特差与各组是否满足。
- 明确失败：全 8bit 仍有组超差、或共同预算内无法让所有组达标时
  抛 PrecisionTargetError（CLI 退出码 3），不自动放宽任何目标。
- 重载复核：转换成功后 edgequant/verify.py 把产物落盘再 load_quant，
  在全部输入组上重放，预测、逐层轨迹、风险计数必须逐元素一致。
- 超校准范围输入：in_calib_range=False 的组照常核对误差，且必须报告
  输入裁剪、激活饱和、int32 累加溢出与 has_risk，不只给看似正常的预测。
- 可追溯：产物含 provenance（plan_id、原始限制、校准来源组、样本指纹、
  激活 scale）；方案库 artifacts/plans.json 只追加、不覆盖旧方案，
  旧方案 / 原限制 / 校准来源长期可查。

## 结构
- edgequant/groups.py   输入组定义/存取（edgequant.groups.v1）+ 样本指纹
- edgequant/quantize.py convert_groups：多组约束下贪心混合精度、
                         逐组误差/风险、校准来源与 provenance
- edgequant/compare.py  两方案逐层精度变化 × 逐组改善/退化比较
- edgequant/plans.py    方案归档（edgequant.plans.v1，只追加）
- edgequant/verify.py   产物重载后重复核对预测与风险
- edgequant/layers.py   浮点/定点语义：激活对称 int8；权重 4bit/8bit；
                         round-half-to-even + 饱和截断；乘加 int64 求值后
                         对 int32 边界显式饱和（不静默回绕）
- edgequant/model.py    模型 / 校准样本 / 转换产物 JSON 独立存取
- edgequant/runner.py   只吃转换产物 JSON 的纯整数推理 + 逐层风险
- edgequant/evaluate.py 逐层误差/风险 + boundary_check + stress_compare
- edgequant/stress.py   大累加、跨数量级尺度通道、异常输入压力小模型
- main.py               CLI（见下）
- tests/                test_quant.py（第一轮）+ test_groups.py（第二轮）

## 固定口径
- 激活：对称 int8，round-half-to-even，饱和截断 [-127,127]
- 权重：逐层混合精度，候选 4bit / 8bit 对称有符号
- 选择策略：全部 4bit 起步，按“跨组最差超差下降 / 额外存储比特”
  贪心跳级；所有输入组必须同时达标；relu 等无权重层误差照常披露
- 存储代价：权重比特 + 32bit 偏置；预算单位为 bit，多组共享
- 校准：激活 scale 只由 in_calib_range=True 的组汇总确定，
  超范围组不污染 scale，但其裁剪/饱和/溢出必须如实报告

## 用法
  python main.py make-model
      # 生成模型 + 旧 calib + 多输入组 *.groups.json
      # （typical / wide-range 校准内；out-of-calib 超校准范围）
  python main.py convert-groups --budget 4000 --plan-id p1
      # 多组约束自动混合精度；成功后自动重载复核并归档到 plans.json
  python main.py convert-groups --weight-bits 8 --plan-id p2
      --quant artifacts/p2.json      # 强制全 8bit 的对照方案
  python main.py compare-plans --plan-a p1 --plan-b p2 [--json]
  python main.py list-plans
  python main.py convert --tol 0.05 [--budget N]      # 单校准集旧口径
  python main.py evaluate / boundary / stress
  python -m pytest tests -q

## 产物
- 转换产物沿用 edgequant.quant.v1：mixed_precision_report 中记录
  input_groups（含样本指纹）、calibration（来源组/汇总指纹/scale）、
  group_reports（逐组逐层 上限/实测/满足 + 风险计数）、
  w_bits_for、total_bits/budget_bits；顶层 provenance 记录原限制。
- 方案库 edgequant.plans.v1：plan_id、产物路径、完整报告与 provenance，
  以及该方案的重载复核结果；同 plan_id 不允许覆盖。
