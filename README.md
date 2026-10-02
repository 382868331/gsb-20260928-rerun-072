边缘模型部署研究系统（多输入组逐层混合精度 + 按组精度路由
             + 逐组误差/共同预算约束 + 部署方案对比 + 饱和风险保留
             + 重载复核与可追溯）

## 结构
- edgequant/layers.py    浮点/定点语义：激活对称 int8；权重逐层 4bit/8bit；
                          round-half-to-even + 饱和截断；乘加 int64 求值后对
                          int32 累加边界显式饱和（不静默回绕）
- edgequant/model.py     模型 / 单组校准 / 多输入组校准 / 转换产物 JSON 存取
- edgequant/quantize.py  多组共同校准、逐组逐层候选精度误差评估与贪心混合
                          精度选择；candidate_plans / compare_plans 做多方案
                          对比；不支持的层抛 UnsupportedLayerError；任一输入
                          组误差或共同预算达不到时抛 PrecisionTargetError，
                          不降低标准、不做组间平均
- edgequant/routing.py   部署前冻结的 JSON 输入条件；只按输入特征做确定性
                          首条规则路由，不读取预测误差；交叠标 ambiguous，
                          未覆盖标 uncovered，均走产物声明的统一回退方案
- edgequant/deployment.py 统一回退方案 + 按输入组独立校准的候选精度方案；
                          预算只允许容纳部分候选时按改善/额外存储预先选择；
                          产物记录实际路径、逐组误差、共同占用存储与边界结果
- edgequant/runner.py    独立运行器：只吃转换产物 JSON，纯整数推理；
                          逐层报告输入截断/激活饱和/int32 累加溢出与风险标记
- edgequant/evaluate.py  逐层 max_abs_err/RMSE/SQNR + 风险；boundary_check
                          超校准范围验证；stress_compare 转换前后压力对比；
                          reverify 产物重载后重复核对（逐组预测复现 + 越界风险）
- edgequant/stress.py    压力小模型与 nominal/wide/edge 三个代表性输入组
- main.py                CLI：make-model / convert / compare / evaluate /
                          deploy / boundary / stress / reverify
- tests/test_quant.py    混合精度选择、预算/误差显式失败、逐组独立达标、
                          方案对比改善/退化、交叠/未覆盖回退、预算仅容纳
                          部分候选、条件变更失效、重载复核、越界风险等用例

## 固定口径
- 激活：对称 int8，round-half-to-even，饱和截断 [-127,127]
- 权重：逐层混合精度，候选 4bit / 8bit 对称有符号
- 多输入组：每个代表性输入组（如 nominal/wide/edge）有自己的样本与逐层
  max_abs_err 上限；所有组共同决定一份 scale（取逐层激活并集上界），
  共享一份存储预算。达标判定逐组逐层独立进行，报告逐组给 PASS/FAIL，
  绝不用组间平均误差掩盖某一组失效
- 选择策略：全部从 4bit 起步，每轮选择让“最坏(组,层)超差裕度 / 额外
  存储比特”下降最多的 dense 层升级到 8bit；relu 等无权重层误差来自
  激活粒度，固定不可降但逐组照常披露
- 存储代价：权重比特 + 32bit 偏置；预算单位为 bit，所有组共享
- 失败语义：最高 8bit 仍有任一输入组超差，或共同预算内无法让全部组
  达标 -> 明确报错（CLI 退出码 3），绝不悄悄放宽要求或改判
- 方案对比：compare 同时评估 all-4bit / greedy-mixed / all-8bit 三套
  部署方案，逐组给出达标情况与存储代价，pairwise 展示每一层精度变化
  让每个输入组的误差 improved / unchanged / degraded；没有方案满足
  全部约束时 feasible=false 并列明失败，不自动选择或放宽目标
- 统一 vs 按组：deploy --grouped 先构建统一方案作为公共回退，再为每个
  输入组独立校准候选。选择只依据保存前校准结果中的误差改善/额外存储，
  运行时只执行冻结条件，不能看过实际预测误差后改选。额外存储会逐项说明
  换来了哪些组的 worst max_abs_err 改善；未入选组仍走统一方案
- 路由去向：规则按数组顺序判断并采用首条命中；同时符合多组时记录
  ambiguous、命中的全部组和实际规则；不符合任何组时记录 uncovered。两种
  情况的实际 plan 都是 route_policy 中声明的 fallback_plan（默认 unified）
- 共同存储：统一回退方案的完整权重计入一次，每个入选分组方案完整计入；
  不对不同方案间的权重或 scale 做去重，总存储超过预算则该候选不入选
- 校准证明范围：保存 policy_sha256_16，内容包括分组样本、容差和条件。
  修改条件边界或分组样本后，verify_deployment 会拒绝沿用旧产物，必须重新
  校准生成新证明
- 重载复核：reverify 只吃转换产物 JSON，重新预测并逐层核对与产物记录
  逐元素一致；越界输入必须报告输入裁剪、激活饱和/累加溢出与风险；
  provenance 保留旧方案限制（容差/预算）、校准来源路径与校准内容指纹
- 越界语义：超出校准范围的输入不允许只返回看似正常的预测，输入截断、
  激活饱和、int32 累加溢出全部计入逐层 risk

## 用法
  python main.py make-model                     # 示例模型 + 单组/多组校准
  python main.py compare --tol 0.05            # 三方案逐组改善/退化对比
  python main.py compare --tol 0.05 --budget 800  # 再加共同存储预算
  python main.py deploy --grouped --tol 0.05 --budget 1200
  python main.py reverify --grouped
  python main.py convert --tol 0.05            # 多组自动混合精度
  python main.py convert --tol 0.05 --budget 800
  python main.py convert --weight-bits 8       # 强制全 8bit（仍逐组校验）
  python main.py reverify                      # 重载产物后重复核对预测+越界
  python main.py evaluate                      # 首个输入组逐层误差/风险
  python main.py boundary                      # 超范围输入边界验证
  python main.py stress --tol 0.02             # 压力多组转换前后+复核
  python -m pytest tests -q

## 产物格式
沿用 edgequant.quant.v1：每层含 w_bits；mixed_precision_report 含逐组
报告 groups（每组逐层 err/tol/satisfied）、共同预算与总存储比特；
provenance 记录 calib_source、各组样本数与容差、校准指纹 calib_sha256_16、
预算与生成时间。校准文件新增 edgequant.calib-groups.v1（旧 v1 单组
仍可读取，按 default 单组处理）。独立 load_quant 后 run_quant 的输出、
风险以及 reverify 重算误差均与转换阶段逐元素一致。

按组部署新增 edgequant.deployment.v2：顶层 plans 保存 unified 与入选的
group:<name> 完整方案；route_policy.rules 保存条件、顺序、实际 plan 与
未入选原因；storage 保存统一方案比特、各入选方案比特、共同占用总量和
预算；deployment_report 保存每组统一/分组误差与新增存储改善。复合产物
也可直接传给 run_quant，返回风险中的 route_assignments 可供逐样本审计。
