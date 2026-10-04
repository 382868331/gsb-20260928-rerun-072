# precswitch — 有状态输入流中的精度方案切换

对连续输入形成的会话比较三种选择方式，并保证切换决策因果、状态可解释、预算完整、校准不越界。

## 三种选择方式（precswitch/engine.py 的 Mode）

- UNIFORM：统一方案，所有会话所有输入都用同一方案，忽略分组。
- INDEPENDENT：每次输入独立选组选方案，不保留会话状态。
- STATEFUL：会话保留内部状态，按 SwitchPolicy（切换条件 + min_hold 最短保持范围）做切换。

## 核心约束

- 因果切换：位置 i 的决策只用输入 0..i 与固定规则；API 不存在按后来误差回选历史路径的入口（test_causal_decisions_only 用相同前缀验证决策一致）。
- 切换预览（preview.py）：说明能否延续（can_continue）、不能延续时必须从哪个明确位置重启（restart_at）、哪些先前输出仍基于旧方案（old_scheme_outputs）。状态无法被当前方案解释时抛 UnexplainableStateError，不允许无声丢状态后声称连续推断。
- 状态不可互换：StateKind 区分 STATELESS / COMPATIBLE（经 migrator 迁移）/ INCOMPATIBLE。不可迁移时引擎给出显式 ROLLED_BACK，保留旧方案与旧状态。
- 预算（budget.py）：共享的已载入方案成本只计一次，按引用计数——没有会话或进行中的迁移再使用时立即释放；每个活跃会话的持久状态成本持续计入，暂时没有新输入也不豁免（test_idle_session_still_counts）。
- 切换资格：`project_switch` 同时给出稳态占用（steady）与迁移峰值（peak）：迁移期间旧状态未释放、新状态已建好，目标方案的共享负载也可能尚未与旧方案重叠；两者任一超预算即拒绝，拒绝先于任何状态改动，旧方案与旧状态保持可用并说明阻塞原因。
- 两阶段切换：`begin_switch` 预留峰值 → 迁移构造新状态（旧状态仍在）→ `commit_switch` 释放旧状态/旧共享负载。迁移器抛错时取消预留并产出显式 `SWITCH_FAILED`；可在提交前中断，此时 `SWITCHED_RESTARTED`，产物中同时保存旧/新状态，重开后按真实保留资源重建预算，再 `resume_pending_switch` 或 `cancel_pending_switch`。
- 校准边界（calibration.py）：校准绑定分组规则版本与保持规则版本；改变分组条件或保持规则后查询旧校准范围抛 CalibrationError，原范围不给新切换路径背书。
- 持久化（persistence.py）：完整产物（会话、状态、历史、规则/策略版本、预算）保存后重开，按原会话路径延续；规则版本变化拒绝载入。

## 核对场景（tests/，47 个用例）

- 往返跨组：min_hold 大时全部 HELD_BY_MIN_HOLD；min_hold=1 时可见抖动且不可迁移处显式回退。
- 规则切换时仍有活跃会话：切换预览列出旧方案输出位置，状态迁移或显式重启位置。
- 存储不足的短输入流：会话建立或切换时 BUDGET_DENIED / 回退，旧状态保持。
- 未覆盖输入：UNCOVERED_INPUT 显式结果，流继续。
- 无法迁移状态：ROLLED_BACK + 预览说明；无法解释状态抛错。
- 回退选择：预算不足或状态不可迁移时均记录显式回退而非静默处理。
- 资源完整轨迹（test_switch_budget_trajectory.py）：预算 50 / 共享 30 / 目标状态 100 时切换被拒绝（稳态 130、峰值 140）；释放无关占用后重试成功；迁移器失败释放预留并恢复旧方案；迁移中断→保存→重开→取消/恢复，候选复用已有共享负载不重复计费，历史切换结果不被重新解释。

## 运行

    python3 -m pytest tests -q
