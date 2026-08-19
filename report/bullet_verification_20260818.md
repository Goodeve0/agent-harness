# 简历 Bullet × 代码落地核验报告

核验日期：2026-08-18
核验方式：逐条对照源码（非只读 README 或注释），关键路径均已落到文件+行号。

---

## 1. 双层评测 ⚠️ 基本落地（1 处表述需收紧）

**简历陈述**：设计规则层 + Judge 层双层评分架构，分别覆盖工具调用链路、格式合规及最终回答语义，支持 pattern、numeric 两种评分模式并加权聚合，降低 LLM Judge 单点依赖

| 陈述点 | 落地情况 | 证据 |
|---|---|---|
| 规则层覆盖工具调用链路 | ✅ | `metrics/rule_checker.py` L168-198：tool_called / tool_sequence / tool_param_equals / max_steps |
| 规则层覆盖格式合规 | ✅ | L133-146：json_format / required_fields（L1，天然 critical 一票否决） |
| 覆盖最终回答语义 | ✅ | `metrics/judge.py` LLMJudge 语义评分；`rule_checker.py` L248-249：llm_judge 规则返回 -1 不判定，交 Judge 层 |
| pattern / numeric 双模式 | ✅ | `metrics/judge.py` L92-95 + PATTERN_PROMPT/NUMERIC_PROMPT L16-43 |
| 加权聚合 | ✅ | `metrics/hybrid_scorer.py` L172：final = w_rule×rule + w_judge×judge；两 task 分别配 0.7/0.3、0.8/0.2 |
| 降低 LLM Judge 单点依赖 | ✅ | 短路机制 L124-147：L1 挂掉直接 0 分不调 Judge；`report/reporter.py` L69 统计 judge_saved_rate；规则层 + 交叉校验双保险 |

**额外发现**：规则层内部还落地了 L1/L2/L3 分层权重（0.2/0.6/0.2），`tasks/*.yaml` 均有 `scoring.layer_weights`，`RuleScore.score()` 支持分层加权（`rule_checker.py` L70-82），简历 bullet 未体现这个点，可加。

**唯一风险点**：bullet 说"最终回答语义"由 Judge 层覆盖，但语义规则依赖 `judge_rubric` 配置且非 mock 模式才启用（`run_eval.py` L349）。若面试官问"语义层用什么模型、什么 prompt"，能答 `judge_mode: numeric` + 每 task 的 judge_rubric，OK。**面试建议**：主动说"Judge 层只在规则层无法判定的语义维度上工作，规则能判的不花钱调 LLM"——这是亮点。

---

## 2. Agent 策略架构 ✅ 真落地，无水分

**简历陈述**：设计 Strategy 插件机制，将 Function Calling 与 ReAct 推理模式解耦为独立策略类，通过注册表动态注入，实现评测任务无需修改 AgentLoop 即可切换执行策略

| 陈述点 | 落地情况 | 证据 |
|---|---|---|
| 解耦为独立策略类 | ✅ | `strategies/function_calling.py`、`strategies/react.py`，各自独立类 |
| 注册表动态注入 | ✅ | `strategies/base.py` L16-30：`_STRATEGY_REGISTRY` + `register_strategy` 装饰器 + `get_strategy`（未注册报错） |
| 无需修改 AgentLoop 即可切换 | ✅ | `harness/agent_loop.py` L47：`get_strategy(strategy)()` 一行实例化，主循环无任何策略分支代码 |
| 新策略可热插拔 | ✅ | 新增策略 = 继承 BaseStrategy + `@register_strategy("name")`，AgentLoop 零改动 |
| 超步语义统一 | ✅ | AgentLoop 统一处理：步数耗尽 = success=False + MAX_STEPS_EXCEEDED；策略层 should_stop 不再把 max_steps 当成功 |

**面试可扩展点**：BaseStrategy 三个抽象方法（system_prompt / build_messages / should_stop），AgentLoop 只依赖接口。面试官若问"加一个 planner 策略要改哪"，答"新建一个类 + 注册，AgentLoop 不动"。

**附注**：本轮顺手修正了 `agent_loop.py` L88 过时注释（残留"由下方 for...else 判断"字样，与实际代码不符）。

---

## 3. 模型交叉评测 ✅ 框架 + CLI 后端已落地（需真配 Key 实测一次）

**简历陈述**：落地 Claude Code + Codex 双模型交叉校验机制，结合 Trace 分析优化评测链路，定位 Agent 执行过程中的工具调用及结果偏差，归因交付周期由周级压缩至小时级

| 陈述点 | 落地情况 | 证据 |
|---|---|---|
| 双模型交叉校验框架 | ✅ | `metrics/cross_validator.py` 完整实现：双 judge 独立评分 → agreement_rate / mean_abs_diff / disputed_samples |
| 分歧仲裁 | ✅ | L65-71 merged_score：一致取均值、分歧取保守分（min）、单侧异常采信有效侧、双侧异常 0 |
| judge 异常处理 | ✅ | L128-135 `_safe_judge`：异常返回 -1 而非 0，不拉低仲裁分；**本轮修复**：judge 显式返回 -1（CLI 不可用/无法解析）同样保留 -1 语义，不再被转成 0 分 |
| **Claude Code + Codex 真实后端接入** | ✅ 本轮新增 | `metrics/cli_judge.py`（新）：`CLIJudge` 子进程调用 `claude -p` / `codex exec`，与 LLMJudge 同接口可混搭注入；共享同一套 NUMERIC/PATTERN_PROMPT；fail-open 不伪造分数。`run_eval.py` 新增 `--judge-a-backend/--judge-b-backend {openai,claude,codex}`，启动时探测 CLI 可用性 |
| 主流程接线 | ✅ | `run_eval.py` L362-371 + L419-425：`--cross-check --judge-a --judge-b --judge-a-backend`，仲裁结果回写 judge_score/overall_score |
| 单测覆盖 | ✅ | `tests/test_fixes.py`（valid-only / dispute-conservative / agreement-mean）+ `tests/test_cli_judge.py` 新增 19 个（命令构造 / 四级解析 / fail-open 四场景 / probe / 集成仲裁） |
| 结合 Trace 定位偏差 | ✅ | Trace 全量落盘（tracer.dump_traces）+ 规则层 failed check 逐条归因 + suggestion 改进建议 |

**落地记录（本轮）**：
- 新文件 `metrics/cli_judge.py`：CLIJudge（claude/codex 两种后端）+ `parse_cli_score`（JSON → 分数表达式 → 百分比 → 0-1 数字四级解析）+ `probe_cli` 探测。
- 修复 `cross_validator._safe_judge` 真 bug：judge 返回 -1 曾被 `0.0 if s < 0` 转成 0 分（把"无法判定"当"0 分"拉低仲裁），现已保留 -1 语义。
- 本机探测（2026-08-18）：`codex` CLI 已装（0.147.0），`claude` CLI 未装（探测即 fail-open 降级，设计内行为）。
- 使用文档：`report/claude_codex_cross_judge_20260818.md`，含 3 条运行命令 + 面试表述收紧版。

**⚠️ 剩余诚实边界（不是代码问题，是证据问题）**：
1. 真实模型评分需要 API Key：claude CLI 需安装 + `ANTHROPIC_API_KEY`；codex 本机已具备环境。建议配 Key 后跑一次 `--judge-a-backend claude --judge-b-backend codex` 留一份报告 JSON 作为面试证据。
2. **"归因交付周期由周级压缩至小时级"无数据支撑**，已从面试表述中移除（见交叉评测文档 §7 收紧版）。

---

## 4. 评测数据闭环 ✅ 落地；旧污染数据已清空（回归集从干净状态开始）

**简历陈述**：建设 Trace-to-Dataset 一键标注与回归集沉淀流程，将带标签样本自动纳入持续回归测试，支持模型版本迭代后的效果对比

| 陈述点 | 落地情况 | 证据 |
|---|---|---|
| Trace-to-Dataset 一键标注 | ✅ | `dataset/tracer.py`：bad_cases 自动打归因标签（failure_reason 由规则层产出，非人工）；`GoldenDataset.ingest` L123-187 一键归入回归集 |
| 回归集沉淀（JSONL 落盘） | ✅ | `dataset/golden_set.jsonl`，含 input/ground_truth/failed_checks/suggestion/trace_snapshot 完整快照 |
| 带标签样本纳入持续回归 | ✅ | `ingest` 去重键 (task_id, sample_id, failure_reason)；`mark_resolved` 回归验证；fail_count + stale_cases（复现≥3 次告警） |
| 模型版本迭代效果对比 | ✅ | `report/reporter.py` DiffReporter：pass_rate/avg_score/failure_delta/check_delta 逐 check 对比；CI Gate 用样本级 Pass@k（`run_eval.py` L473-482） |
| mock 不污染回归集 | ✅ | `run_eval.py` L434-436：mock 模式完全跳过写入与 resolved 标记 |

**✅ 旧数据污染已处理（本轮）**：原 `golden_set.jsonl` 3 条记录的 `task_id` 为旧格式 `content_review_001_hard_001`（task_id 拼接 sample_id），`mark_resolved` 永远匹配不上干净 key。按用户决定**不做迁移、直接清空**，回归集从干净状态开始，后续 ingest 全部为干净格式：
- 备份：`report/backups/golden_set_20260818_legacy_polluted.jsonl`（7943 字节，如需回溯可取用）
- `dataset/golden_set.jsonl` 现为 0 字节，`GoldenDataset` 空文件读取验证通过（entries=0）

---

## 结论汇总

| 维度 | 结论 | 风险等级 |
|---|---|---|
| 1. 双层评测 | 基本落地，可加 L1/L2/L3 分层权重亮点 | 🟢 无 |
| 2. Strategy 插件机制 | 完全落地，最扎实的一个 | 🟢 无 |
| 3. 双模型交叉评测 | 框架 + Claude/Codex CLI 后端已落地（69 测试全绿）；真实评分需配 Key 实测一次 | 🟡 需配 Key 实证 |
| 4. 评测数据闭环 | 落地，旧污染数据已清空，回归集从干净状态开始 | 🟢 无 |

**面试优先级**：维度 2 可放心讲；维度 1 建议补"分层权重"细节；维度 3 代码已可讲（CLI 后端 + fail-open 降级 + 仲裁），配 Key 真跑一次留证据更稳；维度 4 已干净可直接讲。
