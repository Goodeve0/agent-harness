# AgentHarness MVP 代码审查报告

- **审查日期**: 2026-08-18
- **范围**: 全部 6 个模块（harness / metrics / dataset / report / strategies / tasks）+ 主入口
- **测试基线**: `pytest` 30 passed；`--mock-run` 双任务实跑验证

---

## 结论速览

| 级别 | 数量 | 说明 |
|------|------|------|
| 🔴 严重（P0/P1） | 3 | 数据污染、缺陷注入失真、步数耗尽误判成功 |
| 🟡 中等（P2） | 5 | task_id 污染、交叉校验不生效、死代码、口径混淆 |
| 🟢 低风险（P3） | 6 | 易踩坑与可改进项 |

架构设计（双层评分短路、声明式规则、数据飞轮闭环）是清晰的，**方向没问题**，问题集中在「mock 链路的数据可信度」和「统计口径一致性」上。

---

## 🔴 严重问题

### 1. Mock 模式污染 Golden 回归集（实测复现）

`run_eval.py:324-331`：`--mock-run` 下 Bad Case 照常 `golden.ingest()`，且 `mark_resolved()` **无条件执行**——`--no-ingest` 只挡 ingest，挡不住 resolved 标记。

**实测**：一次 `--mock-run` 后 `golden_set.jsonl` 从 3 条 → 9 条，且新条目是 mock 假 Agent 的失败样本。

**后果**：
- mock 数据代表的是「注入的缺陷模式」，不是真实 Agent 能力，喂进回归集让「修复率 / 未修复计数」完全失真；
- 更危险的是 `mark_resolved`：若历史真实 Bad Case 的 key 被 mock 运行「全过」覆盖，会被**误标为已修复**，回归集失去意义。

**修复**：mock 模式下跳过 ingest 与 mark_resolved（或强制等于 `--no-ingest` 并打印警告）。

### 2. Mock 缺陷注入器与 Task Rubric 脱节（实测复现）

`run_eval.py:70-117` 的 `_mock_agent_run` 按难度注入缺陷，但**没有针对具体 task 的 rubric 定制**，实测两个任务行为都失真：

| Task | 注入意图 | 实际行为 |
|------|---------|---------|
| customer_service | trial1 注入「reason 空」 | rubric 无 reason 检查 → **注入无效，该 trial 全对** |
| content_review hard | trial2/3 注入「金额错 / 声称退款」 | 无 `amount`/`submit_refund` → **什么都不注入**，输出缺 `reason` 字段 → 归因为 `MISSING_FIELD`（意图是金额错） |

**后果**：mock 模式的 pass@k / 失败归因分布**没有任何验证价值**，只能证明「链路跑通」。归因标签失真还会传导到 golden 集（同问题 1）。

**修复**：缺陷注入改为声明式（在 task.yaml 定义每个 sample 的注入缺陷），或 mock 模式只做 smoke 断言不产出统计结论。

### 3. AgentLoop 步数耗尽误判成功

`harness/agent_loop.py:86-89`：`should_stop` 中 `step >= max_steps` 返回 True 后，`trace.success = True`。若 Agent 在第 max_steps 轮恰好输出非空 content（同时仍有 tool_calls），**超步被当成成功退出**，`MAX_STEPS_EXCEEDED` 永远不会被触发（只在 final_output 为空时兜底）。

**修复**：区分「正常结束」与「步数耗尽」两种停止原因，超步必须 `success=False + MAX_STEPS_EXCEEDED`。

---

## 🟡 中等问题

### 4. task_id 被 sample_id 污染（隐式约定，脆弱）

`sandbox` 构造时 `task_id=f"{spec['task_id']}_{sample['sample_id']}"`，导致 golden 集里 task_id 变成 `content_review_001_hard_001`（语义错误）。`mark_resolved` 的 key 匹配（`run_eval.py:294` 与 `tracer.py:198`）能对上，**纯属这套拼接约定的巧合**——一旦 sample_id 命名规则变化（含下划线）立即 break。

### 5. 交叉校验结果不参与最终评分

`CrossValidator.merged_scores()` 从未被调用，`cross_check` 只作为展示字段。文档声称「分歧时取保守值、一致时采信均值」，**实现未生效**。要么接入最终分，要么改文档明确「仅报告」。

### 6. 两套评分体系并存，存在死代码

- `harness/eval_agent.py`（EvalAgent 全类）—— 无任何引用
- `metrics/rubric.py`（全部函数，含 L1/L2/L3 评分与 `compute_overall_score`）—— 无任何引用

实际生效的是 `RuleChecker + HybridScorer`。死代码让「默认权重 L1 20% / L2 60% / L3 20%」这类文档描述与实现不一致，建议删除或明确废弃。

### 7. 统计口径混淆：pass_rate 是 trial 级

`Reporter.to_dict()` 的 `pass_rate` / `total_samples` 实际是 **trial 总数**（samples × runs），CI Gate 用的就是这个 trial 级通过率；而 aggregator 的 `pass@k` 是**样本级**。改 `--runs` 会直接改变 CI 结果，不同 runs 的两次评测不可比。

### 8. `--judge-a/--judge-b` 未传时打印 `None × None`

`run_eval.py:261` 直接用 CLI 参数打印，应为实际模型名。

---

## 🟢 低风险 / 可改进项

1. `rule_checker.py:148`：`bool(rule.get("critical", False))` —— YAML 写 `critical: "false"`（字符串）会被转成 True，建议改为 `is True` 判断。
2. `rule_checker.py:166`：`tool_param_equals` 用 `next()` 只查**第一次**匹配的工具调用，同工具多次调用时后续参数不校验（当前用例靠输出字段兜住，但存在漏检面）。
3. `judge.py:100`：`_parse_numeric` 正则 `[\d.]+` 脆弱，模型输出 `0.85.1` 等格式会解析错位。
4. `cross_validator.py:120`：`_safe_judge` 异常时返回 0.0，Judge 故障会被当成 0 分拉低一致率，建议返回 `-1` 并跳过该样本。
5. `reporter.py:125`：`f"{output_dir}/{rid}.json"` 字符串拼接，应使用 `Path`（Windows 兼容）。
6. `requirements.txt` 全部 `>=` 无上限；`aggregation.k` 在 task.yaml 里声明了但被忽略（实际用 `--runs`）。

---

## 建议修复优先级

1. **P0**：mock 模式隔离回归集（问题 1）—— 一行条件即可，阻止数据污染
2. **P0**：AgentLoop 超步判失败（问题 3）
3. **P1**：mock 缺陷注入改声明式（问题 2）—— 与 1 一起做
4. **P2**：统一统计口径（问题 7）、清理死代码（问题 6）、修正 task_id 拼接（问题 4）

> 注：审查过程运行的 mock 评测已还原，`golden_set.jsonl` 保持 3 条原样，未残留污染数据。
