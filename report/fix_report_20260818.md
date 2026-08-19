# AgentHarness 修复报告

日期：2026-08-18
对应审查：`report/code_review_20260818.md`
验证：pytest **50 passed**（30 原有 + 20 新增回归测试）+ 双任务 mock 实跑

---

## 修复总览

| 级别 | 问题 | 状态 |
|---|---|---|
| 🔴 P0 | Mock 模式污染回归集 | ✅ 已修复 |
| 🔴 P0 | 超步误判成功 | ✅ 已修复 |
| 🟡 P1 | Mock 缺陷注入失真 | ✅ 已修复 |
| 🟡 P1 | task_id 被 sample_id 污染（含 mark_resolved key 失效） | ✅ 已修复 |
| 🟡 P1 | 交叉校验不生效 + Judge 异常当 0 分 | ✅ 已修复 |
| 🟡 P1 | 死代码（eval_agent.py / rubric.py） | ✅ 已清理 |
| 🟡 P1 | pass_rate 口径混淆（CI 用 trial 级） | ✅ 已统一 |
| 🟢 低 | critical 字符串、tool_param 只查首调、judge 正则、Windows 路径、aggregation.k 忽略 | ✅ 已修复 |

---

## 逐项改动明细

### 1. 🔴 Mock 模式污染回归集（P0）
**文件**：`run_eval.py`
- `--mock-run` 时**完全跳过** GoldenDataset 写入与 `mark_resolved`（打印「Mock 模式：跳过 Golden Dataset 写入与回归 resolved 标记」）。
- 实测：mock 跑完 `golden_set.jsonl` 保持 3 条原样，`diff` 与审查前一致。

### 2. 🔴 超步误判成功（P0）
**文件**：`harness/agent_loop.py`、`strategies/function_calling.py`、`strategies/react.py`
- 原逻辑：策略 `should_stop` 里 `step >= max_steps` 直接返回 True → 循环跑满必然「成功」。
- 新逻辑：循环层用 `completed` 标志，`for` 循环自然结束 = 步数耗尽 → `success=False` + `MAX_STEPS_EXCEEDED`；策略层只负责「Agent 主动收尾」判断。
- 边界：最后一轮恰好输出最终答案（无 tool_calls）仍算成功。
- 验证（mock OpenAI client 三场景）：耗尽→失败；主动收尾→成功；末轮收尾→成功。

### 3. 🟡 Mock 缺陷注入声明式（P1）
**文件**：`run_eval.py`
- 新增 `_injection_plan(rubric)`：**从 rubric 自动推断注入点**，保证「注入什么缺陷 → 归因什么标签」一一对应：

| 注入缺陷 | 推断来源 | 归因 |
|---|---|---|
| `field_empty` | l3 `field_non_empty` | WRONG_CHOICE |
| `field_wrong` | l2 critical `field_equals` | CRITICAL_FAILURE |
| `tool_param_wrong` | l2 critical `tool_param_equals` | TOOL_PARAM_ERROR |
| `reason_contains_violation` | l3 critical `regex_absent` | SAFETY_VIOLATION |
| `skip_last_tool` | l2 `tool_sequence` | TRAJECTORY_DEVIATION |

- 注入调度 = `[正常] + 缺陷循环`：k=3 → `[正常, 缺陷A, 缺陷B]`；k=4 → 全部三种缺陷。
- 支持 YAML `mock_injection.<difficulty>` 显式覆盖自动推断。
- 双任务实测归因完全正确（content_review：WRONG/CRITICAL/SAFETY；customer_service：CRITICAL/TOOL_PARAM×2/TRAJECTORY）。

### 4. 🟡 task_id 污染 + mark_resolved key 失效（P1）
**文件**：`run_eval.py`、`dataset/tracer.py`
- `MockSandbox` 的 `task_id` 不再拼接 `sample_id`（此前 `content_review_001_easy_001` 污染 golden key）。
- `passed_keys` 格式修正为 `{task_id}::{sample_id}`（**此前格式与 `mark_resolved` 不匹配，回归标记实际从未生效**）。
- `aggregator` 的 key 用干净 `sample_id`。
- 回归测试验证 ingest → mark_resolved 闭环可命中。

### 5. 🟡 交叉校验落地 + Judge 异常处理（P1）
**文件**：`run_eval.py`、`metrics/cross_validator.py`
- `run_trial` 中 cross-check 的 `merged_score`（一致取均值 / **分歧取保守**）**回写** `judge_score` 与 `overall_score`，双模型仲裁真正影响最终分。
- `_safe_judge` 异常返回 **-1**（此前当 0 分，拉低仲裁分）；`merged_score` 只采信有效侧，双侧异常才按 0。
- 报告一致率只统计两侧均有效的样本。

### 6. 🟡 死代码清理（P1）
- 删除 `harness/eval_agent.py`、`metrics/rubric.py`（已 grep 确认零引用）。

### 7. 🟡 口径统一（P1）
**文件**：`run_eval.py`、`report/reporter.py`
- **CI Gate 改用样本级 `mean_pass@k`**（trial 级 pass_rate 随 `--runs` 漂移不可比），并打印所用指标。
- Reporter 输出标注「Pass Rate (trial 级)」，新增 `total_trials` 字段澄清语义。
- `--runs` 默认取 YAML `aggregation.k`（CLI 显式传入优先）。
- 附带修复：`--judge-a/b` 未传时不再打印 `None × None`（显示解析后的模型名）。

### 8. 🟢 低风险项
- **critical 字符串**：新增 `_as_bool`，`critical: "false"` 不再被 `bool()` 误判为 True。
- **tool_param_equals**：校验**全部**调用，任一参数错误即失败（副作用工具重复调用即资损），detail 报 `错误次数/总调用`。
- **judge 正则**：支持 `8/10`、`85%` 等模型常见输出格式，数字匹配收紧为 `\d+(?:\.\d+)?`。
- **layer_weights**：落地「L1 20% / L2 60% / L3 20%」分层权重（YAML `scoring.layer_weights`，缺省保持旧的全 check 加权，向后兼容）。
- **Windows 路径**：`Reporter.save` 改用 `Path` 拼接。
- **requirements**：openai 加 `<2.0` 上限。

---

## 验证结果

- `pytest tests/` → **50 passed**（新增 `tests/test_fixes.py` 20 例：超步 3 场景、mark_resolved 闭环、layer_weights、_as_bool、cross 仲裁、tool_param 全调用）。
- mock 双任务（runs=4/5）归因对照表全部命中预期标签。
- mock 运行后 `golden_set.jsonl` 与审查前 `diff` 一致（3 条，未污染）。
- `--mock-run` 终端明确提示跳过回归集写入。

---

## 遗留说明（重要）

1. **历史 trace 明细丢失**：收尾清理 mock 产物时通配符过宽（`content_review_agent_v1_*`），误删了 **8 月 12 日的两条历史 trace**（`report/traces/*_20260812_*.jsonl`），无备份且项目无 git，**不可恢复**。对应的报告 JSON（`report/output/*_20260812_*.json`）已从 /tmp 备份恢复，聚合结论无损。教训已记入工作日志：删除只精确匹配本次 run_id。
2. mock 模式下 `trace.success` 恒为 True 是刻意的（评分以 `eval_result` 为准），如需「mock 也验证超步路径」，可在后续用声明式注入扩展。
3. `aggregation.k` 现在生效：不传 `--runs` 时两个 task 默认 k=3。

## 后续建议

- 将项目纳入 git，报告/trace/golden 均可回滚。
- 给 `tests/test_fixes.py` 之外的 AgentLoop 真实链路补集成测试（需要 mock LLM 响应序列）。
