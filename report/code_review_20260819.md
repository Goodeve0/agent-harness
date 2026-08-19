# AgentHarness 全量代码审查报告（2026-08-19）

## 审查范围

| 层 | 文件 | 行数 |
|---|---|---|
| 入口 | run_eval.py | 552 |
| Harness | harness/agent_loop.py, harness/sandbox.py | 372 |
| Strategies | strategies/base.py, function_calling.py, react.py | 171 |
| Metrics | rule_checker, hybrid_scorer, judge, cli_judge, cross_validator, aggregators | 1270 |
| Data | dataset/tracer.py | 230 |
| Report | report/reporter.py | 233 |
| Tasks | tasks/content_pipeline, customer_service | 2 YAML |
| Tests | tests/ 下 5 个文件 | 805 |

**验证方式**：全量读码 + `pytest -q`（108 passed）+ `compileall` 全绿。

**修复进展（2026-08-19）**：P1-1 / P1-2 / P2-1 / P2-2 / P2-3 / P2-4 已全部修复并补回归测试（`tests/test_fixes.py`，10 组用例），详见文末「五、修复记录」。

---

## 一、总体评价

架构清晰，分层合理，是一个完成度较高的 Agent 评测框架：

- **分层解耦**：harness（沙箱/循环）→ strategies（策略插件注册表）→ metrics（双层评分）→ dataset（Bad Case 飞轮闭环）→ report（报告/CI Gate），职责边界清楚。
- **设计亮点**：
  1. 双层评分（确定性规则层 + 语义 Judge 层）+ 短路策略，省钱且可归因；
  2. critical 一票否决 + layer_weights 分层加权，评分口径成熟；
  3. Pass@k / Vote@k / Pass^k 区分"能力上界"与"稳定性"，并带 Wilson 置信区间；
  4. Bad Case 自动打标 → Golden Dataset → 回归验证的完整数据飞轮；
  5. mock 模式按 rubric 声明式注入缺陷，链路自检可离线验证；
  6. 代码注释质量高，设计动机交代清楚。

---

## 二、问题清单

### 🔴 P1（建议立即修复）

**P1-1. `CrossValidator.report()` 空记录时缺键 → 主流程 KeyError crash** ✅ 已修复
- 位置：`metrics/cross_validator.py:143-144` × `run_eval.py:495-497`
- 问题：当所有记录均无效（如两侧 CLI 都未安装、或全部 judge 返回 -1）时，`report()` 返回 `{"total": 0, "agreement_rate": 0.0}`，缺少 `mean_abs_diff` / `disputed_count` / `verdict` 键。而 `run_eval.py` 打印时直接索引 `cr['mean_abs_diff']`、`cr['disputed_count']` → **KeyError**。
- 影响：CLI 后端的 fail-open 设计（"不崩主流程"）在此处失效，`--cross-check` 场景下 judge 全部不可用时会直接崩溃。
- 修复：`report()` 空记录时补全所有键（0 值）。

**P1-2. Trace 落盘丢失完整对话历史（messages）** ✅ 已修复
- 位置：`harness/sandbox.py:67-92` `AgentTrace.to_dict()`
- 问题：`to_dict()` 只导出 `steps`（工具调用）和 `final_output`，**未导出 `messages`**（完整多轮对话，含每轮 assistant 消息与 Observation）。`tracer.dump_traces()` 用 `to_dict()` 落盘 → 磁盘上的 Trace 无法回放完整对话，只留下工具调用摘要。
- 影响：与"Trace 落盘（可复现 / 可回放）"的设计目标不符；Golden Dataset 的 `trace_snapshot` 同样缺对话历史；跨版本排查 bad case 时缺关键证据。
- 修复：`to_dict()` 增加 `messages` 字段。

### 🟡 P2（值得改进）

**P2-1. ReAct 策略：输出格式漂移时静默丢步** ✅ 已修复
- 位置：`harness/agent_loop.py:99-110`
- 问题：`extract_action()` 返回 None（既无 Final Answer 也无合法 Action）且无 tool_calls 时，当前轮消息不追加任何内容，直接进入下一轮，直到 max_steps 耗尽。模型"格式漂移"不产生任何失败信号，只表现为 token 浪费 + MAX_STEPS_EXCEEDED。
- 建议：该分支记录一个无效步骤（如 `trace.steps` 追加 error 标记或计数），让规则层可针对"无效输出步"打分。

**P2-2. `LLMJudge._parse_numeric` 兜底分支含死代码与隐含 clamp 行为** ✅ 已修复
- 位置：`metrics/judge.py:120-124`
- 问题：`for tok in reversed(candidates): ... return v / return 1.0` 第一次迭代必然 return，`return 0.0` 永不执行（死代码）；且任何 >1 的数字被 clamp 到 1.0，若模型输出 "8 分"（满分 10 漏写），会直接得 1.0 满分而非 0.8。
- 建议：删除死代码；对 >1 的值优先尝试 `x/10` 或 `x/100` 语义换算，再退化为 clamp。

**P2-3. 宽松模式 JSON 解析的正则贪婪/非贪婪边界** ✅ 已修复
> 注：`rule_checker.py` 的宽松 JSON 提取器经复核为"首个 `{` → 最后一个 `}` 全串尝试 + 反向剥离后缀"的实现，无贪婪截断风险；真正的非贪婪截断隐患在 ReAct 的 Action Input 提取（跨行 JSON 被提前截断），已一并修复。
- 位置：`metrics/rule_checker.py:273-279`
- 问题：`re.search(r"\{.*\}", text, re.S)` 贪婪匹配首个 `{` 到最后一个 `}`。若输出含多个 JSON 块或 JSON 后跟随含花括号的说明文字，会解析失败误判格式错误。
- 建议：改为从末尾反向定位闭合括号，或用 `json.JSONDecoder.raw_decode` 精确截取。

**P2-4. 测试缺口：空记录场景未覆盖** ✅ 已修复
> `tests/test_fixes.py` 补齐：空记录 report() 键完整性、Trace messages 回放、`_parse_numeric` 15 组分制用例、ReAct 多行/嵌套 JSON 提取、格式漂移纠错循环集成（恢复 / 达上限归因 FORMAT_ERROR / 步数耗尽仍归因 MAX_STEPS_EXCEEDED）。
- 正是 P1-1 未被测试发现的原因。建议补：`CrossValidator.report()` 全无效记录、`parse_cli_score` 各种边界、`extract_action` 多行 JSON/畸形输出、sandbox `_conditional` 条件 mock。

### 🟢 P3（优化建议）

- **P3-1** `run_eval.py` 552 行承担 CLI + mock agent + 注入调度 + trial 执行 + 回归合并 + CI 口径，建议按职责拆分（cli.py / mock_agent.py / pipeline.py）。
- **P3-2** `MockSandbox.call_tool` 对非 callable mock 值返回同一对象引用，若被测 Agent 修改 response 会污染后续调用（低风险，可 copy.deepcopy）。
- **P3-3** `ToolCall.is_mock` 字段在 `to_dict()` 未导出（sandbox.py:26 定义了但没序列化）。
- **P3-4** `HybridScorer` 短路 1（critical_failed）与短路 2（l1 json/fields）逻辑重叠，可合并。
- **P3-5** `reporter.to_dict()` 的 `total_samples` 实为 trial 数，靠注释兼容；建议显式命名为 `total_trials` 并废弃旧键。
- **P3-6** `agent_loop.py` 无 API Key 时 `OpenAI(...)` 构造不报错但调用时抛 401，主循环有 try/except 兜底，建议启动时显式探测并给出友好提示。

---

## 三、测试覆盖情况

| 模块 | 覆盖 | 缺口 |
|---|---|---|
| 聚合指标（Pass@k/Pass^k/Vote@k） | ✅ 数学正确性 | — |
| Golden Dataset 闭环（ingest/去重/resolved/stale） | ✅ | — |
| 规则层（L1/L2/L3、critical、权重） | ✅ | — |
| 双层评分合并与短路 | ✅ | — |
| CLI Judge 解析 | ✅（tests/test_cli_judge.py） | report() 空记录 |
| ReAct extract_action | ❌ 无专项测试 | 多行 JSON、畸形输出 |
| AgentLoop 集成（真实 LLM） | ❌（依赖 API，可理解） | — |
| Reporter / DiffReporter | ❌ 无测试 | — |

---

## 四、结论

项目整体质量良好：架构、注释、测试都在线，修复后 **108 测试全绿**，适合作为可扩展的评测基础设施继续演进。

**本轮已修复**：P1-1（KeyError crash）、P1-2（messages 落盘）、P2-1（格式漂移失败信号）、P2-2（数值解析死代码/分制推断）、P2-3（ReAct 跨行 JSON 截断）、P2-4（回归测试补齐）。剩余 P3 优化项不阻塞使用，可按需推进。

**未发现**：密钥硬编码、SQL 注入面（无 DB）、XSS 面（无前端渲染用户输入）、路径穿越风险。

---

## 五、修复记录（2026-08-19）

| 编号 | 问题 | 修复方案 | 涉及文件 |
|---|---|---|---|
| P1-1 | 空记录缺键 KeyError | `report()` 无有效记录时返回键齐全的 0 值 dict（fail-open），修复注释笔误 | `metrics/cross_validator.py` |
| P1-2 | Trace 缺完整对话 | `to_dict()` 导出 `messages` 字段，JSONL / golden `trace_snapshot` 可完整回放 | `harness/sandbox.py` |
| P2-1 | 格式漂移静默丢步 | 新增 `format_error_feedback()`（参照 LangChain ReAct 的解析失败回灌惯例）；`AgentLoop` 连续纠错达 `_MAX_FORMAT_RETRIES=2` 归因 `FORMAT_ERROR`，最后一次反馈先落 trace 再终止 | `strategies/react.py`、`strategies/base.py`、`harness/agent_loop.py` |
| P2-2 | 数值解析死代码 + "8分"误判满分 | 解析顺序：分数表达式 → 百分比 → 中文分制（N>1 才推断）→ [0,1] 取末位 → 兜底分制推断（10/100 分制，计数词"X 个/维度/次/条/步"不伪造分数）；全程 clamp | `metrics/judge.py` |
| P2-3 | ReAct 跨行 JSON 提前截断 | 收尾锚点由 `$`（MULTILINE 下零宽匹配每个换行前）改为 `\Z`（绝对末尾）；嵌套 JSON 用字符串感知的花括号栈 `_extract_json_object()` 提取 | `strategies/react.py` |
| P2-4 | 测试缺口 | 新增 `tests/test_fixes.py`：P1/P2 全部修复点 10 组回归用例 | `tests/test_fixes.py` |
| — | 附加 | `AgentLoop` 客户端惰性化（构造不强制 API Key，首次 `run()` 才建），便于装配与测试注入；修复 docstring 非法转义 `\{` 的 SyntaxWarning | `harness/agent_loop.py`、`strategies/react.py` |

**业界参照**：OpenAI Evals trace 记录完整对话（input + 每步 reasoning/tool_call/result）→ 对应 P1-2；生产级 extract_score 普遍"正则提取 + clamp + 分制推断"→ 对应 P2-2；LangChain ReAct 用 OutputParserException 把解析错误反馈给模型重试（有限次）→ 对应 P2-1。
