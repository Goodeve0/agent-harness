# 功能缺陷核查与修复概览（2026-08-19）

按用户指令（"改动改动……能用就行，该有的得有"）对 7 项核查结论逐条处理，目标：可运行、达到 MVP 水平、不追求完美。

## 逐项处理结果

| # | 核查项 | 处理 | 说明 |
|---|--------|------|------|
| 2 | Golden Dataset 空 | 只看不改 | 链路完整：生成脚本 + 忽略规则均在，数据为运行产物可复现 |
| 3 | Judge 从未真实调用 | 已修复 | 新增 `--judge` 开关，mock 模式也能真实调用 LLM Judge |
| 4 | 缺标注 / Pass^k=0 水印 | 跳过 | 按用户要求不处理 |
| 5 | mock_params 硬编码工具名 | 已修复 | 改为从工具 schema 声明式生成参数 |
| 6 | 多 Agent 名不副实 | 已修复 | 新增 run_chain 多 Agent 协作链路 |
| 7 | function_calling.py 是壳 | 已补强 | 补设计注释 + 强化 system_prompt |

另外：0 分报告（`report/output/content_review_agent_v1_20260819_144435.json`）已从 output 移除。

## 关键改动

### #3 Judge 真实调用（run_eval.py）
- 新增 `--judge`：`allow_real_judge = (not mock_run) or with_judge`
- 缺 `OPENAI_API_KEY` 时构造前拦截（OpenAI SDK 构造即校验 Key），打印黄色警告后权重回落纯规则层（rule=1.0 / judge=0.0），不崩主流程；交叉校验同样 fail-open

### #5 mock_params 声明式（harness/mock_agent.py）
- 签名：`mock_params(tool_name, sample, gt, tools)`
- 从 tools schema 的 `parameters.properties` 遍历生成参数，取值优先级：sample.input 命中 > ground_truth 命中 > 类型占位（string/integer/number/boolean/object/array）
- 新增工具无需改代码

### #6 多 Agent 链路（harness/pipeline.py + tasks/content_chain/task.yaml）
- 新增 `run_chain`：spec 顶层 `agents` 列表驱动，每环 Agent 用自己的 tools/mock_apis/prompt/max_steps，上一环 final_output 作为下一环 upstream 上下文
- 链路 trace 合并为一条：steps/messages 拼接、agent_id 取 chain_id、最终输出取末环、token/延迟求和
- 抽取公共 `_score_trace`，单 Agent 与链路评分口径一致（双层评分 + 交叉校验 + 终端输出），rubric 无需改造
- demo：`tasks/content_chain/task.yaml`（初审 content_review_agent → 终审 content_approval_agent，9 样本）

### #7 function_calling.py
- 补详细设计说明注释（为何是薄壳：策略是 ReAct 循环 + 工具 schema 驱动）
- 强化 system_prompt：严格按工具定义传参、不得臆造参数、不得声称已调用未真实调用的工具、JSON 输出无解释文字

## 验证
- 测试 114 → **121 全绿**，compileall 零警告
- 端到端：mock 跑 content_chain（9 样本，缺陷注入按设计 1/3 trial 失败）、mock 跑 content_pipeline（未破坏）、`--judge`/`--cross-check` 缺 Key 均降级不崩
- git 状态干净，已推送 main（commit `2068dc5`）

## 踩坑记录
- 归因规则 `_REASON_RULES` 按 check 名子串匹配（tool_called/tool_sequence → TRAJECTORY_DEVIATION），rubric check 命名必须含关键词，否则落到 WRONG_CHOICE

## 使用方式
```bash
# 多 Agent 链路（mock）
python run_eval.py --task tasks/content_chain/task.yaml --mock-run

# mock 模式真实调用 LLM Judge（需 OPENAI_API_KEY）
python run_eval.py --task tasks/content_pipeline/task.yaml --mock-run --judge
```
