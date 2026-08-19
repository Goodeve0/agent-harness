# AgentHarness

面向工具调用型 Agent 的评测框架。提供规则 + LLM Judge 双层评分、Function Calling / ReAct 策略切换、Trace 回归闭环。

## 快速开始

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

## 使用

```bash
# 模拟运行（不调用模型 API，验证主链路）
python run_eval.py --task tasks/content_pipeline/task.yaml --mock-run

# 真实评测（复制 .env.example 为 .env 并填写 API Key）
python run_eval.py --task tasks/content_pipeline/task.yaml --runs 3 --ci
```

## 目录结构

```
harness/      Agent 运行循环与沙箱
strategies/   策略插件（Function Calling / ReAct）
metrics/      规则检查、LLM 评分、聚合
dataset/      Trace 与 Golden Set 回归数据
report/       评测报告与 Trace 输出
tasks/        评测任务定义
tests/        测试
```
