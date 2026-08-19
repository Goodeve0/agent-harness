"""
CLI judge（Claude Code / Codex）双模型交叉评测测试

覆盖：
  · 两种后端命令构造（claude -p / codex exec）
  · stdout 解析：JSON / 分数表达式 / 百分比 / 0-1 数字
  · pattern 模式 YES/NO 判定
  · fail-open：CLI 缺失 / 超时 / 非零退出 / 无法解析 → -1.0
  · probe_cli 探测逻辑
  · 与 CrossValidator 集成仲裁（双侧有效 / 单侧异常不拉低分）
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import patch

from metrics.cli_judge import CLIJudge, parse_cli_score, probe_cli
from metrics.cross_validator import CrossValidator


# ─── 命令构造 ─────────────────────────────────────────────────────────────────

class _FakeProc:
    """模拟 subprocess.CompletedProcess"""
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def _captured_cmd(mock_run):
    """返回最近一次 subprocess.run 的 ([argv], kwargs)"""
    args, kwargs = mock_run.call_args
    return args[0], kwargs


def test_claude_command_and_prompt_contains_rubric():
    judge = CLIJudge(backend="claude")
    with patch("metrics.cli_judge.subprocess.run", return_value=_FakeProc(stdout="0.85")) as m:
        score = judge.judge(text="输出内容", rubric="标准：清晰", reference="参考答案")
    argv, kwargs = _captured_cmd(m)
    assert argv[0] == "claude" and argv[1] == "-p"
    prompt = argv[2]
    assert "输出内容" in prompt and "标准：清晰" in prompt and "参考答案" in prompt
    assert kwargs["timeout"] == 180
    assert score == pytest.approx(0.85)


def test_codex_command():
    judge = CLIJudge(backend="codex")
    with patch("metrics.cli_judge.subprocess.run", return_value=_FakeProc(stdout="0.9")) as m:
        judge.judge(text="t", rubric="r")
    argv, _ = _captured_cmd(m)
    assert argv[0] == "codex" and argv[1] == "exec"


# ─── 输出解析 ─────────────────────────────────────────────────────────────────

def test_parse_json_score():
    assert parse_cli_score('{"score": 0.85, "reason": "good"}') == pytest.approx(0.85)


def test_parse_json_score_string():
    assert parse_cli_score('{"score": "0.7"}') == pytest.approx(0.7)


def test_parse_json_nested_result():
    # claude --output-format json 形态：{"type":"result","result":"0.82"}
    assert parse_cli_score('{"type": "result", "result": "0.82"}') == pytest.approx(0.82)


def test_parse_fraction():
    assert parse_cli_score("评分：8/10") == pytest.approx(0.8)


def test_parse_percent():
    assert parse_cli_score("综合得分 85%") == pytest.approx(0.85)


def test_parse_plain_number_last():
    assert parse_cli_score("内容不错 0.75") == pytest.approx(0.75)


def test_parse_empty_and_garbage():
    assert parse_cli_score("") == -1.0
    assert parse_cli_score("无法判断") == -1.0
    assert parse_cli_score("出错了 2 个维度") == -1.0  # 只有 >1 的数字，无 0-1 值


def test_judge_pattern_yes_no():
    judge = CLIJudge(backend="claude", mode="pattern")
    with patch("metrics.cli_judge.subprocess.run", return_value=_FakeProc(stdout="YES")) as m:
        assert judge.judge(text="t", rubric="r") == 1.0
    with patch("metrics.cli_judge.subprocess.run", return_value=_FakeProc(stdout="NO")) as m:
        assert judge.judge(text="t", rubric="r") == 0.0
    with patch("metrics.cli_judge.subprocess.run", return_value=_FakeProc(stdout="不清楚")) as m:
        assert judge.judge(text="t", rubric="r") == -1.0


# ─── fail-open：异常 → -1.0，绝不伪造分数 ────────────────────────────────────

def test_cli_missing_returns_minus_one():
    judge = CLIJudge(backend="claude")
    with patch("metrics.cli_judge.subprocess.run", side_effect=FileNotFoundError):
        assert judge.judge(text="t", rubric="r") == -1.0
    assert "未安装" in (judge.last_error or "")


def test_cli_timeout_returns_minus_one():
    import subprocess
    judge = CLIJudge(backend="codex", timeout=30)
    with patch("metrics.cli_judge.subprocess.run",
               side_effect=subprocess.TimeoutExpired(cmd="codex", timeout=30)):
        assert judge.judge(text="t", rubric="r") == -1.0
    assert "超时" in (judge.last_error or "")


def test_cli_nonzero_exit_returns_minus_one():
    judge = CLIJudge(backend="claude")
    with patch("metrics.cli_judge.subprocess.run",
               return_value=_FakeProc(stdout="", stderr="auth failed", returncode=1)):
        assert judge.judge(text="t", rubric="r") == -1.0
    assert "退出码 1" in (judge.last_error or "")


def test_cli_unparseable_returns_minus_one():
    judge = CLIJudge(backend="claude")
    with patch("metrics.cli_judge.subprocess.run", return_value=_FakeProc(stdout="天书")):
        assert judge.judge(text="t", rubric="r") == -1.0
    assert "无法解析" in (judge.last_error or "")


# ─── 可用性探测 ───────────────────────────────────────────────────────────────

def test_probe_cli_installed():
    with patch("metrics.cli_judge.shutil.which", return_value="/usr/local/bin/claude"), \
         patch("metrics.cli_judge.subprocess.run", return_value=_FakeProc(stdout="1.0.10\n")):
        ok, info = probe_cli("claude")
    assert ok and "claude" in info


def test_probe_cli_missing():
    with patch("metrics.cli_judge.shutil.which", return_value=None):
        ok, info = probe_cli("codex")
    assert not ok and "未找到" in info


# ─── CrossValidator 集成仲裁 ─────────────────────────────────────────────────

def test_cross_validator_claude_x_codex_agreed_mean():
    judge_a = CLIJudge(backend="claude")
    judge_b = CLIJudge(backend="codex")
    cv = CrossValidator(judge_a=judge_a, judge_b=judge_b, threshold=0.6)
    with patch("metrics.cli_judge.subprocess.run",
               side_effect=[_FakeProc(stdout="0.85"), _FakeProc(stdout="0.95")]):
        rec = cv.validate("s1", text="t", rubric="r")
    assert rec.valid
    assert rec.agreed                      # 两侧都判 pass
    assert rec.merged_score == pytest.approx(0.9)   # 一致取均值
    rpt = cv.report()
    assert rpt["agreement_rate"] == pytest.approx(1.0)
    assert rpt["model_a"] == "claude-cli"
    assert rpt["model_b"] == "codex-cli"


def test_cross_validator_single_side_fail_uses_valid_side():
    judge_a = CLIJudge(backend="claude")
    judge_b = CLIJudge(backend="codex")
    cv = CrossValidator(judge_a=judge_a, judge_b=judge_b, threshold=0.6)
    # B 侧 CLI 崩溃（FileNotFoundError）→ -1，不拉低仲裁分
    with patch("metrics.cli_judge.subprocess.run",
               side_effect=[_FakeProc(stdout="0.8"), FileNotFoundError]):
        rec = cv.validate("s2", text="t", rubric="r")
    assert not rec.valid
    assert rec.score_a == 0.8 and rec.score_b == -1.0
    assert rec.merged_score == pytest.approx(0.8)   # 只采信有效侧
    assert rec.abs_diff == 0.0


def test_cross_validator_both_sides_fail_zero():
    judge_a = CLIJudge(backend="claude")
    judge_b = CLIJudge(backend="codex")
    cv = CrossValidator(judge_a=judge_a, judge_b=judge_b)
    with patch("metrics.cli_judge.subprocess.run",
               side_effect=[FileNotFoundError, FileNotFoundError]):
        rec = cv.validate("s3", text="t", rubric="r")
    assert rec.merged_score == 0.0
    assert cv.report()["total"] == 0   # 无效记录不进一致率统计
