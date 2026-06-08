"""Tests for trace.py — JSON logger of agent runs."""
import json
import tempfile
from pathlib import Path


class TestTrace:

    def test_creates_file_with_metadata(self):
        from multitool.trace import Trace

        with tempfile.TemporaryDirectory() as tmpdir:
            t = Trace(directory=tmpdir, question="What is 2+2?", provider="groq", model="llama-3.3-70b-versatile")
            t.flush()
            assert Path(t.path).exists()
            data = json.loads(Path(t.path).read_text())
            assert data["question"] == "What is 2+2?"
            assert data["provider"] == "groq"
            assert data["model"] == "llama-3.3-70b-versatile"
            assert "started_at" in data
            assert "run_id" in data

    def test_log_step(self):
        from multitool.trace import Trace
        from multitool.llm_client import ToolCall

        with tempfile.TemporaryDirectory() as tmpdir:
            t = Trace(directory=tmpdir, question="Q", provider="groq", model="m")
            call = ToolCall(id="1", name="calculator", arguments={"expression": "2+2"})
            t.log_step(step=0, call=call, result="4")
            t.flush()
            data = json.loads(Path(t.path).read_text())
            assert len(data["steps"]) == 1
            assert data["steps"][0]["step"] == 0
            assert data["steps"][0]["tool_calls"][0]["name"] == "calculator"
            assert data["steps"][0]["results"][0] == "4"

    def test_log_final(self):
        from multitool.trace import Trace

        with tempfile.TemporaryDirectory() as tmpdir:
            t = Trace(directory=tmpdir, question="Q", provider="groq", model="m")
            t.log_final(step=1, final_answer="The answer is 4.")
            t.flush()
            data = json.loads(Path(t.path).read_text())
            assert data["final_answer"] == "The answer is 4."
            assert data["total_steps"] == 2  # 1 zero-indexed step + 1 (final)
