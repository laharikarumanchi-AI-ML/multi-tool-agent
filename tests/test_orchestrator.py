"""Tests for the orchestrator agent loop."""
import pytest


class TestOrchestratorSkeleton:

    def test_construct(self, tmp_path, mocker):
        from multitool.orchestrator import Orchestrator
        from multitool.trace import Trace
        mock_llm = mocker.MagicMock()
        trace = Trace(directory=str(tmp_path), question="Q", provider="groq", model="m")
        orch = Orchestrator(llm=mock_llm, trace=trace)
        assert orch.llm is mock_llm
        assert orch.trace is trace

    def test_agent_result_dataclass(self):
        from multitool.orchestrator import AgentResult
        r = AgentResult(answer="42", steps_taken=2, tool_calls=[], error=None, trace_path="/tmp/x.json")
        assert r.answer == "42"
        assert r.steps_taken == 2
        assert r.error is None


class TestDispatchWithRetry:

    def test_successful_call_returns_result(self, tmp_path, mocker):
        from multitool.orchestrator import Orchestrator
        from multitool.llm_client import ToolCall
        from multitool.trace import Trace
        from multitool.tools import TOOL_REGISTRY

        TOOL_REGISTRY["fake_tool"] = {"fn": lambda x: f"OK:{x}", "schema": {}}
        trace = Trace(directory=str(tmp_path), question="Q", provider="g", model="m")
        orch = Orchestrator(llm=mocker.MagicMock(), trace=trace)

        call = ToolCall(id="1", name="fake_tool", arguments={"x": "hello"})
        result = orch._dispatch_with_retry(call)
        assert result == "OK:hello"

    def test_failing_call_retries_then_returns_error(self, tmp_path, mocker):
        from multitool.orchestrator import Orchestrator
        from multitool.llm_client import ToolCall
        from multitool.trace import Trace
        from multitool.tools import TOOL_REGISTRY

        # Always-failing tool
        TOOL_REGISTRY["always_fails"] = {
            "fn": lambda: (_ for _ in ()).throw(RuntimeError("boom")),
            "schema": {},
        }
        trace = Trace(directory=str(tmp_path), question="Q", provider="g", model="m")
        orch = Orchestrator(llm=mocker.MagicMock(), trace=trace)

        call = ToolCall(id="1", name="always_fails", arguments={})
        result = orch._dispatch_with_retry(call)
        assert "Tool error" in result
        assert "boom" in result
