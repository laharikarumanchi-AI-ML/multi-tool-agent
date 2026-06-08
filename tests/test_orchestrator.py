"""Tests for the orchestrator agent loop."""


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


class TestOrchestratorRun:

    def test_returns_final_answer_when_model_responds_with_content(self, tmp_path, mocker):
        """If first LLM call returns content (no tool_calls), that's the final answer."""
        from multitool.orchestrator import Orchestrator
        from multitool.llm_client import ToolResponse
        from multitool.trace import Trace

        mock_llm = mocker.MagicMock()
        mock_llm.chat_with_tools.return_value = ToolResponse(content="2+2=4", tool_calls=[])

        trace = Trace(directory=str(tmp_path), question="What is 2+2?", provider="g", model="m")
        orch = Orchestrator(llm=mock_llm, trace=trace)
        result = orch.run("What is 2+2?")

        assert result.answer == "2+2=4"
        assert result.steps_taken == 1
        assert result.error is None

    def test_max_steps_reached_returns_error(self, tmp_path, mocker):
        """If model keeps calling tools and never answers, exit with max_steps_reached."""
        from multitool.orchestrator import Orchestrator
        from multitool.llm_client import ToolResponse, ToolCall
        from multitool.tools import TOOL_REGISTRY
        from multitool.trace import Trace

        TOOL_REGISTRY["echo"] = {"fn": lambda x: f"echoed:{x}", "schema": {}}

        mock_llm = mocker.MagicMock()
        # Always return a tool_call, never content
        mock_llm.chat_with_tools.return_value = ToolResponse(
            content=None,
            tool_calls=[ToolCall(id="1", name="echo", arguments={"x": "loop"})],
        )

        trace = Trace(directory=str(tmp_path), question="Q", provider="g", model="m")
        orch = Orchestrator(llm=mock_llm, trace=trace)
        result = orch.run("Q")

        assert result.answer is None
        assert result.error == "max_steps_reached"
        assert result.steps_taken == Orchestrator.MAX_STEPS


class TestRetryBudgetAndStepCeiling:
    """Verify the spec §3.6 'repeated-failed-call loop' decision: per-dispatch
    retry resets each step; step ceiling is the backstop."""

    def test_per_dispatch_retry_budget_resets_each_step(self, tmp_path, mocker):
        """Each step starts with a fresh MAX_TOOL_RETRIES budget."""
        from multitool.orchestrator import Orchestrator
        from multitool.llm_client import ToolResponse, ToolCall
        from multitool.tools import TOOL_REGISTRY
        from multitool.trace import Trace

        call_count = {"n": 0}
        def counting_fn():
            call_count["n"] += 1
            raise RuntimeError("always fails")
        TOOL_REGISTRY["fail_tool"] = {"fn": counting_fn, "schema": {}}

        mock_llm = mocker.MagicMock()
        mock_llm.chat_with_tools.return_value = ToolResponse(
            content=None,
            tool_calls=[ToolCall(id="1", name="fail_tool", arguments={})],
        )

        trace = Trace(directory=str(tmp_path), question="Q", provider="g", model="m")
        orch = Orchestrator(llm=mock_llm, trace=trace)
        orch.run("Q")
        # 10 steps × 3 attempts per dispatch = 30 fn invocations
        assert call_count["n"] == Orchestrator.MAX_STEPS * (Orchestrator.MAX_TOOL_RETRIES + 1)

    def test_repeated_same_call_terminates_at_step_ceiling(self, tmp_path, mocker):
        """Model can loop on same (name, args) all day; we just exit cleanly at MAX_STEPS."""
        from multitool.orchestrator import Orchestrator
        from multitool.llm_client import ToolResponse, ToolCall
        from multitool.tools import TOOL_REGISTRY
        from multitool.trace import Trace

        TOOL_REGISTRY["broken"] = {
            "fn": lambda: (_ for _ in ()).throw(RuntimeError("nope")),
            "schema": {},
        }

        mock_llm = mocker.MagicMock()
        mock_llm.chat_with_tools.return_value = ToolResponse(
            content=None,
            tool_calls=[ToolCall(id="1", name="broken", arguments={})],
        )

        trace = Trace(directory=str(tmp_path), question="Q", provider="g", model="m")
        orch = Orchestrator(llm=mock_llm, trace=trace)
        result = orch.run("Q")
        assert result.error == "max_steps_reached"
        assert result.steps_taken == Orchestrator.MAX_STEPS


class TestTerminationSemantics:
    """I-1 + M-7 fixes from code-quality review."""

    def test_empty_content_returns_distinct_error_not_max_steps(self, tmp_path, mocker):
        """I-1: a model returning content=None + tool_calls=[] should NOT
        burn through 10 steps and report max_steps_reached. The eval's
        failure-mode breakdown needs 'empty_response' to be distinguishable."""
        from multitool.orchestrator import Orchestrator
        from multitool.llm_client import ToolResponse
        from multitool.trace import Trace

        mock_llm = mocker.MagicMock()
        mock_llm.chat_with_tools.return_value = ToolResponse(
            content=None,
            tool_calls=[],
        )

        trace = Trace(directory=str(tmp_path), question="Q", provider="g", model="m")
        orch = Orchestrator(llm=mock_llm, trace=trace)
        result = orch.run("Q")

        assert result.error == "empty_response"
        assert result.error != "max_steps_reached"
        assert result.steps_taken == 1  # Exit immediately, not after 10
        # Critical: only ONE LLM call, not 10
        assert mock_llm.chat_with_tools.call_count == 1

    def test_empty_string_content_treated_as_final_answer(self, tmp_path, mocker):
        """I-1 corollary: content="" is a real (if unhelpful) response — return
        it as the final answer. The eval scorer will mark it wrong, but the
        failure mode is 'empty answer' not 'agent failed to terminate'."""
        from multitool.orchestrator import Orchestrator
        from multitool.llm_client import ToolResponse
        from multitool.trace import Trace

        mock_llm = mocker.MagicMock()
        mock_llm.chat_with_tools.return_value = ToolResponse(
            content="",
            tool_calls=[],
        )

        trace = Trace(directory=str(tmp_path), question="Q", provider="g", model="m")
        orch = Orchestrator(llm=mock_llm, trace=trace)
        result = orch.run("Q")

        assert result.answer == ""
        assert result.error is None
        assert result.steps_taken == 1

    def test_multiple_tool_calls_in_one_step(self, tmp_path, mocker):
        """M-7: model returns 2 tool_calls in a single response.
        Verify all dispatch correctly + trace groups them under same step."""
        from multitool.orchestrator import Orchestrator
        from multitool.llm_client import ToolResponse, ToolCall
        from multitool.tools import TOOL_REGISTRY
        from multitool.trace import Trace
        import json
        from pathlib import Path

        TOOL_REGISTRY["echo_a"] = {"fn": lambda x: f"A:{x}", "schema": {}}
        TOOL_REGISTRY["echo_b"] = {"fn": lambda x: f"B:{x}", "schema": {}}

        # Step 0: 2 parallel tool calls. Step 1: final answer.
        responses = [
            ToolResponse(content=None, tool_calls=[
                ToolCall(id="t1", name="echo_a", arguments={"x": "alpha"}),
                ToolCall(id="t2", name="echo_b", arguments={"x": "beta"}),
            ]),
            ToolResponse(content="Done", tool_calls=[]),
        ]
        mock_llm = mocker.MagicMock()
        mock_llm.chat_with_tools.side_effect = responses

        trace = Trace(directory=str(tmp_path), question="Q", provider="g", model="m")
        orch = Orchestrator(llm=mock_llm, trace=trace)
        result = orch.run("Q")

        assert result.answer == "Done"
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["result"] == "A:alpha"
        assert result.tool_calls[1]["result"] == "B:beta"

        # Trace: both calls should be grouped under step 0, not split across steps
        data = json.loads(Path(trace.path).read_text())
        step_0 = data["steps"][0]
        assert step_0["step"] == 0
        assert len(step_0["tool_calls"]) == 2
        assert len(step_0["results"]) == 2

    def test_assistant_message_omits_content_when_none(self, tmp_path, mocker):
        """Regression: when the model returns tool_calls with content=None,
        we omit the `content` field rather than sending `content: null`.

        This mirrors what the OpenAI Python SDK emits for the same case
        and is the canonical OpenAI-compatible shape. Some OpenAI-compatible
        providers have been observed to reject `content: null` paired with
        tool_calls with an opaque HTTP 400; omitting the field sidesteps
        that entire class of provider-specific edge cases.
        """
        from multitool.orchestrator import Orchestrator
        from multitool.llm_client import ToolResponse, ToolCall
        from multitool.tools import TOOL_REGISTRY
        from multitool.trace import Trace

        TOOL_REGISTRY["echo"] = {"fn": lambda x: f"echoed:{x}", "schema": {}}

        captured_messages: list = []

        def fake_chat_with_tools(messages, tools, **kwargs):
            # Snapshot the messages on the SECOND call (after the tool result
            # has been appended), where the assistant-with-tool_calls message
            # is what previously broke Groq.
            captured_messages.append([dict(m) for m in messages])
            if len(captured_messages) == 1:
                return ToolResponse(
                    content=None,
                    tool_calls=[ToolCall(id="t1", name="echo", arguments={"x": "hi"})],
                )
            return ToolResponse(content="done", tool_calls=[])

        mock_llm = mocker.MagicMock()
        mock_llm.chat_with_tools.side_effect = fake_chat_with_tools

        trace = Trace(directory=str(tmp_path), question="Q", provider="g", model="m")
        orch = Orchestrator(llm=mock_llm, trace=trace)
        orch.run("Q")

        # Second call's messages list must contain the assistant message,
        # and that message MUST NOT have a `content` key set to None.
        second_call_messages = captured_messages[1]
        assistant_msgs = [m for m in second_call_messages if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        am = assistant_msgs[0]
        assert "tool_calls" in am
        # Either `content` is absent OR it's a non-None string. Sending
        # `content: null` is what triggered the 400.
        if "content" in am:
            assert am["content"] is not None, (
                "assistant message must not send `content: null` when tool_calls "
                "are present — Groq rejects with HTTP 400"
            )

    def test_assistant_message_keeps_content_when_string(self, tmp_path, mocker):
        """If the model returns BOTH content and tool_calls (rare but legal
        per the OpenAI spec), we must preserve the content string."""
        from multitool.orchestrator import Orchestrator
        from multitool.llm_client import ToolResponse, ToolCall
        from multitool.tools import TOOL_REGISTRY
        from multitool.trace import Trace

        TOOL_REGISTRY["echo2"] = {"fn": lambda x: f"echoed:{x}", "schema": {}}
        captured: list = []

        def fake(messages, tools, **kwargs):
            captured.append([dict(m) for m in messages])
            if len(captured) == 1:
                return ToolResponse(
                    content="thinking...",
                    tool_calls=[ToolCall(id="t1", name="echo2", arguments={"x": "y"})],
                )
            return ToolResponse(content="final", tool_calls=[])

        mock_llm = mocker.MagicMock()
        mock_llm.chat_with_tools.side_effect = fake

        trace = Trace(directory=str(tmp_path), question="Q", provider="g", model="m")
        orch = Orchestrator(llm=mock_llm, trace=trace)
        orch.run("Q")

        am = [m for m in captured[1] if m["role"] == "assistant"][0]
        assert am.get("content") == "thinking..."

    def test_hallucinated_tool_fast_paths_to_error(self, tmp_path, mocker):
        """M-5: model picks a tool name not in TOOL_REGISTRY.
        Should return error string without burning 3 KeyError attempts."""
        from multitool.orchestrator import Orchestrator
        from multitool.llm_client import ToolCall
        from multitool.trace import Trace

        trace = Trace(directory=str(tmp_path), question="Q", provider="g", model="m")
        orch = Orchestrator(llm=mocker.MagicMock(), trace=trace)

        call = ToolCall(id="x", name="this_tool_does_not_exist", arguments={})
        result = orch._dispatch_with_retry(call)
        assert "unknown tool" in result
        assert "this_tool_does_not_exist" in result
