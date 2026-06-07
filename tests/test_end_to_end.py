"""End-to-end agent loop with a scripted mock LLM. No real API calls."""
import pytest


class TestEndToEnd:

    def test_two_step_search_then_calc(self, tmp_path, mocker):
        """Mock LLM returns: search-call → search-result-in-prompt → calc-call → calc-result-in-prompt → final answer."""
        from multitool.orchestrator import Orchestrator
        from multitool.llm_client import ToolResponse, ToolCall
        from multitool.tools import TOOL_REGISTRY
        from multitool.trace import Trace

        # Stub two tools
        TOOL_REGISTRY["search_fake"] = {
            "fn": lambda q: f"Population of {q.split()[0]}: 1000000",
            "schema": {},
        }
        TOOL_REGISTRY["calc_fake"] = {
            "fn": lambda e: str(eval(e)),  # ok in test; not in production
            "schema": {},
        }

        # Script the LLM responses
        responses = [
            ToolResponse(content=None, tool_calls=[
                ToolCall(id="c1", name="search_fake", arguments={"q": "Chicago population"})
            ]),
            ToolResponse(content=None, tool_calls=[
                ToolCall(id="c2", name="calc_fake", arguments={"e": "1000000 / 50000"})
            ]),
            ToolResponse(content="The answer is 20.0", tool_calls=[]),
        ]
        mock_llm = mocker.MagicMock()
        mock_llm.chat_with_tools.side_effect = responses

        trace = Trace(directory=str(tmp_path), question="Q", provider="g", model="m")
        orch = Orchestrator(llm=mock_llm, trace=trace)
        result = orch.run("How many people per $50k of GDP in Chicago?")

        assert result.answer == "The answer is 20.0"
        assert result.steps_taken == 3
        assert len(result.tool_calls) == 2  # 2 tool calls, 1 final answer = 3 steps
