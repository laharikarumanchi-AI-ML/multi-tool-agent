"""Tests for the chat_with_tools() extension and ToolCall/ToolResponse types."""
import pytest


class TestToolDataclasses:

    def test_tool_call_has_fields(self):
        from multitool.llm_client import ToolCall

        c = ToolCall(id="abc", name="search", arguments={"query": "test"})
        assert c.id == "abc"
        assert c.name == "search"
        assert c.arguments == {"query": "test"}

    def test_tool_response_with_content(self):
        from multitool.llm_client import ToolResponse

        r = ToolResponse(content="Final answer", tool_calls=[])
        assert r.content == "Final answer"
        assert r.tool_calls == []

    def test_tool_response_with_tool_calls(self):
        from multitool.llm_client import ToolResponse, ToolCall

        r = ToolResponse(content=None, tool_calls=[
            ToolCall(id="1", name="search", arguments={"q": "x"})
        ])
        assert r.content is None
        assert len(r.tool_calls) == 1
