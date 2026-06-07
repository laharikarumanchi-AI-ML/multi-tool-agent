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


import json


class TestGroqChatWithTools:

    def test_returns_tool_calls_when_model_picks_a_tool(self, mocker):
        from multitool.llm_client import GroqClient, ToolResponse

        # Mock requests.post to return a Groq-shaped tool_calls response
        mock_resp = mocker.MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "tavily_search",
                            "arguments": json.dumps({"query": "Chicago population"}),
                        },
                    }],
                }
            }]
        }
        mocker.patch("multitool.llm_client.requests.post", return_value=mock_resp)

        client = GroqClient(api_key="dummy")
        result = client.chat_with_tools(
            messages=[{"role": "user", "content": "What's Chicago's population?"}],
            tools=[{"type": "function", "function": {"name": "tavily_search", "parameters": {}}}],
        )
        assert isinstance(result, ToolResponse)
        assert result.content is None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_abc"
        assert result.tool_calls[0].name == "tavily_search"
        assert result.tool_calls[0].arguments == {"query": "Chicago population"}  # parsed dict!

    def test_returns_content_when_model_answers_directly(self, mocker):
        from multitool.llm_client import GroqClient, ToolResponse

        mock_resp = mocker.MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{
                "message": {"role": "assistant", "content": "Hello", "tool_calls": None}
            }]
        }
        mocker.patch("multitool.llm_client.requests.post", return_value=mock_resp)

        client = GroqClient(api_key="dummy")
        result = client.chat_with_tools(
            messages=[{"role": "user", "content": "Say hello"}],
            tools=[],
        )
        assert result.content == "Hello"
        assert result.tool_calls == []

    def test_arguments_always_parsed_to_dict(self, mocker):
        """Critical contract: arguments is dict, never str."""
        from multitool.llm_client import GroqClient

        mock_resp = mocker.MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{
                "message": {
                    "content": None,
                    "tool_calls": [{
                        "id": "x",
                        "function": {"name": "f", "arguments": json.dumps({"a": 1, "b": "two"})},
                    }],
                }
            }]
        }
        mocker.patch("multitool.llm_client.requests.post", return_value=mock_resp)

        client = GroqClient(api_key="dummy")
        result = client.chat_with_tools(messages=[], tools=[])
        assert isinstance(result.tool_calls[0].arguments, dict)
        assert result.tool_calls[0].arguments == {"a": 1, "b": "two"}


class TestGeminiChatWithTools:

    def test_synthesizes_call_id(self, mocker):
        """Gemini doesn't issue call IDs; client must synthesize them."""
        from multitool.llm_client import GeminiClient

        mock_resp = mocker.MagicMock()
        mock_resp.status_code = 200
        mock_resp.url = "https://example.com/foo"  # so the scrub line doesn't error
        mock_resp.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{
                        "functionCall": {
                            "name": "tavily_search",
                            "args": {"query": "Chicago"},
                        }
                    }]
                }
            }]
        }
        mocker.patch("multitool.llm_client.requests.post", return_value=mock_resp)

        client = GeminiClient(api_key="dummy")
        result = client.chat_with_tools(messages=[], tools=[])
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id.startswith("gemini-call-")
        assert result.tool_calls[0].name == "tavily_search"
        assert result.tool_calls[0].arguments == {"query": "Chicago"}  # already dict
