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
        from multitool.llm_client import GroqClient

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
        # Full 32-hex uuid (gemini-call- prefix is 12 chars + 32 hex = 44)
        assert len(result.tool_calls[0].id) == len("gemini-call-") + 32
        assert result.tool_calls[0].name == "tavily_search"
        assert result.tool_calls[0].arguments == {"query": "Chicago"}  # already dict

    def test_returns_content_when_model_answers_directly(self, mocker):
        """Parallel to Groq's content-path test."""
        from multitool.llm_client import GeminiClient

        mock_resp = mocker.MagicMock()
        mock_resp.status_code = 200
        mock_resp.url = "https://example.com/foo?key=secret"  # exercises the scrub
        mock_resp.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{"text": "Hello from Gemini"}]
                }
            }]
        }
        mocker.patch("multitool.llm_client.requests.post", return_value=mock_resp)

        client = GeminiClient(api_key="dummy")
        result = client.chat_with_tools(messages=[], tools=[])
        assert result.content == "Hello from Gemini"
        assert result.tool_calls == []

    def test_arguments_always_dict_for_gemini(self, mocker):
        """Parallel to Groq's arguments-as-dict test. Gemini already gives dict."""
        from multitool.llm_client import GeminiClient

        mock_resp = mocker.MagicMock()
        mock_resp.status_code = 200
        mock_resp.url = "https://example.com/foo"
        mock_resp.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{
                        "functionCall": {
                            "name": "calculator",
                            "args": {"expression": "2+2", "verbose": True},
                        }
                    }]
                }
            }]
        }
        mocker.patch("multitool.llm_client.requests.post", return_value=mock_resp)

        client = GeminiClient(api_key="dummy")
        result = client.chat_with_tools(messages=[], tools=[])
        assert isinstance(result.tool_calls[0].arguments, dict)
        assert result.tool_calls[0].arguments == {"expression": "2+2", "verbose": True}


class TestChatWithToolsContractGuards:
    """C1 + I1 fixes from code-quality review."""

    def test_groq_4xx_raises_immediately_no_retry(self, mocker):
        """C1: bad auth (401) shouldn't burn 5 retries × ~62s of backoff."""
        from multitool.llm_client import GroqClient

        mock_resp = mocker.MagicMock()
        mock_resp.status_code = 401  # Unauthorized
        mock_resp.raise_for_status.side_effect = __import__("requests").HTTPError(
            "401 Client Error", response=mock_resp
        )
        post_mock = mocker.patch("multitool.llm_client.requests.post", return_value=mock_resp)

        client = GroqClient(api_key="bad_key")
        with pytest.raises(Exception):
            client.chat_with_tools(messages=[], tools=[])

        # Critical: only ONE attempt, not 5
        assert post_mock.call_count == 1

    def test_gemini_4xx_raises_immediately_no_retry(self, mocker):
        """C1 for Gemini: 400 malformed function schema shouldn't burn 5 retries."""
        from multitool.llm_client import GeminiClient

        mock_resp = mocker.MagicMock()
        mock_resp.status_code = 400  # Bad Request
        mock_resp.url = "https://example.com/foo"
        mock_resp.raise_for_status.side_effect = __import__("requests").HTTPError(
            "400 Client Error", response=mock_resp
        )
        post_mock = mocker.patch("multitool.llm_client.requests.post", return_value=mock_resp)

        client = GeminiClient(api_key="dummy")
        with pytest.raises(Exception):
            client.chat_with_tools(messages=[], tools=[])

        assert post_mock.call_count == 1

    def test_groq_reserved_kwarg_raises_value_error(self):
        """I1: caller cannot override 'tools' / 'messages' / 'model' via kwargs."""
        from multitool.llm_client import GroqClient

        client = GroqClient(api_key="dummy")
        with pytest.raises(ValueError, match="reserved kwargs"):
            client.chat_with_tools(messages=[], tools=[], model="other-model")

    def test_gemini_reserved_kwarg_raises_value_error(self):
        """Symmetric with Groq guard — passing `model` via kwargs is the
        realistic case (caller splats a config dict)."""
        from multitool.llm_client import GeminiClient

        client = GeminiClient(api_key="dummy")
        # Simulate someone splatting a config dict that has `model` in it
        bad_config = {"model": "gemini-1.5-flash"}
        with pytest.raises(ValueError, match="reserved kwargs"):
            client.chat_with_tools(messages=[], tools=[], **bad_config)
