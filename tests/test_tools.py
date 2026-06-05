"""Tests for the @tool decorator + TOOL_REGISTRY."""
import pytest


class TestPyToJsonType:
    """Maps Python type hints to JSON Schema type strings."""

    def test_str_maps_to_string(self):
        from multitool.tools import _py_to_json_type
        assert _py_to_json_type(str) == "string"

    def test_int_maps_to_integer(self):
        from multitool.tools import _py_to_json_type
        assert _py_to_json_type(int) == "integer"

    def test_float_maps_to_number(self):
        from multitool.tools import _py_to_json_type
        assert _py_to_json_type(float) == "number"

    def test_bool_maps_to_boolean(self):
        from multitool.tools import _py_to_json_type
        assert _py_to_json_type(bool) == "boolean"

    def test_unsupported_type_raises(self):
        from multitool.tools import _py_to_json_type, UnsupportedToolTypeError
        with pytest.raises(UnsupportedToolTypeError):
            _py_to_json_type(list)


class TestPyToJsonTypeOptional:
    """Optional[X] / X | None unwraps to inner type."""

    def test_optional_str(self):
        from multitool.tools import _py_to_json_type
        assert _py_to_json_type(str | None) == "string"

    def test_optional_int(self):
        from multitool.tools import _py_to_json_type
        assert _py_to_json_type(int | None) == "integer"

    def test_optional_float(self):
        from multitool.tools import _py_to_json_type
        assert _py_to_json_type(float | None) == "number"

    def test_union_of_multiple_non_none_types_raises(self):
        from multitool.tools import _py_to_json_type, UnsupportedToolTypeError
        with pytest.raises(UnsupportedToolTypeError):
            _py_to_json_type(str | int)  # not Optional — multiple non-None


class TestToolDecorator:
    """The @tool decorator registers functions and builds JSON Schema."""

    def setup_method(self):
        """Clear TOOL_REGISTRY before each test (decorator side-effects)."""
        from multitool.tools import TOOL_REGISTRY
        TOOL_REGISTRY.clear()

    def test_decorator_registers_function(self):
        from multitool.tools import tool, TOOL_REGISTRY

        @tool
        def example_tool(query: str) -> str:
            """Example tool docstring."""
            return f"echo: {query}"

        assert "example_tool" in TOOL_REGISTRY
        assert TOOL_REGISTRY["example_tool"]["fn"] is example_tool

    def test_schema_has_correct_shape(self):
        from multitool.tools import tool, TOOL_REGISTRY

        @tool
        def example(query: str) -> str:
            """Example docstring used as description."""
            return query

        schema = TOOL_REGISTRY["example"]["schema"]
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "example"
        assert schema["function"]["description"] == "Example docstring used as description."
        assert schema["function"]["parameters"]["type"] == "object"
        assert schema["function"]["parameters"]["properties"] == {"query": {"type": "string"}}
        assert schema["function"]["parameters"]["required"] == ["query"]

    def test_multiple_required_params(self):
        from multitool.tools import tool, TOOL_REGISTRY

        @tool
        def add(a: int, b: int) -> int:
            """Add two integers."""
            return a + b

        params = TOOL_REGISTRY["add"]["schema"]["function"]["parameters"]
        assert params["properties"] == {
            "a": {"type": "integer"},
            "b": {"type": "integer"},
        }
        assert set(params["required"]) == {"a", "b"}


class TestToolDecoratorBehaviors:

    def setup_method(self):
        from multitool.tools import TOOL_REGISTRY
        TOOL_REGISTRY.clear()

    def test_decorator_requires_docstring(self):
        from multitool.tools import tool

        with pytest.raises(ValueError, match="docstring required"):
            @tool
            def no_doc(query: str) -> str:
                return query

    def test_optional_param_not_required(self):
        from multitool.tools import tool, TOOL_REGISTRY

        @tool
        def search(query: str, limit: int = 5) -> str:
            """Search with optional limit."""
            return query

        params = TOOL_REGISTRY["search"]["schema"]["function"]["parameters"]
        assert params["required"] == ["query"]  # limit excluded
        assert "limit" in params["properties"]  # but still in properties

    def test_optional_str_schema(self):
        from multitool.tools import tool, TOOL_REGISTRY

        @tool
        def event(name: str, when: str | None = None) -> str:
            """Optional when parameter."""
            return f"{name} at {when}"

        params = TOOL_REGISTRY["event"]["schema"]["function"]["parameters"]
        assert params["properties"]["when"] == {"type": "string"}
        assert params["required"] == ["name"]


class TestTavilySearch:
    """tavily_search tool. Real API calls are @pytest.mark.slow; defaults are mocked."""

    @pytest.fixture(autouse=True)
    def _reset_client(self):
        """Reset the cached Tavily client before AND after every test in this
        class. Without the post-cleanup, a successful real-API test (in the
        sibling TestTavilySearchReal class) leaves a real TavilyClient in the
        cache; a subsequent test that forgets to patch would silently hit
        the network. PR #4's tool test classes should follow this pattern."""
        import multitool.tools.search as search_mod
        search_mod._client = None
        yield
        search_mod._client = None

    def test_returns_formatted_string(self, mocker):
        from multitool.tools.search import tavily_search

        mock_client = mocker.MagicMock()
        mock_client.search.return_value = {
            "answer": "Chicago's population in 2023 was 2,664,452.",
            "results": [
                {"url": "https://example.com/chi", "title": "Chicago demographics", "content": "Population in 2023..."},
                {"url": "https://example.com/chi2", "title": "Census", "content": "Per Census Bureau..."},
            ],
        }
        mocker.patch("multitool.tools.search._get_client", return_value=mock_client)

        result = tavily_search("Chicago population 2023")
        assert "Chicago's population in 2023 was 2,664,452" in result
        assert "https://example.com/chi" in result

    def test_handles_no_results(self, mocker):
        from multitool.tools.search import tavily_search

        mock_client = mocker.MagicMock()
        mock_client.search.return_value = {"answer": None, "results": []}
        mocker.patch("multitool.tools.search._get_client", return_value=mock_client)

        result = tavily_search("query with no results")
        assert "No results" in result or "no results" in result

    def test_raises_on_missing_api_key(self, mocker, monkeypatch):
        from multitool.tools.search import tavily_search

        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        # Force re-create the cached client (so it re-reads env)
        import multitool.tools.search as search_mod
        search_mod._client = None

        with pytest.raises(RuntimeError, match="TAVILY_API_KEY"):
            tavily_search("anything")


class TestTavilySearchReal:
    """Real-API smoke test. Run with: pytest -m slow"""

    @pytest.fixture(autouse=True)
    def _reset_client(self):
        """Same cleanup fixture as TestTavilySearch — ensures the real client
        doesn't leak into subsequent tests in the session."""
        import multitool.tools.search as search_mod
        search_mod._client = None
        yield
        search_mod._client = None

    @pytest.mark.slow
    def test_real_search_returns_something(self):
        import os
        if not os.environ.get("TAVILY_API_KEY"):
            pytest.skip("TAVILY_API_KEY not set; skipping real-API test")

        from multitool.tools.search import tavily_search
        result = tavily_search("What is the capital of France?")
        assert "Paris" in result


class TestCalculator:

    def test_simple_arithmetic(self):
        from multitool.tools.calculator import calculator
        assert calculator("2 + 2") == "4"

    def test_floating_point(self):
        from multitool.tools.calculator import calculator
        result = float(calculator("2664452 / 81632"))
        assert abs(result - 32.64) < 0.01

    def test_handles_syntax_error(self):
        from multitool.tools.calculator import calculator
        result = calculator("2 +")
        assert "error" in result.lower() or "invalid" in result.lower()

    def test_rejects_unsafe_expressions(self):
        """numexpr doesn't execute arbitrary Python — verify this."""
        from multitool.tools.calculator import calculator
        result = calculator("__import__('os').system('ls')")
        assert "error" in result.lower()


class TestDatetimeTool:

    def test_years_between(self):
        from multitool.tools.datetime_tool import datetime_tool
        # Apple founded 1976, iPhone launched 2007
        result = datetime_tool("years_between", "1976", "2007")
        assert "31" in result

    def test_add_years(self):
        from multitool.tools.datetime_tool import datetime_tool
        result = datetime_tool("add_years", "2024", "5")
        assert "2029" in result

    def test_day_of_week(self):
        from multitool.tools.datetime_tool import datetime_tool
        # 2024-07-04 was a Thursday
        result = datetime_tool("day_of_week", "2024-07-04")
        assert "Thursday" in result

    def test_unknown_operation(self):
        from multitool.tools.datetime_tool import datetime_tool
        result = datetime_tool("invalid_op", "2024")
        assert "error" in result.lower() or "unknown" in result.lower()
