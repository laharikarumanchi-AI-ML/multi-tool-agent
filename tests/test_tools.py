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
