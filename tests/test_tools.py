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
