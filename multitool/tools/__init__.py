"""Tool registry. Decorator-driven tool registration with auto-generated JSON Schema."""
import inspect
from typing import Callable, Union, get_type_hints, get_origin, get_args
from types import NoneType

TOOL_REGISTRY: dict[str, dict] = {}

# Closed set of supported types. Anything else raises UnsupportedToolTypeError.
PY_TO_JSON_TYPE: dict[type, str] = {
    str:   "string",
    int:   "integer",
    float: "number",
    bool:  "boolean",
}


class UnsupportedToolTypeError(TypeError):
    """A tool's parameter has a type the decorator cannot map to JSON Schema."""


def _py_to_json_type(hint) -> str:
    """Map a Python type hint to a JSON Schema type string."""
    if hint in PY_TO_JSON_TYPE:
        return PY_TO_JSON_TYPE[hint]
    raise UnsupportedToolTypeError(f"Unsupported tool parameter type: {hint!r}")
