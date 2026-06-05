"""Tool registry. Decorator-driven tool registration with auto-generated JSON Schema."""
import inspect
from typing import Callable, Union, get_type_hints, get_origin, get_args
from types import NoneType, UnionType

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
    """Map a Python type hint to a JSON Schema type string.
    Handles Optional[X] (= X | None) by stripping NoneType and recursing on X."""
    if get_origin(hint) in (Union, UnionType):
        args = tuple(a for a in get_args(hint) if a is not NoneType)
        if len(args) == 1:
            return _py_to_json_type(args[0])
        raise UnsupportedToolTypeError(f"Union of multiple non-None types: {hint}")
    if hint in PY_TO_JSON_TYPE:
        return PY_TO_JSON_TYPE[hint]
    raise UnsupportedToolTypeError(f"Unsupported tool parameter type: {hint!r}")
