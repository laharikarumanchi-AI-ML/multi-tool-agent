"""Unit conversion tool using pint.

Pint's UnitRegistry loads ~1000 unit definitions on construction (~50ms),
so we use the same lazy + cached pattern as tavily_search and wikipedia:
module-level _ureg, _get_ureg() initializes on first call.
Errors returned as Observation strings.
"""
from . import tool

_ureg = None


def _get_ureg():
    """Lazy-initialize the pint UnitRegistry. The `import pint` is deferred
    to first call so pytest collection works without pint installed."""
    global _ureg
    if _ureg is None:
        import pint
        _ureg = pint.UnitRegistry()
    return _ureg


@tool
def unit_convert(value: float, from_unit: str, to_unit: str) -> str:
    """Convert a value from one unit to another.

    Accepts most physical units pint knows: meter, mile, kilometer, second,
    minute, hour, kilogram, pound, celsius, fahrenheit, and combinations
    like meter/second, kilogram*meter/second**2.

    Examples:
      unit_convert(60, "mile/hour", "meter/second") -> "26.82..."
      unit_convert(100, "kilometer", "mile")        -> "62.14..."
      unit_convert(212, "fahrenheit", "celsius")    -> "100.0..."
    """
    try:
        ureg = _get_ureg()
        quantity = value * ureg(from_unit)
        converted = quantity.to(to_unit)
        return str(converted.magnitude)
    except Exception as e:
        return f"Unit convert error: {type(e).__name__}: {e}"
