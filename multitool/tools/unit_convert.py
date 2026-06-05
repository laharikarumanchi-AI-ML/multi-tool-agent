"""Unit conversion tool using pint.

Pint instantiates a UnitRegistry on first call; this is cheap so we
inline it inside the function. Stateless from the tool's perspective.
Errors returned as Observation strings.
"""
from . import tool


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
        import pint
        ureg = pint.UnitRegistry()
        quantity = value * ureg(from_unit)
        converted = quantity.to(to_unit)
        return str(converted.magnitude)
    except Exception as e:
        return f"Unit convert error: {type(e).__name__}: {e}"
