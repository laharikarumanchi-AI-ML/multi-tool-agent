"""Calculator tool using numexpr (safe math expression evaluator).

Error-wrapping convention: this tool returns error STRINGS for invalid
expressions (instead of raising). The orchestrator treats the string as
an Observation; the model decides whether to retry with adjusted args.
This is per-tool variation from the `raise raw` convention used by tools
that wrap external clients — for stateless math evaluation, an error
string is the right Observation for the model.
"""
from . import tool


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression. Supports + - * / ** %, parentheses, and
    common functions (sin, cos, log, exp, sqrt, abs). Returns the numeric
    result as a string, or an error message.

    Examples:
      calculator("2664452 / 81632") -> "32.638..."
      calculator("sqrt(144)")        -> "12.0"
      calculator("2 ** 10")          -> "1024"
    """
    try:
        import numexpr
        result = numexpr.evaluate(expression)
        # numexpr returns ndarray; convert to Python scalar
        return str(result.item() if hasattr(result, "item") else result)
    except Exception as e:
        return f"Calculator error: {type(e).__name__}: {e}"
