"""Date arithmetic tool (years between, add years, day of week).

Stdlib-only — no external client. Errors returned as Observation strings
(same convention as calculator), not raised exceptions.
"""
from datetime import datetime
from . import tool


@tool
def datetime_tool(
    operation: str,
    date_or_year: str,
    extra: str | None = None,
) -> str:
    """Date arithmetic.

    Operations:
      - "years_between": years from `date_or_year` to `extra` (both as year strings or YYYY-MM-DD).
      - "add_years": `date_or_year` (a year string) + `extra` (an integer year count).
      - "day_of_week": weekday name for `date_or_year` (YYYY-MM-DD).

    Examples:
      datetime_tool("years_between", "1976", "2007") -> "31 years"
      datetime_tool("add_years", "2024", "5")        -> "2029"
      datetime_tool("day_of_week", "2024-07-04")     -> "Thursday"
    """
    try:
        if operation == "years_between":
            if extra is None:
                return "Error: years_between requires extra=second date"
            start = int(date_or_year[:4])
            end = int(extra[:4])
            return f"{end - start} years"
        if operation == "add_years":
            if extra is None:
                return "Error: add_years requires extra=year count"
            year = int(date_or_year[:4])
            return str(year + int(extra))
        if operation == "day_of_week":
            d = datetime.strptime(date_or_year, "%Y-%m-%d")
            return d.strftime("%A")
        return f"Error: unknown operation {operation!r}. Valid: years_between, add_years, day_of_week."
    except Exception as e:
        return f"Datetime error: {type(e).__name__}: {e}"
