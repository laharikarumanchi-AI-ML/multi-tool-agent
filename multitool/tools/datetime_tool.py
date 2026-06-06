"""Date arithmetic tool (years between, add years, day of week).

Stdlib-only — no external client. Errors returned as Observation strings
(same convention as calculator), not raised exceptions.
"""
import re
from datetime import datetime
from . import tool


# Strict year matcher: exactly 4 digits, must NOT have leading/trailing junk
# beyond an optional ISO date suffix. "1976" OK; "1976-07-04" OK; "19" REJECTED.
_YEAR_OR_ISO_RE = re.compile(r"^(\d{4})(?:-\d{2}-\d{2})?$")


def _parse_year(value: str) -> int:
    """Parse a year from "YYYY" or "YYYY-MM-DD". Raises ValueError on garbage."""
    m = _YEAR_OR_ISO_RE.match(value.strip())
    if not m:
        raise ValueError(
            f"expected year as 'YYYY' or 'YYYY-MM-DD', got {value!r}"
        )
    return int(m.group(1))


@tool
def datetime_tool(
    operation: str,
    date_or_year: str,
    extra: str | None = None,
) -> str:
    """Date arithmetic. The 'extra' parameter's meaning depends on 'operation':
    for years_between it's a second year/date; for add_years it's an integer
    count; for day_of_week it's ignored.

    Operations:
      - "years_between": years from `date_or_year` to `extra` (both as "YYYY" or "YYYY-MM-DD").
        Inputs must be valid years; "19" or "abc1976" raise.
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
            start = _parse_year(date_or_year)
            end = _parse_year(extra)
            return f"{end - start} years"
        if operation == "add_years":
            if extra is None:
                return "Error: add_years requires extra=year count"
            year = _parse_year(date_or_year)
            # extra must be a small integer count, not a year/date — validate explicitly.
            try:
                count = int(extra)
            except ValueError:
                raise ValueError(f"add_years expects extra as integer count, got {extra!r}")
            return str(year + count)
        if operation == "day_of_week":
            d = datetime.strptime(date_or_year, "%Y-%m-%d")
            return d.strftime("%A")
        return f"Error: unknown operation {operation!r}. Valid: years_between, add_years, day_of_week."
    except Exception as e:
        return f"Datetime error: {type(e).__name__}: {e}"
