"""Scorer for eval results.

_parse_number: first-float-in-string heuristic. Spec §3.8.
"""
import re
from typing import Any

# Matches first float-or-int in a string. Handles negatives, commas,
# scientific notation. Doesn't match numbers embedded in words.
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z])-?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?"
)


def _parse_number(text: str) -> float | None:
    """Extract the first numeric value from prose. Returns None if no match."""
    m = _NUMBER_RE.search(text)
    if m is None:
        return None
    try:
        return float(m.group().replace(",", ""))
    except ValueError:
        return None


def score(
    predicted: str,
    gold: Any,
    kind: str,
    tolerance: float | None = None,
) -> dict:
    """Score a prediction. Returns dict with at least `passed: bool`.

    kind:
      - "numeric": parse first number, compare |parsed - gold| <= tolerance
      - "string": case-insensitive substring match (gold in predicted)
      - "list":   set comparison
    """
    if kind == "numeric":
        parsed = _parse_number(predicted)
        if parsed is None:
            return {"passed": False, "predicted": predicted, "parse_error": "no_number_found"}
        return {"passed": abs(parsed - gold) <= tolerance, "predicted": parsed}
    if kind == "string":
        return {"passed": gold.lower() in predicted.lower(), "predicted": predicted}
    if kind == "list":
        return {"passed": set(predicted) == set(gold), "predicted": predicted}
    raise ValueError(f"Unknown answer_kind: {kind!r}")
