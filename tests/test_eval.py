"""Tests for the eval scorer + JSONL loader."""
import pytest


class TestParseNumber:

    def test_simple_float(self):
        from multitool.eval.scorer import _parse_number
        assert _parse_number("32.64") == 32.64

    def test_negative(self):
        from multitool.eval.scorer import _parse_number
        assert _parse_number("-12.5") == -12.5

    def test_with_commas(self):
        from multitool.eval.scorer import _parse_number
        assert _parse_number("2,664,452") == 2664452.0

    def test_picks_first_when_ambiguous(self):
        """spec §3.8: first-float-in-string heuristic."""
        from multitool.eval.scorer import _parse_number
        assert _parse_number("About 32.64 (or 33 depending on source)") == 32.64

    def test_returns_none_on_no_number(self):
        from multitool.eval.scorer import _parse_number
        assert _parse_number("I could not find this information.") is None

    def test_scientific_notation(self):
        from multitool.eval.scorer import _parse_number
        assert _parse_number("1.5e9 people") == 1.5e9


class TestScore:

    def test_numeric_within_tolerance(self):
        from multitool.eval.scorer import score
        r = score(predicted="32.64", gold=32.6, kind="numeric", tolerance=0.5)
        assert r["passed"] is True

    def test_numeric_outside_tolerance(self):
        from multitool.eval.scorer import score
        r = score(predicted="40", gold=32.6, kind="numeric", tolerance=0.5)
        assert r["passed"] is False

    def test_numeric_unparseable(self):
        from multitool.eval.scorer import score
        r = score(predicted="I don't know", gold=32.6, kind="numeric", tolerance=0.5)
        assert r["passed"] is False
        assert r["parse_error"] == "no_number_found"

    def test_string_substring_match(self):
        from multitool.eval.scorer import score
        r = score(predicted="The author is from Norway (Jon Fosse).", gold="Norway", kind="string")
        assert r["passed"] is True

    def test_string_case_insensitive(self):
        from multitool.eval.scorer import score
        r = score(predicted="thursday", gold="Thursday", kind="string")
        assert r["passed"] is True
