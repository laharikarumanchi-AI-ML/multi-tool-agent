"""Tests for the eval scorer + JSONL loader."""


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

    def test_parses_unformatted_4plus_digit_numbers(self):
        """Regression: pre-fix regex truncated '10000' to '100'."""
        from multitool.eval.scorer import _parse_number
        assert _parse_number("10000") == 10000.0
        assert _parse_number("8336817") == 8336817.0
        assert _parse_number("population is 1234567") == 1234567.0


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

    def test_numeric_tolerance_none_means_exact_match(self):
        """Regression: tolerance=None previously crashed with TypeError."""
        from multitool.eval.scorer import score
        result = score(predicted="5", gold=5.0, kind="numeric", tolerance=None)
        assert result["passed"] is True
        result2 = score(predicted="5.1", gold=5.0, kind="numeric", tolerance=None)
        assert result2["passed"] is False


class TestRunner:

    def test_load_test_set(self, tmp_path):
        from multitool.eval.run import load_test_set

        path = tmp_path / "test_set.jsonl"
        path.write_text(
            '{"id":"q01","question":"Q1","gold_answer":1.0,"tolerance":0.1,"answer_kind":"numeric","expected_tools":[],"category":"x","difficulty":"easy"}\n'
            '{"id":"q02","question":"Q2","gold_answer":"hello","tolerance":null,"answer_kind":"string","expected_tools":[],"category":"y","difficulty":"easy"}\n'
        )
        queries = load_test_set(str(path))
        assert len(queries) == 2
        assert queries[0]["id"] == "q01"
        assert queries[1]["answer_kind"] == "string"


class TestRunEval:

    def test_empty_queries_does_not_crash(self, tmp_path):
        """Regression: empty queries previously hit ZeroDivisionError."""
        from multitool.eval.run import run_eval
        results_path = tmp_path / "results.json"

        def factory():
            raise AssertionError("factory should not be called for empty queries")

        summary = run_eval([], factory, str(results_path))
        assert summary["total_attempted"] == 0
        assert summary["total_passed"] == 0
        assert summary["pass_rate"] == 0.0
