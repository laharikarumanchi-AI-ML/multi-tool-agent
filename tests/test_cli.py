"""Tests for the CLI entrypoint (``multitool ask "<question>"``).

These tests never make real network calls — the ``_run_agent`` seam
is monkeypatched in every test that exercises the orchestrator path.
The argparse-error tests (missing question, unknown subcommand) don't
even reach ``_run_agent``, so they need no mock.
"""
import pytest

from multitool import cli
from multitool.orchestrator import AgentResult


def _fake_result(
    answer: str = "42",
    steps_taken: int = 1,
    tool_calls: list[dict] | None = None,
    error: str | None = None,
    trace_path: str = "traces/abc123.json",
) -> AgentResult:
    """Construct a real AgentResult with sensible defaults — using the
    actual dataclass (not a MagicMock) keeps tests honest about the shape
    of what the orchestrator returns."""
    return AgentResult(
        answer=answer,
        steps_taken=steps_taken,
        tool_calls=tool_calls if tool_calls is not None else [],
        error=error,
        trace_path=trace_path,
    )


class TestCliArgparseErrors:
    """Tests for cases where argparse rejects input before we touch the
    agent. These should always exit with code 2 (argparse's convention)."""

    def test_ask_missing_question_exits_with_error(self):
        """`multitool ask` with no question is an argparse usage error."""
        with pytest.raises(SystemExit) as exc_info:
            cli.main(["ask"])
        assert exc_info.value.code == 2

    def test_unknown_subcommand_shows_help(self, capsys):
        """`multitool bogus` is an argparse usage error → exit 2, usage on stderr."""
        with pytest.raises(SystemExit) as exc_info:
            cli.main(["bogus"])
        assert exc_info.value.code == 2
        captured = capsys.readouterr()
        # argparse prints usage to stderr on parse errors
        assert "usage" in captured.err.lower() or "invalid choice" in captured.err.lower()


class TestCliRuntimeErrors:
    """Tests for cases where the parser succeeds but execution fails —
    these should exit with code 1 (NOT 2) and print a friendly message."""

    def test_ask_missing_api_key_exits_with_friendly_error(
        self, monkeypatch, capsys
    ):
        """Missing GROQ_API_KEY → friendly stderr message + exit 1,
        NOT a raw KeyError traceback."""
        monkeypatch.delenv("GROQ_API_KEY", raising=False)

        rc = cli.main(["ask", "what is 2+2?"])

        assert rc == 1
        captured = capsys.readouterr()
        # Stderr should mention the missing var by name and be actionable.
        assert "GROQ_API_KEY" in captured.err
        # Verify it's a FRIENDLY message — no traceback markers should leak.
        assert "Traceback" not in captured.err
        assert "KeyError" not in captured.err

    def test_orchestrator_exception_prints_friendly_error(
        self, monkeypatch, capsys
    ):
        """If _run_agent raises (e.g. network failure), the user sees a
        clean stderr message + exit 1 — not a traceback (unless --verbose)."""
        def boom(*args, **kwargs):
            raise ConnectionError("groq is down")

        monkeypatch.setattr(cli, "_run_agent", boom)

        rc = cli.main(["ask", "anything"])

        assert rc == 1
        captured = capsys.readouterr()
        assert "ConnectionError" in captured.err
        assert "groq is down" in captured.err
        assert "Traceback" not in captured.err

    def test_verbose_flag_propagates_orchestrator_exception(
        self, monkeypatch
    ):
        """With --verbose, the original exception must be re-raised so
        the user sees the full traceback for debugging."""
        def boom(*args, **kwargs):
            raise ConnectionError("groq is down")

        monkeypatch.setattr(cli, "_run_agent", boom)

        with pytest.raises(ConnectionError, match="groq is down"):
            cli.main(["ask", "anything", "--verbose"])


class TestCliHappyPath:
    """Tests covering the success path: orchestrator returns a final
    answer; CLI prints it to stdout and exits 0."""

    def test_ask_calls_orchestrator_and_prints_answer(
        self, monkeypatch, capsys
    ):
        """Stdout should contain ONLY the answer (no decoration)."""
        monkeypatch.setattr(
            cli, "_run_agent",
            lambda **kwargs: _fake_result(answer="42"),
        )

        rc = cli.main(["ask", "what is the meaning of life?"])

        assert rc == 0
        captured = capsys.readouterr()
        assert "42" in captured.out
        # Bare stdout — no [trace: ...] decoration unless --verbose
        assert "trace" not in captured.out.lower()
        assert "steps" not in captured.out.lower()

    def test_ask_forwards_args_to_run_agent(self, monkeypatch):
        """CLI flags should reach _run_agent unmangled — covers wiring of
        --provider, --model, --trace."""
        captured_kwargs: dict = {}

        def capture(**kwargs):
            captured_kwargs.update(kwargs)
            return _fake_result(answer="ok")

        monkeypatch.setattr(cli, "_run_agent", capture)

        rc = cli.main([
            "ask", "hello",
            "--provider", "gemini",
            "--model", "gemini-pro",
            "--trace", "/tmp/run.json",
        ])

        assert rc == 0
        assert captured_kwargs["question"] == "hello"
        assert captured_kwargs["provider"] == "gemini"
        assert captured_kwargs["model"] == "gemini-pro"
        assert captured_kwargs["trace_path"] == "/tmp/run.json"


class TestCliVerboseTrace:
    """Tests covering --verbose tool-call logging."""

    def test_verbose_flag_prints_tool_calls(self, monkeypatch, capsys):
        """`-v` should print each tool invocation to STDERR
        (stderr keeps stdout clean for piping the answer)."""
        result = _fake_result(
            answer="5",
            tool_calls=[
                {"name": "calculator", "args": {"expression": "2+3"}, "result": "5"},
            ],
        )
        monkeypatch.setattr(cli, "_run_agent", lambda **kwargs: result)

        rc = cli.main(["ask", "what is 2+3?", "-v"])

        assert rc == 0
        captured = capsys.readouterr()
        # Answer on stdout
        assert "5" in captured.out
        # Tool call name on stderr
        assert "calculator" in captured.err
        assert "2+3" in captured.err

    def test_verbose_truncates_long_tool_results(self, monkeypatch, capsys):
        """A 5000-char Wikipedia summary shouldn't flood the terminal."""
        long_result = "x" * 5000
        result = _fake_result(
            answer="ok",
            tool_calls=[
                {"name": "wikipedia", "args": {"topic": "Python"}, "result": long_result},
            ],
        )
        monkeypatch.setattr(cli, "_run_agent", lambda **kwargs: result)

        cli.main(["ask", "tell me about python", "-v"])

        captured = capsys.readouterr()
        # Truncation marker should appear; full 5000-char string should not.
        assert "chars]" in captured.err
        assert long_result not in captured.err


class TestCliMaxStepsReached:
    """Tests for the partial-answer-on-max-steps path."""

    def test_max_steps_prints_warning_to_stderr_and_exits_nonzero(
        self, monkeypatch, capsys
    ):
        """`max_steps_reached` is a soft failure: print any partial answer
        on stdout, warn on stderr, exit 1."""
        result = _fake_result(
            answer="best guess: 42",
            steps_taken=10,
            error="max_steps_reached",
        )
        monkeypatch.setattr(cli, "_run_agent", lambda **kwargs: result)

        rc = cli.main(["ask", "hard question"])

        assert rc == 1
        captured = capsys.readouterr()
        assert "best guess: 42" in captured.out  # partial answer on stdout
        assert "max steps" in captured.err.lower()  # warning on stderr
