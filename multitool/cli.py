"""Command-line entrypoint: ``multitool ask "<question>"``.

Subcommand-style argparse (no Click) so future commands (``multitool eval``,
``multitool demo``) can be added cleanly. Imports all 5 tool modules at the
top of ``_run_agent`` so their ``@tool`` decorators register into
``TOOL_REGISTRY`` before the orchestrator is constructed — mirrors the
pattern in ``multitool/eval/run.py``.

Error-handling philosophy (project-wide):
- Missing required env var → friendly stderr + exit 1 (NOT a stack trace).
- Orchestrator raises → friendly stderr + exit 1; the raw traceback is only
  re-raised when ``--verbose`` is set, so library bugs are debuggable.
- Orchestrator returns ``error="max_steps_reached"`` → print warning to
  stderr, exit 1 (orchestrator does not surface partial state).
- argparse errors (missing question, unknown subcommand) bubble up as
  ``SystemExit(2)`` — argparse's own default — so shells see the standard
  "usage error" exit code.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional


_TRUNCATE_RESULT_AT = 200  # chars; verbose tool-call log shouldn't dump full results


def _resolve_trace_dir(trace_path: Optional[str]) -> str:
    """Resolve and ensure the trace directory exists.

    ``--trace PATH`` is ALWAYS treated as a directory. If the user passes
    ``--trace mytrace``, traces land in ``mytrace/``, not CWD. This avoids
    the foot-gun where ``os.path.dirname("mytrace")`` returns ``""`` and
    traces silently drop into the current working directory.

    The directory is created if missing (both for the user-supplied path
    and the default ``traces/``)."""
    trace_dir = trace_path if trace_path else "traces"
    os.makedirs(trace_dir, exist_ok=True)
    return trace_dir


def _run_agent(
    question: str,
    provider: str,
    model: Optional[str] = None,
    trace_path: Optional[str] = None,
):
    """Build orchestrator + run. Factored out so tests can monkeypatch
    a single seam instead of mocking the whole import graph."""
    from multitool.orchestrator import Orchestrator
    from multitool.llm_client import GroqClient, GeminiClient
    from multitool.trace import Trace

    # Trigger @tool decorator side effects so TOOL_REGISTRY is populated
    # before Orchestrator inspects it.
    import multitool.tools.search        # noqa: F401
    import multitool.tools.calculator    # noqa: F401
    import multitool.tools.datetime_tool # noqa: F401
    import multitool.tools.unit_convert  # noqa: F401
    import multitool.tools.wikipedia     # noqa: F401

    if provider == "groq":
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            # Caught by main() and turned into a friendly stderr message.
            raise _MissingEnvVar("GROQ_API_KEY")
        llm = GroqClient(api_key=api_key, model=model) if model else GroqClient(api_key=api_key)
    elif provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise _MissingEnvVar("GEMINI_API_KEY")
        llm = GeminiClient(api_key=api_key, model=model) if model else GeminiClient(api_key=api_key)
    else:
        # argparse `choices=` already prevents this, but belt-and-suspenders.
        raise ValueError(f"Unknown provider: {provider!r}")

    # Trace directory resolution: --trace PATH is always treated as a directory
    # (created if missing). See _resolve_trace_dir for the foot-gun this avoids.
    # The Trace class always derives its own run_id-based filename inside the
    # directory; --trace lets the user redirect WHERE traces land, not rename them.
    trace_dir = _resolve_trace_dir(trace_path)
    trace = Trace(
        directory=trace_dir,
        question=question,
        provider=provider,
        model=llm._model,
    )
    orch = Orchestrator(llm=llm, trace=trace)
    return orch.run(question)


class _MissingEnvVar(RuntimeError):
    """Sentinel for missing required env vars; caught + reformatted in main()."""
    def __init__(self, name: str):
        super().__init__(name)
        self.var_name = name


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="multitool",
        description="ReAct-style multi-tool agent. Function-calling-native; no LangChain.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True, metavar="<command>")

    ask = sub.add_parser(
        "ask",
        help='Ask the agent a question (e.g. multitool ask "what is 2+2?")',
        description="Run the agent on a single question and print the final answer to stdout.",
    )
    ask.add_argument("question", help="The question to ask the agent (wrap in quotes).")
    ask.add_argument(
        "--provider",
        default="groq",
        choices=["groq", "gemini"],
        help="LLM provider. Default: groq (requires GROQ_API_KEY).",
    )
    ask.add_argument(
        "--model",
        default=None,
        help="Override the default model for the chosen provider.",
    )
    ask.add_argument(
        "--trace",
        dest="trace_path",
        default=None,
        metavar="PATH",
        help="Directory to write trace JSON files into (one file per run, named "
             "<run_id>.json). Created if it doesn't exist. Default: traces/",
    )
    ask.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print each tool call (name, args, truncated result) to stderr. "
             "Also re-raises orchestrator exceptions with the full traceback.",
    )
    return parser


def _print_tool_calls(tool_calls: list[dict], stream=None) -> None:
    """Format the orchestrator's flat tool_calls log for --verbose output.
    Truncates each result so a giant Wikipedia summary doesn't flood the terminal.

    ``stream`` defaults to None (resolved to ``sys.stderr`` at call time, not
    import time — this matters for pytest's ``capsys``, which swaps
    ``sys.stderr`` per test)."""
    if stream is None:
        stream = sys.stderr
    for i, call in enumerate(tool_calls, start=1):
        name = call.get("name", "?")
        args = call.get("args", {})
        result = str(call.get("result", ""))
        if len(result) > _TRUNCATE_RESULT_AT:
            result = result[:_TRUNCATE_RESULT_AT] + f"... [+{len(result) - _TRUNCATE_RESULT_AT} chars]"
        print(f"[tool {i}] {name}({args}) -> {result}", file=stream)


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point for ``multitool`` console script.

    Returns the process exit code so tests can assert without needing to
    catch SystemExit. ``sys.exit()`` is only called via argparse's own
    error path (unknown subcommand, missing positional arg → exit code 2).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)  # SystemExit(2) on parse error

    if args.cmd == "ask":
        try:
            result = _run_agent(
                question=args.question,
                provider=args.provider,
                model=args.model,
                trace_path=args.trace_path,
            )
        except _MissingEnvVar as e:
            print(
                f"error: required environment variable {e.var_name} is not set.\n"
                f"hint: export {e.var_name}=<your-key> (or add to your .env).",
                file=sys.stderr,
            )
            return 1
        except Exception as e:
            if args.verbose:
                raise  # full traceback for debugging
            print(f"error: agent failed: {type(e).__name__}: {e}", file=sys.stderr)
            return 1

        if args.verbose and result.tool_calls:
            _print_tool_calls(result.tool_calls)

        if result.error == "max_steps_reached":
            # Orchestrator hard-codes answer=None on max_steps_reached, so there
            # is no partial state to surface — just warn on stderr and exit 1
            # so scripts can detect the truncation.
            print(
                f"warning: agent hit max steps ({result.steps_taken}) without finalizing. "
                f"Trace: {result.trace_path}",
                file=sys.stderr,
            )
            return 1
        if result.error:
            print(f"error: {result.error} (trace: {result.trace_path})", file=sys.stderr)
            return 1

        print(result.answer if result.answer is not None else "")
        if args.verbose:
            print(
                f"[done in {result.steps_taken} steps; trace: {result.trace_path}]",
                file=sys.stderr,
            )
        return 0

    # argparse's required=True on subparsers means we never reach this for
    # an unknown subcommand (it exits with code 2 first). Guard anyway.
    parser.print_help(sys.stderr)
    return 2
