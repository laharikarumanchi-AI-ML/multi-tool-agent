"""Smoke tests for the Streamlit demo (app.py).

Uses Streamlit's AppTest harness (1.28+) — runs the script in-process and
exposes the rendered widget tree as Python objects. No browser, no network.

These are intentionally smoke-level: they verify the page boots, the API-key
gate fires, the example chips render, and the submit button is wired through
to the orchestrator (mocked) without surfacing an exception. The Orchestrator
itself is covered exhaustively in tests/test_orchestrator.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

# Path to app.py at the repo root. AppTest resolves relative paths from CWD,
# which pytest runs from the repo root anyway, but be explicit so the tests
# pass regardless of where they're invoked from.
APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def no_api_keys(monkeypatch):
    """Strip all provider keys so we can test the missing-key gate."""
    for var in ("GROQ_API_KEY", "GEMINI_API_KEY", "TAVILY_API_KEY"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def stub_api_keys(monkeypatch):
    """Inject fake keys so the gate passes — useful for tests that don't
    actually call the orchestrator (or mock it)."""
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")


def test_app_loads_without_exception(no_api_keys):
    """app.py imports + initial render works without raising. This catches
    syntax errors, missing imports, and bad Streamlit API usage."""
    at = AppTest.from_file(APP_PATH)
    at.run()
    assert not at.exception, f"App raised: {[e.value for e in at.exception]}"


def test_missing_api_key_surfaces_friendly_error(no_api_keys):
    """When GROQ_API_KEY is absent, a user-facing st.error appears rather
    than a stack trace."""
    at = AppTest.from_file(APP_PATH)
    at.run()
    assert not at.exception
    error_texts = [e.value for e in at.error]
    assert any("GROQ_API_KEY" in e for e in error_texts), (
        f"Expected GROQ_API_KEY error message, got: {error_texts}"
    )


def test_example_chips_render(no_api_keys):
    """At least 5 example-question chips are rendered as clickable buttons."""
    at = AppTest.from_file(APP_PATH)
    at.run()
    assert not at.exception
    # Buttons whose key starts with "example_" are the chips. The submit
    # button has a different key ("Ask the agent" auto-key).
    example_buttons = [b for b in at.button if b.key and b.key.startswith("example_")]
    assert len(example_buttons) >= 5, (
        f"Expected at least 5 example chips, got {len(example_buttons)}"
    )
    assert len(example_buttons) <= 7, (
        f"Expected at most 7 example chips, got {len(example_buttons)}"
    )


def test_submit_disabled_without_question(stub_api_keys):
    """Submit button is disabled when the question text area is empty."""
    at = AppTest.from_file(APP_PATH)
    at.run()
    assert not at.exception
    # The primary "Ask the agent" button should be disabled — there's no
    # question yet and no API-key error blocking it.
    primary_buttons = [b for b in at.button if b.label == "Ask the agent"]
    assert len(primary_buttons) == 1
    assert primary_buttons[0].disabled is True


def test_clicking_example_chip_populates_question(no_api_keys):
    """Clicking an example chip writes the full question into session_state,
    which the text area then renders on the next run."""
    at = AppTest.from_file(APP_PATH)
    at.run()
    assert not at.exception
    example_buttons = [b for b in at.button if b.key and b.key.startswith("example_")]
    assert example_buttons, "no example chips found"
    # Click the first chip and re-run.
    example_buttons[0].click()
    at.run()
    assert not at.exception
    # session_state["question"] should now hold the chip's full question
    # (the help= attribute on each chip carries the full text).
    assert at.session_state["question"] == example_buttons[0].help


def test_submit_invokes_orchestrator(stub_api_keys, monkeypatch):
    """When the user enters a question and clicks submit, the orchestrator
    runs and its result is rendered. We monkeypatch Orchestrator.run + the
    LLM constructors so no network happens.

    AppTest re-executes the script source each .run(), so patches on the
    `app` module itself get wiped. Patching the upstream symbols in
    multitool.* sticks because those modules are import-cached."""
    from multitool import orchestrator as orch_mod
    from multitool import llm_client as llm_mod

    @dataclass
    class _FakeResult:
        answer: str = "**The answer is 42.**"
        steps_taken: int = 1
        tool_calls: list = None
        error: str = None
        trace_path: str = "/tmp/fake-trace.json"

        def __post_init__(self):
            if self.tool_calls is None:
                self.tool_calls = [
                    {"name": "calculator", "args": {"expression": "6*7"}, "result": "42"}
                ]

    class _FakeLLM:
        _model = "fake-model"

        def __init__(self, *args, **kwargs):
            pass

    # Patch the LLM clients so no real HTTP calls go out, and the
    # Orchestrator.run so we get a deterministic answer back.
    monkeypatch.setattr(llm_mod, "GroqClient", _FakeLLM)
    monkeypatch.setattr(llm_mod, "GeminiClient", _FakeLLM)
    # app.py imports these names into its module namespace at import time;
    # re-execution by AppTest grabs them again from llm_client, so the patch
    # above propagates. But to be safe in case the import order changed,
    # patch the orchestrator method directly.
    monkeypatch.setattr(orch_mod.Orchestrator, "run", lambda self, q: _FakeResult())

    at = AppTest.from_file(APP_PATH)
    at.run()
    assert not at.exception

    # Type into the text area; calling .run() re-executes the script with
    # the new session_state.
    at.text_area(key="question").set_value("What is 6 times 7?")
    at.run()
    assert not at.exception

    primary_buttons = [b for b in at.button if b.label == "Ask the agent"]
    assert primary_buttons, "Submit button missing"
    assert not primary_buttons[0].disabled, "Submit button still disabled with API key + question set"
    primary_buttons[0].click()
    at.run()
    assert not at.exception

    # The fake answer should appear in the rendered markdown.
    rendered_md = "\n".join(m.value for m in at.markdown)
    assert "42" in rendered_md, f"answer not rendered; markdown: {rendered_md!r}"
