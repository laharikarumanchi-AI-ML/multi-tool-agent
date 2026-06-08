"""Shared pytest fixtures for the multitool test suite.

Most fixtures here are autouse — they apply to every test in the suite
unless explicitly opted out.
"""
import pytest

# Force-import all 5 real tools at conftest load time so they're registered
# in TOOL_REGISTRY BEFORE the autouse isolation fixture takes its baseline
# snapshot. Otherwise the first test triggers the import, the fixture's
# "before" snapshot misses the new entries, and teardown wipes them out
# of the cached module — leaving subsequent tests with an empty registry.
import multitool.tools.search       # noqa: F401  registers tavily_search
import multitool.tools.calculator   # noqa: F401  registers calculator
import multitool.tools.datetime_tool # noqa: F401 registers datetime_tool
import multitool.tools.unit_convert # noqa: F401  registers unit_convert
import multitool.tools.wikipedia    # noqa: F401  registers wikipedia


@pytest.fixture(autouse=True)
def _isolate_tool_registry():
    """Snapshot TOOL_REGISTRY's full state before each test, restore it after.

    Why this matters: orchestrator + end-to-end tests register fake tools
    ('fake_tool', 'echo', 'broken', 'search_fake', 'calc_fake') as
    TOOL_REGISTRY-level side effects. Some tests (TestToolDecorator) also
    `.clear()` the registry in their setup_method. Without isolation, the 5
    real tools registered at conftest load time would either get clobbered
    by setup_method, or leak fakes across to other tests, depending on order.

    Snapshot + full restore handles both: any mutation a test makes is
    undone at teardown, restoring exactly the state we had at conftest
    import (5 real tools, no fakes).
    """
    from multitool.tools import TOOL_REGISTRY
    before = dict(TOOL_REGISTRY)  # shallow copy of {name: {fn, schema}}
    yield
    TOOL_REGISTRY.clear()
    TOOL_REGISTRY.update(before)


@pytest.fixture(autouse=True)
def _stub_load_project_env(monkeypatch):
    """Stub multitool._env.load_project_env to a no-op for the entire test suite.

    cli.main() and eval/run.main() call load_project_env() at entry to
    auto-load .env from the project root. During tests this would side-load
    real env vars (e.g. a developer's GROQ_API_KEY in repo-root .env) and
    sabotage tests that intentionally delenv() those vars to exercise the
    "missing API key" code paths. Stub it globally so tests get a clean
    environment regardless of what's in the repo's .env.
    """
    from multitool import _env
    monkeypatch.setattr(_env, "load_project_env", lambda: None)
