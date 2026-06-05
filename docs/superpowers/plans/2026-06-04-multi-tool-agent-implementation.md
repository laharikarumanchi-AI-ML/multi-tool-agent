# Multi-Tool AI Agent — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a from-scratch ReAct-style autonomous agent (no LangChain) that uses Groq's function-calling API to dispatch 5 tools (Tavily search, calculator, datetime, unit conversion, Wikipedia) with multi-step reasoning and per-tool retry. Evaluated on a hand-curated 25-query test set.

**Architecture:** Python 3.11 package `multitool/` with `tools/` (decorator-registered functions), `llm_client.py` (copied from DA Agent + new `chat_with_tools()`), `orchestrator.py` (~250-line function-calling loop), `eval/` (harness + 25 JSONL queries + scorer), and `demo/` (Streamlit). 10 PRs sequentially merged to `main` with CI pytest on each.

**Tech Stack:** Python 3.11, pytest, requests, numexpr, pint, wikipedia-api, tavily-python, Streamlit. No LangChain, no LlamaIndex, no agent framework.

**Spec:** [`docs/superpowers/specs/2026-06-04-multi-tool-agent-design.md`](../specs/2026-06-04-multi-tool-agent-design.md)

---

## A note on TDD here

Unlike the portfolio (Astro/CSS, no test culture), this is a real Python project with pytest from PR #1. TDD per task is real:

1. **Write the failing test** — assertion captures the behavior we want
2. **Run pytest to verify it fails** — output should be RED with a specific error
3. **Implement the minimum code to pass** — no over-engineering
4. **Run pytest to verify it passes** — output GREEN
5. **Commit** — one commit per task with a focused subject

Most tasks below follow this pattern. Where a task is pure config (scaffold, CI setup) the TDD steps are replaced with "create file" + "verify build/install."

For the orchestrator, `pytest-mock`'s `mocker` fixture is used to mock the `LLMClient` so tests don't hit real APIs. Real-API tests are marked `@pytest.mark.slow` and excluded from default CI.

---

## File structure (full inventory)

### New files (created across PRs)

| Path | PR | Responsibility |
|---|---|---|
| `pyproject.toml` | #1 | Package config: name, deps, dev/demo extras, pytest config |
| `requirements.txt` | #1 | HF Spaces deploy compatibility (no `-e .` per DA Agent's HF lesson) |
| `.gitignore` | #1 | Already exists; verify Python entries present |
| `.github/workflows/test.yml` | #1 | pytest CI on push + PR (mirror RAG project's workflow) |
| `.github/PULL_REQUEST_TEMPLATE.md` | #1 | PR description template (mirror RAG project) |
| `multitool/__init__.py` | #1 | Package marker; `__version__` |
| `multitool/__main__.py` | #1 | Forwarder to `cli.py` (`python -m multitool`) |
| `multitool/cli.py` | #1 (skeleton), #8 (real impl) | `multitool ask "<question>"` entrypoint |
| `multitool/llm_client.py` | #5 | Copied from DA Agent + `chat_with_tools()` + `ToolCall`/`ToolResponse` dataclasses |
| `multitool/orchestrator.py` | #6 | Function-calling agent loop |
| `multitool/trace.py` | #6 | JSON trace logger |
| `multitool/prompts/system.txt` | #6 | The system prompt for the agent |
| `multitool/tools/__init__.py` | #2 | `@tool` decorator, `TOOL_REGISTRY`, type mapping |
| `multitool/tools/search.py` | #3 | `tavily_search` |
| `multitool/tools/calculator.py` | #4 | `calculator` (numexpr) |
| `multitool/tools/datetime_tool.py` | #4 | `datetime_tool` (years_between, add_years, day_of_week) |
| `multitool/tools/unit_convert.py` | #4 | `unit_convert` (pint) |
| `multitool/tools/wikipedia.py` | #4 | `wikipedia` (wikipedia-api) |
| `multitool/eval/__init__.py` | #7 | empty marker |
| `multitool/eval/test_set.jsonl` | #7 | 25 hand-curated queries with gold answers |
| `multitool/eval/scorer.py` | #7 | `score()` + `_parse_number` |
| `multitool/eval/run.py` | #7 | Eval runner with checkpointing |
| `tests/__init__.py` | #1 | empty marker |
| `tests/test_tools.py` | #2, #3, #4 | `@tool` decorator + each tool's happy/error path |
| `tests/test_llm_client.py` | #5 | `chat_with_tools()` per-provider + parsing |
| `tests/test_orchestrator.py` | #6 | Loop logic, retry budget, termination |
| `tests/test_trace.py` | #6 | JSON serialization + key-scrub |
| `tests/test_end_to_end.py` | #6 | Full loop with scripted mock LLM |
| `tests/test_eval.py` | #7 | Loader + scorer correctness |
| `tests/conftest.py` | #1 | pytest fixtures (sample queries, mock LLM factory) |
| `demo/app.py` | #9 | Streamlit demo with trace UI |
| `README.md` | #10 | Project writeup with headline result, architecture, eval table |

### Modified files

None initially — this is a greenfield repo. The only currently-committed files are `.gitignore` and the spec doc.

---

## Branching strategy

```
main ──┬──── feat/scaffold              (PR #1) ──┐
       │                                            ├─► merge ─► main
       │     feat/tool-decorator         (PR #2) ──┤
       │     feat/tavily-search          (PR #3) ──┤
       │     feat/remaining-tools        (PR #4) ──┤
       │     feat/llm-client             (PR #5) ──┤
       │     feat/orchestrator           (PR #6) ──┤
       │     feat/eval                   (PR #7) ──┤
       │     feat/cli                    (PR #8) ──┤
       │     feat/demo                   (PR #9) ──┤
       │     docs/readme                 (PR #10) ─┘
       └────────────────────────────────────────────►
```

Non-stacked. Each branch is created off the latest `main` after the previous PR merges. PR-to-PR mergeability is checked at PR-open time, not maintained as a stacked chain.

---

## Pre-PR-1 user actions

**Required before Task 1.1:**

- [ ] **Sign up for Tavily API key** — <https://tavily.com> → register → copy free-tier key (1000 queries/month)
- [ ] **Verify Groq key still works** — should be in `/Users/anilkumar/Lahari/1.password.env` from DA Agent setup; reuse for this project
- [ ] **Verify Gemini key still works** — same source
- [ ] **Decide on local credentials file location** — recommendation: `/Users/anilkumar/multi-tool-agent/.env` (gitignored), containing:

```bash
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxxxxxxxxxx
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxx
GEMINI_API_KEY=AQ.xxxxxxxxxxxxxxxxxxxxxxxx
```

- [ ] **Create venv + activate** before any pytest commands:

```bash
cd /Users/anilkumar/multi-tool-agent
python3.11 -m venv .venv
source .venv/bin/activate
```

The implementer subagent will operate inside this venv. All `pip install` and `pytest` commands assume it's active.

---

# Phase 1 — PR #1: Scaffold + CI

**Branch:** `feat/scaffold`
**Goal:** Empty but installable Python package + pytest config + GHA CI workflow. First green CI run.
**Estimated LoC:** ~150
**Estimated time:** 30–45 min

### Task 1.1: Branch + create pyproject.toml

**Files:**
- Create: `pyproject.toml`

- [ ] **Step 1: Branch from main**

```bash
cd /Users/anilkumar/multi-tool-agent
git checkout main && git pull origin main
git checkout -b feat/scaffold
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[project]
name = "multitool"
version = "0.1.0"
description = "ReAct-style autonomous agent with function-calling-native tool dispatch — no LangChain."
requires-python = ">=3.11"
dependencies = [
    "requests>=2.31.0",
    "tavily-python>=0.5.0",
    "numexpr>=2.10.0",
    "pint>=0.24.0",
    "wikipedia-api>=0.7.0",
]

[project.optional-dependencies]
demo = ["streamlit>=1.31.0"]
dev = [
    "pytest>=8.0.0",
    "pytest-mock>=3.12.0",
]

[project.scripts]
multitool = "multitool.cli:main"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["."]
include = ["multitool*"]

[tool.setuptools.package-data]
"*" = ["prompts/*.txt"]

[tool.pytest.ini_options]
addopts = "-ra -m 'not slow'"
testpaths = ["tests"]
log_cli = false
markers = [
    "slow: tests that hit real APIs (run with: pytest -m slow)",
]
```

- [ ] **Step 3: Verify install works**

```bash
pip install -e ".[dev,demo]"
```

Expected: success, all 5 runtime deps + pytest/streamlit installed.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "$(cat <<'EOF'
feat(scaffold): pyproject.toml + dev/demo extras

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 1.2: Create package skeleton

**Files:**
- Create: `multitool/__init__.py`
- Create: `multitool/__main__.py`
- Create: `multitool/cli.py` (skeleton; real impl in PR #8)
- Create: `multitool/tools/__init__.py` (skeleton; real impl in PR #2)
- Create: `multitool/eval/__init__.py` (empty)
- Create: `multitool/prompts/.gitkeep` (placeholder for prompts/system.txt added in PR #6)
- Create: `tests/__init__.py` (empty)
- Create: `tests/conftest.py` (empty placeholder; fixtures added in later PRs)
- Create: `tests/test_smoke.py` (single test confirming import works)

- [ ] **Step 1: Create `multitool/__init__.py`**

```python
"""multitool — ReAct-style autonomous agent with function-calling-native tool dispatch."""
__version__ = "0.1.0"
```

- [ ] **Step 2: Create `multitool/__main__.py`**

```python
"""Allow `python -m multitool`."""
from multitool.cli import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create skeleton `multitool/cli.py`**

```python
"""Command-line entrypoint. Real impl in PR #8."""

def main() -> int:
    """Entrypoint stub. Replaced in PR #8 with arg parsing + orchestrator dispatch."""
    print("multitool: CLI not yet implemented (PR #8)")
    return 0
```

- [ ] **Step 4: Create skeleton `multitool/tools/__init__.py`**

```python
"""Tool registry. Real impl in PR #2."""
```

- [ ] **Step 5: Create empty markers**

```bash
mkdir -p multitool/eval multitool/prompts tests
touch multitool/eval/__init__.py
touch multitool/prompts/.gitkeep
touch tests/__init__.py
touch tests/conftest.py
```

- [ ] **Step 6: Create `tests/test_smoke.py`**

```python
"""Smoke test: package imports + skeleton CLI runs."""

def test_package_imports():
    import multitool
    assert multitool.__version__ == "0.1.0"

def test_cli_main_exits_zero():
    from multitool.cli import main
    assert main() == 0
```

- [ ] **Step 7: Run pytest — both tests should PASS**

```bash
pytest -v
```

Expected: 2 passed.

- [ ] **Step 8: Commit**

```bash
git add multitool tests
git commit -m "$(cat <<'EOF'
feat(scaffold): package skeleton + smoke test

multitool/{__init__,__main__,cli,tools/__init__,eval/__init__}.py
tests/{__init__,conftest,test_smoke}.py
multitool/prompts/.gitkeep

Smoke test verifies import + skeleton CLI exit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 1.3: Create requirements.txt for HF Spaces

**Files:**
- Create: `requirements.txt`

Per DA Agent's HF Spaces lesson: HF Spaces installs `requirements.txt` before mounting source, so `-e .` fails. Use sys.path bootstrap in `demo/app.py` (added in PR #9); for requirements just list the deps explicitly.

- [ ] **Step 1: Create `requirements.txt`**

```
# Requirements for Hugging Face Spaces deployment of the Streamlit demo.
# HF reads this file at build time to install dependencies into the Space.
# Local development uses pyproject.toml's [dev,demo] extras instead.
#
# Note: we deliberately do NOT install the multitool package via pip here.
# HF Spaces mounts requirements.txt at /tmp at install time, but the
# repo source isn't available yet at that step — so any `-e .` reference
# fails. Instead, demo/app.py adds the repo root to sys.path explicitly.

streamlit>=1.31.0
requests>=2.31.0
tavily-python>=0.5.0
numexpr>=2.10.0
pint>=0.24.0
wikipedia-api>=0.7.0
```

- [ ] **Step 2: Commit**

```bash
git add requirements.txt
git commit -m "$(cat <<'EOF'
feat(scaffold): requirements.txt for HF Spaces deploy

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 1.4: GitHub Actions CI

**Files:**
- Create: `.github/workflows/test.yml`
- Create: `.github/PULL_REQUEST_TEMPLATE.md`

- [ ] **Step 1: Create `.github/workflows/test.yml`**

```yaml
name: tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install package + dev + demo extras
        run: pip install -e ".[dev,demo]"

      - name: Run pytest
        run: pytest -v
```

- [ ] **Step 2: Create `.github/PULL_REQUEST_TEMPLATE.md`**

```markdown
## Summary

<!-- 1-3 sentences on what this PR does and why -->

## Test plan

- [ ] CI green (`pytest -v`)
- [ ] <feature-specific manual check 1>
- [ ] <feature-specific manual check 2>

## Spec / plan reference

Spec §: <spec section that maps to this PR>
Plan §: Phase N — PR #N
```

- [ ] **Step 3: Commit**

```bash
git add .github
git commit -m "$(cat <<'EOF'
feat(scaffold): GitHub Actions pytest CI + PR template

CI runs pytest -v on every push/PR with Python 3.11 + dev/demo extras.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 1.5: Push + open PR #1

- [ ] **Step 1: Push the branch**

```bash
git push -u origin feat/scaffold
```

- [ ] **Step 2: Open PR**

```bash
gh pr create --base main --title "feat(scaffold): Python package + pyproject + CI" --body "$(cat <<'EOF'
## Summary

PR #1 of the multi-tool-agent build. Sets up:
- \`pyproject.toml\` with all 5 runtime deps + dev/demo extras + pytest config
- Package skeleton: \`multitool/__init__.py\`, \`__main__.py\`, \`cli.py\` (stub), \`tools/__init__.py\` (stub), \`eval/__init__.py\`
- \`requirements.txt\` for future HF Spaces deploy (no \`-e .\` per DA Agent's lesson)
- GitHub Actions pytest workflow + PR template
- Smoke test verifying import + CLI stub

## Test plan

- [ ] CI green (one smoke test: 2 passed)
- [ ] \`pip install -e ".[dev,demo]"\` succeeds locally

## Spec / plan reference

Spec §3.1 (repo layout)
Plan Phase 1

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: After CI green, squash-merge**

```bash
gh pr merge --squash --delete-branch
git checkout main && git pull origin main
```

---

# Phase 2 — PR #2: `@tool` decorator + tool registry

**Branch:** `feat/tool-decorator`
**Goal:** The decorator that registers functions in `TOOL_REGISTRY` with auto-generated JSON Schema. Closed type-set with Optional handling.
**Estimated LoC:** ~150 + ~150 tests
**Estimated time:** 60–90 min

Per spec §3.2, the decorator must:
- Read function signature + docstring
- Auto-build OpenAI-compatible JSON Schema
- Require docstrings (raise ValueError if missing)
- Exclude optional parameters from `required` list
- Handle `Optional[X]` / `X | None` by stripping NoneType
- Support exactly: `str`, `int`, `float`, `bool`, and their Optional variants. Anything else raises `UnsupportedToolTypeError`.

### Task 2.1: Branch + write the type-mapping table

**Files:**
- Modify: `multitool/tools/__init__.py`

- [ ] **Step 1: Branch from latest main**

```bash
cd /Users/anilkumar/multi-tool-agent
git checkout main && git pull origin main
git checkout -b feat/tool-decorator
```

- [ ] **Step 2: Write failing test for the type mapping**

In `tests/test_tools.py`:

```python
"""Tests for the @tool decorator + TOOL_REGISTRY."""
import pytest


class TestPyToJsonType:
    """Maps Python type hints to JSON Schema type strings."""

    def test_str_maps_to_string(self):
        from multitool.tools import _py_to_json_type
        assert _py_to_json_type(str) == "string"

    def test_int_maps_to_integer(self):
        from multitool.tools import _py_to_json_type
        assert _py_to_json_type(int) == "integer"

    def test_float_maps_to_number(self):
        from multitool.tools import _py_to_json_type
        assert _py_to_json_type(float) == "number"

    def test_bool_maps_to_boolean(self):
        from multitool.tools import _py_to_json_type
        assert _py_to_json_type(bool) == "boolean"

    def test_unsupported_type_raises(self):
        from multitool.tools import _py_to_json_type, UnsupportedToolTypeError
        with pytest.raises(UnsupportedToolTypeError):
            _py_to_json_type(list)
```

- [ ] **Step 3: Run pytest — expect 5 failures**

```bash
pytest tests/test_tools.py::TestPyToJsonType -v
```

Expected: 5 errors (`_py_to_json_type` not defined / can't import).

- [ ] **Step 4: Implement `_py_to_json_type` (no Optional handling yet)**

Replace `multitool/tools/__init__.py` contents with:

```python
"""Tool registry. Decorator-driven tool registration with auto-generated JSON Schema."""
import inspect
from typing import Callable, Union, get_type_hints, get_origin, get_args
from types import NoneType

TOOL_REGISTRY: dict[str, dict] = {}

# Closed set of supported types. Anything else raises UnsupportedToolTypeError.
PY_TO_JSON_TYPE: dict[type, str] = {
    str:   "string",
    int:   "integer",
    float: "number",
    bool:  "boolean",
}


class UnsupportedToolTypeError(TypeError):
    """A tool's parameter has a type the decorator cannot map to JSON Schema."""


def _py_to_json_type(hint) -> str:
    """Map a Python type hint to a JSON Schema type string."""
    if hint in PY_TO_JSON_TYPE:
        return PY_TO_JSON_TYPE[hint]
    raise UnsupportedToolTypeError(f"Unsupported tool parameter type: {hint!r}")
```

- [ ] **Step 5: Run pytest — expect 5 passed**

```bash
pytest tests/test_tools.py::TestPyToJsonType -v
```

- [ ] **Step 6: Commit**

```bash
git add multitool/tools/__init__.py tests/test_tools.py
git commit -m "$(cat <<'EOF'
feat(tools): _py_to_json_type for primitive types (str/int/float/bool)

Closed set of 4 primitives mapped to JSON Schema strings.
UnsupportedToolTypeError raised on anything outside the set.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 2.2: Optional / Union[X, None] handling

**Files:**
- Modify: `multitool/tools/__init__.py`
- Modify: `tests/test_tools.py`

- [ ] **Step 1: Write failing tests for Optional**

Append to `tests/test_tools.py`:

```python
class TestPyToJsonTypeOptional:
    """Optional[X] / X | None unwraps to inner type."""

    def test_optional_str(self):
        from multitool.tools import _py_to_json_type
        assert _py_to_json_type(str | None) == "string"

    def test_optional_int(self):
        from multitool.tools import _py_to_json_type
        assert _py_to_json_type(int | None) == "integer"

    def test_optional_float(self):
        from multitool.tools import _py_to_json_type
        assert _py_to_json_type(float | None) == "number"

    def test_union_of_multiple_non_none_types_raises(self):
        from multitool.tools import _py_to_json_type, UnsupportedToolTypeError
        with pytest.raises(UnsupportedToolTypeError):
            _py_to_json_type(str | int)  # not Optional — multiple non-None
```

- [ ] **Step 2: Run pytest — expect 4 failures**

```bash
pytest tests/test_tools.py::TestPyToJsonTypeOptional -v
```

- [ ] **Step 3: Extend `_py_to_json_type` with Union handling**

In `multitool/tools/__init__.py`, replace the existing `_py_to_json_type` with:

```python
def _py_to_json_type(hint) -> str:
    """Map a Python type hint to a JSON Schema type string.
    Handles Optional[X] (= X | None) by stripping NoneType and recursing on X."""
    if get_origin(hint) is Union:
        args = tuple(a for a in get_args(hint) if a is not NoneType)
        if len(args) == 1:
            return _py_to_json_type(args[0])
        raise UnsupportedToolTypeError(f"Union of multiple non-None types: {hint}")
    if hint in PY_TO_JSON_TYPE:
        return PY_TO_JSON_TYPE[hint]
    raise UnsupportedToolTypeError(f"Unsupported tool parameter type: {hint!r}")
```

- [ ] **Step 4: Run pytest — expect 4 passed (+5 from before = 9)**

```bash
pytest tests/test_tools.py -v
```

- [ ] **Step 5: Commit**

```bash
git add multitool/tools/__init__.py tests/test_tools.py
git commit -m "$(cat <<'EOF'
feat(tools): _py_to_json_type handles Optional[X] / X | None

Strip NoneType from Union types, recurse on inner. Reject unions of
multiple non-None types.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 2.3: `@tool` decorator — basic schema generation

**Files:**
- Modify: `multitool/tools/__init__.py`
- Modify: `tests/test_tools.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_tools.py`:

```python
class TestToolDecorator:
    """The @tool decorator registers functions and builds JSON Schema."""

    def setup_method(self):
        """Clear TOOL_REGISTRY before each test (decorator side-effects)."""
        from multitool.tools import TOOL_REGISTRY
        TOOL_REGISTRY.clear()

    def test_decorator_registers_function(self):
        from multitool.tools import tool, TOOL_REGISTRY

        @tool
        def example_tool(query: str) -> str:
            """Example tool docstring."""
            return f"echo: {query}"

        assert "example_tool" in TOOL_REGISTRY
        assert TOOL_REGISTRY["example_tool"]["fn"] is example_tool

    def test_schema_has_correct_shape(self):
        from multitool.tools import tool, TOOL_REGISTRY

        @tool
        def example(query: str) -> str:
            """Example docstring used as description."""
            return query

        schema = TOOL_REGISTRY["example"]["schema"]
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "example"
        assert schema["function"]["description"] == "Example docstring used as description."
        assert schema["function"]["parameters"]["type"] == "object"
        assert schema["function"]["parameters"]["properties"] == {"query": {"type": "string"}}
        assert schema["function"]["parameters"]["required"] == ["query"]

    def test_multiple_required_params(self):
        from multitool.tools import tool, TOOL_REGISTRY

        @tool
        def add(a: int, b: int) -> int:
            """Add two integers."""
            return a + b

        params = TOOL_REGISTRY["add"]["schema"]["function"]["parameters"]
        assert params["properties"] == {
            "a": {"type": "integer"},
            "b": {"type": "integer"},
        }
        assert set(params["required"]) == {"a", "b"}
```

- [ ] **Step 2: Run pytest — expect 3 failures**

```bash
pytest tests/test_tools.py::TestToolDecorator -v
```

- [ ] **Step 3: Implement the `@tool` decorator**

In `multitool/tools/__init__.py`, append:

```python
def tool(fn: Callable) -> Callable:
    """Decorator. Registers fn in TOOL_REGISTRY with auto-generated JSON Schema."""
    hints = get_type_hints(fn)
    sig = inspect.signature(fn)
    required = [
        name for name, param in sig.parameters.items()
        if param.default is inspect.Parameter.empty
    ]
    schema = {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": inspect.getdoc(fn) or "",
            "parameters": {
                "type": "object",
                "properties": {
                    name: {"type": _py_to_json_type(hints[name])}
                    for name in sig.parameters
                },
                "required": required,
            },
        },
    }
    TOOL_REGISTRY[fn.__name__] = {"fn": fn, "schema": schema}
    return fn
```

- [ ] **Step 4: Run pytest — expect 3 passed (+9 from before = 12)**

```bash
pytest tests/test_tools.py -v
```

- [ ] **Step 5: Commit**

```bash
git add multitool/tools/__init__.py tests/test_tools.py
git commit -m "$(cat <<'EOF'
feat(tools): @tool decorator with auto-generated JSON Schema

Reads function signature + docstring, builds OpenAI-compatible schema.
Required params inferred from missing defaults.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 2.4: Decorator behaviors — docstring required, optionals not required

**Files:**
- Modify: `multitool/tools/__init__.py`
- Modify: `tests/test_tools.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_tools.py`:

```python
class TestToolDecoratorBehaviors:

    def setup_method(self):
        from multitool.tools import TOOL_REGISTRY
        TOOL_REGISTRY.clear()

    def test_decorator_requires_docstring(self):
        from multitool.tools import tool

        with pytest.raises(ValueError, match="docstring required"):
            @tool
            def no_doc(query: str) -> str:
                return query

    def test_optional_param_not_required(self):
        from multitool.tools import tool, TOOL_REGISTRY

        @tool
        def search(query: str, limit: int = 5) -> str:
            """Search with optional limit."""
            return query

        params = TOOL_REGISTRY["search"]["schema"]["function"]["parameters"]
        assert params["required"] == ["query"]  # limit excluded
        assert "limit" in params["properties"]  # but still in properties

    def test_optional_str_schema(self):
        from multitool.tools import tool, TOOL_REGISTRY

        @tool
        def event(name: str, when: str | None = None) -> str:
            """Optional when parameter."""
            return f"{name} at {when}"

        params = TOOL_REGISTRY["event"]["schema"]["function"]["parameters"]
        assert params["properties"]["when"] == {"type": "string"}
        assert params["required"] == ["name"]
```

- [ ] **Step 2: Run pytest — expect 3 failures**

```bash
pytest tests/test_tools.py::TestToolDecoratorBehaviors -v
```

- [ ] **Step 3: Update `tool` to require docstring**

In `multitool/tools/__init__.py`, modify the `tool` function — add docstring check at the top:

```python
def tool(fn: Callable) -> Callable:
    """Decorator. Registers fn in TOOL_REGISTRY with auto-generated JSON Schema.
    Requires a docstring (raises ValueError if missing — tool descriptions
    are load-bearing for tool-selection accuracy)."""
    if not inspect.getdoc(fn):
        raise ValueError(
            f"@tool {fn.__name__}: docstring required (used as tool description)"
        )
    hints = get_type_hints(fn)
    sig = inspect.signature(fn)
    required = [
        name for name, param in sig.parameters.items()
        if param.default is inspect.Parameter.empty
    ]
    schema = {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": inspect.getdoc(fn),
            "parameters": {
                "type": "object",
                "properties": {
                    name: {"type": _py_to_json_type(hints[name])}
                    for name in sig.parameters
                },
                "required": required,
            },
        },
    }
    TOOL_REGISTRY[fn.__name__] = {"fn": fn, "schema": schema}
    return fn
```

- [ ] **Step 4: Run all tests — expect 15 passed**

```bash
pytest tests/test_tools.py -v
```

- [ ] **Step 5: Commit**

```bash
git add multitool/tools/__init__.py tests/test_tools.py
git commit -m "$(cat <<'EOF'
feat(tools): @tool docstring required + optional params excluded from required

- Raises ValueError if @tool used on a function without docstring
- Optional params (with defaults) appear in properties but not required
- str | None correctly unwraps to "string" via _py_to_json_type

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 2.5: Push + open PR #2

- [ ] **Step 1: Final test run + push**

```bash
pytest -v
git push -u origin feat/tool-decorator
```

- [ ] **Step 2: Open PR**

```bash
gh pr create --base main --title "feat(tools): @tool decorator + TOOL_REGISTRY" --body "$(cat <<'EOF'
## Summary

PR #2 of the multi-tool-agent build. Ships the \`@tool\` decorator:
- Reads function signature + docstring, auto-builds OpenAI-compatible JSON Schema
- Closed type set: str, int, float, bool, plus their \`X | None\` Optional variants
- Required parameters inferred from missing defaults
- Docstring required at decoration time (raises ValueError otherwise)
- \`UnsupportedToolTypeError\` for unsupported types (list, dict, custom classes, multi-non-None unions)

## Test plan

- [ ] CI green (15 tests in TestPyToJsonType + TestPyToJsonTypeOptional + TestToolDecorator + TestToolDecoratorBehaviors)
- [ ] \`@tool def foo(x: list[str]): ...\` raises UnsupportedToolTypeError (manual check)

## Spec / plan reference

Spec §3.2 (Tool registry — the @tool decorator)
Plan Phase 2

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: After CI green, squash-merge**

```bash
gh pr merge --squash --delete-branch
git checkout main && git pull origin main
```

---

# Phase 3 — PR #3: `tavily_search` (first real tool)

**Branch:** `feat/tavily-search`
**Goal:** First production tool. Calls Tavily API, formats results for LLM consumption. Real-API path marked `@pytest.mark.slow`; default tests use `pytest-mock`.
**Estimated LoC:** ~120 + ~80 tests
**Estimated time:** 60–75 min

### Task 3.1: Branch + write the Tavily search tool

**Files:**
- Create: `multitool/tools/search.py`
- Modify: `tests/test_tools.py`

- [ ] **Step 1: Branch**

```bash
git checkout main && git pull origin main
git checkout -b feat/tavily-search
```

- [ ] **Step 2: Write failing tests (mocked Tavily client)**

Append to `tests/test_tools.py`:

```python
class TestTavilySearch:
    """tavily_search tool. Real API calls are @pytest.mark.slow; defaults are mocked."""

    def test_returns_formatted_string(self, mocker):
        from multitool.tools.search import tavily_search

        mock_client = mocker.MagicMock()
        mock_client.search.return_value = {
            "answer": "Chicago's population in 2023 was 2,664,452.",
            "results": [
                {"url": "https://example.com/chi", "title": "Chicago demographics", "content": "Population in 2023..."},
                {"url": "https://example.com/chi2", "title": "Census", "content": "Per Census Bureau..."},
            ],
        }
        mocker.patch("multitool.tools.search._get_client", return_value=mock_client)

        result = tavily_search("Chicago population 2023")
        assert "Chicago's population in 2023 was 2,664,452" in result
        assert "https://example.com/chi" in result

    def test_handles_no_results(self, mocker):
        from multitool.tools.search import tavily_search

        mock_client = mocker.MagicMock()
        mock_client.search.return_value = {"answer": None, "results": []}
        mocker.patch("multitool.tools.search._get_client", return_value=mock_client)

        result = tavily_search("query with no results")
        assert "No results" in result or "no results" in result

    def test_raises_on_missing_api_key(self, mocker, monkeypatch):
        from multitool.tools.search import tavily_search

        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        # Force re-create the cached client (so it re-reads env)
        import multitool.tools.search as search_mod
        search_mod._client = None

        with pytest.raises(RuntimeError, match="TAVILY_API_KEY"):
            tavily_search("anything")
```

- [ ] **Step 3: Run pytest — expect 3 errors (module doesn't exist)**

```bash
pytest tests/test_tools.py::TestTavilySearch -v
```

- [ ] **Step 4: Implement `multitool/tools/search.py`**

```python
"""Tavily search tool. Free-tier 1000 queries/month at https://tavily.com."""
import os
from . import tool

# Cached client so we don't re-instantiate per call. Reset to None in tests.
_client = None


def _get_client():
    """Lazy-initialize the Tavily client. Reads API key from env on first call."""
    global _client
    if _client is None:
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError(
                "TAVILY_API_KEY environment variable not set. "
                "Sign up at https://tavily.com for a free-tier key."
            )
        from tavily import TavilyClient
        _client = TavilyClient(api_key=api_key)
    return _client


@tool
def tavily_search(query: str) -> str:
    """Search the web for current information.
    Returns Tavily's pre-summarized answer plus up to 5 result URLs and snippets.
    Use this when you need facts you don't already know — current events, names,
    dates, numbers, geographic information.
    """
    client = _get_client()
    response = client.search(query, max_results=5, include_answer=True)

    answer = response.get("answer")
    results = response.get("results") or []

    if not answer and not results:
        return f"No results found for query: {query!r}"

    lines = []
    if answer:
        lines.append(f"Summary: {answer}")
    if results:
        lines.append("\nSources:")
        for r in results:
            title = r.get("title", "(no title)")
            url = r.get("url", "")
            content = (r.get("content") or "")[:200]  # truncate snippets
            lines.append(f"- {title} <{url}>\n  {content}")
    return "\n".join(lines)
```

- [ ] **Step 5: Run pytest — expect 3 passed**

```bash
pytest tests/test_tools.py::TestTavilySearch -v
```

- [ ] **Step 6: Commit**

```bash
git add multitool/tools/search.py tests/test_tools.py
git commit -m "$(cat <<'EOF'
feat(tools): tavily_search with mocked tests

Lazy-initializes TavilyClient; raises RuntimeError if TAVILY_API_KEY missing.
Returns formatted string: answer + top 5 result URLs with truncated snippets.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 3.2: Add a slow real-API smoke test (optional but recommended)

**Files:**
- Modify: `tests/test_tools.py`

- [ ] **Step 1: Add the slow test**

```python
class TestTavilySearchReal:
    """Real-API smoke test. Run with: pytest -m slow"""

    @pytest.mark.slow
    def test_real_search_returns_something(self):
        import os
        if not os.environ.get("TAVILY_API_KEY"):
            pytest.skip("TAVILY_API_KEY not set; skipping real-API test")

        from multitool.tools.search import tavily_search
        result = tavily_search("What is the capital of France?")
        assert "Paris" in result
```

- [ ] **Step 2: Run with -m slow locally to verify it works**

```bash
set -a; source .env; set +a
pytest -m slow tests/test_tools.py::TestTavilySearchReal -v
```

Expected: 1 passed (real Tavily call returns Paris).

- [ ] **Step 3: Commit**

```bash
git add tests/test_tools.py
git commit -m "$(cat <<'EOF'
test(search): add @pytest.mark.slow real-API smoke test

Skipped by default; run explicitly with `pytest -m slow`. Verifies
Tavily key + network path actually work end-to-end.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 3.3: Push + open PR #3

- [ ] **Step 1: Final test run + push**

```bash
pytest -v  # default = -m 'not slow', should pass without TAVILY_API_KEY
git push -u origin feat/tavily-search
```

- [ ] **Step 2: Open PR**

```bash
gh pr create --base main --title "feat(tools): tavily_search" --body "$(cat <<'EOF'
## Summary

PR #3 of the multi-tool-agent build. First real tool implementation:
- \`tavily_search(query: str) -> str\` — calls Tavily REST API
- Returns formatted string: pre-summarized answer + top 5 source URLs
- Lazy client initialization; raises RuntimeError if \`TAVILY_API_KEY\` env var missing
- 3 mocked tests + 1 \`@pytest.mark.slow\` real-API smoke test

## Test plan

- [ ] CI green (3 mocked tests; real-API test skipped by default)
- [ ] Locally: \`set -a; source .env; set +a; pytest -m slow tests/test_tools.py::TestTavilySearchReal\` returns "Paris"

## Spec / plan reference

Spec §3.3 (tavily_search row)
Plan Phase 3

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: Squash-merge after green**

---

# Phase 4 — PR #4: Calculator + datetime + unit_convert + wikipedia

**Branch:** `feat/remaining-tools`
**Goal:** The other 4 tools. Same pattern as `tavily_search` but each is much smaller (mostly stdlib + one library wrap).
**Estimated LoC:** ~300 + ~200 tests
**Estimated time:** 90–120 min

### Task 4.1: Branch + calculator tool

**Files:**
- Create: `multitool/tools/calculator.py`
- Modify: `tests/test_tools.py`

- [ ] **Step 1: Branch**

```bash
git checkout main && git pull origin main
git checkout -b feat/remaining-tools
```

- [ ] **Step 2: Write failing tests**

Append to `tests/test_tools.py`:

```python
class TestCalculator:

    def test_simple_arithmetic(self):
        from multitool.tools.calculator import calculator
        assert calculator("2 + 2") == "4"

    def test_floating_point(self):
        from multitool.tools.calculator import calculator
        result = float(calculator("2664452 / 81632"))
        assert abs(result - 32.64) < 0.01

    def test_handles_syntax_error(self):
        from multitool.tools.calculator import calculator
        result = calculator("2 +")
        assert "error" in result.lower() or "invalid" in result.lower()

    def test_rejects_unsafe_expressions(self):
        """numexpr doesn't execute arbitrary Python — verify this."""
        from multitool.tools.calculator import calculator
        result = calculator("__import__('os').system('ls')")
        assert "error" in result.lower()
```

- [ ] **Step 3: Run pytest — 4 failures**

```bash
pytest tests/test_tools.py::TestCalculator -v
```

- [ ] **Step 4: Implement `multitool/tools/calculator.py`**

```python
"""Calculator tool using numexpr (safe math expression evaluator)."""
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
```

- [ ] **Step 5: Run pytest — 4 passed**

```bash
pytest tests/test_tools.py::TestCalculator -v
```

- [ ] **Step 6: Commit**

```bash
git add multitool/tools/calculator.py tests/test_tools.py
git commit -m "$(cat <<'EOF'
feat(tools): calculator (numexpr-based)

Safe math expression evaluator. Rejects arbitrary Python (no eval()).
Returns numeric result as string or "Calculator error: ..." message.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 4.2: datetime_tool

**Files:**
- Create: `multitool/tools/datetime_tool.py`
- Modify: `tests/test_tools.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_tools.py`:

```python
class TestDatetimeTool:

    def test_years_between(self):
        from multitool.tools.datetime_tool import datetime_tool
        # Apple founded 1976, iPhone launched 2007
        result = datetime_tool("years_between", "1976", "2007")
        assert "31" in result

    def test_add_years(self):
        from multitool.tools.datetime_tool import datetime_tool
        result = datetime_tool("add_years", "2024", "5")
        assert "2029" in result

    def test_day_of_week(self):
        from multitool.tools.datetime_tool import datetime_tool
        # 2024-07-04 was a Thursday
        result = datetime_tool("day_of_week", "2024-07-04")
        assert "Thursday" in result

    def test_unknown_operation(self):
        from multitool.tools.datetime_tool import datetime_tool
        result = datetime_tool("invalid_op", "2024")
        assert "error" in result.lower() or "unknown" in result.lower()
```

- [ ] **Step 2: Run pytest — 4 failures**

- [ ] **Step 3: Implement `multitool/tools/datetime_tool.py`**

```python
"""Date arithmetic tool (years between, add years, day of week)."""
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
```

- [ ] **Step 4: Run pytest — 4 passed**

- [ ] **Step 5: Commit**

```bash
git add multitool/tools/datetime_tool.py tests/test_tools.py
git commit -m "$(cat <<'EOF'
feat(tools): datetime_tool (years_between / add_years / day_of_week)

Stdlib datetime, no extra deps. Three operations covering the most
common date-math questions in eval queries.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 4.3: unit_convert tool

**Files:**
- Create: `multitool/tools/unit_convert.py`
- Modify: `tests/test_tools.py`

- [ ] **Step 1: Write failing tests**

```python
class TestUnitConvert:

    def test_mph_to_ms(self):
        from multitool.tools.unit_convert import unit_convert
        result = float(unit_convert(60.0, "mile/hour", "meter/second"))
        assert abs(result - 26.82) < 0.1

    def test_km_to_mile(self):
        from multitool.tools.unit_convert import unit_convert
        result = float(unit_convert(100.0, "kilometer", "mile"))
        assert abs(result - 62.14) < 0.1

    def test_unknown_unit_returns_error(self):
        from multitool.tools.unit_convert import unit_convert
        result = unit_convert(1.0, "florblegorps", "meter")
        assert "error" in result.lower() or "undefined" in result.lower()
```

- [ ] **Step 2: Implement `multitool/tools/unit_convert.py`**

```python
"""Unit conversion tool using pint."""
from . import tool


@tool
def unit_convert(value: float, from_unit: str, to_unit: str) -> str:
    """Convert a value from one unit to another.

    Accepts most physical units pint knows: meter, mile, kilometer, second,
    minute, hour, kilogram, pound, celsius, fahrenheit, and combinations
    like meter/second, kilogram*meter/second**2.

    Examples:
      unit_convert(60, "mile/hour", "meter/second") -> "26.82..."
      unit_convert(100, "kilometer", "mile")        -> "62.14..."
      unit_convert(212, "fahrenheit", "celsius")    -> "100.0..."
    """
    try:
        import pint
        ureg = pint.UnitRegistry()
        quantity = value * ureg(from_unit)
        converted = quantity.to(to_unit)
        return str(converted.magnitude)
    except Exception as e:
        return f"Unit convert error: {type(e).__name__}: {e}"
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/test_tools.py::TestUnitConvert -v
git add multitool/tools/unit_convert.py tests/test_tools.py
git commit -m "$(cat <<'EOF'
feat(tools): unit_convert (pint-based)

Handles SI + imperial + temperature + compound units.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 4.4: wikipedia tool

**Files:**
- Create: `multitool/tools/wikipedia.py`
- Modify: `tests/test_tools.py`

- [ ] **Step 1: Write failing tests (mocked wikipedia-api)**

```python
class TestWikipedia:

    def test_returns_summary(self, mocker):
        from multitool.tools.wikipedia import wikipedia as wiki_tool

        mock_page = mocker.MagicMock()
        mock_page.exists.return_value = True
        mock_page.summary = (
            "Python is a high-level programming language. "
            "Created by Guido van Rossum in 1991. "
            "It emphasizes code readability. "
            "Python supports multiple programming paradigms."
        )

        mock_client = mocker.MagicMock()
        mock_client.page.return_value = mock_page
        mocker.patch("multitool.tools.wikipedia._get_client", return_value=mock_client)

        result = wiki_tool("Python")
        assert "Python is a high-level programming language" in result

    def test_returns_only_n_sentences(self, mocker):
        from multitool.tools.wikipedia import wikipedia as wiki_tool

        mock_page = mocker.MagicMock()
        mock_page.exists.return_value = True
        mock_page.summary = "One. Two. Three. Four. Five."

        mock_client = mocker.MagicMock()
        mock_client.page.return_value = mock_page
        mocker.patch("multitool.tools.wikipedia._get_client", return_value=mock_client)

        result = wiki_tool("anything", sentences=2)
        # Should contain first 2 sentences but not the rest
        assert "One" in result
        assert "Two" in result
        assert "Four" not in result

    def test_not_found(self, mocker):
        from multitool.tools.wikipedia import wikipedia as wiki_tool

        mock_page = mocker.MagicMock()
        mock_page.exists.return_value = False

        mock_client = mocker.MagicMock()
        mock_client.page.return_value = mock_page
        mocker.patch("multitool.tools.wikipedia._get_client", return_value=mock_client)

        result = wiki_tool("ThisTopicDoesNotExistOnWikipedia")
        assert "not found" in result.lower() or "no page" in result.lower()
```

- [ ] **Step 2: Implement `multitool/tools/wikipedia.py`**

```python
"""Wikipedia summary tool using wikipedia-api."""
import re
from . import tool

_client = None


def _get_client():
    global _client
    if _client is None:
        import wikipediaapi
        _client = wikipediaapi.Wikipedia(
            user_agent="multitool-agent/0.1 (https://github.com/laharikarumanchi-AI-ML/multi-tool-agent)",
            language="en",
        )
    return _client


@tool
def wikipedia(topic: str, sentences: int = 3) -> str:
    """Look up a Wikipedia article and return the first N sentences of its
    summary. Use this for background facts, biographical info, and historical
    context that doesn't need to be current.

    Examples:
      wikipedia("Python (programming language)") -> first 3 sentences of Python article
      wikipedia("Albert Einstein", sentences=5)  -> first 5 sentences of Einstein article
    """
    client = _get_client()
    page = client.page(topic)
    if not page.exists():
        return f"Wikipedia: no page found for {topic!r}"
    # Split into sentences using a simple regex (good enough for English).
    # wikipedia-api summary is already a single string; split on . ! ?
    sentence_re = re.compile(r"(?<=[.!?])\s+")
    parts = sentence_re.split(page.summary)
    return " ".join(parts[:sentences])
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/test_tools.py::TestWikipedia -v
git add multitool/tools/wikipedia.py tests/test_tools.py
git commit -m "$(cat <<'EOF'
feat(tools): wikipedia (first-N-sentences of article summary)

Uses wikipedia-api with proper User-Agent. Simple regex sentence split.
Default 3 sentences; configurable via `sentences` param.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 4.5: Verify all 5 tools registered + push PR #4

- [ ] **Step 1: Smoke test the registry**

Append to `tests/test_tools.py`:

```python
class TestAllToolsRegistered:
    """After importing each tool module, all 5 tools should appear in TOOL_REGISTRY."""

    def test_all_five_tools_registered(self):
        from multitool.tools import TOOL_REGISTRY
        # Force imports
        import multitool.tools.search  # noqa: F401
        import multitool.tools.calculator  # noqa: F401
        import multitool.tools.datetime_tool  # noqa: F401
        import multitool.tools.unit_convert  # noqa: F401
        import multitool.tools.wikipedia  # noqa: F401

        expected = {"tavily_search", "calculator", "datetime_tool", "unit_convert", "wikipedia"}
        assert expected.issubset(set(TOOL_REGISTRY.keys()))
```

- [ ] **Step 2: Run full test suite — expect ~35 passed**

```bash
pytest -v
```

- [ ] **Step 3: Push + open PR**

```bash
git push -u origin feat/remaining-tools
gh pr create --base main --title "feat(tools): calculator + datetime + unit_convert + wikipedia" --body "$(cat <<'EOF'
## Summary

PR #4 of the multi-tool-agent build. Ships the remaining 4 tools:
- \`calculator(expression)\` — numexpr-based safe math evaluator
- \`datetime_tool(operation, date_or_year, extra)\` — years_between / add_years / day_of_week
- \`unit_convert(value, from_unit, to_unit)\` — pint-based unit conversion
- \`wikipedia(topic, sentences=3)\` — wikipedia-api summary

All 5 tools now registered in TOOL_REGISTRY (verified by TestAllToolsRegistered).

## Test plan

- [ ] CI green (~35 tests total)
- [ ] Each tool has happy-path + error-path tests

## Spec / plan reference

Spec §3.3 (rows 2-5)
Plan Phase 4

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 4: Squash-merge after green**

---

# Phase 5 — PR #5: LLMClient + chat_with_tools()

**Branch:** `feat/llm-client`
**Goal:** Copy DA Agent's `Lahari/agent/llm_client.py` verbatim (with attribution), extend with `chat_with_tools()` + `ToolCall`/`ToolResponse` dataclasses. Both Groq and Gemini implementations.
**Estimated LoC:** ~250 + ~100 tests
**Estimated time:** 90–120 min

### Task 5.1: Branch + copy DA Agent client + attribution header

**Files:**
- Create: `multitool/llm_client.py`

- [ ] **Step 1: Branch**

```bash
git checkout main && git pull origin main
git checkout -b feat/llm-client
```

- [ ] **Step 2: Copy DA Agent's llm_client.py**

```bash
cp /Users/anilkumar/Lahari/agent/llm_client.py multitool/llm_client.py
```

- [ ] **Step 3: Add attribution header to the copied file**

Prepend to `multitool/llm_client.py`:

```python
"""LLM client abstraction (Groq + Gemini).

Originally written for the data-analysis-agent project; copied here verbatim
with this attribution header. The original lives at:
  https://github.com/laharikarumanchi-AI-ML/superpowers/blob/main/agent/llm_client.py

Extended here with chat_with_tools() for function-calling-native agent loops.
"""
```

- [ ] **Step 4: Verify the copy is correct**

```bash
pytest tests/test_smoke.py -v  # should still pass
python -c "from multitool.llm_client import GroqClient, GeminiClient; print('OK')"
```

- [ ] **Step 5: Commit**

```bash
git add multitool/llm_client.py
git commit -m "$(cat <<'EOF'
feat(llm): copy LLMClient from DA Agent + attribution header

Verbatim copy of /Users/anilkumar/Lahari/agent/llm_client.py with a
header pointing back to the original. The chat_with_tools() extension
follows in the next commits.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 5.2: ToolCall + ToolResponse dataclasses

**Files:**
- Modify: `multitool/llm_client.py`

- [ ] **Step 1: Write failing test**

In `tests/test_llm_client.py`:

```python
"""Tests for the chat_with_tools() extension and ToolCall/ToolResponse types."""
import pytest


class TestToolDataclasses:

    def test_tool_call_has_fields(self):
        from multitool.llm_client import ToolCall

        c = ToolCall(id="abc", name="search", arguments={"query": "test"})
        assert c.id == "abc"
        assert c.name == "search"
        assert c.arguments == {"query": "test"}

    def test_tool_response_with_content(self):
        from multitool.llm_client import ToolResponse

        r = ToolResponse(content="Final answer", tool_calls=[])
        assert r.content == "Final answer"
        assert r.tool_calls == []

    def test_tool_response_with_tool_calls(self):
        from multitool.llm_client import ToolResponse, ToolCall

        r = ToolResponse(content=None, tool_calls=[
            ToolCall(id="1", name="search", arguments={"q": "x"})
        ])
        assert r.content is None
        assert len(r.tool_calls) == 1
```

- [ ] **Step 2: Run pytest — 3 errors (import fails)**

- [ ] **Step 3: Add the dataclasses to `multitool/llm_client.py`**

Append to the file (after the existing classes):

```python
from dataclasses import dataclass


@dataclass
class ToolCall:
    """A structured tool invocation from the LLM.

    Note: arguments is ALWAYS a parsed dict, never a JSON string. Groq's raw
    API returns arguments as a JSON-encoded string; chat_with_tools()
    is responsible for json.loads()-ing it before constructing the ToolCall.
    """
    id: str          # Provider-issued ID; for Gemini, synthesized by the client
    name: str        # Tool function name (matches a key in TOOL_REGISTRY)
    arguments: dict  # Already-parsed kwargs dict


@dataclass
class ToolResponse:
    """A response from chat_with_tools(). Exactly one of:
    - content is set, tool_calls is empty → model produced a final answer
    - content is None, tool_calls has items → model requested tool invocations
    """
    content: str | None
    tool_calls: list[ToolCall]
```

- [ ] **Step 4: Run pytest — 3 passed**

- [ ] **Step 5: Commit**

```bash
git add multitool/llm_client.py tests/test_llm_client.py
git commit -m "$(cat <<'EOF'
feat(llm): ToolCall + ToolResponse dataclasses

arguments is always parsed dict (chat_with_tools handles json.loads).
ToolResponse is a tagged union: either content (final answer) OR
tool_calls (requested invocations).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 5.3: Extend the LLMClient Protocol

**Files:**
- Modify: `multitool/llm_client.py`

- [ ] **Step 1: Add `chat_with_tools` to the Protocol**

Find the existing `class LLMClient(Protocol):` block (should be near the top of the file after the attribution header) and extend it:

```python
class LLMClient(Protocol):
    def chat(self, messages: list[dict], **kwargs) -> str: ...
    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        **kwargs,
    ) -> ToolResponse: ...
```

- [ ] **Step 2: Build (no runtime error expected since concrete clients haven't implemented yet)**

```bash
python -c "from multitool.llm_client import LLMClient, ToolResponse; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add multitool/llm_client.py
git commit -m "$(cat <<'EOF'
feat(llm): extend LLMClient Protocol with chat_with_tools()

Concrete implementations follow in next 2 commits.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 5.4: Implement chat_with_tools() on GroqClient

**Files:**
- Modify: `multitool/llm_client.py`
- Modify: `tests/test_llm_client.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_llm_client.py`:

```python
import json


class TestGroqChatWithTools:

    def test_returns_tool_calls_when_model_picks_a_tool(self, mocker):
        from multitool.llm_client import GroqClient, ToolResponse

        # Mock requests.post to return a Groq-shaped tool_calls response
        mock_resp = mocker.MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "tavily_search",
                            "arguments": json.dumps({"query": "Chicago population"}),
                        },
                    }],
                }
            }]
        }
        mocker.patch("multitool.llm_client.requests.post", return_value=mock_resp)

        client = GroqClient(api_key="dummy")
        result = client.chat_with_tools(
            messages=[{"role": "user", "content": "What's Chicago's population?"}],
            tools=[{"type": "function", "function": {"name": "tavily_search", "parameters": {}}}],
        )
        assert isinstance(result, ToolResponse)
        assert result.content is None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_abc"
        assert result.tool_calls[0].name == "tavily_search"
        assert result.tool_calls[0].arguments == {"query": "Chicago population"}  # parsed dict!

    def test_returns_content_when_model_answers_directly(self, mocker):
        from multitool.llm_client import GroqClient, ToolResponse

        mock_resp = mocker.MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{
                "message": {"role": "assistant", "content": "Hello", "tool_calls": None}
            }]
        }
        mocker.patch("multitool.llm_client.requests.post", return_value=mock_resp)

        client = GroqClient(api_key="dummy")
        result = client.chat_with_tools(
            messages=[{"role": "user", "content": "Say hello"}],
            tools=[],
        )
        assert result.content == "Hello"
        assert result.tool_calls == []

    def test_arguments_always_parsed_to_dict(self, mocker):
        """Critical contract: arguments is dict, never str."""
        from multitool.llm_client import GroqClient

        mock_resp = mocker.MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{
                "message": {
                    "content": None,
                    "tool_calls": [{
                        "id": "x",
                        "function": {"name": "f", "arguments": json.dumps({"a": 1, "b": "two"})},
                    }],
                }
            }]
        }
        mocker.patch("multitool.llm_client.requests.post", return_value=mock_resp)

        client = GroqClient(api_key="dummy")
        result = client.chat_with_tools(messages=[], tools=[])
        assert isinstance(result.tool_calls[0].arguments, dict)
        assert result.tool_calls[0].arguments == {"a": 1, "b": "two"}
```

- [ ] **Step 2: Run pytest — 3 failures**

- [ ] **Step 3: Implement `GroqClient.chat_with_tools`**

In `multitool/llm_client.py`, find the `class GroqClient:` block and add this method (after `chat()`):

```python
    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        **kwargs,
    ) -> ToolResponse:
        """Call Groq's chat completions API with the tools= parameter.
        Returns a ToolResponse — either content (final answer) or tool_calls
        (parsed arguments dict, not JSON string)."""
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
        }
        # Allow caller to override via kwargs (max_tokens, temperature, etc.)
        payload.update(kwargs)

        # Reuse the existing chat()'s retry loop pattern. For now, do a
        # straightforward call; the retry behavior is shared via the underlying
        # _post helper if/when we refactor. Here we inline a simple version.
        for attempt in range(self.MAX_ATTEMPTS):
            response = requests.post(
                self.URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=60,
            )
            if response.status_code == 200:
                msg = response.json()["choices"][0]["message"]
                content = msg.get("content")
                raw_tool_calls = msg.get("tool_calls") or []
                tool_calls = [
                    ToolCall(
                        id=tc["id"],
                        name=tc["function"]["name"],
                        arguments=json.loads(tc["function"]["arguments"]),
                    )
                    for tc in raw_tool_calls
                ]
                return ToolResponse(content=content, tool_calls=tool_calls)
            # On non-200, sleep + retry using existing _sleep_seconds helper
            time.sleep(self._sleep_seconds(response, attempt))
        response.raise_for_status()
```

Also at the top of the file, make sure `import json` is present (it may already be).

- [ ] **Step 4: Run pytest — 3 passed**

- [ ] **Step 5: Commit**

```bash
git add multitool/llm_client.py tests/test_llm_client.py
git commit -m "$(cat <<'EOF'
feat(llm): GroqClient.chat_with_tools() — function-calling API

Wraps Groq's tools= parameter. Parses JSON-string arguments into dict
before returning. Reuses MAX_ATTEMPTS + _sleep_seconds for retries.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 5.5: Implement chat_with_tools() on GeminiClient

**Files:**
- Modify: `multitool/llm_client.py`
- Modify: `tests/test_llm_client.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_llm_client.py`:

```python
class TestGeminiChatWithTools:

    def test_synthesizes_call_id(self, mocker):
        """Gemini doesn't issue call IDs; client must synthesize them."""
        from multitool.llm_client import GeminiClient

        mock_resp = mocker.MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{
                        "functionCall": {
                            "name": "tavily_search",
                            "args": {"query": "Chicago"},
                        }
                    }]
                }
            }]
        }
        mocker.patch("multitool.llm_client.requests.post", return_value=mock_resp)

        client = GeminiClient(api_key="dummy")
        result = client.chat_with_tools(messages=[], tools=[])
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id.startswith("gemini-call-")
        assert result.tool_calls[0].name == "tavily_search"
        assert result.tool_calls[0].arguments == {"query": "Chicago"}  # already dict
```

- [ ] **Step 2: Implement `GeminiClient.chat_with_tools`**

In `multitool/llm_client.py`, find `class GeminiClient:` and add this method:

```python
    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        **kwargs,
    ) -> ToolResponse:
        """Call Gemini's generateContent with function-call tools.
        Synthesizes call IDs (Gemini doesn't issue them) so downstream
        code can use call.id uniformly."""
        from uuid import uuid4

        # Gemini's tools format uses functionDeclarations
        gemini_tools = [{"functionDeclarations": [t["function"]] for t in tools}] if tools else []
        if not tools:
            gemini_tools = []
        else:
            gemini_tools = [{"functionDeclarations": [t["function"] for t in tools]}]

        payload = {
            "contents": self._to_gemini_format(messages)["contents"],
            "tools": gemini_tools,
        }
        payload.update(kwargs)

        for attempt in range(self.MAX_ATTEMPTS):
            self._throttle()
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
                params={"key": self.api_key},
                json=payload,
                timeout=60,
            )
            # Scrub URL key from any future error messages
            response.url = response.url.split("?")[0] + "?key=[REDACTED]"
            if response.status_code == 200:
                data = response.json()
                content = None
                tool_calls = []
                parts = data["candidates"][0]["content"]["parts"]
                for part in parts:
                    if "text" in part:
                        content = part["text"]
                    elif "functionCall" in part:
                        fc = part["functionCall"]
                        tool_calls.append(ToolCall(
                            id=f"gemini-call-{uuid4().hex[:8]}",
                            name=fc["name"],
                            arguments=fc.get("args", {}),  # Gemini returns dict, not string
                        ))
                return ToolResponse(content=content, tool_calls=tool_calls)
            time.sleep(self._sleep_seconds(response, attempt))
        response.raise_for_status()
```

- [ ] **Step 3: Run pytest — passed**

- [ ] **Step 4: Commit**

```bash
git add multitool/llm_client.py tests/test_llm_client.py
git commit -m "$(cat <<'EOF'
feat(llm): GeminiClient.chat_with_tools() with synthesized call IDs

Gemini's API uses functionDeclarations / functionCall / args (dict, not
JSON string). Client synthesizes IDs so call.id is uniformly populated
across providers. URL key-scrub preserved from existing chat().

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 5.6: Push + open PR #5

```bash
pytest -v
git push -u origin feat/llm-client
gh pr create --base main --title "feat(llm): LLMClient copied from DA Agent + chat_with_tools()" --body "$(cat <<'EOF'
## Summary

PR #5 of the multi-tool-agent build. Provider-agnostic LLM layer with function-calling support.
- Copied \`Lahari/agent/llm_client.py\` verbatim with attribution header
- Added \`ToolCall\` + \`ToolResponse\` dataclasses
- Extended Protocol with \`chat_with_tools()\`
- Implemented on both \`GroqClient\` and \`GeminiClient\`
- Gemini synthesizes call IDs (uniform call.id field across providers)
- Arguments always parsed to dict (Groq: json.loads; Gemini: native)

## Test plan

- [ ] CI green (~10 new tests)
- [ ] \`arguments\` is dict in every test (the critical contract)

## Spec / plan reference

Spec §3.5
Plan Phase 5

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

# Phase 6 — PR #6: Orchestrator + Trace

**Branch:** `feat/orchestrator`
**Goal:** The agent loop. Function-calling-native; per-tool retry budget; step ceiling; trace logging.
**Estimated LoC:** ~280 + ~150 tests
**Estimated time:** 2-3 hours

### Task 6.1: Branch + create the system prompt

**Files:**
- Create: `multitool/prompts/system.txt`

- [ ] **Step 1: Branch**

```bash
git checkout main && git pull origin main
git checkout -b feat/orchestrator
```

- [ ] **Step 2: Write the system prompt**

```
You are a research assistant with access to tools. For any question that requires external information or computation, call the appropriate tools. Chain multiple tool calls when needed.

Available tool categories:
- Web search (tavily_search): for current facts, current events, names, dates, populations, prices
- Calculator: for arithmetic and math expressions
- Datetime: for date arithmetic (years between, add years, day of week)
- Unit conversion: for converting between units (mph to m/s, km to mile, fahrenheit to celsius)
- Wikipedia: for background facts, biographical info, historical context

Once you have enough information to answer the user's question, respond with your final answer in plain text. Do NOT call any tools in that final response.

If a tool returns an error, you can:
- Retry with adjusted arguments (e.g., different query phrasing, different unit names)
- Switch to a different tool
- Explain what you could not answer and why, if no recovery is possible

Be concise. State the final answer clearly. When you compute a numeric answer, lead with the number.
```

Save to `multitool/prompts/system.txt`.

- [ ] **Step 3: Commit**

```bash
git add multitool/prompts/system.txt
git commit -m "$(cat <<'EOF'
feat(agent): system prompt for the orchestrator

Minimal prompt — function-calling does the heavy lifting. No
Thought:/Action: scaffold. Lists tool categories so model knows
the menu without re-reading every schema description.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 6.2: Trace logger

**Files:**
- Create: `multitool/trace.py`
- Modify: `tests/test_trace.py`

- [ ] **Step 1: Write failing tests**

In `tests/test_trace.py`:

```python
"""Tests for trace.py — JSON logger of agent runs."""
import json
import tempfile
from pathlib import Path
import pytest


class TestTrace:

    def test_creates_file_with_metadata(self):
        from multitool.trace import Trace

        with tempfile.TemporaryDirectory() as tmpdir:
            t = Trace(directory=tmpdir, question="What is 2+2?", provider="groq", model="llama-3.3-70b-versatile")
            t.flush()
            assert Path(t.path).exists()
            data = json.loads(Path(t.path).read_text())
            assert data["question"] == "What is 2+2?"
            assert data["provider"] == "groq"
            assert data["model"] == "llama-3.3-70b-versatile"
            assert "started_at" in data
            assert "run_id" in data

    def test_log_step(self):
        from multitool.trace import Trace
        from multitool.llm_client import ToolCall

        with tempfile.TemporaryDirectory() as tmpdir:
            t = Trace(directory=tmpdir, question="Q", provider="groq", model="m")
            call = ToolCall(id="1", name="calculator", arguments={"expression": "2+2"})
            t.log_step(step=0, call=call, result="4")
            t.flush()
            data = json.loads(Path(t.path).read_text())
            assert len(data["steps"]) == 1
            assert data["steps"][0]["step"] == 0
            assert data["steps"][0]["tool_calls"][0]["name"] == "calculator"
            assert data["steps"][0]["results"][0] == "4"

    def test_log_final(self):
        from multitool.trace import Trace

        with tempfile.TemporaryDirectory() as tmpdir:
            t = Trace(directory=tmpdir, question="Q", provider="groq", model="m")
            t.log_final(step=1, final_answer="The answer is 4.")
            t.flush()
            data = json.loads(Path(t.path).read_text())
            assert data["final_answer"] == "The answer is 4."
            assert data["total_steps"] == 2  # 1 zero-indexed step + 1 (final)
```

- [ ] **Step 2: Implement `multitool/trace.py`**

```python
"""JSON trace logger for agent runs. One file per run; flushed after every step
so a mid-run crash leaves a partial-but-valid log on disk."""
import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class Trace:
    """A single agent-run's trace. Writes JSON to <directory>/<run_id>.json
    after every step. Use .flush() to force-write after the final answer."""

    def __init__(
        self,
        directory: str,
        question: str,
        provider: str,
        model: str,
    ):
        os.makedirs(directory, exist_ok=True)
        self.run_id = uuid.uuid4().hex[:12]
        self.path = str(Path(directory) / f"{self.run_id}.json")
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.question = question
        self.provider = provider
        self.model = model
        self.steps: list[dict] = []
        self.final_answer: str | None = None
        self.total_steps: int = 0
        self.error: str | None = None
        self.flush()

    def log_step(self, step: int, call, result: str) -> None:
        """Record one tool invocation in step N."""
        # Find existing step entry or create
        existing = next((s for s in self.steps if s["step"] == step), None)
        if existing is None:
            existing = {"step": step, "tool_calls": [], "results": []}
            self.steps.append(existing)
        existing["tool_calls"].append({
            "id": call.id,
            "name": call.name,
            "args": call.arguments,
        })
        existing["results"].append(result)
        self.flush()

    def log_final(self, step: int, final_answer: str) -> None:
        """Record the model's final answer (returned without tool_calls)."""
        self.final_answer = final_answer
        self.total_steps = step + 1
        self.flush()

    def log_error(self, error: str) -> None:
        self.error = error
        self.flush()

    def flush(self) -> None:
        data = {
            "run_id": self.run_id,
            "question": self.question,
            "started_at": self.started_at,
            "provider": self.provider,
            "model": self.model,
            "steps": self.steps,
            "final_answer": self.final_answer,
            "total_steps": self.total_steps,
            "error": self.error,
        }
        Path(self.path).write_text(json.dumps(data, indent=2))
```

- [ ] **Step 3: Run pytest — 3 passed**

```bash
pytest tests/test_trace.py -v
```

- [ ] **Step 4: Commit**

```bash
git add multitool/trace.py tests/test_trace.py
git commit -m "$(cat <<'EOF'
feat(agent): Trace — JSON logger flushed after every step

One file per run at <dir>/<run_id>.json. log_step appends tool calls
to the matching step entry; log_final records the final answer and
total step count.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 6.3: Orchestrator skeleton + AgentResult

**Files:**
- Create: `multitool/orchestrator.py`
- Create: `tests/test_orchestrator.py`

- [ ] **Step 1: Write the first failing test (skeleton can be constructed)**

In `tests/test_orchestrator.py`:

```python
"""Tests for the orchestrator agent loop."""
import pytest


class TestOrchestratorSkeleton:

    def test_construct(self, tmp_path, mocker):
        from multitool.orchestrator import Orchestrator
        from multitool.trace import Trace
        mock_llm = mocker.MagicMock()
        trace = Trace(directory=str(tmp_path), question="Q", provider="groq", model="m")
        orch = Orchestrator(llm=mock_llm, trace=trace)
        assert orch.llm is mock_llm
        assert orch.trace is trace

    def test_agent_result_dataclass(self):
        from multitool.orchestrator import AgentResult
        r = AgentResult(answer="42", steps_taken=2, tool_calls=[], error=None, trace_path="/tmp/x.json")
        assert r.answer == "42"
        assert r.steps_taken == 2
        assert r.error is None
```

- [ ] **Step 2: Implement minimal skeleton**

```python
"""The agent loop. Function-calling-native; per-tool retry budget; step ceiling."""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from multitool.llm_client import LLMClient, ToolCall, ToolResponse
from multitool.tools import TOOL_REGISTRY
from multitool.trace import Trace


def _load_system_prompt() -> str:
    path = Path(__file__).parent / "prompts" / "system.txt"
    return path.read_text()


SYSTEM_PROMPT = _load_system_prompt()


@dataclass
class AgentResult:
    answer: str | None
    steps_taken: int
    tool_calls: list[dict]      # Flat log of every tool invocation for eval scoring
    error: str | None           # 'max_steps_reached' / 'tool_failed_permanently' / None
    trace_path: str


class Orchestrator:
    MAX_STEPS = 10
    MAX_TOOL_RETRIES = 2

    def __init__(self, llm: LLMClient, trace: Trace):
        self.llm = llm
        self.trace = trace
```

- [ ] **Step 3: Run pytest — 2 passed**

- [ ] **Step 4: Commit**

```bash
git add multitool/orchestrator.py tests/test_orchestrator.py
git commit -m "$(cat <<'EOF'
feat(agent): Orchestrator + AgentResult skeleton

Loads system prompt from multitool/prompts/system.txt. Constants for
step ceiling + per-tool retry budget. Real run() implementation
follows.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 6.4: `_dispatch_with_retry`

**Files:**
- Modify: `multitool/orchestrator.py`
- Modify: `tests/test_orchestrator.py`

- [ ] **Step 1: Write failing tests**

```python
class TestDispatchWithRetry:

    def test_successful_call_returns_result(self, tmp_path, mocker):
        from multitool.orchestrator import Orchestrator
        from multitool.llm_client import ToolCall
        from multitool.trace import Trace
        from multitool.tools import TOOL_REGISTRY

        TOOL_REGISTRY["fake_tool"] = {"fn": lambda x: f"OK:{x}", "schema": {}}
        trace = Trace(directory=str(tmp_path), question="Q", provider="g", model="m")
        orch = Orchestrator(llm=mocker.MagicMock(), trace=trace)

        call = ToolCall(id="1", name="fake_tool", arguments={"x": "hello"})
        result = orch._dispatch_with_retry(call)
        assert result == "OK:hello"

    def test_failing_call_retries_then_returns_error(self, tmp_path, mocker):
        from multitool.orchestrator import Orchestrator
        from multitool.llm_client import ToolCall
        from multitool.trace import Trace
        from multitool.tools import TOOL_REGISTRY

        # Always-failing tool
        TOOL_REGISTRY["always_fails"] = {
            "fn": lambda: (_ for _ in ()).throw(RuntimeError("boom")),
            "schema": {},
        }
        trace = Trace(directory=str(tmp_path), question="Q", provider="g", model="m")
        orch = Orchestrator(llm=mocker.MagicMock(), trace=trace)

        call = ToolCall(id="1", name="always_fails", arguments={})
        result = orch._dispatch_with_retry(call)
        assert "Tool error" in result
        assert "boom" in result
```

- [ ] **Step 2: Implement `_dispatch_with_retry`**

Append to `Orchestrator`:

```python
    def _dispatch_with_retry(self, call: ToolCall) -> str:
        """Per-tool retry: 2 attempts. If both fail, surface error as Observation
        — the model decides whether to retry, switch tools, or give up."""
        for attempt in range(self.MAX_TOOL_RETRIES + 1):
            try:
                fn = TOOL_REGISTRY[call.name]["fn"]
                return str(fn(**call.arguments))
            except Exception as e:
                if attempt == self.MAX_TOOL_RETRIES:
                    return f"Tool error after {self.MAX_TOOL_RETRIES} retries: {type(e).__name__}: {e}"
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/test_orchestrator.py::TestDispatchWithRetry -v
git add multitool/orchestrator.py tests/test_orchestrator.py
git commit -m "$(cat <<'EOF'
feat(agent): _dispatch_with_retry with 2-attempt budget

Per-call retry budget; surfaces error as Observation if all retries fail.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 6.5: The `run()` method

**Files:**
- Modify: `multitool/orchestrator.py`
- Modify: `tests/test_orchestrator.py`

- [ ] **Step 1: Write failing tests**

```python
class TestOrchestratorRun:

    def test_returns_final_answer_when_model_responds_with_content(self, tmp_path, mocker):
        """If first LLM call returns content (no tool_calls), that's the final answer."""
        from multitool.orchestrator import Orchestrator
        from multitool.llm_client import ToolResponse
        from multitool.trace import Trace

        mock_llm = mocker.MagicMock()
        mock_llm.chat_with_tools.return_value = ToolResponse(content="2+2=4", tool_calls=[])

        trace = Trace(directory=str(tmp_path), question="What is 2+2?", provider="g", model="m")
        orch = Orchestrator(llm=mock_llm, trace=trace)
        result = orch.run("What is 2+2?")

        assert result.answer == "2+2=4"
        assert result.steps_taken == 1
        assert result.error is None

    def test_max_steps_reached_returns_error(self, tmp_path, mocker):
        """If model keeps calling tools and never answers, exit with max_steps_reached."""
        from multitool.orchestrator import Orchestrator
        from multitool.llm_client import ToolResponse, ToolCall
        from multitool.tools import TOOL_REGISTRY
        from multitool.trace import Trace

        TOOL_REGISTRY["echo"] = {"fn": lambda x: f"echoed:{x}", "schema": {}}

        mock_llm = mocker.MagicMock()
        # Always return a tool_call, never content
        mock_llm.chat_with_tools.return_value = ToolResponse(
            content=None,
            tool_calls=[ToolCall(id="1", name="echo", arguments={"x": "loop"})],
        )

        trace = Trace(directory=str(tmp_path), question="Q", provider="g", model="m")
        orch = Orchestrator(llm=mock_llm, trace=trace)
        result = orch.run("Q")

        assert result.answer is None
        assert result.error == "max_steps_reached"
        assert result.steps_taken == Orchestrator.MAX_STEPS
```

- [ ] **Step 2: Implement `run()`**

Append to `Orchestrator`:

```python
    def _all_schemas(self) -> list[dict]:
        return [info["schema"] for info in TOOL_REGISTRY.values()]

    def run(self, question: str) -> AgentResult:
        """Run the agent loop until final answer OR step ceiling."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        tool_calls_log: list[dict] = []

        for step in range(self.MAX_STEPS):
            response = self.llm.chat_with_tools(messages, tools=self._all_schemas())

            # Termination: content + no tool_calls = final answer
            if response.content and not response.tool_calls:
                self.trace.log_final(step, response.content)
                return AgentResult(
                    answer=response.content,
                    steps_taken=step + 1,
                    tool_calls=tool_calls_log,
                    error=None,
                    trace_path=self.trace.path,
                )

            # Append assistant message with tool_calls to history
            messages.append({
                "role": "assistant",
                "content": response.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in response.tool_calls
                ],
            })

            for call in response.tool_calls:
                result = self._dispatch_with_retry(call)
                tool_calls_log.append({"name": call.name, "args": call.arguments, "result": result})
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result,
                })
                self.trace.log_step(step, call, result)

        # Step ceiling reached without final answer
        self.trace.log_error("max_steps_reached")
        return AgentResult(
            answer=None,
            steps_taken=self.MAX_STEPS,
            tool_calls=tool_calls_log,
            error="max_steps_reached",
            trace_path=self.trace.path,
        )
```

- [ ] **Step 3: Run pytest — 2 passed**

- [ ] **Step 4: Commit**

```bash
git add multitool/orchestrator.py tests/test_orchestrator.py
git commit -m "$(cat <<'EOF'
feat(agent): Orchestrator.run() — main function-calling loop

Terminates on content+no_tool_calls (final answer) or MAX_STEPS=10
(error="max_steps_reached"). Per-tool dispatch via _dispatch_with_retry.
Every step logged to trace.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 6.6: Per-dispatch retry isolation test + repeated-call test

**Files:**
- Modify: `tests/test_orchestrator.py`

- [ ] **Step 1: Add the spec §3.6 tests**

```python
class TestRetryBudgetAndStepCeiling:
    """Verify the spec §3.6 'repeated-failed-call loop' decision: per-dispatch
    retry resets each step; step ceiling is the backstop."""

    def test_per_dispatch_retry_budget_resets_each_step(self, tmp_path, mocker):
        """Each step starts with a fresh MAX_TOOL_RETRIES budget."""
        from multitool.orchestrator import Orchestrator
        from multitool.llm_client import ToolResponse, ToolCall
        from multitool.tools import TOOL_REGISTRY
        from multitool.trace import Trace

        call_count = {"n": 0}
        def counting_fn():
            call_count["n"] += 1
            raise RuntimeError("always fails")
        TOOL_REGISTRY["fail_tool"] = {"fn": counting_fn, "schema": {}}

        mock_llm = mocker.MagicMock()
        mock_llm.chat_with_tools.return_value = ToolResponse(
            content=None,
            tool_calls=[ToolCall(id="1", name="fail_tool", arguments={})],
        )

        trace = Trace(directory=str(tmp_path), question="Q", provider="g", model="m")
        orch = Orchestrator(llm=mock_llm, trace=trace)
        orch.run("Q")
        # 10 steps × 3 attempts per dispatch = 30 fn invocations
        assert call_count["n"] == Orchestrator.MAX_STEPS * (Orchestrator.MAX_TOOL_RETRIES + 1)

    def test_repeated_same_call_terminates_at_step_ceiling(self, tmp_path, mocker):
        """Model can loop on same (name, args) all day; we just exit cleanly at MAX_STEPS."""
        from multitool.orchestrator import Orchestrator
        from multitool.llm_client import ToolResponse, ToolCall
        from multitool.tools import TOOL_REGISTRY
        from multitool.trace import Trace

        TOOL_REGISTRY["broken"] = {
            "fn": lambda: (_ for _ in ()).throw(RuntimeError("nope")),
            "schema": {},
        }

        mock_llm = mocker.MagicMock()
        mock_llm.chat_with_tools.return_value = ToolResponse(
            content=None,
            tool_calls=[ToolCall(id="1", name="broken", arguments={})],
        )

        trace = Trace(directory=str(tmp_path), question="Q", provider="g", model="m")
        orch = Orchestrator(llm=mock_llm, trace=trace)
        result = orch.run("Q")
        assert result.error == "max_steps_reached"
        assert result.steps_taken == Orchestrator.MAX_STEPS
```

- [ ] **Step 2: Run + commit**

```bash
pytest tests/test_orchestrator.py::TestRetryBudgetAndStepCeiling -v
git add tests/test_orchestrator.py
git commit -m "$(cat <<'EOF'
test(agent): per-dispatch retry budget + step-ceiling backstop

Pins the spec §3.6 'repeated-failed-call loop' decision: budget resets
each step; step ceiling is the only backstop against pathological loops.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 6.7: End-to-end test with scripted mock LLM

**Files:**
- Modify: `tests/test_end_to_end.py`

- [ ] **Step 1: Write the end-to-end test**

In `tests/test_end_to_end.py`:

```python
"""End-to-end agent loop with a scripted mock LLM. No real API calls."""
import pytest


class TestEndToEnd:

    def test_two_step_search_then_calc(self, tmp_path, mocker):
        """Mock LLM returns: search-call → search-result-in-prompt → calc-call → calc-result-in-prompt → final answer."""
        from multitool.orchestrator import Orchestrator
        from multitool.llm_client import ToolResponse, ToolCall
        from multitool.tools import TOOL_REGISTRY
        from multitool.trace import Trace

        # Stub two tools
        TOOL_REGISTRY["search_fake"] = {
            "fn": lambda q: f"Population of {q.split()[0]}: 1000000",
            "schema": {},
        }
        TOOL_REGISTRY["calc_fake"] = {
            "fn": lambda e: str(eval(e)),  # ok in test; not in production
            "schema": {},
        }

        # Script the LLM responses
        responses = [
            ToolResponse(content=None, tool_calls=[
                ToolCall(id="c1", name="search_fake", arguments={"q": "Chicago population"})
            ]),
            ToolResponse(content=None, tool_calls=[
                ToolCall(id="c2", name="calc_fake", arguments={"e": "1000000 / 50000"})
            ]),
            ToolResponse(content="The answer is 20.0", tool_calls=[]),
        ]
        mock_llm = mocker.MagicMock()
        mock_llm.chat_with_tools.side_effect = responses

        trace = Trace(directory=str(tmp_path), question="Q", provider="g", model="m")
        orch = Orchestrator(llm=mock_llm, trace=trace)
        result = orch.run("How many people per $50k of GDP in Chicago?")

        assert result.answer == "The answer is 20.0"
        assert result.steps_taken == 3
        assert len(result.tool_calls) == 2  # 2 tool calls, 1 final answer = 3 steps
```

- [ ] **Step 2: Run + commit**

```bash
pytest tests/test_end_to_end.py -v
git add tests/test_end_to_end.py
git commit -m "$(cat <<'EOF'
test(agent): end-to-end with scripted mock LLM

Verifies the full search → calc → answer loop with stubbed tools.
No real API calls. Confirms multi-step orchestration works.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 6.8: Push + open PR #6

```bash
pytest -v
git push -u origin feat/orchestrator
gh pr create --base main --title "feat(agent): orchestrator + trace + end-to-end loop" --body "$(cat <<'EOF'
## Summary

PR #6 of the multi-tool-agent build. The core agent loop:
- \`Orchestrator.run(question)\` → \`AgentResult\`
- Function-calling-native (uses chat_with_tools from PR #5)
- MAX_STEPS = 10, MAX_TOOL_RETRIES = 2 (per-dispatch, resets each step)
- Trace logger flushes JSON after every step
- System prompt at multitool/prompts/system.txt
- 8 orchestrator tests + 3 trace tests + 1 end-to-end test (scripted mock LLM)

## Test plan

- [ ] CI green (~12 new tests)
- [ ] End-to-end test verifies multi-step search → calc → answer

## Spec / plan reference

Spec §3.4 + §3.6 + §3.7
Plan Phase 6

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

# Phase 7 — PR #7: Eval test set + scorer + run.py

**Branch:** `feat/eval`
**Goal:** The 25 hand-curated queries + scorer + harness with checkpointing.
**Estimated LoC:** ~250 + ~80 tests
**Estimated time:** 2-3 hours (test-set authoring is the slow part)

### Task 7.1: Branch + author the 25 queries

**Files:**
- Create: `multitool/eval/test_set.jsonl`

- [ ] **Step 1: Branch**

```bash
git checkout main && git pull origin main
git checkout -b feat/eval
```

- [ ] **Step 2: Hand-author 25 queries**

Per spec §3.8, balanced across 5 categories. Each query MUST be multi-tool (≥2 tools expected). Each JSON object on its own line.

This is the most editorial step in the project — the queries you pick shape the eval signal. Here are 25 starter candidates; **edit, replace, or expand based on your judgment**:

Save to `multitool/eval/test_set.jsonl`:

```jsonl
{"id":"q01","question":"What is the population of Chicago divided by the US GDP per capita in 2023?","gold_answer":32.6,"tolerance":1.0,"answer_kind":"numeric","expected_tools":["tavily_search","tavily_search","calculator"],"category":"search-then-compute","difficulty":"medium"}
{"id":"q02","question":"How many seconds are in a year on Mars?","gold_answer":59354304,"tolerance":100000,"answer_kind":"numeric","expected_tools":["tavily_search","unit_convert"],"category":"unit-conversion","difficulty":"medium"}
{"id":"q03","question":"What is the speed of light in km/h?","gold_answer":1079252848,"tolerance":1000000,"answer_kind":"numeric","expected_tools":["unit_convert"],"category":"unit-conversion","difficulty":"easy"}
{"id":"q04","question":"What year was Albert Einstein born, and how many years before the founding of Apple Inc.?","gold_answer":97,"tolerance":1,"answer_kind":"numeric","expected_tools":["wikipedia","tavily_search","datetime_tool"],"category":"datetime-reasoning","difficulty":"medium"}
{"id":"q05","question":"What is the boiling point of water in Fahrenheit?","gold_answer":212,"tolerance":1,"answer_kind":"numeric","expected_tools":["unit_convert"],"category":"unit-conversion","difficulty":"easy"}
{"id":"q06","question":"How many years between the iPhone launch and the iPad launch?","gold_answer":3,"tolerance":0,"answer_kind":"numeric","expected_tools":["tavily_search","tavily_search","datetime_tool"],"category":"datetime-reasoning","difficulty":"medium"}
{"id":"q07","question":"What is the area of the United States in square kilometers, divided by Texas's area in square kilometers?","gold_answer":13.6,"tolerance":1.0,"answer_kind":"numeric","expected_tools":["tavily_search","tavily_search","calculator"],"category":"search-then-compute","difficulty":"medium"}
{"id":"q08","question":"What is the average distance from Earth to the Moon in miles?","gold_answer":238900,"tolerance":5000,"answer_kind":"numeric","expected_tools":["tavily_search","unit_convert"],"category":"unit-conversion","difficulty":"easy"}
{"id":"q09","question":"What day of the week was July 20, 1969 (the moon landing)?","gold_answer":"Sunday","tolerance":null,"answer_kind":"string","expected_tools":["datetime_tool"],"category":"datetime-reasoning","difficulty":"easy"}
{"id":"q10","question":"Who wrote 'One Hundred Years of Solitude' and what year were they born?","gold_answer":"1927","tolerance":null,"answer_kind":"string","expected_tools":["wikipedia"],"category":"multi-search-synthesis","difficulty":"easy"}
{"id":"q11","question":"What is the population of Tokyo divided by the population of New York City?","gold_answer":1.6,"tolerance":0.5,"answer_kind":"numeric","expected_tools":["tavily_search","tavily_search","calculator"],"category":"search-then-compute","difficulty":"easy"}
{"id":"q12","question":"Convert 70 miles per hour to meters per second.","gold_answer":31.3,"tolerance":0.5,"answer_kind":"numeric","expected_tools":["unit_convert"],"category":"unit-conversion","difficulty":"easy"}
{"id":"q13","question":"What is the GDP of Japan in 2023, divided by Japan's population, divided by the average annual salary in the US?","gold_answer":0.5,"tolerance":0.3,"answer_kind":"numeric","expected_tools":["tavily_search","tavily_search","tavily_search","calculator","calculator"],"category":"multi-tool-freestyle","difficulty":"hard"}
{"id":"q14","question":"What year was the Eiffel Tower built, and how many years ago was that from 2024?","gold_answer":135,"tolerance":1,"answer_kind":"numeric","expected_tools":["wikipedia","datetime_tool"],"category":"datetime-reasoning","difficulty":"easy"}
{"id":"q15","question":"How many minutes would it take to drive 2,800 miles at an average highway speed of 65 mph?","gold_answer":2585,"tolerance":50,"answer_kind":"numeric","expected_tools":["calculator","unit_convert"],"category":"multi-tool-freestyle","difficulty":"medium"}
{"id":"q16","question":"What is the population of Mumbai, and is it higher or lower than the population of Mexico City?","gold_answer":"higher","tolerance":null,"answer_kind":"string","expected_tools":["tavily_search","tavily_search"],"category":"multi-search-synthesis","difficulty":"easy"}
{"id":"q17","question":"How many years between the publishing of 'Pride and Prejudice' and 'Wuthering Heights'?","gold_answer":34,"tolerance":1,"answer_kind":"numeric","expected_tools":["wikipedia","wikipedia","datetime_tool"],"category":"multi-search-synthesis","difficulty":"medium"}
{"id":"q18","question":"What is the diameter of Jupiter divided by the diameter of Earth?","gold_answer":11.2,"tolerance":0.5,"answer_kind":"numeric","expected_tools":["tavily_search","tavily_search","calculator"],"category":"search-then-compute","difficulty":"medium"}
{"id":"q19","question":"How many days between January 1, 2000 and January 1, 2024?","gold_answer":8766,"tolerance":2,"answer_kind":"numeric","expected_tools":["datetime_tool"],"category":"datetime-reasoning","difficulty":"easy"}
{"id":"q20","question":"What is 50 kg in pounds?","gold_answer":110.2,"tolerance":0.5,"answer_kind":"numeric","expected_tools":["unit_convert"],"category":"unit-conversion","difficulty":"easy"}
{"id":"q21","question":"What is the population density of Singapore in people per square kilometer?","gold_answer":8500,"tolerance":500,"answer_kind":"numeric","expected_tools":["tavily_search","tavily_search","calculator"],"category":"search-then-compute","difficulty":"medium"}
{"id":"q22","question":"What is the average distance from Earth to the Sun in millions of kilometers?","gold_answer":150,"tolerance":5,"answer_kind":"numeric","expected_tools":["tavily_search","unit_convert"],"category":"unit-conversion","difficulty":"easy"}
{"id":"q23","question":"Who won the 2023 Nobel Prize in Literature and what country are they from?","gold_answer":"Norway","tolerance":null,"answer_kind":"string","expected_tools":["tavily_search"],"category":"multi-search-synthesis","difficulty":"easy"}
{"id":"q24","question":"What is the square root of the population of Sydney, Australia?","gold_answer":2354,"tolerance":100,"answer_kind":"numeric","expected_tools":["tavily_search","calculator"],"category":"search-then-compute","difficulty":"easy"}
{"id":"q25","question":"How many gigaseconds in 100 years?","gold_answer":3.15,"tolerance":0.05,"answer_kind":"numeric","expected_tools":["unit_convert","calculator"],"category":"multi-tool-freestyle","difficulty":"medium"}
```

- [ ] **Step 3: Commit**

```bash
git add multitool/eval/test_set.jsonl
git commit -m "$(cat <<'EOF'
feat(eval): 25-query hand-curated test set, balanced across 5 categories

Each query is multi-tool (≥2 expected tool invocations). Categories:
- search-then-compute (6)
- unit-conversion (6)
- datetime-reasoning (5)
- multi-search-synthesis (4)
- multi-tool-freestyle (4)

Per-question tolerance is calibrated. Numeric questions use absolute
tolerance; string questions use case-insensitive substring match.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 7.2: Scorer with `_parse_number`

**Files:**
- Create: `multitool/eval/scorer.py`
- Create: `tests/test_eval.py`

- [ ] **Step 1: Write failing tests**

In `tests/test_eval.py`:

```python
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
        r = score(predicted="The author is Norwegian (Jon Fosse).", gold="Norway", kind="string")
        assert r["passed"] is True

    def test_string_case_insensitive(self):
        from multitool.eval.scorer import score
        r = score(predicted="thursday", gold="Thursday", kind="string")
        assert r["passed"] is True
```

- [ ] **Step 2: Implement `multitool/eval/scorer.py`**

```python
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
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/test_eval.py -v
git add multitool/eval/scorer.py tests/test_eval.py
git commit -m "$(cat <<'EOF'
feat(eval): scorer with first-float-in-string heuristic

Spec §3.8 numeric/string/list scoring. _parse_number handles commas,
negatives, scientific notation. 11 tests pinning behavior.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 7.3: Eval runner

**Files:**
- Create: `multitool/eval/run.py`
- Modify: `tests/test_eval.py`

- [ ] **Step 1: Write failing test for the JSONL loader**

```python
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
```

- [ ] **Step 2: Implement the runner**

```python
"""Eval runner. Loads test set, runs agent on each query, scores, checkpoints
after every task (resumable on quota crash, like DA Agent's eval)."""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from multitool.eval.scorer import score


def load_test_set(path: str) -> list[dict]:
    """Load JSONL test set; one dict per line."""
    queries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            queries.append(json.loads(line))
    return queries


def run_eval(
    queries: list[dict],
    orchestrator_factory,        # Callable[[], Orchestrator]; fresh agent per query
    results_path: str,
) -> dict:
    """Run agent on each query, score, write checkpoint after each.

    Returns summary: {total_attempted, total_passed, pass_rate}.

    Checkpoint format (overwritten after every task):
      {
        "started_at": ISO,
        "results": [{id, question, passed, predicted, gold, ...}],
        "total_attempted": N,
        "total_passed": M,
      }
    """
    results = []
    summary = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "total_attempted": 0,
        "total_passed": 0,
    }

    def checkpoint():
        Path(results_path).write_text(json.dumps(summary, indent=2))

    checkpoint()

    for q in queries:
        try:
            orch = orchestrator_factory()
            agent_result = orch.run(q["question"])
            predicted = agent_result.answer or ""
            scored = score(
                predicted=predicted,
                gold=q["gold_answer"],
                kind=q["answer_kind"],
                tolerance=q.get("tolerance"),
            )
            entry = {
                "id": q["id"],
                "question": q["question"],
                "gold": q["gold_answer"],
                "predicted": predicted,
                "passed": scored["passed"],
                "category": q.get("category"),
                "tool_calls": agent_result.tool_calls,
                "error": agent_result.error,
                "parse_error": scored.get("parse_error"),
            }
        except Exception as e:
            entry = {
                "id": q["id"],
                "question": q["question"],
                "gold": q["gold_answer"],
                "predicted": "",
                "passed": False,
                "category": q.get("category"),
                "error": f"crash: {type(e).__name__}: {e}",
            }
        results.append(entry)
        summary["total_attempted"] = len(results)
        summary["total_passed"] = sum(1 for r in results if r["passed"])
        checkpoint()

    summary["pass_rate"] = summary["total_passed"] / summary["total_attempted"]
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    checkpoint()
    return summary


def main():
    """CLI entrypoint: python -m multitool.eval.run --test-set X --results Y"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-set", default="multitool/eval/test_set.jsonl")
    parser.add_argument("--results", default="eval_results.json")
    args = parser.parse_args()

    # Build a fresh-orchestrator factory
    from multitool.orchestrator import Orchestrator
    from multitool.llm_client import GroqClient
    from multitool.trace import Trace

    def factory():
        llm = GroqClient(api_key=os.environ["GROQ_API_KEY"])
        trace = Trace(directory="traces", question="", provider="groq", model=llm.model)
        return Orchestrator(llm=llm, trace=trace)

    queries = load_test_set(args.test_set)
    summary = run_eval(queries, factory, args.results)
    print(f"\nFinal: {summary['total_passed']}/{summary['total_attempted']} = {summary['pass_rate']:.1%}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/test_eval.py -v
git add multitool/eval/run.py tests/test_eval.py
git commit -m "$(cat <<'EOF'
feat(eval): run.py harness with per-task checkpointing

Resumable on quota crash. CLI entrypoint:
  python -m multitool.eval.run --test-set X --results Y

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 7.4: Push + open PR #7

```bash
pytest -v
git push -u origin feat/eval
gh pr create --base main --title "feat(eval): test set + scorer + run.py harness" --body "$(cat <<'EOF'
## Summary

PR #7 of the multi-tool-agent build. The eval system:
- 25 hand-curated multi-step queries balanced across 5 categories
- Scorer with first-float-in-string heuristic (spec §3.8)
- Runner with per-task checkpointing (resumable on quota crash)
- CLI: \`python -m multitool.eval.run --test-set X --results Y\`

## Test plan

- [ ] CI green (~12 new tests)
- [ ] Manual: \`set -a; source .env; set +a; python -m multitool.eval.run\` runs against real Groq (warning: burns API quota)

## Spec / plan reference

Spec §3.8
Plan Phase 7

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

# Phase 8 — PR #8: CLI

**Branch:** `feat/cli`
**Goal:** Replace the PR #1 CLI stub with real arg parsing + orchestrator dispatch.
**Estimated LoC:** ~80 + ~40 tests

### Task 8.1: Branch + implement real CLI

**Files:**
- Modify: `multitool/cli.py`
- Create/modify: `tests/test_cli.py`

- [ ] **Step 1: Branch**

```bash
git checkout main && git pull origin main
git checkout -b feat/cli
```

- [ ] **Step 2: Write failing test**

In `tests/test_cli.py`:

```python
"""Tests for the CLI entrypoint."""
import pytest


class TestCli:

    def test_ask_command_runs_agent(self, mocker, capsys):
        from multitool import cli

        # Mock the orchestrator + LLM so no real API calls
        mock_result = mocker.MagicMock()
        mock_result.answer = "42"
        mock_result.steps_taken = 1
        mock_result.error = None
        mock_result.trace_path = "/tmp/x.json"

        mocker.patch.object(cli, "_run_agent", return_value=mock_result)

        rc = cli.main(["ask", "What is the meaning of life?"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "42" in captured.out

    def test_no_args_prints_help(self, capsys):
        from multitool import cli
        rc = cli.main([])
        assert rc != 0  # help-and-exit returns nonzero
        captured = capsys.readouterr()
        assert "usage" in captured.out.lower() or "usage" in captured.err.lower()
```

- [ ] **Step 3: Implement `multitool/cli.py`**

```python
"""CLI entrypoint: `multitool ask "<question>"`."""
import argparse
import os
import sys
from typing import Optional


def _run_agent(question: str, provider: str):
    """Build orchestrator + run. Separated so tests can mock."""
    from multitool.orchestrator import Orchestrator
    from multitool.llm_client import GroqClient, GeminiClient
    from multitool.trace import Trace
    # Trigger tool registrations
    import multitool.tools.search       # noqa: F401
    import multitool.tools.calculator   # noqa: F401
    import multitool.tools.datetime_tool # noqa: F401
    import multitool.tools.unit_convert # noqa: F401
    import multitool.tools.wikipedia    # noqa: F401

    if provider == "groq":
        llm = GroqClient(api_key=os.environ["GROQ_API_KEY"])
    elif provider == "gemini":
        llm = GeminiClient(api_key=os.environ["GEMINI_API_KEY"])
    else:
        raise SystemExit(f"Unknown provider: {provider}")
    trace = Trace(directory="traces", question=question, provider=provider, model=llm.model)
    orch = Orchestrator(llm=llm, trace=trace)
    return orch.run(question)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="multitool")
    sub = parser.add_subparsers(dest="cmd", required=True)
    ask = sub.add_parser("ask", help="Ask the agent a question")
    ask.add_argument("question", help="The question to answer")
    ask.add_argument("--provider", default="groq", choices=["groq", "gemini"])

    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        parser.print_help()
        return 1

    args = parser.parse_args(argv)
    if args.cmd == "ask":
        result = _run_agent(args.question, args.provider)
        if result.error:
            print(f"Error: {result.error}", file=sys.stderr)
            return 1
        print(result.answer)
        print(f"\n[steps: {result.steps_taken}, trace: {result.trace_path}]", file=sys.stderr)
        return 0
    return 0
```

- [ ] **Step 4: Run + commit + push + PR**

```bash
pytest tests/test_cli.py -v
git add multitool/cli.py tests/test_cli.py
git commit -m "$(cat <<'EOF'
feat(cli): multitool ask "<question>" entrypoint

Real implementation replaces the PR #1 stub. Imports all 5 tools to
trigger @tool decorator side effects. Provider selectable via --provider.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
git push -u origin feat/cli
gh pr create --base main --title "feat(cli): multitool ask command" --body "PR #8. Real CLI replaces PR #1 stub. ~4 tests."
```

---

# Phase 9 — PR #9: Streamlit demo

**Branch:** `feat/demo`
**Goal:** Web UI with text input + trace UI + provider selector + example queries gallery.
**Estimated LoC:** ~250

### Task 9.1: Branch + implement Streamlit app

**Files:**
- Create: `demo/app.py`

- [ ] **Step 1: Branch + create demo/app.py**

```bash
git checkout main && git pull origin main
git checkout -b feat/demo
mkdir -p demo
```

Then create `demo/app.py`:

```python
"""Streamlit demo for the multi-tool agent.

Uses the same sys.path bootstrap pattern as DA Agent / RAG to handle
HF Spaces' install-requirements-before-source flow.
"""
import os
import sys
from pathlib import Path

# Bootstrap: add repo root to sys.path so `import multitool` works on HF Spaces
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from multitool.orchestrator import Orchestrator
from multitool.llm_client import GroqClient, GeminiClient
from multitool.trace import Trace
# Trigger tool registrations
import multitool.tools.search       # noqa: F401
import multitool.tools.calculator   # noqa: F401
import multitool.tools.datetime_tool # noqa: F401
import multitool.tools.unit_convert # noqa: F401
import multitool.tools.wikipedia    # noqa: F401


EXAMPLE_QUERIES = [
    "What is the population of Chicago divided by the US GDP per capita in 2023?",
    "How many years between the iPhone launch and the iPad launch?",
    "Convert 70 miles per hour to meters per second.",
    "What day of the week was July 20, 1969?",
    "Who won the 2023 Nobel Prize in Literature and what country are they from?",
]


def main():
    st.set_page_config(page_title="Multi-Tool Agent", layout="wide")
    st.title("Multi-Tool AI Agent")
    st.caption(
        "ReAct-style autonomous agent with function-calling-native dispatch. "
        "5 tools: web search, calculator, datetime, unit conversion, Wikipedia."
    )

    with st.sidebar:
        st.subheader("Settings")
        provider = st.selectbox("LLM provider", options=["groq", "gemini"], index=0)
        st.markdown("---")
        st.subheader("Example queries")
        for q in EXAMPLE_QUERIES:
            if st.button(q, key=q):
                st.session_state.question = q

    # Initialize session state
    if "question" not in st.session_state:
        st.session_state.question = ""

    question = st.text_input(
        "Ask a question that needs multiple tools:",
        value=st.session_state.question,
        placeholder="e.g., What's the population of Chicago divided by US GDP per capita 2023?",
    )

    if st.button("Ask", type="primary") and question.strip():
        # Check for required keys
        if provider == "groq" and not os.environ.get("GROQ_API_KEY"):
            st.error("GROQ_API_KEY not set. Add it as a Space secret or .env entry.")
            st.stop()
        if provider == "gemini" and not os.environ.get("GEMINI_API_KEY"):
            st.error("GEMINI_API_KEY not set. Add it as a Space secret or .env entry.")
            st.stop()
        if not os.environ.get("TAVILY_API_KEY"):
            st.error("TAVILY_API_KEY not set. Add it as a Space secret or .env entry.")
            st.stop()

        with st.spinner("Running agent..."):
            if provider == "groq":
                llm = GroqClient(api_key=os.environ["GROQ_API_KEY"])
            else:
                llm = GeminiClient(api_key=os.environ["GEMINI_API_KEY"])
            trace = Trace(directory="traces", question=question, provider=provider, model=llm.model)
            orch = Orchestrator(llm=llm, trace=trace)
            result = orch.run(question)

        # Render the answer
        if result.answer:
            st.success(result.answer)
        if result.error:
            st.error(f"Agent error: {result.error}")

        # Render the trace
        with st.expander(f"View trace ({result.steps_taken} steps)", expanded=False):
            for i, call_log in enumerate(result.tool_calls):
                st.markdown(f"**Step {i+1}: `{call_log['name']}`**")
                st.code(f"args: {call_log['args']}", language="json")
                result_str = call_log.get("result", "")
                if len(result_str) > 500:
                    result_str = result_str[:500] + "\n... (truncated)"
                st.text(result_str)
                st.markdown("---")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Local smoke test**

```bash
streamlit run demo/app.py
```

Open `http://localhost:8501`, click an example query, click Ask, verify it works.

- [ ] **Step 3: Commit + push + open PR #9**

```bash
git add demo/app.py
git commit -m "$(cat <<'EOF'
feat(demo): Streamlit app with trace UI + example queries

Sidebar: provider selector + 5 clickable example queries.
Main: text input, Ask button, success/error rendering, expandable trace.

sys.path bootstrap handles HF Spaces install-before-source flow.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
git push -u origin feat/demo
gh pr create --base main --title "feat(demo): Streamlit app" --body "PR #9. Streamlit demo for local + HF Spaces."
```

---

# Phase 10 — PR #10: README + headline result

**Branch:** `docs/readme`
**Goal:** Project README with headline eval number from PR #7. Tells the project's story.
**Estimated time:** 60–90 min of focused writing

### Task 10.1: Branch + write README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Branch**

```bash
git checkout main && git pull origin main
git checkout -b docs/readme
```

- [ ] **Step 2: Run the eval to get the headline number**

```bash
set -a; source .env; set +a
python -m multitool.eval.run
# Note the X/25 = Y% from the output
```

- [ ] **Step 3: Write the README**

Structure (mirror RAG project's README pattern):

```markdown
# Multi-Tool AI Agent

[![tests](https://github.com/laharikarumanchi-AI-ML/multi-tool-agent/actions/workflows/test.yml/badge.svg)](https://github.com/laharikarumanchi-AI-ML/multi-tool-agent/actions/workflows/test.yml)

A ReAct-style autonomous agent that decomposes multi-step questions and dispatches the right tool for each step. **No LangChain** — built from-scratch in ~250 lines using Groq's function-calling API.

---

## Headline result

**X/25 = Y% pass rate** on a hand-curated multi-tool eval set, Llama-3.3-70B via Groq.

[full breakdown by category + worked examples table]

## Quick start

[code block: clone, venv, pip install, .env setup, `multitool ask "..."`]

## Architecture

[ASCII diagram of orchestrator → tools[5] / llm_client / trace]

## Design choices

- **No LangChain.** ~250 lines of orchestrator code.
- **Function-calling-native.** Groq's `tools=` API, not ReAct text-format parsing.
- **5 tools.** Search (Tavily), calculator (numexpr), datetime, unit_convert (pint), Wikipedia.
- **Eval-first.** 25-query hand-curated test set with calibrated tolerances.

## Tools

[table of 5 tools with brief descriptions]

## Eval

[how to reproduce the headline number; describe scorer methodology]

## Limitations

- ...

## What I'd do differently

[honest reflection — same voice as DA Agent README]

## Acknowledgments

LLMClient copied from sibling project [data-analysis-agent](https://github.com/laharikarumanchi-AI-ML/superpowers) with attribution.
```

Write the prose. Aim for ~2,000 words. Use first-person voice matching DA Agent's README.

- [ ] **Step 4: Commit + push + open PR #10**

```bash
git add README.md
git commit -m "$(cat <<'EOF'
docs: README with headline X/25 = Y% pass rate

Project writeup mirroring DA Agent + RAG project README patterns.
First-person, honest, eval-grounded.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
git push -u origin docs/readme
gh pr create --base main --title "docs: README with headline result" --body "PR #10. Final piece — writeup with the X/25 = Y% from the actual eval run."
```

---

# Cross-cutting reference

## Quality gates (every PR)

1. **CI green**: `pytest -v` passes on Python 3.11 via GitHub Actions
2. **Diff readable**: PR description + commit messages tell the story
3. **No fabrications**: every behavior claim has a corresponding test
4. **No real-API hits in default tests**: `@pytest.mark.slow` on anything that touches a real provider

## Skills to reference during execution

- `@superpowers:subagent-driven-development` — recommended execution mode for this plan
- `@superpowers:executing-plans` — alternative for inline execution
- `@superpowers:test-driven-development` — every task in this plan is a TDD cycle

## Risks + mitigations (from spec §5, restated)

| Risk | Mitigation |
|---|---|
| Tavily free-tier consumed during eval iteration | Eval runner checkpoints after every task; HF Spaces uses separate key |
| Groq quota crashes mid-eval | Resumable eval harness; switch to Gemini for completion |
| Function-calling format differs between Groq + Gemini | `chat_with_tools()` normalizes both into `ToolResponse` |
| Model loops on same failed (name, args) call until step ceiling | Accepted; step ceiling is backstop. See spec §3.6. |
| Tool returns prose model misinterprets as structured data | Document expected output in tool docstrings; monitor during eval |

## What's out of scope (from spec §8, restated)

Agent memory across runs · Parallel tool calls · Tool-call streaming in Streamlit · GAIA / AgentBench full eval · Fine-tuning · Docker isolation for demo · Multi-user Streamlit session state · Custom user-supplied tools at runtime

## Post-PR-10 follow-ups

After all 10 PRs merge:
1. Deploy to HF Spaces: `huggingface.co/spaces/laharikarumanchi/multi-tool-agent` (separate `deploy/hf-spaces` workflow branch, same pattern as DA Agent + RAG)
2. Update portfolio's `multi-tool-agent.mdx` stub when portfolio PR #7 of the v2 replication lands — rewrite Approach paragraph + flip `techStack` frontmatter to match the from-scratch + Groq + Tavily reality
3. Eval iteration: rerun on Gemini for comparison; add ablation rows (retry-off, single-provider, etc.) to the README results table
