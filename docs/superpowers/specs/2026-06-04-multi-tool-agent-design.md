# Multi-Tool AI Agent — design

**Author:** Lahari Karumanchi (paired w/ Claude)
**Date:** 2026-06-04
**Status:** Design — ready to plan
**Related portfolio MDX:** `portfolio/src/content/projects/multi-tool-agent.mdx` (currently a stub committing to LangChain ReAct — this design intentionally diverges; the MDX will be rewritten when portfolio PR #7 of the v2 replication lands)

## 1. Context

The third agent in the personal-portfolio agent series, alongside:
- `data-analysis-agent` (Lahari/) — code-as-action agent for CSV analysis, evaluated on InfiAgent-DABench
- `document-qa-rag` (document-qa-rag/) — retrieval-augmented Q&A with page-aware chunking + threshold abstention

This project is a **ReAct-style autonomous agent** that decomposes user queries into steps and selects the right tool (web search, calculator, datetime, unit conversion, Wikipedia summary) for each step. Multi-step tool reasoning with retry/error handling. The core question it answers: *"given a query that requires multiple distinct tools — search to find facts, math to combine them, maybe a unit conversion to make them comparable — can a small Python agent loop do this reliably without LangChain?"*

The signature engineering choice is the same as DA Agent: **no framework**. Build the loop. The signature *differentiation* from DA Agent is the dispatch mechanism: **function-calling-native** (Groq's `tools=` API) rather than text-format ReAct parsing. The portfolio's existing MDX stub for this project literally predicted this comparison: *"I'd compare against a function-calling-native loop... to see how much of the framework I actually need."* This project executes that comparison as a built artifact.

## 2. Locked-in decisions

| Decision | Choice | Rejected alternatives |
|---|---|---|
| **Framework** | From-scratch (~250 LoC); no LangChain | LangChain ReAct (matches stub); Hybrid prose+function-calling |
| **Loop architecture** | Function-calling-native (Groq `tools=` API; structured `tool_calls` array) | Pure ReAct text-format (DA Agent's parser); Hybrid Thought+function-calling |
| **Search API** | Tavily (1000 queries/month free, agent-optimized snippets) | Brave Search API; DuckDuckGo HTML scrape; multi-provider abstraction |
| **LLM provider** | DA Agent's `LLMClient` copied + extended with `chat_with_tools()` (Groq + Gemini) | Groq-only; Anthropic Claude; OpenAI |
| **Calculator** | `numexpr` (safe math expression evaluator, no `eval()`) | `eval()`; sympy; restricted-ast |
| **Tool registration** | `@tool` decorator pattern, auto-schema from signature + docstring | JSON schema files; subclass-based; manual registration |
| **Tool count for v1** | 5 (search + calculator + datetime + unit_convert + wikipedia) | 2-3 minimal; 8+ ambitious |
| **Eval approach** | Custom hand-curated test set of 25 multi-step queries with gold answers | HotpotQA subset; GAIA subset; no formal eval |
| **Demo** | Streamlit; HF Spaces deploy mirroring DA Agent / RAG pattern | CLI-only; FastAPI + custom UI |
| **Voice anchor** | DA Agent's README (first-person, honest about scope, no marketing) | New project-specific voice |

### 2.1 Net engineering identity

"No framework, function-calling-native, honest small-N eval" — same backbone as DA Agent + RAG, but the *specific* design choice this project executes is the ReAct-vs-function-calling comparison the portfolio stub already promised. That comparison is the story.

## 3. Design

### 3.1 Repo layout

Mirrors `document-qa-rag/`:

```
multi-tool-agent/
├── pyproject.toml
├── README.md
├── requirements.txt              # Streamlit + HF Spaces compatible
├── .github/workflows/test.yml    # pytest on push + PR
├── multitool/                    # the package
│   ├── __init__.py
│   ├── __main__.py               # python -m multitool ask "<question>"
│   ├── cli.py
│   ├── llm_client.py             # Copied from Lahari/agent/ + chat_with_tools()
│   ├── orchestrator.py           # Function-calling-native loop (~250 LoC)
│   ├── trace.py                  # JSON log of every (prompt, tool_calls, observations)
│   ├── tools/
│   │   ├── __init__.py           # @tool decorator + TOOL_REGISTRY
│   │   ├── search.py             # tavily_search
│   │   ├── calculator.py         # numexpr-based
│   │   ├── datetime_tool.py      # date math: "years between X and Y"
│   │   ├── unit_convert.py       # pint library
│   │   └── wikipedia.py          # wikipedia-api summary
│   └── eval/
│       ├── run.py                # Runs agent against test set, scores, writes JSON
│       ├── test_set.jsonl        # 25 hand-curated multi-step queries
│       └── scorer.py             # Numeric / string / list comparison
├── tests/                        # pytest (TDD per-PR, same as DA Agent)
│   ├── test_orchestrator.py
│   ├── test_llm_client.py
│   ├── test_tools.py
│   ├── test_trace.py
│   ├── test_end_to_end.py
│   └── test_eval.py
├── demo/
│   └── app.py                    # Streamlit, deployed to HF Spaces
└── docs/superpowers/
    ├── specs/                    # This document
    └── plans/                    # Generated by writing-plans skill
```

### 3.2 Tool registry — the `@tool` decorator

The decorator reads function signature + docstring to auto-build a JSON schema compatible with Groq's `tools=` API (OpenAI-compatible format). Adding a 6th tool later is a no-boilerplate operation:

```python
# multitool/tools/__init__.py
import inspect
from typing import Callable, Union, get_type_hints, get_origin, get_args
from types import NoneType

TOOL_REGISTRY: dict[str, dict] = {}

# Supported Python types → JSON Schema types. Closed set; anything else raises.
PY_TO_JSON_TYPE: dict[type, str] = {
    str:   "string",
    int:   "integer",
    float: "number",
    bool:  "boolean",
}

class UnsupportedToolTypeError(TypeError):
    """A tool's parameter has a type the decorator cannot map to JSON Schema."""

def _py_to_json_type(hint) -> str:
    """Map a Python type hint to a JSON Schema type string.
    Handles Optional[X] (= X | None) by stripping NoneType."""
    if get_origin(hint) is Union:
        args = tuple(a for a in get_args(hint) if a is not NoneType)
        if len(args) == 1:
            return _py_to_json_type(args[0])
        raise UnsupportedToolTypeError(f"Union of multiple non-None types: {hint}")
    if hint in PY_TO_JSON_TYPE:
        return PY_TO_JSON_TYPE[hint]
    raise UnsupportedToolTypeError(f"Unsupported tool parameter type: {hint!r}")

def tool(fn: Callable):
    """Decorator. Registers fn in TOOL_REGISTRY with auto-generated schema.
    Requires a docstring (raises ValueError if missing — tool descriptions
    are load-bearing for tool-selection accuracy)."""
    if not inspect.getdoc(fn):
        raise ValueError(
            f"@tool {fn.__name__}: docstring required (used as tool description)"
        )
    hints = get_type_hints(fn)
    sig = inspect.signature(fn)
    # `required` excludes parameters that have defaults (they're optional).
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

# Usage
@tool
def tavily_search(query: str) -> str:
    """Search the web for current information.
    Returns top results with URL + snippet + Tavily's pre-summarized answer."""
    ...
```

**Supported parameter types** (closed set; anything else raises `UnsupportedToolTypeError`):

| Python type | JSON Schema | Notes |
|---|---|---|
| `str` | `"string"` | |
| `int` | `"integer"` | |
| `float` | `"number"` | |
| `bool` | `"boolean"` | |
| `str \| None` (= `Optional[str]`) | `"string"` | NoneType stripped from union; outer field also excluded from `required` since it has default `None` |
| `int \| None`, `float \| None`, `bool \| None` | same as non-Optional variants | same rule |
| `list[X]`, `dict[X, Y]`, custom classes, unions of multiple non-None types | — | Raise `UnsupportedToolTypeError`. Out of scope for v1. |

**Decorator behaviors:**

- **Docstring required.** `@tool def foo(): ...` without docstring raises `ValueError` at decoration time. Test: `test_tools.py::test_decorator_requires_docstring`.
- **Optionals excluded from `required`.** Parameters with default values are listed in `properties` but excluded from `required`. Test: `test_tools.py::test_optional_param_not_required`.
- **Optional unions handled.** `str | None` → schema type `"string"` + excluded from `required`. Test: `test_tools.py::test_optional_str_schema`.

### 3.3 The 5 tools

| Tool | Function signature | Notes |
|---|---|---|
| `tavily_search` | `(query: str) -> str` | Tavily API; returns formatted results with URL + snippet + pre-summarized answer. API key from `TAVILY_API_KEY` env var. |
| `calculator` | `(expression: str) -> float \| str` | `numexpr.evaluate(expression)`; returns float or error message. Handles operators + - * / ** and common functions (sin, cos, log). |
| `datetime_tool` | `(operation: str, date_or_year: str, extra: str \| None = None) -> str` | Operations: `"years_between"`, `"add_years"`, `"day_of_week"`. Uses stdlib `datetime`. |
| `unit_convert` | `(value: float, from_unit: str, to_unit: str) -> float \| str` | `pint` library handles unit parsing + conversion. Returns float or error. |
| `wikipedia` | `(topic: str, sentences: int = 3) -> str` | `wikipedia-api` package; returns first N sentences of summary. |

Each tool is a single function in its own file under `multitool/tools/`. Each has a happy-path test + error-path test in `tests/test_tools.py`.

### 3.4 Agent loop (orchestrator)

```python
# multitool/orchestrator.py — function-calling-native

@dataclass
class AgentResult:
    answer: str | None
    steps_taken: int
    tool_calls: list[dict]      # for eval scoring + debugging
    error: str | None           # 'max_steps_reached' / 'tool_failed_permanently'
    trace_path: str

class Orchestrator:
    MAX_STEPS = 10
    MAX_TOOL_RETRIES = 2        # per-call retry budget

    def __init__(self, llm: LLMClient, trace: Trace):
        self.llm = llm
        self.trace = trace

    def run(self, question: str) -> AgentResult:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        tool_calls_log: list[dict] = []

        for step in range(self.MAX_STEPS):
            response = self.llm.chat_with_tools(messages, tools=self._all_schemas())

            # Termination: model returned content without tool_calls = final answer
            if response.content and not response.tool_calls:
                self.trace.log_final(step, response.content)
                return AgentResult(
                    answer=response.content,
                    steps_taken=step + 1,
                    tool_calls=tool_calls_log,
                    error=None,
                    trace_path=self.trace.path,
                )

            # Append assistant message with tool calls to history
            messages.append({"role": "assistant", "tool_calls": response.tool_calls})

            for call in response.tool_calls:
                result = self._dispatch_with_retry(call)
                tool_calls_log.append({"name": call.name, "args": call.arguments, "result": result})
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result,
                })
                self.trace.log_step(step, call, result)

        # Step ceiling reached without a final answer
        return AgentResult(
            answer=None,
            steps_taken=self.MAX_STEPS,
            tool_calls=tool_calls_log,
            error="max_steps_reached",
            trace_path=self.trace.path,
        )

    def _dispatch_with_retry(self, call) -> str:
        """Per-tool retry: 2 attempts. If both fail, surface error as Observation —
        the model decides whether to retry, switch tools, or give up."""
        for attempt in range(self.MAX_TOOL_RETRIES + 1):
            try:
                fn = TOOL_REGISTRY[call.name]["fn"]
                return str(fn(**call.arguments))
            except Exception as e:
                if attempt == self.MAX_TOOL_RETRIES:
                    return f"Tool error after {self.MAX_TOOL_RETRIES} retries: {e}"
```

System prompt is intentionally minimal — function-calling does the heavy lifting:

```
You are a research assistant with access to tools. For any question that requires
external information or computation, call the appropriate tools. Chain multiple
tool calls when needed. Once you have enough information to answer, respond
with your final answer in plain text (no tool calls).

If a tool returns an error, you can retry with adjusted arguments, switch to a
different tool, or — if no recovery is possible — explain what you could not
answer and why.
```

No `Thought:` / `Action:` scaffold. Groq's `tools=` API enforces the call format; we trust the model + the structured response.

### 3.5 LLMClient extension

`multitool/llm_client.py` is the DA Agent's `Lahari/agent/llm_client.py` copied verbatim (with attribution header pointing back to the source commit). Auditing the source: the existing Protocol exposes a **single method**, `chat()`. There is no `with_budget()` or `name()` — earlier drafts of this spec misrepresented the source as having those; corrected here.

Three changes to the copied file:

1. **Extend the Protocol** to declare `chat_with_tools()`.
2. **Implement `chat_with_tools()` on `GroqClient` and `GeminiClient`** — each handles its provider's tool-call response format and normalizes to the project's `ToolResponse` dataclass.
3. **No `name()` method needed.** For demo + trace purposes, the project uses `type(client).__name__` (yields `"GroqClient"` / `"GeminiClient"`) or an explicit `provider_label: str` passed alongside the client where a human-friendly name is needed.

```python
# Final Protocol shape after the extension
class LLMClient(Protocol):
    def chat(self, messages: list[dict], **kwargs) -> str: ...
    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        **kwargs,
    ) -> ToolResponse: ...

@dataclass
class ToolCall:
    id: str          # Provider-issued ID. Forwarded in the next message's tool_call_id.
    name: str        # Tool function name (matches a key in TOOL_REGISTRY).
    arguments: dict  # ALWAYS a parsed dict — chat_with_tools() handles json.loads.

@dataclass
class ToolResponse:
    content: str | None         # None if model used tool_calls only
    tool_calls: list[ToolCall]  # Empty list if model produced content directly
```

**Critical: `arguments` is always a parsed `dict`, never a string.** Groq's raw API response returns `"arguments"` as a JSON-encoded string (`{"function": {"arguments": "{\"query\": \"...\"}"}}`). The `chat_with_tools()` implementation is responsible for calling `json.loads(raw_arguments)` BEFORE constructing the `ToolCall`. Tests verify this — `test_llm_client.py::test_chat_with_tools_parses_arguments_to_dict`.

The orchestrator then calls `fn(**call.arguments)` without any further parsing.

#### Provider format normalization

Groq's response format ([docs](https://console.groq.com/docs/tool-use)) — OpenAI-compatible:

```json
{"choices": [{"message": {"tool_calls": [
    {"id": "call_abc", "function": {"name": "tavily_search", "arguments": "{\"query\": \"...\"}"}}
]}}]}
```

Gemini's format ([docs](https://ai.google.dev/api/docs/function-calling)) uses `functionCall` (camelCase, dict arguments not JSON string):

```json
{"candidates": [{"content": {"parts": [
    {"functionCall": {"name": "tavily_search", "args": {"query": "..."}}}
]}}]}
```

Gemini does NOT issue tool-call IDs; the Gemini client synthesizes them as `f"gemini-call-{uuid4().hex[:8]}"` so the rest of the loop (which uses `call.id` for the `tool_call_id` field) works uniformly. Test: `test_llm_client.py::test_gemini_synthesizes_call_ids`.

### 3.6 Error handling — 3 tiers

| Tier | Trigger | Response |
|---|---|---|
| **Tool errors — network class** (Tavily 5xx, Wikipedia network blip) | Per-call exception RAISED by the tool | Retry up to 2× with same args (might succeed on retry). If still failing, surface as `Observation`; model decides next step. |
| **Tool errors — deterministic class** (numexpr syntax error, pint UndefinedUnitError, datetime parse error) | Tool RETURNS an error STRING (no exception) | Single attempt. The error string IS the Observation; the model sees it and picks a different arg shape on the next step. Retrying with the same broken args is wasted Groq tokens — these errors don't fix themselves. |
| **LLM errors** (rate limit 429, timeout, network blip) | API call exception | DA Agent's `LLMClient` handles this — 5 attempts with `Retry-After`-aware backoff. Direct reuse, no new code. |
| **Format errors** (model returns malformed tool_calls JSON) | Parse exception in `chat_with_tools()` | Re-prompt with format reminder (`"Your previous response had malformed JSON. Please use the tools= schema."`); counts toward step ceiling. Rare with function-calling. |

#### Per-tool error convention (codified)

PR #4's code-quality review surfaced that the 5 tools split into two error-handling classes by their failure mode. The convention below is **load-bearing** — the orchestrator's `_dispatch_with_retry` relies on it:

| Tool | Convention | Why |
|---|---|---|
| `tavily_search` | RAISES on network errors | 5xx / connection-reset / timeout — retrying might succeed |
| `wikipedia` | RAISES on network errors | Same — wikipedia-api wraps a real HTTP client |
| `calculator` | RETURNS error string on numexpr errors | A syntax error doesn't fix itself on retry; model needs to see the error and rewrite the expression |
| `datetime_tool` | RETURNS error string on parse errors | Same — bad year format doesn't self-heal; model picks a different format |
| `unit_convert` | RETURNS error string on undefined unit | Same — `pint.UndefinedUnitError` doesn't self-heal; model picks a different unit name |

The PR #4 module docstrings already document this per-tool. The orchestrator should NOT wrap deterministic tools in additional try/except — they return their own error strings as valid string returns, which the orchestrator treats as Observations (not retries).

#### Repeated-failed-call loop — decision

The `MAX_TOOL_RETRIES = 2` budget is **per-dispatch, not per-(name, args)**. It resets every step. That means the model could, in theory, call the same broken tool with the same arguments 10 times in a row, burn through the step ceiling, and exit with `error="max_steps_reached"`.

**This is intentional.** The `MAX_STEPS = 10` ceiling is the only backstop. Two reasons:

1. **The model often legitimately re-tries the same tool with different args after seeing the first failure.** A "we already saw this (name, args) and it failed" guard would block that recovery path. The model getting *new information* (the error message as an Observation) often does change its next call.
2. **The repeat-call-with-same-args pathology is empirically rare** in function-calling-native loops. The model reads the error in the Observation and adjusts. If it doesn't, that's a finding worth surfacing as a `max_steps_reached` error, not silently masking with a guard.

Tests pin this down:
- `test_orchestrator.py::test_repeated_same_call_terminates_at_step_ceiling` — verifies the loop exits cleanly with `error="max_steps_reached"` when the model misbehaves.
- `test_orchestrator.py::test_per_dispatch_retry_budget_resets_each_step` — verifies the 2-attempt budget is per-call, not lifetime.

If empirically (during eval) the model does loop on the same call > 1× of total tasks, we revisit and add a per-(name, args) guard. Documented as a deferred improvement in §5 risk register.

### 3.7 Trace

Every run writes `traces/<run_id>.json`:

```json
{
  "run_id": "uuid",
  "question": "What is the population of Chicago divided by US GDP per capita 2023?",
  "started_at": "2026-06-04T15:30:00Z",
  "provider": "groq",
  "model": "llama-3.3-70b-versatile",
  "steps": [
    {
      "step": 0,
      "tool_calls": [{"name": "tavily_search", "args": {"query": "Chicago population 2023"}}],
      "results": ["{\"results\": [...], \"answer\": \"Chicago population in 2023 was 2,664,452\"}"]
    },
    {
      "step": 1,
      "tool_calls": [{"name": "tavily_search", "args": {"query": "US GDP per capita 2023"}}],
      "results": ["..."]
    },
    {
      "step": 2,
      "tool_calls": [{"name": "calculator", "args": {"expression": "2664452 / 81632"}}],
      "results": ["32.64"]
    }
  ],
  "final_answer": "About 32.64 people per dollar of GDP per capita.",
  "total_steps": 3,
  "error": null,
  "duration_ms": 4250
}
```

Used by:
- Eval runner — feeds final_answer + tool_calls to scorer
- Streamlit demo — renders trace as collapsible UI sections
- `__repr__` of `AgentResult` — for CLI users

### 3.8 Eval

#### Test set

`multitool/eval/test_set.jsonl` — 25 hand-curated queries, JSONL one-per-line:

```json
{
  "id": "q01",
  "question": "What is the population of Chicago divided by the US GDP per capita in 2023?",
  "gold_answer": 32.64,
  "tolerance": 1.0,
  "answer_kind": "numeric",
  "expected_tools": ["tavily_search", "tavily_search", "calculator"],
  "category": "search-then-compute",
  "difficulty": "medium"
}
```

**Per-question tolerance.** The example above uses `tolerance: 1.0` for illustration only; real tolerances are calibrated when the test set is authored. For ratio answers like q01 (gold ≈ 32.6), `tolerance: 0.5` is a defensible default — accommodates "32.64", "32.6", "33" rounding while rejecting "32" (which would imply meaningfully wrong inputs). The eval JSONL author MUST think about tolerance per question, not copy 1.0 across the board.

Balanced across 5 categories:

| Category | Count | Example | Tools tested |
|---|---|---|---|
| Search-then-compute | 6 | "Chicago population / US GDP per capita 2023" | search × N, calculator |
| Multi-search synthesis | 6 | "City where 2024 Nobel literature winner was born + its population" | search × N |
| Datetime reasoning | 5 | "How many years between Apple's founding and the iPhone launch?" | search, datetime |
| Unit conversion + facts | 4 | "Is the speed of sound at sea level higher than a Boeing 747's top speed in m/s?" | search, unit_convert, calculator |
| Multi-tool freestyle | 4 | "How many minutes would it take to drive NYC→LA at average highway speed?" | search, unit_convert, calculator |

Every query is multi-tool by design (≥2 tool calls expected). Single-tool queries are out — they don't test the loop's value prop.

#### Scorer

```python
# multitool/eval/scorer.py
import re
from typing import Any

# Captures the FIRST float-or-int in a string. Handles negatives, commas
# in big numbers ("2,664,452"), and scientific notation. Does NOT match
# numbers embedded in words (e.g., "iPhone3").
_NUMBER_RE = re.compile(r"(?<![A-Za-z])-?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?")

def _parse_number(text: str) -> float | None:
    """Extract the FIRST numeric value from a prose answer.

    Decision rationale: agent answers are typically headline-first
    ("About 32.64..."), so the first number is almost always the intended
    answer. Last-number / largest-number heuristics had higher error rates
    in informal testing on the DA Agent's trace JSONs.
    """
    m = _NUMBER_RE.search(text)
    if m is None:
        return None
    try:
        return float(m.group().replace(",", ""))
    except ValueError:
        return None

def score(predicted: str, gold: Any, kind: str, tolerance: float | None = None) -> dict:
    if kind == "numeric":
        parsed = _parse_number(predicted)
        if parsed is None:
            return {"passed": False, "predicted": predicted, "parse_error": "no_number_found"}
        return {"passed": abs(parsed - gold) <= tolerance, "predicted": parsed}
    if kind == "string":
        return {"passed": gold.lower() in predicted.lower(), "predicted": predicted}
    if kind == "list":
        return {"passed": set(predicted) == set(gold), "predicted": predicted}
```

**`_parse_number` decision: first-float-in-string.** When a prose answer contains multiple numbers (e.g., *"About 32.64 (or 33 depending on source)"*), the scorer takes the first match. Rationale:

- Agent answers from function-calling-native models are typically headline-first (the answer-number leads, qualifications follow).
- Last-number and largest-number heuristics had higher error rates in informal testing on DA Agent trace JSONs.
- The behavior is testable: `test_eval.py::test_parse_number_picks_first_when_ambiguous` pins it.

If `_parse_number` finds zero numbers in the response (e.g., the model answered *"I could not find this information"*), the result is `{"passed": false, "parse_error": "no_number_found"}`. The eval runner counts these toward the denominator.

No partial credit. Strict like DA Agent's official InfiAgent scorer.

#### Headline metric

`X / 25 = Y% pass rate`. Same shape as DA Agent's 75% ABQ. Honest reporting of failure modes (which categories were weakest, which tools the agent over-relied on) carries the writeup as much as the headline number.

#### Eval runner

`multitool/eval/run.py`:
- Loads test set
- Runs agent on each question
- Scores each answer
- Writes per-run JSON: `{run_id, model, results: [...], total_passed, total_attempted}`
- Checkpoints after every task (resumable on quota crash, like DA Agent's eval)

### 3.9 Tests (pytest)

| File | ~Tests | Covers |
|---|---|---|
| `test_orchestrator.py` | 8-10 | `run()` returns `AgentResult`; step ceiling; per-tool retry budget; final-answer detection (content + no tool_calls); trace integration |
| `test_llm_client.py` | 6-8 | New `chat_with_tools()` method; tool_calls parsing for Groq + Gemini; preserved retry/throttle behaviors from DA Agent client |
| `test_tools.py` | 10-12 | `@tool` decorator: schema generation, type mapping, required params; each tool's happy + error path; argument validation |
| `test_trace.py` | 3-4 | JSON serialization; URL key-scrub (Tavily API key won't leak via Response.url) |
| `test_end_to_end.py` | 3-4 | Full agent loop against a scripted mock LLM (no real API calls; mock returns canned tool_calls then final answer) |
| `test_eval.py` | 3-4 | JSONL loader; scorer correctness on known numeric/string/list cases; edge: parse failure → `passed: false` |

Target ~35-40 tests total. CI runs `pytest -v` on every push and PR.

### 3.10 Demo

`demo/app.py` — Streamlit app with:
- Single text input for the user's question
- "Ask" button → runs agent, shows live progress
- Expandable trace UI: each step as a collapsible section showing `(tool_name, args, result)`
- Final answer rendered at top once available
- Side panel: 5-6 example queries the user can click to populate the input
- Provider selector: Groq (default) / Gemini. Implemented as a Streamlit `selectbox` mapping label → `LLMClient` factory function (no `name()` method on the client needed; the label lives in the Streamlit code).

Deployed to HF Spaces using the same `sys.path` bootstrap pattern as DA Agent / RAG to avoid `-e .` install issues.

## 4. Implementation plan (rough PR sequence)

| PR | Title | Scope | LoC est |
|---|---|---|---|
| #1 | `feat: scaffold Python package + pyproject.toml + CI` | Empty `multitool/` skeleton, pytest config, `.github/workflows/test.yml` | ~150 |
| #2 | `feat(tools): @tool decorator + tool registry` | `tools/__init__.py` with decorator + auto-schema + 8-10 tests | ~150 |
| #3 | `feat(tools): tavily_search` | First tool implementation + 3 tests | ~120 |
| #4 | `feat(tools): calculator + datetime + unit_convert + wikipedia` | Remaining 4 tools + ~12 tests | ~300 |
| #5 | `feat(llm): LLMClient copied from DA Agent + chat_with_tools()` | Provider abstraction with tool support + 6 tests | ~250 |
| #6 | `feat(agent): orchestrator function-calling loop + retry budget` | Main loop + per-tool retry + trace integration + 8 tests | ~280 |
| #7 | `feat(eval): test set + scorer + run.py harness` | 25 JSONL queries + scorer + runner + 4 tests | ~250 |
| #8 | `feat(cli): python -m multitool ask + trace logging` | CLI entry point + 3 tests | ~80 |
| #9 | `feat(demo): Streamlit app with worked-examples gallery` | The web demo | ~250 |
| #10 | `docs: README + headline result + architecture diagram` | Writeup with eval numbers from PR #7 | docs only |

~10 PRs. RAG was 9 PRs; this is similar. Detailed task-by-task plan written by `writing-plans` skill.

After all 10 PRs:
- Deploy to HF Spaces (separate workflow branch `deploy/hf-spaces`, same pattern as DA Agent + RAG)
- Embed demo iframe in portfolio's multi-tool-agent.mdx (portfolio v2 PR #7 territory)

## 5. Risk register

| Risk | Mitigation |
|---|---|
| Tavily free-tier 1000 queries/month gets consumed during eval iteration | Eval runner checkpoints after every task (no re-running successful ones); HF Spaces demo uses separate Tavily key from local-dev key |
| Groq quota crashes mid-eval (same problem DA Agent hit) | Resumable eval harness; provider abstraction lets you switch to Gemini for completion |
| Function-calling format differs between Groq + Gemini | `LLMClient.chat_with_tools()` normalizes both into a single `ToolResponse` dataclass; per-provider impl handles the translation. Gemini synthesizes call IDs since the API doesn't issue them. |
| Model loops on same failed (name, args) call until step ceiling | Accepted (see §3.6 "Repeated-failed-call loop — decision"). If empirically >1% of eval tasks exhibit this pathology, add a per-(name, args) seen-set guard as a deferred follow-up. |
| Tool returns prose that model misinterprets as structured data | Tavily's pre-summarized `answer` field is freeform prose; the orchestrator just stringifies tool results. Tools document expected output shape in their docstrings; the model has been observed reliably treating Tavily snippets as prose, not JSON. Worth monitoring during eval. |
| Tool error masquerades as success (e.g., Tavily returns "no results found" — agent treats it as a fact) | Eval set explicitly includes queries where the right answer is "I don't know" (1 in v1); orchestrator system prompt allows graceful "I could not find this" answers |
| Agent gets stuck in loop calling the same failed tool | Per-tool retry capped at 2; step ceiling at 10; eval logs catch these as `max_steps_reached` errors |
| Portfolio MDX stub says "LangChain" but the real project doesn't use it | Documented in §1; portfolio PR #7 rewrites the Approach paragraph + flips `techStack` frontmatter to match reality |

## 6. Success criteria

The project succeeds when:

1. Eval headline number lands somewhere honest (50-75% pass rate). Anything below 40% suggests the design is wrong; anything above 85% suggests the test set is too easy.
2. The Streamlit demo handles all 5 example queries from the gallery without error.
3. Trace logs are readable enough that you can debug a failing eval question by reading the JSON.
4. CI is green (all 35-40 tests passing).
5. HF Spaces deploy lives at `huggingface.co/spaces/laharikarumanchi/multi-tool-agent`.
6. The portfolio's multi-tool-agent.mdx, when rewritten in PR #7 of the portfolio v2 replication, has a real headline number, a real architecture diagram, and a "Worked examples" section drawing from the actual eval JSON traces — the same shape as DA Agent's portfolio writeup.

## 7. Open questions

None — all design questions resolved in brainstorming. Implementation details (exact commit boundaries, system prompt wording, scorer parse heuristics) deferred to the implementation plan.

## 8. Out of scope (explicit)

| Item | Why excluded |
|---|---|
| Agent memory across runs | Each `ask` is stateless; conversational memory is a separate project |
| Parallel tool calls | Groq supports it; loop dispatches sequentially in v1 for simpler reasoning |
| Tool-call streaming in Streamlit | Adds UI complexity; defer |
| GAIA / AgentBench full eval | 466+ questions, prohibitive API costs; custom 25-query set is honest enough |
| Fine-tuning | Out of scope for "agent design + honest evaluation" framing |
| Docker isolation for demo | Multi-Tool's tools (search, calc, datetime, etc.) don't execute arbitrary code — safer than DA Agent's Jupyter sandbox by default |
| Multi-user session state in Streamlit | Single-session demo only |
| API key management beyond `.env` + HF secret | Sufficient for portfolio-tier shipping |
| Custom user-supplied tools at runtime | Adds plugin loading complexity; v2 territory |
