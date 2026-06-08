---
title: Multi-Tool AI Agent
emoji: 🤖
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: 1.31.0
app_file: app.py
pinned: false
---

# Multi-Tool AI Agent

> Function-calling-native agent that picks tools and chains them — multi-step
> reasoning over web search, calculation, datetime math, unit conversion,
> and Wikipedia.

**Tests:** 114 passing · **Python:** 3.11 · **Provider:** Groq (Llama-3.3-70B) or Gemini 2.0 Flash

---

## What this is

The third agent in my portfolio series (after
[data-analysis-agent](https://github.com/laharikarumanchi-AI-ML/superpowers)
and [document-qa-rag](https://github.com/laharikarumanchi-AI-ML/document-qa-rag)).
It answers one question: *given a query that needs multiple distinct tools —
search to find facts, math to combine them, maybe a unit conversion to make
them comparable — can a small Python agent loop do this reliably without
LangChain?*

The signature engineering choice is function-calling-native dispatch (Groq's
`tools=` API and Gemini's `functionDeclarations`), not text-format ReAct
parsing and not a framework. The agent loop is ~140 lines; the `@tool`
decorator auto-builds JSON Schema from type hints; per-tool error handling
follows a documented convention.

This is a portfolio project, not a production system. The eval is a
hand-curated 25-query set, not GAIA. The headline is honest about scope
(see [Evaluation](#evaluation)).

---

## Quick start

```bash
git clone https://github.com/laharikarumanchi-AI-ML/multi-tool-agent.git
cd multi-tool-agent
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .

# Two free-tier keys. Both auto-load from .env via python-dotenv.
echo "GROQ_API_KEY=gsk_..." > .env       # https://console.groq.com/keys
echo "TAVILY_API_KEY=tvly-..." >> .env   # https://app.tavily.com/

multitool ask "What's the square root of Sydney's population?"
# → "About 2,354."
```

Optional: `GEMINI_API_KEY` if you want to switch providers with
`--provider gemini`.

---

## The 5 tools

| Tool | Signature | Error convention |
|---|---|---|
| `tavily_search` | `(query: str) -> str` | RAISES on network errors (retried up to 2× per dispatch). |
| `calculator` | `(expression: str) -> str` | RETURNS error string on numexpr syntax errors. No retry — syntax errors don't self-heal. |
| `datetime_tool` | `(operation: str, date_or_year: str, extra: str \| None) -> str` | RETURNS error string on parse failures. |
| `unit_convert` | `(value: float, from_unit: str, to_unit: str) -> str` | RETURNS error string on `UndefinedUnitError`. |
| `wikipedia` | `(topic: str, sentences: int = 3) -> str` | RAISES on network errors (retried). |

The split between "raise" and "return error string" is load-bearing — see
[spec §3.6](docs/superpowers/specs/2026-06-04-multi-tool-agent-design.md#36-error-handling--3-tiers)
for why. Short version: network failures might fix themselves on retry;
syntax errors won't, and the model handles them better as Observations than
as exceptions.

Adding a 6th tool is a no-boilerplate operation: a function with a
docstring and type hints, decorated with `@tool`. The decorator generates
the OpenAI-compatible JSON Schema from the signature.

---

## Architecture

The orchestrator is a single `for step in range(MAX_STEPS)` loop. Each
iteration calls `llm.chat_with_tools(messages, tools=schemas)` and
inspects the response. Three-way termination:

1. **`tool_calls` present** → dispatch each call, append results as
   `role: tool` messages, continue.
2. **`content is not None`, no `tool_calls`** → final answer; return.
3. **Neither** → model produced nothing usable; return with
   `error="empty_response"` so the eval can distinguish this from a
   step-ceiling hit.

`MAX_STEPS = 10` is the hard backstop. Per-dispatch retry budget is
`MAX_TOOL_RETRIES = 2` (resets each step — the model might legitimately
re-call the same tool with different args after seeing an error). Tools
that follow the "return error string" convention skip the retry path —
the string IS the Observation.

The `@tool` decorator (`multitool/tools/__init__.py`) walks the function's
`inspect.signature()`, maps Python types to JSON Schema types via a closed
set (`str → string`, `int → integer`, `float → number`, `bool → boolean`,
plus `Optional[X]`), and registers the function in `TOOL_REGISTRY`. Missing
docstrings raise at decoration time — tool descriptions are load-bearing
for tool-selection accuracy.

### `tool_use_failed` retry

Groq parses the model's emitted function-call syntax server-side. When the
model emits raw Python expressions as arguments (`pint.Quantity("100°C")`
instead of valid JSON) or stacks multiple `<function=...>` blocks in one
assistant turn, Groq returns HTTP 400 with `error.code == "tool_use_failed"`.

The initial eval surfaced this as nine inscrutable `HTTPError` crashes —
`raise_for_status()` discards the JSON error body, so the cause was
invisible until we added `_log_http_error_body` and saw the actual code.
PR #17 promoted `tool_use_failed` to the retryable class (same 5-attempt
budget as 429/5xx); the next sample from the same model typically parses
cleanly. The classifier is precise — a 401 with `code == "invalid_api_key"`
still raises immediately.

[Full design spec](docs/superpowers/specs/2026-06-04-multi-tool-agent-design.md).

---

## CLI

```bash
multitool ask "<question>" [--provider groq|gemini] [--model NAME] [--trace DIR] [-v]
```

| Flag | Default | Purpose |
|---|---|---|
| `--provider` | `groq` | LLM provider. Requires `GROQ_API_KEY` or `GEMINI_API_KEY`. |
| `--model` | provider default | Override the model (e.g. `--model llama-3.1-8b-instant` for cheap smoke tests). |
| `--trace` | `traces/` | Directory for trace JSON files (always treated as a directory; created if missing). |
| `-v`, `--verbose` | off | Print each tool call to stderr; re-raise orchestrator exceptions with full traceback. |

Sample verbose run:

```
$ multitool ask "Convert 70 mph to m/s, then how many seconds to cover a marathon?" -v
About 605 seconds (≈10 minutes).
[tool 1] unit_convert({'value': 70, 'from_unit': 'mph', 'to_unit': 'm/s'}) -> 31.2928
[tool 2] tavily_search({'query': 'marathon distance in meters'}) -> {"results": [...], "answer": "A marathon is 42,195 metres..."}... [+842 chars]
[tool 3] calculator({'expression': '42195 / 31.2928'}) -> 1348.39
[done in 4 steps; trace: traces/9c3e2b1a-....json]
```

Exit codes: `0` success · `1` missing env var, agent raised, or
`max_steps_reached` · `2` argparse usage error.

---

## Demo

Streamlit app at `app.py`. **HF Spaces deploy is in progress** — once
deployed, the URL lands here. Run locally:

```bash
pip install -e ".[demo]"
streamlit run app.py
```

Sidebar selects provider; the page shows a final-answer block above an
expandable per-step tool-call timeline, with a "Download trace (JSON)"
button after each run.

---

## Evaluation

### Methodology

`multitool/eval/test_set.jsonl` is 25 hand-curated multi-step queries,
distributed across 5 categories:

| Category | Count | Example |
|---|---|---|
| search-then-compute | 6 | "What is the population of Chicago divided by the US GDP per capita in 2023?" |
| unit-conversion | 7 | "What is the speed of light in km/h?" |
| datetime-reasoning | 5 | "How many years between the iPhone launch and the iPad launch?" |
| multi-search-synthesis | 4 | "Who won the 2023 Nobel Prize in Literature and what country are they from?" |
| multi-tool-freestyle | 3 | "How many minutes would it take to drive 2,800 miles at an average highway speed of 65 mph?" |

Every query is multi-tool by design (≥2 expected tool calls). Single-tool
queries are out — they don't test the loop's value prop.

The scorer (`multitool/eval/scorer.py`) uses a first-float-in-string
heuristic for numeric answers (regex captures the first signed
number with optional commas and scientific notation; matches against
`gold ± tolerance`). String answers are case-insensitive substring
match. List answers are set equality. No partial credit; strict like DA
Agent's official InfiAgent scorer. The eval runner checkpoints after every
task — a quota crash mid-run is resumable.

### Current state

Initial eval against Llama-3.3-70B (Groq) scored **9 / 25**. The honest
read: **9 of the 16 failures were HTTP 400 crashes from Groq with
`error.code = "tool_use_failed"`**. That's a provider-side parser
rejecting malformed function-call syntax from the model, not the agent
loop being wrong about anything — and it's exactly the kind of failure
that retrying-the-same-sample fixes most of the time.

PR #17 added retry-on-`tool_use_failed` to `chat_with_tools()` (same
5-attempt budget as 429/5xx) plus the `_log_http_error_body` diagnostic
that surfaced the issue in the first place. The corrected headline number
lands after Groq's daily token budget resets and we re-run; the rerun is
queued.

Run it yourself:

```bash
python -m multitool.eval.run \
  --test-set multitool/eval/test_set.jsonl \
  --results eval_runs/my-run.json
```

---

## Project layout

```
multi-tool-agent/
├── app.py                          # Streamlit demo (HF Spaces entry point)
├── pyproject.toml
├── requirements.txt                # HF Spaces install target
├── multitool/
│   ├── cli.py                      # `multitool ask` entrypoint
│   ├── orchestrator.py             # The agent loop (~140 LoC)
│   ├── llm_client.py               # Groq + Gemini + chat_with_tools()
│   ├── trace.py                    # JSON trace log
│   ├── _env.py                     # .env loader (python-dotenv)
│   ├── prompts/system.txt          # Minimal system prompt
│   ├── tools/
│   │   ├── __init__.py             # @tool decorator + TOOL_REGISTRY
│   │   ├── search.py               # Tavily
│   │   ├── calculator.py           # numexpr
│   │   ├── datetime_tool.py        # stdlib datetime
│   │   ├── unit_convert.py         # pint
│   │   └── wikipedia.py            # wikipedia-api
│   └── eval/
│       ├── run.py                  # Eval runner (checkpointed)
│       ├── scorer.py               # First-float heuristic + substring match
│       └── test_set.jsonl          # 25 queries
├── tests/                          # pytest — 114 tests
└── docs/superpowers/
    ├── specs/                      # Design doc (single source of truth)
    └── plans/                      # Per-PR task breakdown
```

---

## Testing

```bash
.venv/bin/pytest -q
# 114 passed, 1 deselected
```

114 tests across orchestrator, LLM clients (Groq + Gemini, including the
`tool_use_failed` classifier), each of the 5 tools (happy + error path),
the `@tool` decorator's schema generation and type mapping, trace JSON
serialization (with API-key scrub for Tavily/Gemini URL leaks), CLI flags
and exit codes, end-to-end with a scripted mock LLM, the eval scorer's
parse-number edge cases, and the Streamlit app's import and run wiring.

CI runs `pytest -q` on every push and PR via
[`.github/workflows/test.yml`](.github/workflows/test.yml).

---

## What I'd do differently

Instrument the integration boundary from day one. The initial eval surfaced
nine inscrutable `HTTPError` crashes, and the actual root cause — Groq's
`tool_use_failed` 400 — was sitting in the response body the whole time.
`raise_for_status()` was throwing away that body. The
one-line `_log_http_error_body(response, label)` helper that prints
the first 2 KB of any 4xx/5xx body to stderr before the raise should have
shipped from the first commit, not after the eval revealed an inscrutable
crash that took a cURL session to diagnose. The same principle applies to
the Gemini URL-key scrub — it was added reactively after seeing the key
in an exception message. Both are now permanent diagnostics; neither
should have been retroactive.

---

## Links

- **GitHub:** <https://github.com/laharikarumanchi-AI-ML/multi-tool-agent>
- **HF Spaces demo:** TBD (deploy in progress)
- **Design spec:** [`docs/superpowers/specs/2026-06-04-multi-tool-agent-design.md`](docs/superpowers/specs/2026-06-04-multi-tool-agent-design.md)
- **Implementation plan:** [`docs/superpowers/plans/2026-06-04-multi-tool-agent-implementation.md`](docs/superpowers/plans/2026-06-04-multi-tool-agent-implementation.md)
- **Author:** [Lahari Karumanchi](https://laharikarumanchi.vercel.app/)
