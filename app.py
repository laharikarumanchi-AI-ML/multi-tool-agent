"""Streamlit demo for the multi-tool agent.

Run locally:
    pip install -e ".[demo]"
    export GROQ_API_KEY=gsk_...
    export TAVILY_API_KEY=tvly-...
    streamlit run app.py

Lives at the repo root (not under demo/) per the Hugging Face Spaces
convention — Spaces looks for app.py at the root by default.

Mirrors the DA Agent / RAG demo style: sidebar settings, primary text
input, expandable per-step tool-call timeline, friendly error handling
instead of stack traces. Tool calls are batch-rendered after run()
returns; streaming would need a bigger change to the orchestrator.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Bootstrap: HF Spaces installs requirements.txt BEFORE mounting the source
# tree, so `pip install -e .` is unavailable. Add the repo root to sys.path
# so `from multitool.*` resolves regardless of CWD. Harmless locally.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from multitool.llm_client import GeminiClient, GroqClient
from multitool.orchestrator import Orchestrator
from multitool.trace import Trace

# Tool registration side-effects (each module's @tool decorator populates
# multitool.tools.TOOL_REGISTRY at import time). Without these the
# orchestrator sees an empty registry.
import multitool.tools.calculator    # noqa: F401
import multitool.tools.datetime_tool  # noqa: F401
import multitool.tools.search        # noqa: F401
import multitool.tools.unit_convert  # noqa: F401
import multitool.tools.wikipedia     # noqa: F401


# ───────────────────────── constants ───────────────────────────────────────

GITHUB_URL = "https://github.com/laharikarumanchi-AI-ML/multi-tool-agent"

# Curated example questions from multitool/eval/test_set.jsonl, picked for
# category variety so the demo showcases more than one tool path.
EXAMPLE_QUESTIONS = [
    "What is the population of Chicago divided by the US GDP per capita in 2023?",
    "How many years between the iPhone launch and the iPad launch?",
    "Convert 70 miles per hour to meters per second.",
    "What day of the week was July 20, 1969 (the moon landing)?",
    "Who won the 2023 Nobel Prize in Literature and what country are they from?",
    "What is the speed of light in km/h?",
    "How many years between the publishing of 'Pride and Prejudice' and 'Wuthering Heights'?",
]

_TRUNCATE_RESULT_AT = 500  # chars; per-step result preview length in the timeline


# ───────────────────────── helpers ─────────────────────────────────────────


def _has_api_key(provider: str) -> bool:
    """True if the environment has the provider's required key."""
    var = "GROQ_API_KEY" if provider == "groq" else "GEMINI_API_KEY"
    return bool(os.environ.get(var))


def _truncate(text: str, limit: int = _TRUNCATE_RESULT_AT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... (truncated, +{len(text) - limit} chars)"


def _build_llm(provider: str, model: str | None):
    """Instantiate the chosen LLM client. Caller must verify the API key
    is present first (we raise KeyError otherwise — handled upstream)."""
    if provider == "groq":
        api_key = os.environ["GROQ_API_KEY"]
        return GroqClient(api_key=api_key, model=model) if model else GroqClient(api_key=api_key)
    api_key = os.environ["GEMINI_API_KEY"]
    return GeminiClient(api_key=api_key, model=model) if model else GeminiClient(api_key=api_key)


def _run_agent(question: str, provider: str, model: str | None):
    """Build trace + orchestrator + run. Returns (AgentResult, Trace) so
    the caller can offer the trace file as a download."""
    llm = _build_llm(provider, model)
    trace = Trace(
        directory="traces",
        question=question,
        provider=provider,
        model=llm._model,
    )
    orch = Orchestrator(llm=llm, trace=trace)
    result = orch.run(question)
    return result, trace


# ───────────────────────── page setup ──────────────────────────────────────


st.set_page_config(
    page_title="Multi-Tool AI Agent",
    page_icon="🤖",
    layout="wide",
)

st.title("Multi-Tool AI Agent")
st.caption(
    "Ask a multi-step question; the agent picks tools and chains them. "
    "5 tools available: web search (Tavily), calculator, datetime, unit conversion, Wikipedia. "
    f"[Source on GitHub →]({GITHUB_URL})"
)


# ───────────────────────── sidebar ─────────────────────────────────────────


with st.sidebar:
    st.subheader("Settings")
    provider = st.radio(
        "LLM provider",
        options=["groq", "gemini"],
        index=0,
        format_func=lambda p: "Groq (Llama-3.3-70B)" if p == "groq" else "Gemini (2.0-flash)",
        help="Both providers use the same function-calling-native loop.",
    )
    with st.expander("Advanced", expanded=False):
        custom_model = st.text_input(
            "Model override (optional)",
            value="",
            placeholder="e.g. llama-3.1-8b-instant",
            help="Leave blank to use the provider default.",
        )
    custom_model = custom_model.strip() or None

    st.markdown("---")
    st.subheader("How it works")
    st.caption(
        "ReAct-style loop. Each step the model either calls a tool or returns "
        "a final answer. Max 10 steps. Built from scratch — no LangChain."
    )


# ───────────────────────── API key gate ────────────────────────────────────


key_present = _has_api_key(provider)
tavily_present = bool(os.environ.get("TAVILY_API_KEY"))

if not key_present:
    var = "GROQ_API_KEY" if provider == "groq" else "GEMINI_API_KEY"
    st.error(
        f"{var} is not configured. Add it in Settings → Secrets on HF Spaces, "
        f"or set as an environment variable locally."
    )
if not tavily_present:
    st.warning(
        "TAVILY_API_KEY is not set. The web-search tool will fail; other tools "
        "(calculator, datetime, unit conversion, Wikipedia) still work."
    )


# ───────────────────────── example chips ───────────────────────────────────


st.markdown("**Try an example:**")
chip_cols = st.columns(min(len(EXAMPLE_QUESTIONS), 4))
for i, example in enumerate(EXAMPLE_QUESTIONS):
    col = chip_cols[i % len(chip_cols)]
    # Short label for the button; full question is the tooltip + the value
    # we populate the text area with on click.
    label = example if len(example) <= 60 else example[:57] + "..."
    if col.button(label, key=f"example_{i}", help=example, use_container_width=True):
        st.session_state["question"] = example


# ───────────────────────── question input ──────────────────────────────────


if "question" not in st.session_state:
    st.session_state["question"] = ""

question = st.text_area(
    "Your question",
    key="question",
    height=100,
    placeholder="e.g. What is the population of Tokyo divided by the population of New York City?",
)

submit_disabled = not key_present or not question.strip()
submit_clicked = st.button(
    "Ask the agent",
    type="primary",
    disabled=submit_disabled,
    help="Disabled until you enter a question and configure the API key.",
)


# ───────────────────────── run ─────────────────────────────────────────────


if submit_clicked:
    with st.spinner("Agent is thinking..."):
        try:
            result, trace = _run_agent(question.strip(), provider, custom_model)
        except Exception as exc:
            st.error(
                f"Agent run failed: {type(exc).__name__}: {exc}. "
                f"Check your API keys and try again."
            )
            st.stop()

    # ─── final answer ───
    st.markdown("### Answer")
    if result.error == "max_steps_reached":
        st.warning(
            f"Agent hit the {result.steps_taken}-step limit without finalizing. "
            f"Try rephrasing or breaking the question into smaller steps."
        )
    elif result.error == "empty_response":
        st.warning("Model returned an empty response. Try again or rephrase.")
    elif not result.answer or not result.answer.strip():
        st.warning("Agent finished without producing an answer. Try rephrasing.")
    else:
        st.markdown(result.answer)

    # ─── tool-call timeline ───
    if result.tool_calls:
        st.markdown(f"### Tool-call timeline ({len(result.tool_calls)} step{'s' if len(result.tool_calls) != 1 else ''})")
        for i, call in enumerate(result.tool_calls, start=1):
            name = call.get("name", "?")
            args = call.get("args", {})
            result_str = str(call.get("result", ""))
            with st.expander(f"Step {i} · `{name}`", expanded=False):
                st.markdown("**args**")
                st.code(json.dumps(args, indent=2, ensure_ascii=False), language="json")
                st.markdown("**result**")
                st.text(_truncate(result_str))

    # ─── trace download ───
    trace_path = Path(trace.path)
    if trace_path.exists():
        try:
            trace_bytes = trace_path.read_bytes()
            st.download_button(
                "Download trace (JSON)",
                data=trace_bytes,
                file_name=trace_path.name,
                mime="application/json",
                help="Full structured trace of this run: every tool call, args, result, and timing.",
            )
        except OSError:
            # The trace file should always exist after run() — but on HF Spaces'
            # ephemeral filesystem there's no guarantee, so don't crash the UI.
            pass
