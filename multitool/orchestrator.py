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

    def _dispatch_with_retry(self, call: ToolCall) -> str:
        """Per-tool retry: 2 attempts. If both fail, surface error as Observation
        — the model decides whether to retry, switch tools, or give up.

        Per spec §3.6: tools that return error STRINGS (calculator, datetime,
        unit_convert) flow through here unchanged — no retry, the string IS
        the Observation. Tools that RAISE (tavily_search, wikipedia) trigger
        the retry path, then surface as Observation on permanent failure."""
        for attempt in range(self.MAX_TOOL_RETRIES + 1):
            try:
                fn = TOOL_REGISTRY[call.name]["fn"]
                return str(fn(**call.arguments))
            except Exception as e:
                if attempt == self.MAX_TOOL_RETRIES:
                    return f"Tool error after {self.MAX_TOOL_RETRIES} retries: {type(e).__name__}: {e}"
