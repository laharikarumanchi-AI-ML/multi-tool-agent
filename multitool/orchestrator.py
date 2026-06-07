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
