"""The agent loop. Function-calling-native; per-tool retry budget; step ceiling."""
import json
from dataclasses import dataclass
from pathlib import Path
from multitool.llm_client import LLMClient, ToolCall
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
    error: str | None           # 'max_steps_reached' / 'empty_response' / None
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
        # Fast-path: if the model hallucinated a tool name (rare with function-calling
        # but possible), don't burn the retry budget on a guaranteed KeyError.
        if call.name not in TOOL_REGISTRY:
            return f"Tool error: unknown tool {call.name!r}"

        for attempt in range(self.MAX_TOOL_RETRIES + 1):
            try:
                fn = TOOL_REGISTRY[call.name]["fn"]
                return str(fn(**call.arguments))
            except Exception as e:
                if attempt == self.MAX_TOOL_RETRIES:
                    return f"Tool error after {self.MAX_TOOL_RETRIES} retries: {type(e).__name__}: {e}"

    def _all_schemas(self) -> list[dict]:
        return [info["schema"] for info in TOOL_REGISTRY.values()]

    def run(self, question: str) -> AgentResult:
        """Run the agent loop until final answer OR step ceiling."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        tool_calls_log: list[dict] = []
        # Schemas are fixed at run start (TOOL_REGISTRY doesn't mutate during a run).
        # Hoist out of the loop to signal this and avoid recomputing 10x.
        tools = self._all_schemas()

        for step in range(self.MAX_STEPS):
            response = self.llm.chat_with_tools(messages, tools=tools)

            # Three-way termination check (explicit, not truthy fallthrough):
            # 1. tool_calls present → dispatch path
            # 2. content is not None (including "") → final answer
            # 3. neither → model produced nothing usable, log + exit
            if response.tool_calls:
                pass  # fall through to dispatch
            elif response.content is not None:
                # Final answer (possibly empty string — that's still a real response)
                self.trace.log_final(step, response.content)
                return AgentResult(
                    answer=response.content,
                    steps_taken=step + 1,
                    tool_calls=tool_calls_log,
                    error=None,
                    trace_path=self.trace.path,
                )
            else:
                # Both content and tool_calls are empty — model produced nothing.
                # Distinguishes "empty response" from "max_steps_reached" in the
                # eval's failure-mode breakdown.
                self.trace.log_error("empty_response")
                return AgentResult(
                    answer=None,
                    steps_taken=step + 1,
                    tool_calls=tool_calls_log,
                    error="empty_response",
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
