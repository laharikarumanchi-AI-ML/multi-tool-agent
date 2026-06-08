"""LLM client abstraction (Groq + Gemini).

Originally written for the data-analysis-agent project; copied here verbatim
with this attribution header. The original lives at:
  https://github.com/laharikarumanchi-AI-ML/superpowers/blob/main/agent/llm_client.py

Extended here with chat_with_tools() for function-calling-native agent loops.
"""
from dataclasses import dataclass
from typing import Protocol
from uuid import uuid4
import json
import sys
import time
import requests

# Status codes worth retrying on. Permanent failures (400, 401, 403, 404)
# don't fix themselves and shouldn't burn the retry budget.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

# Reserved kwargs that chat_with_tools() callers must NOT override —
# `model` is the one realistic conflict (caller could pass it via **kwargs
# from a config dict). `messages` and `tools` are method parameters, so
# Python's arg dispatcher catches duplicates before our guard runs. We
# guard `model` explicitly because the orchestrator could accidentally
# splat a config dict containing a stale model name.
_RESERVED_KWARGS = frozenset({"model"})


@dataclass
class ToolCall:
    """A structured tool invocation from the LLM.

    Note: arguments is ALWAYS a parsed dict, never a JSON string. Groq's raw
    API returns arguments as a JSON-encoded string; chat_with_tools()
    is responsible for json.loads()-ing it before constructing the ToolCall.

    For Gemini, `id` is synthesized client-side (Gemini doesn't issue them).
    The id is INTERNAL to the orchestrator's bookkeeping — Gemini's API
    keys tool responses by `name`, not `id`, so we don't round-trip it.
    """
    id: str          # Provider-issued (Groq) OR client-synthesized (Gemini)
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


def _log_http_error_body(response, source: str) -> None:
    """If the response is a 4xx/5xx, print the body to stderr before the
    caller raises. Groq's `raise_for_status()` discards the JSON error body,
    which made multi-turn tool-call bugs invisible — this restores visibility.

    Truncates at 2000 chars to keep stderr tidy on giant payloads.
    """
    if response is None or response.ok:
        return
    try:
        body = response.text[:2000]
    except Exception:
        body = "<unreadable body>"
    print(
        f"[llm_client] {source} HTTP {response.status_code}: {body}",
        file=sys.stderr,
    )


def _check_reserved_kwargs(kwargs: dict) -> None:
    """Raise ValueError if caller passed any kwarg that would clobber a
    contract-defined field of chat_with_tools()."""
    bad = _RESERVED_KWARGS & set(kwargs)
    if bad:
        raise ValueError(
            f"chat_with_tools(): cannot override reserved kwargs {sorted(bad)}; "
            f"use the method's named parameters instead."
        )


class LLMClient(Protocol):
    def chat(self, messages: list[dict], **kwargs) -> str: ...
    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        **kwargs,
    ) -> ToolResponse: ...


class GroqClient:
    URL = "https://api.groq.com/openai/v1/chat/completions"
    MAX_ATTEMPTS = 5
    BACKOFF_BASE_SECONDS = 2.0
    MAX_BACKOFF_SECONDS = 60.0

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile") -> None:
        self._api_key = api_key
        self._model = model

    def chat(self, messages: list[dict], **kwargs) -> str:
        payload = {"model": self._model, "messages": messages, **kwargs}
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        last_exc: Exception | None = None
        for attempt in range(self.MAX_ATTEMPTS):
            try:
                resp = requests.post(self.URL, headers=headers, json=payload, timeout=60)
                _log_http_error_body(resp, "Groq.chat")
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except requests.HTTPError as exc:
                last_exc = exc
                status = getattr(exc.response, "status_code", None)
                if status in _RETRYABLE_STATUS and attempt < self.MAX_ATTEMPTS - 1:
                    time.sleep(self._sleep_seconds(exc.response, attempt))
                    continue
                raise
        assert last_exc is not None
        raise last_exc

    def _sleep_seconds(self, response, attempt: int) -> float:
        """Respect server's Retry-After if present; otherwise exponential backoff."""
        if response is not None:
            ra = response.headers.get("Retry-After") if hasattr(response, "headers") else None
            if ra:
                try:
                    return min(float(ra), self.MAX_BACKOFF_SECONDS)
                except (TypeError, ValueError):
                    pass
        return min(self.BACKOFF_BASE_SECONDS * (2 ** attempt), self.MAX_BACKOFF_SECONDS)

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        **kwargs,
    ) -> ToolResponse:
        """Call Groq's chat completions API with the tools= parameter.
        Returns a ToolResponse — either content (final answer) or tool_calls
        (parsed arguments dict, not JSON string).

        kwargs may include `temperature`, `max_tokens`, `tool_choice`, etc.
        but MAY NOT override `model`, `messages`, or `tools` (raises ValueError).

        Retries ONLY on retryable statuses (429/5xx). 4xx (e.g., bad auth, malformed
        tool schema) raises immediately — retrying won't help and burns time.
        """
        _check_reserved_kwargs(kwargs)
        payload = {
            "model": self._model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            **kwargs,  # tool_choice override allowed (e.g., "required")
        }

        # NOTE: attrs are _api_key and _model (underscore-prefixed) on the
        # existing GroqClient — not api_key/model. Don't drift.
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        last_response = None
        for attempt in range(self.MAX_ATTEMPTS):
            response = requests.post(self.URL, headers=headers, json=payload, timeout=60)
            last_response = response
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
            # Only retry on retryable statuses; otherwise raise immediately.
            if response.status_code in _RETRYABLE_STATUS and attempt < self.MAX_ATTEMPTS - 1:
                time.sleep(self._sleep_seconds(response, attempt))
                continue
            _log_http_error_body(response, "Groq.chat_with_tools")
            response.raise_for_status()
        # Loop exhausted on retryable status; raise final response's error.
        if last_response is not None:
            _log_http_error_body(last_response, "Groq.chat_with_tools")
            last_response.raise_for_status()


class GeminiClient:
    """Google Gemini chat client.

    Free tier of gemini-2.0-flash is 15 RPM — much tighter than I assumed.
    Use min_seconds_between_calls to throttle to the published rate.
    """
    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    MAX_ATTEMPTS = 5
    BACKOFF_BASE_SECONDS = 2.0
    MAX_BACKOFF_SECONDS = 60.0

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash",
                 min_seconds_between_calls: float = 0.0) -> None:
        self._api_key = api_key
        self._model = model
        self._min_gap = min_seconds_between_calls
        self._last_call_monotonic: float = 0.0

    def _throttle(self) -> None:
        if self._min_gap <= 0:
            return
        elapsed = time.monotonic() - self._last_call_monotonic
        if elapsed < self._min_gap:
            time.sleep(self._min_gap - elapsed)

    @staticmethod
    def _to_gemini_format(messages: list[dict]) -> dict:
        """Convert OpenAI-style chat messages to Gemini's API shape:
        - system messages lift into top-level `system_instruction`
        - assistant role -> 'model'; user role -> 'user'
        - consecutive same-role turns are merged (Gemini rejects them otherwise)
        """
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        body: list[dict] = []
        for m in messages:
            if m["role"] == "system":
                continue
            role = "user" if m["role"] == "user" else "model"
            if body and body[-1]["role"] == role:
                body[-1]["parts"][0]["text"] += "\n\n" + m["content"]
            else:
                body.append({"role": role, "parts": [{"text": m["content"]}]})
        out: dict = {"contents": body}
        if system_parts:
            out["system_instruction"] = {
                "parts": [{"text": "\n\n".join(system_parts)}]
            }
        return out

    def chat(self, messages: list[dict], **kwargs) -> str:
        payload = self._to_gemini_format(messages)
        gen_cfg: dict = {}
        if "temperature" in kwargs:
            gen_cfg["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            gen_cfg["maxOutputTokens"] = kwargs["max_tokens"]
        if gen_cfg:
            payload["generationConfig"] = gen_cfg

        url = self.URL.format(model=self._model)
        last_exc: Exception | None = None
        for attempt in range(self.MAX_ATTEMPTS):
            self._throttle()
            try:
                resp = requests.post(url, params={"key": self._api_key},
                                     json=payload, timeout=60)
                self._last_call_monotonic = time.monotonic()
                # SCRUB: Gemini puts the API key in the URL query string.
                # `raise_for_status()` builds its message from resp.url, which
                # leaks the key to logs/exceptions/results files. Sanitize first.
                if "?key=" in resp.url or "&key=" in resp.url:
                    resp.url = resp.url.split("?")[0] + "?key=[REDACTED]"
                _log_http_error_body(resp, "Gemini.chat")
                resp.raise_for_status()
                data = resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except requests.HTTPError as exc:
                last_exc = exc
                status = getattr(exc.response, "status_code", None)
                if status in _RETRYABLE_STATUS and attempt < self.MAX_ATTEMPTS - 1:
                    time.sleep(self._sleep_seconds(exc.response, attempt))
                    continue
                raise
        assert last_exc is not None
        raise last_exc

    def _sleep_seconds(self, response, attempt: int) -> float:
        if response is not None:
            ra = response.headers.get("Retry-After") if hasattr(response, "headers") else None
            if ra:
                try:
                    return min(float(ra), self.MAX_BACKOFF_SECONDS)
                except (TypeError, ValueError):
                    pass
        return min(self.BACKOFF_BASE_SECONDS * (2 ** attempt), self.MAX_BACKOFF_SECONDS)

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        **kwargs,
    ) -> ToolResponse:
        """Call Gemini's generateContent with function-call tools.
        Synthesizes call IDs (Gemini doesn't issue them) so downstream
        code can use call.id uniformly. The id is INTERNAL to orchestrator
        bookkeeping — Gemini's tool responses are keyed by `name`, not `id`,
        so we don't round-trip it.

        kwargs may include `temperature`, `max_tokens`, etc. but MAY NOT
        override `model`, `messages`, or `tools` (raises ValueError).

        Retries ONLY on retryable statuses (429/5xx). 4xx raises immediately.
        """
        _check_reserved_kwargs(kwargs)

        # Gemini's tools format uses functionDeclarations.
        # NOTE: attrs are _api_key and _model (underscore-prefixed) on the
        # existing GeminiClient — not api_key/model. Don't drift.
        if not tools:
            gemini_tools = []
        else:
            gemini_tools = [{"functionDeclarations": [t["function"] for t in tools]}]

        # Preserve system_instruction routing from _to_gemini_format
        gemini_payload = self._to_gemini_format(messages)
        payload: dict = {
            "contents": gemini_payload["contents"],
            "tools": gemini_tools,
        }
        if "system_instruction" in gemini_payload:
            payload["system_instruction"] = gemini_payload["system_instruction"]
        payload.update(kwargs)

        url = self.URL.format(model=self._model)
        last_response = None
        for attempt in range(self.MAX_ATTEMPTS):
            self._throttle()
            response = requests.post(
                url,
                params={"key": self._api_key},
                json=payload,
                timeout=60,
            )
            self._last_call_monotonic = time.monotonic()
            # Guarded scrub — only mutate if the URL actually has the key.
            if "?key=" in response.url or "&key=" in response.url:
                response.url = response.url.split("?")[0] + "?key=[REDACTED]"
            last_response = response

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
                            id=f"gemini-call-{uuid4().hex}",  # full 32 hex chars
                            name=fc["name"],
                            arguments=fc.get("args", {}),  # Gemini returns dict, not string
                        ))
                return ToolResponse(content=content, tool_calls=tool_calls)
            # Only retry on retryable statuses; otherwise raise immediately.
            if response.status_code in _RETRYABLE_STATUS and attempt < self.MAX_ATTEMPTS - 1:
                time.sleep(self._sleep_seconds(response, attempt))
                continue
            _log_http_error_body(response, "Gemini.chat_with_tools")
            response.raise_for_status()
        if last_response is not None:
            _log_http_error_body(last_response, "Gemini.chat_with_tools")
            last_response.raise_for_status()
