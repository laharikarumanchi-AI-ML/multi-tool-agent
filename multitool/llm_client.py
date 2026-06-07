"""LLM client abstraction (Groq + Gemini).

Originally written for the data-analysis-agent project; copied here verbatim
with this attribution header. The original lives at:
  https://github.com/laharikarumanchi-AI-ML/superpowers/blob/main/agent/llm_client.py

Extended here with chat_with_tools() for function-calling-native agent loops.
"""
from typing import Protocol
import json
import time
import requests


class LLMClient(Protocol):
    def chat(self, messages: list[dict], **kwargs) -> str: ...
    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        **kwargs,
    ) -> "ToolResponse": ...


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
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except requests.HTTPError as exc:
                last_exc = exc
                status = getattr(exc.response, "status_code", None)
                if status in (429, 500, 502, 503, 504) and attempt < self.MAX_ATTEMPTS - 1:
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
    ) -> "ToolResponse":
        """Call Groq's chat completions API with the tools= parameter.
        Returns a ToolResponse — either content (final answer) or tool_calls
        (parsed arguments dict, not JSON string)."""
        payload = {
            "model": self._model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
        }
        # Allow caller to override via kwargs (max_tokens, temperature, etc.)
        payload.update(kwargs)

        # NOTE: attrs are _api_key and _model (underscore-prefixed) on the
        # existing GroqClient — not api_key/model. Don't drift.
        for attempt in range(self.MAX_ATTEMPTS):
            response = requests.post(
                self.URL,
                headers={"Authorization": f"Bearer {self._api_key}"},
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
                resp.raise_for_status()
                data = resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except requests.HTTPError as exc:
                last_exc = exc
                status = getattr(exc.response, "status_code", None)
                if status in (429, 500, 502, 503, 504) and attempt < self.MAX_ATTEMPTS - 1:
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
