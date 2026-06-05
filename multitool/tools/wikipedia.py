"""Wikipedia summary tool using wikipedia-api.

Same cached-client pattern as tavily_search (PR #3): module-level _client,
lazy init in _get_client(), `global _client` because assignment otherwise
shadows the module attribute. Wikipedia-api doesn't require an API key
(just a User-Agent string).

Error-wrapping convention: tool raises if the underlying library raises
(network errors etc.); orchestrator's _dispatch_with_retry catches and
formats as Observation. The 'not found' case is a normal control-flow
return string, not an error.
"""
import re
from . import tool

_client = None


def _get_client():
    """Lazy-initialize the Wikipedia client.
    The `from wikipediaapi import ...` is deferred to first call so pytest
    collection works on a fresh checkout without wikipedia-api installed."""
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
    sentence_re = re.compile(r"(?<=[.!?])\s+")
    parts = sentence_re.split(page.summary)
    return " ".join(parts[:sentences])
