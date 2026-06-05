"""Tavily search tool. Free-tier 1000 queries/month at https://tavily.com.

Error-wrapping convention for all tools in this package:
this tool raises whatever the underlying client raises (network errors,
JSON parse errors, etc.). The orchestrator's _dispatch_with_retry catches
Exception, formats it as an Observation string, and lets the model decide
to retry / switch tools / give up. Don't wrap in custom ToolError types here.
"""
import os
from . import tool

# Cached client so we don't re-instantiate per call. Reset to None in tests.
# Use `global _client` inside _get_client() — assignment-without-global would
# shadow the module attribute and the cache would never persist.
_client = None


def _get_client():
    """Lazy-initialize the Tavily client. Reads API key from env on first call.
    The `from tavily import TavilyClient` is deferred to first call so that
    pytest collection works on a fresh checkout without tavily-python installed."""
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
        if answer:
            lines.append("")  # blank-line separator between Summary and Sources
        lines.append("Sources:")
        for r in results:
            # Use `or` (not dict's default arg) for None-safety: dict.get("k", d)
            # only triggers d when the key is ABSENT, not when the value is None.
            # Tavily doesn't document that any field is guaranteed non-null.
            title = r.get("title") or "(no title)"
            url = r.get("url") or ""
            content = (r.get("content") or "")[:200]  # truncate snippets
            lines.append(f"- {title} <{url}>\n  {content}")
    return "\n".join(lines)
