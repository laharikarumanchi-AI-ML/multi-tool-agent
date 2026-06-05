"""Tavily search tool. Free-tier 1000 queries/month at https://tavily.com."""
import os
from . import tool

# Cached client so we don't re-instantiate per call. Reset to None in tests.
_client = None


def _get_client():
    """Lazy-initialize the Tavily client. Reads API key from env on first call."""
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
        lines.append("\nSources:")
        for r in results:
            title = r.get("title", "(no title)")
            url = r.get("url", "")
            content = (r.get("content") or "")[:200]  # truncate snippets
            lines.append(f"- {title} <{url}>\n  {content}")
    return "\n".join(lines)
