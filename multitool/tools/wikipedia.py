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
    return _truncate_to_sentences(page.summary, sentences)


# Common English abbreviations whose trailing "." should NOT trigger a
# sentence break. Lowercased for case-insensitive matching of the word
# immediately preceding a period. Doesn't need to be exhaustive — covers
# the high-frequency cases that wreck typical Wikipedia summaries.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr",
    "st", "mt", "fr", "rev",
    "u.s", "u.k", "u.s.a", "u.k.a", "e.g", "i.e", "etc", "vs", "no",
    "inc", "ltd", "co", "corp", "ave", "blvd",
}


def _truncate_to_sentences(text: str, n: int) -> str:
    """Return the first n sentences of text, skipping false-positive splits
    at common abbreviations like "U.S.", "Dr.", "i.e." that the naive
    regex `(?<=[.!?])\\s+` would otherwise mangle.

    Strategy: split at every `[.!?]\\s+` candidate, then merge backward
    whenever the token immediately before the punctuation is a known
    abbreviation. This is intentionally heuristic — perfect English
    sentence tokenization is unsolved; the goal is "don't produce
    a fragment in the middle of 'U.S.A.'" for a typical Wikipedia summary.
    """
    # Candidates: every sentence-ending punctuation followed by whitespace.
    candidates = list(re.finditer(r"([.!?])\s+", text))
    if not candidates:
        return text  # no splits to make; return everything

    sentence_ends: list[int] = []   # index in text where each sentence ends (exclusive)
    for m in candidates:
        # The word immediately preceding the punctuation
        end = m.start()  # index of the [.!?]
        # Walk backward to find the start of the preceding word/abbrev
        word_start = end
        while word_start > 0 and (text[word_start - 1].isalpha() or text[word_start - 1] == "."):
            word_start -= 1
        preceding_word = text[word_start:end].lower()
        if preceding_word in _ABBREVIATIONS:
            continue  # false-positive split; skip this candidate
        sentence_ends.append(m.end())  # end of whitespace = start of next sentence

    if not sentence_ends:
        return text

    # Take the first n sentence breaks; if we want n sentences, that's
    # sentence_ends[n-1] (end of the nth sentence including its trailing space).
    take = min(n, len(sentence_ends))
    cut = sentence_ends[take - 1]
    return text[:cut].rstrip()
