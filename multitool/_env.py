"""Helper to auto-load .env from project root when running CLI / eval entry points.

Intentionally placed at entry points (multitool/cli.py, multitool/eval/run.py)
rather than in library modules like llm_client.py. Loading environment side
effects from a library would surprise programmatic consumers of GroqClient;
.env loading is an entry-point convenience.
"""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv  # type: ignore[import-not-found]


def load_project_env() -> None:
    """Load .env from the nearest parent directory containing one.

    Searches upward from CWD for a .env file (so it works whether the user
    runs `multitool ask` from the repo root or a subdirectory). Silently
    no-ops if no .env is found — keeps existing os.environ behavior intact.
    """
    env_path = _find_dotenv()
    if env_path:
        load_dotenv(env_path, override=False)  # don't override real env vars


def _find_dotenv() -> Path | None:
    """Walk up from CWD looking for a .env file. Return path or None."""
    cwd = Path.cwd().resolve()
    for parent in (cwd, *cwd.parents):
        candidate = parent / ".env"
        if candidate.is_file():
            return candidate
    return None
