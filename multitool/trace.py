"""JSON trace logger for agent runs. One file per run; flushed after every step
so a mid-run crash leaves a partial-but-valid log on disk."""
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path


class Trace:
    """A single agent-run's trace. Writes JSON to <directory>/<run_id>.json
    after every step. Use .flush() to force-write after the final answer."""

    def __init__(
        self,
        directory: str,
        question: str,
        provider: str,
        model: str,
    ):
        os.makedirs(directory, exist_ok=True)
        self.run_id = uuid.uuid4().hex[:12]
        self.path = str(Path(directory) / f"{self.run_id}.json")
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.question = question
        self.provider = provider
        self.model = model
        self.steps: list[dict] = []
        self.final_answer: str | None = None
        self.total_steps: int = 0
        self.error: str | None = None
        self.flush()

    def log_step(self, step: int, call, result: str) -> None:
        """Record one tool invocation in step N."""
        # Find existing step entry or create
        existing = next((s for s in self.steps if s["step"] == step), None)
        if existing is None:
            existing = {"step": step, "tool_calls": [], "results": []}
            self.steps.append(existing)
        existing["tool_calls"].append({
            "id": call.id,
            "name": call.name,
            "args": call.arguments,
        })
        existing["results"].append(result)
        self.flush()

    def log_final(self, step: int, final_answer: str) -> None:
        """Record the model's final answer (returned without tool_calls)."""
        self.final_answer = final_answer
        self.total_steps = step + 1
        self.flush()

    def log_error(self, error: str) -> None:
        self.error = error
        self.flush()

    def flush(self) -> None:
        data = {
            "run_id": self.run_id,
            "question": self.question,
            "started_at": self.started_at,
            "provider": self.provider,
            "model": self.model,
            "steps": self.steps,
            "final_answer": self.final_answer,
            "total_steps": self.total_steps,
            "error": self.error,
        }
        Path(self.path).write_text(json.dumps(data, indent=2))
