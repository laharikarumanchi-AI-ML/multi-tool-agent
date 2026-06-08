"""Eval runner. Loads test set, runs agent on each query, scores, checkpoints
after every task (for inspection — re-running starts fresh; manually filter
completed ids if you need to resume)."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from multitool.eval.scorer import score


def load_test_set(path: str) -> list[dict]:
    """Load JSONL test set; one dict per line."""
    queries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            queries.append(json.loads(line))
    return queries


def run_eval(
    queries: list[dict],
    orchestrator_factory,        # Callable[[], Orchestrator]; fresh agent per query
    results_path: str,
) -> dict:
    """Run agent on each query, score, write checkpoint after each.

    Returns summary: {total_attempted, total_passed, pass_rate}.

    Checkpoint format (overwritten after every task — checkpointed for
    inspection only; re-running starts fresh, manually filter completed ids
    if you need to resume):
      {
        "started_at": ISO,
        "results": [{id, question, passed, predicted, gold, ...}],
        "total_attempted": N,
        "total_passed": M,
      }
    """
    results = []
    summary = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "total_attempted": 0,
        "total_passed": 0,
    }

    def checkpoint():
        Path(results_path).write_text(json.dumps(summary, indent=2))

    checkpoint()

    for q in queries:
        try:
            orch = orchestrator_factory()
            agent_result = orch.run(q["question"])
            predicted = agent_result.answer or ""
            scored = score(
                predicted=predicted,
                gold=q["gold_answer"],
                kind=q["answer_kind"],
                tolerance=q.get("tolerance"),
            )
            entry = {
                "id": q["id"],
                "question": q["question"],
                "gold": q["gold_answer"],
                "predicted": predicted,
                "passed": scored["passed"],
                "category": q.get("category"),
                "tool_calls": agent_result.tool_calls,
                "error": agent_result.error,
                "parse_error": scored.get("parse_error"),
            }
        except Exception as e:
            entry = {
                "id": q["id"],
                "question": q["question"],
                "gold": q["gold_answer"],
                "predicted": "",
                "passed": False,
                "category": q.get("category"),
                "error": f"crash: {type(e).__name__}: {e}",
            }
        results.append(entry)
        summary["total_attempted"] = len(results)
        summary["total_passed"] = sum(1 for r in results if r["passed"])
        checkpoint()

    summary["pass_rate"] = (
        summary["total_passed"] / summary["total_attempted"]
        if summary["total_attempted"]
        else 0.0
    )
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    checkpoint()
    return summary


def main():
    """CLI entrypoint: python -m multitool.eval.run --test-set X --results Y"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-set", default="multitool/eval/test_set.jsonl")
    parser.add_argument("--results", default="eval_results.json")
    args = parser.parse_args()

    # Build a fresh-orchestrator factory
    from multitool.orchestrator import Orchestrator
    from multitool.llm_client import GroqClient
    from multitool.trace import Trace
    # Trigger tool registrations
    import multitool.tools.search       # noqa: F401
    import multitool.tools.calculator   # noqa: F401
    import multitool.tools.datetime_tool # noqa: F401
    import multitool.tools.unit_convert # noqa: F401
    import multitool.tools.wikipedia    # noqa: F401

    def factory():
        llm = GroqClient(api_key=os.environ["GROQ_API_KEY"])
        trace = Trace(directory="traces", question="", provider="groq", model=llm._model)
        return Orchestrator(llm=llm, trace=trace)

    queries = load_test_set(args.test_set)
    summary = run_eval(queries, factory, args.results)
    print(f"\nFinal: {summary['total_passed']}/{summary['total_attempted']} = {summary['pass_rate']:.1%}")


if __name__ == "__main__":
    main()
