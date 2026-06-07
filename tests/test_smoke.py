"""Smoke test: package imports + CLI is wired up."""
import pytest


def test_package_imports():
    import multitool
    assert multitool.__version__ == "0.1.0"


def test_cli_main_is_callable():
    """The CLI's main() should be importable + callable. Invoking with
    --help exits via argparse (SystemExit(0)), which is the cheapest proof
    the parser is wired up without hitting any LLM/orchestrator code."""
    from multitool.cli import main
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
