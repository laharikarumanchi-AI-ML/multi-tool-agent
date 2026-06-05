"""Smoke test: package imports + skeleton CLI runs."""

def test_package_imports():
    import multitool
    assert multitool.__version__ == "0.1.0"

def test_cli_main_exits_zero():
    from multitool.cli import main
    assert main() == 0
