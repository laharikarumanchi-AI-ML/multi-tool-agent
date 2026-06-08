"""Tests for multitool._env (auto-loading .env at entry points).

The helper walks up from CWD looking for a .env file and loads it without
overriding real env vars. Tests mock Path.cwd to control the walk root.
"""
from __future__ import annotations

import os
from pathlib import Path

from multitool import _env


class TestFindDotenv:
    def test_find_dotenv_returns_path_when_present(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("FOO=bar\n")

        monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: tmp_path))

        result = _env._find_dotenv()
        assert result == env_file.resolve()

    def test_find_dotenv_walks_up_directory_tree(self, tmp_path, monkeypatch):
        # .env at the top, CWD several levels deeper
        env_file = tmp_path / ".env"
        env_file.write_text("FOO=bar\n")

        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)

        monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: deep))

        result = _env._find_dotenv()
        assert result == env_file.resolve()

    def test_find_dotenv_returns_none_when_no_dotenv(self, tmp_path, monkeypatch):
        # tmp_path has no .env. We bound the search to the tmp subtree by
        # monkeypatching _find_dotenv with a variant that stops at tmp_path's
        # ancestor — otherwise the walk would escape to the real filesystem
        # where a .env might exist.
        sub = tmp_path / "no_env_here"
        sub.mkdir()

        def bounded_find():
            cwd = sub.resolve()
            tmp_root = tmp_path.resolve()
            for parent in (cwd, *cwd.parents):
                if not str(parent).startswith(str(tmp_root)):
                    break
                candidate = parent / ".env"
                if candidate.is_file():
                    return candidate
            return None

        monkeypatch.setattr(_env, "_find_dotenv", bounded_find)
        assert _env._find_dotenv() is None


class TestLoadProjectEnv:
    def test_load_project_env_no_op_when_no_dotenv(self, tmp_path, monkeypatch):
        # Force _find_dotenv to return None and verify load_project_env is a
        # no-op (does not raise, does not mutate os.environ).
        monkeypatch.setattr(_env, "_find_dotenv", lambda: None)

        before = dict(os.environ)
        _env.load_project_env()
        after = dict(os.environ)
        assert before == after

    def test_load_project_env_does_not_override_existing_env_vars(
        self, tmp_path, monkeypatch
    ):
        # Set a real env var, then write a .env with a conflicting value and
        # load it. The real env var must win (override=False).
        env_file = tmp_path / ".env"
        env_file.write_text("MULTITOOL_TEST_VAR=from_dotenv\n")

        monkeypatch.setenv("MULTITOOL_TEST_VAR", "from_real_env")
        monkeypatch.setattr(_env, "_find_dotenv", lambda: env_file)

        _env.load_project_env()

        assert os.environ["MULTITOOL_TEST_VAR"] == "from_real_env"
