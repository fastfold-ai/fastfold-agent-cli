"""Tests for Windows bundled Claude exe short-path workaround."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

from ct.agent import runner as runner_mod


def test_claude_sdk_cli_path_unix_no_env_returns_none(monkeypatch):
    monkeypatch.delenv("FASTFOLD_CLAUDE_CODE_CLI", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_CLI_PATH", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    assert runner_mod._claude_sdk_cli_path() is None


def test_windows_prefers_explicit_env_over_cache(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(sys, "platform", "win32")

    shim = tmp_path / "mine.exe"
    shim.write_bytes(b"MZ")
    monkeypatch.setenv("FASTFOLD_CLAUDE_CODE_CLI", str(shim))

    assert runner_mod._claude_sdk_cli_path() == str(shim.resolve())


def test_windows_fallback_copies_bundled_to_localappdata(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("FASTFOLD_CLAUDE_CODE_CLI", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_CLI_PATH", raising=False)
    monkeypatch.setattr(sys, "platform", "win32")

    bundled = tmp_path / "bundled.exe"
    bundled.write_bytes(b"MZZZ")

    local_root = tmp_path / "Local"
    monkeypatch.setenv("LOCALAPPDATA", str(local_root))

    with patch.object(runner_mod.shutil, "which", return_value=None):
        with patch.object(
            runner_mod,
            "_claude_sdk_bundled_exe_path_win32",
            return_value=bundled,
        ):
            out = runner_mod._claude_sdk_cli_path()

    assert out == str((local_root / "FastFoldAgent" / "claude_sdk_bundled.exe").resolve())
    dest = Path(out)
    assert dest.is_file()
    assert dest.read_bytes() == b"MZZZ"


def test_windows_fallback_recopies_when_bundled_changes(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))

    bundled = tmp_path / "bundled.exe"
    bundled.write_bytes(b"MZZZ")

    with patch.object(
        runner_mod,
        "_claude_sdk_bundled_exe_path_win32",
        return_value=bundled,
    ):
        p1 = runner_mod._windows_short_cache_bundled_claude_exe()
        assert Path(p1).read_bytes() == b"MZZZ"

        bundled.write_bytes(b"MZ99")
        ta, tm = bundled.stat().st_atime + 3, bundled.stat().st_mtime + 3
        os.utime(bundled, (ta, tm))

        p2 = runner_mod._windows_short_cache_bundled_claude_exe()

    assert p1 == p2
    assert Path(p2).read_bytes() == b"MZ99"


def test_windows_short_cache_returns_none_without_bundled(monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(Path.home() / "tmp-localappdata"))
    monkeypatch.setattr(
        runner_mod,
        "_claude_sdk_bundled_exe_path_win32",
        lambda: None,
    )
    assert runner_mod._windows_short_cache_bundled_claude_exe() is None
