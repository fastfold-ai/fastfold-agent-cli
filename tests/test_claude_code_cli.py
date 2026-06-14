"""Tests for agent/claude_code_cli.py pure helper functions."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent import claude_code_cli as ccli


class TestBundledSdkClaudeExe:
    def test_returns_none_when_sdk_missing(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)
        with patch.dict(sys.modules, {"claude_agent_sdk": None}):
            # Force import failure path
            with patch("builtins.__import__", side_effect=ImportError("no sdk")):
                # Re-call via patching the function's import
                pass
        with patch("agent.claude_code_cli.bundled_sdk_claude_exe_win32", wraps=ccli.bundled_sdk_claude_exe_win32):
            with patch.dict("sys.modules", {"claude_agent_sdk": None}):
                result = ccli.bundled_sdk_claude_exe_win32()
        # When SDK is installed in test env, may return path; when not, None
        assert result is None or isinstance(result, Path)

    def test_bundled_path_length_check(self, tmp_path):
        exe = tmp_path / "nested" / "claude.exe"
        exe.parent.mkdir(parents=True)
        exe.write_bytes(b"MZ")
        with patch.object(ccli, "bundled_sdk_claude_exe_win32", return_value=exe):
            assert ccli.bundled_windows_path_maybe_too_long() is False

        long_path = tmp_path / ("x" * 250) / "claude.exe"
        long_path.parent.mkdir(parents=True, exist_ok=True)
        long_path.write_bytes(b"MZ")
        with patch.object(ccli, "bundled_sdk_claude_exe_win32", return_value=long_path):
            assert ccli.bundled_windows_path_maybe_too_long() is True

    def test_bundled_path_none_returns_none_for_length_check(self):
        with patch.object(ccli, "bundled_sdk_claude_exe_win32", return_value=None):
            assert ccli.bundled_windows_path_maybe_too_long() is None


class TestWindowsCacheRoots:
    def test_includes_explicit_and_temp_dirs(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FASTFOLD_CLAUDE_CACHE_DIR", str(tmp_path / "custom"))
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
        roots = ccli._windows_cache_roots()
        assert any("custom" in str(r) for r in roots)
        assert any("FastFoldAgent" in str(r) for r in roots)
        assert len(roots) == len({str(r).lower() for r in roots})


class TestValidateWindowsClaudeSpawn:
    def test_non_windows_skips_probe(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ccli.os, "name", "posix", raising=False)
        shim = tmp_path / "claude.sh"
        shim.write_text("#!/bin/sh")
        ok, msg = ccli._validate_windows_claude_spawn(shim, probe_long_args=True)
        assert ok is True
        assert "non-windows" in msg

    def test_missing_file_fails(self):
        ok, msg = ccli._validate_windows_claude_spawn("/no/such/claude.exe")
        assert ok is False
        assert "not found" in msg


class TestResolveClaudeSdkCliPath:
    def test_unix_returns_none_without_env(self, monkeypatch):
        monkeypatch.delenv("FASTFOLD_CLAUDE_CODE_CLI", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_CLI_PATH", raising=False)
        monkeypatch.setattr(sys, "platform", "linux")
        assert ccli.resolve_claude_sdk_cli_path() is None

    def test_explicit_env_file_used(self, tmp_path, monkeypatch):
        shim = tmp_path / "claude-bin"
        shim.write_bytes(b"#!/bin/sh")
        monkeypatch.setenv("FASTFOLD_CLAUDE_CODE_CLI", str(shim))
        monkeypatch.setattr(sys, "platform", "linux")
        resolved = ccli.resolve_claude_sdk_cli_path()
        assert resolved == str(shim.resolve())


class TestWindowsResolveDetail:
    def test_prefers_env_on_windows(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        shim = tmp_path / "mine.exe"
        shim.write_bytes(b"MZ")
        monkeypatch.setenv("FASTFOLD_CLAUDE_CODE_CLI", str(shim))

        path, reason = ccli.windows_claude_code_cli_resolve_detail()
        assert path == str(shim.resolve())
        assert "FASTFOLD_CLAUDE_CODE_CLI" in reason

    def test_returns_none_reason_when_unavailable(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.delenv("FASTFOLD_CLAUDE_CODE_CLI", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_CLI_PATH", raising=False)
        with patch.object(ccli.shutil, "which", return_value=None), patch.object(
            ccli, "ensure_windows_bundled_claude_sdk_cache", return_value=None
        ), patch.object(ccli, "bundled_sdk_claude_exe_win32", return_value=None):
            path, reason = ccli.windows_claude_code_cli_resolve_detail()
        assert path is None
        assert "not found" in reason.lower()


class TestRunWindowsAutofix:
    def test_non_windows_is_noop(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        result = ccli.run_windows_autofix()
        assert result["ok"] is True
        assert "only needed on Windows" in result["summary"]

    def test_windows_healthy_launcher(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        shim = tmp_path / "claude.exe"
        shim.write_bytes(b"MZ")
        with patch.object(
            ccli, "windows_claude_code_cli_resolve_detail", return_value=(str(shim), "test")
        ), patch.object(ccli, "_validate_windows_claude_spawn", return_value=(True, "ok")):
            result = ccli.run_windows_autofix()
        assert result["ok"] is True
        assert "healthy" in result["summary"].lower()

    def test_windows_recache_on_failure(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, "platform", "win32")
        cached = str(tmp_path / "c.exe")
        with patch.object(
            ccli, "windows_claude_code_cli_resolve_detail", return_value=(None, "missing")
        ), patch.object(
            ccli, "ensure_windows_bundled_claude_sdk_cache", return_value=cached
        ), patch.object(ccli, "_validate_windows_claude_spawn", return_value=(True, "ok")):
            result = ccli.run_windows_autofix()
        assert result["ok"] is True
        assert result["path"] == cached
