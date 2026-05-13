"""Claude Agent SDK subprocess CLI resolution (Windows path-length safeguards)."""

from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
from contextlib import suppress
from pathlib import Path

logger = logging.getLogger("ct.claude_code_cli")

_RESOLVER_ENV = ("FASTFOLD_CLAUDE_CODE_CLI", "CLAUDE_CODE_CLI_PATH")


def bundled_sdk_claude_exe_win32() -> Path | None:
    """Return ``claude_agent_sdk/_bundled/claude.exe`` if present."""

    try:
        import claude_agent_sdk as cas
    except ImportError:
        return None

    exe = Path(cas.__file__).resolve().parent / "_bundled" / "claude.exe"
    return exe if exe.is_file() else None


def ensure_windows_bundled_claude_sdk_cache() -> str | None:
    """Copy bundled ``claude.exe`` to a shorter path under **%LOCALAPPDATA%**.

    Deep ``site-packages`` under ``uv`` Roaming prefixes can overflow Windows
    **CreateProcess** limits (**WinError 206**). Returns cached path if usable.
    """

    src = bundled_sdk_claude_exe_win32()
    if src is None:
        return None

    root = Path(os.environ.get("LOCALAPPDATA") or tempfile.gettempdir())
    dest = root / "FastFoldAgent" / "claude_sdk_bundled.exe"
    tmp = root / "FastFoldAgent" / "_claude_sdk_bundled.exe.tmp"

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        need_copy = not dest.is_file()
        if not need_copy:
            ss, sd = src.stat(), dest.stat()
            need_copy = ss.st_size != sd.st_size or int(ss.st_mtime) != int(
                sd.st_mtime
            )

        if need_copy:
            with suppress(OSError):
                tmp.unlink(missing_ok=True)
            shutil.copyfile(src, tmp)
            os.replace(tmp, dest)
            shutil.copystat(src, dest)
        return str(dest.resolve())
    except OSError as e:
        with suppress(OSError):
            tmp.unlink(missing_ok=True)
        logger.debug("Could not cache bundled Claude Code to short path: %s", e)
        return None


def windows_claude_code_cli_resolve_detail() -> tuple[str | None, str]:
    """Return ``(effective_path_or_none, reason)`` mirroring resolver order (Windows)."""

    for key in _RESOLVER_ENV:
        raw = os.environ.get(key)
        if not raw:
            continue
        s = raw.strip().strip('"').strip("'")
        if not s:
            continue
        p = Path(s)
        with suppress(OSError):
            if p.is_file():
                return str(p.resolve()), f"{key} (file)"
        w = shutil.which(s)
        if w:
            return w, f"{key} resolves to PATH"
        continue

    w = shutil.which("claude")
    if w:
        return w, "claude on PATH"

    npm_global = Path(os.environ.get("APPDATA", "")) / "npm" / "claude.cmd"
    if npm_global.is_file():
        return str(npm_global), "npm global claude.cmd"

    cached = ensure_windows_bundled_claude_sdk_cache()
    if cached:
        return cached, "cached bundled exe (%LOCALAPPDATA%\\FastFoldAgent)"

    bundled = bundled_sdk_claude_exe_win32()
    if bundled is not None:
        return None, (
            "Bundled sdk claude.exe exists but LOCALAPPDATA cache failed "
            "(check permissions/disk)"
        )

    return None, "claude-agent-sdk bundled claude.exe not found"


def bundled_windows_path_maybe_too_long() -> bool | None:
    """True when bundled exe path exceeds a conservative CreateProcess-risk length."""

    p = bundled_sdk_claude_exe_win32()
    if p is None:
        return None
    return len(str(p)) > 230


def resolve_claude_sdk_cli_path() -> str | None:
    """Explicit ``cli_path`` when SDK default path is risky; ``None`` on Unix."""

    for key in _RESOLVER_ENV:
        raw = os.environ.get(key)
        if raw:
            s = raw.strip().strip('"').strip("'")
            p = Path(s)
            with suppress(OSError):
                if p.is_file():
                    return str(p.resolve())
            found = shutil.which(s)
            if found:
                return found

    if sys.platform != "win32":
        return None

    found = shutil.which("claude")
    if found:
        return found

    npm_global = Path(os.environ.get("APPDATA", "")) / "npm" / "claude.cmd"
    if npm_global.is_file():
        return str(npm_global)

    return ensure_windows_bundled_claude_sdk_cache()
