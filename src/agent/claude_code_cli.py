"""Claude Agent SDK subprocess CLI resolution (Windows path-length safeguards)."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import suppress
from pathlib import Path

logger = logging.getLogger("claude_code_cli")

_RESOLVER_ENV = ("FASTFOLD_CLAUDE_CODE_CLI", "CLAUDE_CODE_CLI_PATH")


def bundled_sdk_claude_exe_win32() -> Path | None:
    """Return ``claude_agent_sdk/_bundled/claude.exe`` if present."""

    try:
        import claude_agent_sdk as cas
    except ImportError:
        return None

    exe = Path(cas.__file__).resolve().parent / "_bundled" / "claude.exe"
    return exe if exe.is_file() else None


def _validate_windows_claude_spawn(path: str | Path, probe_long_args: bool = True) -> tuple[bool, str]:
    """Check whether Windows can spawn Claude executable from this path."""

    p = Path(path)
    if not p.is_file():
        return False, f"launcher not found: {p}"
    if os.name != "nt":
        return True, "non-windows platform; spawn probe skipped"

    try:
        subprocess.run(
            [str(p), "--version"],
            capture_output=True,
            timeout=8,
            text=True,
            shell=False,
        )
    except OSError as e:
        return False, f"spawn failed for --version: {e}"
    except Exception as e:
        return False, f"version probe failed: {e}"

    if probe_long_args:
        # Probe long argument command-lines to catch WinError 206 ahead of real SDK runs.
        try:
            subprocess.run(
                [str(p), "x" * 24000],
                capture_output=True,
                timeout=8,
                text=True,
                shell=False,
            )
        except OSError as e:
            return False, f"spawn failed for long-arg probe: {e}"
        except Exception:
            # Non-spawn failures are fine; we only care about CreateProcess viability.
            pass

    return True, "spawn probes passed"


def _windows_cache_roots() -> list[Path]:
    roots: list[Path] = []
    explicit = os.environ.get("FASTFOLD_CLAUDE_CACHE_DIR")
    if explicit:
        roots.append(Path(explicit))

    system_drive = os.environ.get("SystemDrive", "C:")
    roots.append(Path(system_drive) / "ff")

    local = os.environ.get("LOCALAPPDATA")
    if local:
        roots.append(Path(local) / "FastFoldAgent")

    roots.append(Path(tempfile.gettempdir()) / "ff")

    uniq: list[Path] = []
    seen: set[str] = set()
    for r in roots:
        k = str(r).lower()
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    return uniq


def ensure_windows_bundled_claude_sdk_cache(force_recopy: bool = False) -> str | None:
    """Copy bundled ``claude.exe`` to a very short path and validate spawnability.

    Deep ``site-packages`` under ``uv`` Roaming prefixes can overflow Windows
    **CreateProcess** limits (**WinError 206**). Returns cached path if usable.
    """

    src = bundled_sdk_claude_exe_win32()
    if src is None:
        return None

    for root in _windows_cache_roots():
        dest = root / "c.exe"
        tmp = root / "_c.exe.tmp"
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            need_copy = force_recopy or not dest.is_file()
            if not need_copy:
                ss, sd = src.stat(), dest.stat()
                need_copy = ss.st_size != sd.st_size or int(ss.st_mtime) != int(sd.st_mtime)

            if need_copy:
                with suppress(OSError):
                    tmp.unlink(missing_ok=True)
                shutil.copyfile(src, tmp)
                os.replace(tmp, dest)
                shutil.copystat(src, dest)

            ok, _ = _validate_windows_claude_spawn(dest, probe_long_args=True)
            if ok:
                return str(dest.resolve())
        except OSError as e:
            with suppress(OSError):
                tmp.unlink(missing_ok=True)
            logger.debug("Could not cache bundled Claude Code to %s: %s", root, e)
            continue
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

    cached = ensure_windows_bundled_claude_sdk_cache(force_recopy=False)
    if cached:
        return cached, "cached bundled exe (short local path)"

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
        ok, _ = _validate_windows_claude_spawn(found, probe_long_args=True)
        if ok:
            return found

    npm_global = Path(os.environ.get("APPDATA", "")) / "npm" / "claude.cmd"
    if npm_global.is_file():
        ok, _ = _validate_windows_claude_spawn(npm_global, probe_long_args=True)
        if ok:
            return str(npm_global)

    return ensure_windows_bundled_claude_sdk_cache(force_recopy=False)


def run_windows_autofix() -> dict[str, str | bool]:
    """Attempt to self-heal Windows Claude launcher resolution/spawn issues."""

    if sys.platform != "win32":
        return {"ok": True, "summary": "Autofix is only needed on Windows."}

    current, label = windows_claude_code_cli_resolve_detail()
    if current:
        ok, _ = _validate_windows_claude_spawn(current, probe_long_args=True)
        if ok:
            return {"ok": True, "summary": f"Claude launcher already healthy ({label}).", "path": current}

    cached = ensure_windows_bundled_claude_sdk_cache(force_recopy=True)
    if cached:
        ok, _ = _validate_windows_claude_spawn(cached, probe_long_args=True)
        if ok:
            return {
                "ok": True,
                "summary": "Repaired launcher by recaching bundled Claude binary to a short path.",
                "path": cached,
            }

    return {
        "ok": False,
        "summary": (
            "Autofix could not produce a spawnable launcher. Install Claude Code globally "
            "(npm install -g @anthropic-ai/claude-code) and set FASTFOLD_CLAUDE_CODE_CLI "
            "to that claude.cmd/claude.exe path."
        ),
    }
