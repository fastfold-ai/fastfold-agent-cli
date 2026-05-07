"""
AgentRunner: query entry point using the Claude Agent SDK.

Replaces the Plan-then-Execute architecture (Planner → Executor → Synthesis)
with a single agentic loop where Claude directly orchestrates all domain tools.

Uses ``ClaudeSDKClient`` (not ``query()``) because only the client supports
custom MCP tools.
"""

import asyncio
from contextlib import suppress
import json
import logging
import os
import re
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from ct.agent.types import ExecutionResult, Plan, Step

logger = logging.getLogger("ct.runner")


# ------------------------------------------------------------------
# Testable message processing (extracted from _run_async)
# ------------------------------------------------------------------


def _sdk_msg_field(msg, field_name: str, default=None):
    """Read SDK message fields across typed and generic SystemMessage payloads."""
    value = getattr(msg, field_name, None)
    if value is not None:
        return value
    data = getattr(msg, "data", None)
    if isinstance(data, dict):
        return data.get(field_name, default)
    return default


def _sdk_msg_subtype(msg) -> str:
    """Return SDK message subtype when present."""
    subtype = getattr(msg, "subtype", None)
    if subtype:
        return str(subtype)
    data = getattr(msg, "data", None)
    if isinstance(data, dict) and data.get("subtype"):
        return str(data.get("subtype"))
    return ""


def _extract_task_event(msg) -> dict[str, Any] | None:
    """Normalize SDK task lifecycle messages into a common event shape."""
    subtype = _sdk_msg_subtype(msg)
    if subtype not in {"task_started", "task_progress", "task_notification"}:
        return None

    event: dict[str, Any] = {
        "type": subtype,
        "task_id": _sdk_msg_field(msg, "task_id", "") or "",
        "description": _sdk_msg_field(msg, "description", "") or "",
        "tool_use_id": _sdk_msg_field(msg, "tool_use_id", None),
    }
    if subtype == "task_started":
        event["task_type"] = _sdk_msg_field(msg, "task_type", None)
        return event
    if subtype == "task_progress":
        event["usage"] = _sdk_msg_field(msg, "usage", None)
        event["last_tool_name"] = _sdk_msg_field(msg, "last_tool_name", None)
        return event
    # task_notification
    event["status"] = _sdk_msg_field(msg, "status", "") or ""
    event["summary"] = _sdk_msg_field(msg, "summary", "") or ""
    event["output_file"] = _sdk_msg_field(msg, "output_file", "") or ""
    event["usage"] = _sdk_msg_field(msg, "usage", None)
    return event


def _extract_task_output_paths_from_text(full_text: list[str]) -> dict[str, str]:
    """Best-effort extraction of task output file paths from assistant text."""
    joined = "\n".join(full_text or [])
    pattern = re.compile(r"(?P<path>/\S*/tasks/(?P<task_id>[A-Za-z0-9_-]+)\.output)")
    mapping: dict[str, str] = {}
    for match in pattern.finditer(joined):
        path = str(match.group("path") or "").strip().rstrip(".,;)")
        task_id = str(match.group("task_id") or "").strip()
        if path and task_id:
            mapping[task_id] = path
    return mapping


def _default_local_task_output_path(task_id: str) -> str:
    """Best-guess Claude local task output path for current working dir."""
    try:
        uid = os.getuid()
    except Exception:
        uid = None
    if uid is None:
        return ""
    cwd = Path.cwd().resolve().as_posix()
    sanitized = "-" + cwd.lstrip("/").replace("/", "-")
    return f"/private/tmp/claude-{uid}/{sanitized}/tasks/{task_id}.output"


def _discover_local_task_output_file(task_id: str) -> str:
    """Try to locate a local Claude task output file by task id."""
    if not task_id:
        return ""

    direct = _default_local_task_output_path(task_id)
    if direct and Path(direct).exists():
        return direct

    try:
        uid = os.getuid()
    except Exception:
        return ""

    for root in (Path(f"/private/tmp/claude-{uid}"), Path(f"/tmp/claude-{uid}")):
        if not root.exists():
            continue
        # Common layout: /private/tmp/claude-<uid>/<sanitized-cwd>/tasks/<task_id>.output
        matches = list(root.glob(f"*/tasks/{task_id}.output"))
        if matches:
            return str(matches[0])
    return ""


def _parse_task_probe_json(raw_text: str) -> dict[str, str]:
    """Parse strict JSON map from TaskOutput probe response text."""
    text = str(raw_text or "").strip()
    if not text:
        return {}
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    # Try full text first, then first {...} object slice.
    candidates = [text]
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        candidates.append(m.group(0))

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        out: dict[str, str] = {}
        for k, v in payload.items():
            key = str(k or "").strip()
            val = str(v or "").strip().lower()
            if not key:
                continue
            if val not in {"running", "completed", "failed", "stopped", "unknown"}:
                val = "unknown"
            out[key] = val
        if out:
            return out
    return {}


def _is_warp_terminal_env(env: dict[str, str] | None = None) -> bool:
    """Detect whether current terminal is Warp."""
    source = env if env is not None else os.environ
    term_program = str(source.get("TERM_PROGRAM") or "").lower()
    if "warp" in term_program:
        return True
    if str(source.get("WARP_IS_LOCAL_SHELL_SESSION") or "") == "1":
        return True
    if str(source.get("WARP_SESSION_ID") or "").strip():
        return True
    return False


def _sanitize_notification_text(text: str, max_len: int = 220) -> str:
    """Sanitize text for OSC notification payloads."""
    cleaned = str(text or "").replace("\r", " ").replace("\n", " ").replace(";", ",").strip()
    if len(cleaned) > max_len:
        cleaned = cleaned[: max_len - 3] + "..."
    return cleaned


def _clean_sdk_env(session, model: str) -> dict[str, str]:
    """Build env for lightweight SDK watcher/probe calls."""
    strip_vars = {
        "CLAUDECODE",
        "CLAUDE_CODE_SESSION_ID",
        "CLAUDE_CODE_PARENT_SESSION_ID",
    }
    clean_env = {k: v for k, v in os.environ.items() if k not in strip_vars}
    try:
        api_key = session.config.llm_api_key("anthropic")
    except Exception:
        api_key = None
    if api_key:
        clean_env["ANTHROPIC_API_KEY"] = api_key
    try:
        fastfold_cloud_key = session.config.get("api.fastfold_cloud_key")
    except Exception:
        fastfold_cloud_key = None
    if fastfold_cloud_key:
        clean_env["FASTFOLD_API_KEY"] = str(fastfold_cloud_key)

    if any(clean_env.get(v) for v in (
        "ANTHROPIC_FOUNDRY_API_KEY",
        "ANTHROPIC_FOUNDRY_RESOURCE",
        "ANTHROPIC_FOUNDRY_BASE_URL",
    )):
        clean_env["CLAUDE_CODE_USE_FOUNDRY"] = "1"
        clean_env.setdefault("ANTHROPIC_DEFAULT_SONNET_MODEL", model)
        clean_env.setdefault("ANTHROPIC_DEFAULT_OPUS_MODEL", model)
        clean_env.setdefault("ANTHROPIC_DEFAULT_HAIKU_MODEL", model)
    return clean_env

async def process_messages(
    messages_iter,
    trace_renderer=None,
    headless=False,
    trace_events: list[dict] | None = None,
    thinking_status=None,
    runner=None,
    on_activity=None,
):
    """Process an async iterable of SDK messages into structured results.

    This is extracted from ``AgentRunner._run_async`` so it can be tested
    with mock message streams without a live SDK client.

    Args:
        messages_iter: Async iterable of SDK messages.
        trace_renderer: Optional TraceRenderer for console output.
        headless: If True, suppress console output.
        trace_events: Optional list to append trace events to. When provided,
            each TextBlock, ToolUseBlock, and ToolResultBlock produces a
            trace event dict for downstream notebook/export consumers.
        thinking_status: Optional ThinkingStatus to stop on first message.

    Returns:
        dict with keys: full_text, tool_calls, result_msg, streamed_len
    """
    # Lazy imports — these may not be available in unit tests without
    # the SDK installed, but callers pass mock objects anyway.
    try:
        from claude_agent_sdk import (
            AssistantMessage,
            ResultMessage,
            TextBlock,
            ToolUseBlock,
        )
    except ImportError:
        # If SDK is missing entirely, preserve previous behavior and fail
        # at runtime when isinstance checks are attempted with missing types.
        from claude_agent_sdk import (
            AssistantMessage,
            ResultMessage,
            TextBlock,
            ToolUseBlock,
        )

    try:
        from claude_agent_sdk import ToolResultBlock
    except ImportError:
        ToolResultBlock = None

    try:
        from claude_agent_sdk import StreamEvent
    except ImportError:
        StreamEvent = None

    full_text: list[str] = []
    tool_calls: list[dict] = []
    inflight: dict[str, dict] = {}  # tool_use_id → {name, input, start_time}
    background_tasks: dict[str, dict] = {}  # task_id -> task metadata
    completed_background_tasks: list[dict] = []
    _last_progress_emit: dict[str, float] = {}
    result_msg = None
    streamed_len = 0  # characters already displayed via StreamEvent

    async for message in messages_iter:

        # --- StreamEvent (partial streaming) ---
        if StreamEvent is not None and isinstance(message, StreamEvent):
            event = getattr(message, "event", None) or {}
            if isinstance(event, dict):
                delta = event.get("delta", {})
                if isinstance(delta, dict) and delta.get("type") == "text_delta":
                    text = delta.get("text", "")
                    if text:
                        # Track streamed length but don't print raw text —
                        # the full TextBlock will be rendered as markdown
                        streamed_len += len(text)
            continue

        # --- Task/System messages ---
        task_event = _extract_task_event(message)
        if task_event and task_event["type"] == "task_started":
            task_id = task_event["task_id"]
            description = task_event["description"]
            task_type = task_event.get("task_type")
            tool_use_id = task_event.get("tool_use_id")
            now = time.time()
            if task_id:
                background_tasks[task_id] = {
                    "task_id": task_id,
                    "description": description,
                    "task_type": task_type,
                    "tool_use_id": tool_use_id,
                    "started_at": now,
                    "status": "running",
                }
            if trace_events is not None:
                trace_events.append({
                    "type": "task_started",
                    "task_id": task_id,
                    "description": description,
                    "task_type": task_type,
                    "tool_use_id": tool_use_id,
                    "timestamp": now,
                })
            if not headless and trace_renderer:
                trace_renderer.render_task_started(task_id, description, task_type)
            if on_activity and description:
                snippet = description.replace("\n", " ")[:40]
                on_activity(f"⌛ {snippet}")
            continue

        if task_event and task_event["type"] == "task_progress":
            task_id = task_event["task_id"]
            description = task_event["description"]
            usage = task_event.get("usage")
            last_tool_name = task_event.get("last_tool_name")
            now = time.time()
            if task_id and task_id in background_tasks:
                background_tasks[task_id]["description"] = description or background_tasks[task_id].get("description", "")
                background_tasks[task_id]["last_tool_name"] = last_tool_name
                background_tasks[task_id]["usage"] = usage
                background_tasks[task_id]["last_progress_at"] = now
            if trace_events is not None:
                trace_events.append({
                    "type": "task_progress",
                    "task_id": task_id,
                    "description": description,
                    "usage": usage,
                    "last_tool_name": last_tool_name,
                    "timestamp": now,
                })
            # Throttle console progress updates to avoid noisy streams.
            prev_emit = _last_progress_emit.get(task_id, 0.0)
            if not headless and trace_renderer and (now - prev_emit >= 2.0):
                trace_renderer.render_task_progress(task_id, description, usage, last_tool_name)
                _last_progress_emit[task_id] = now
            continue

        if task_event and task_event["type"] == "task_notification":
            task_id = task_event["task_id"]
            status = task_event.get("status", "")
            output_file = task_event.get("output_file", "")
            summary = task_event.get("summary", "")
            usage = task_event.get("usage")
            tool_use_id = task_event.get("tool_use_id")
            now = time.time()

            task_meta = background_tasks.pop(task_id, None)
            completed_background_tasks.append({
                "task_id": task_id,
                "status": status,
                "summary": summary,
                "output_file": output_file,
                "usage": usage,
                "tool_use_id": tool_use_id,
                "description": (task_meta or {}).get("description", ""),
                "task_type": (task_meta or {}).get("task_type"),
                "timestamp": now,
            })
            if trace_events is not None:
                trace_events.append({
                    "type": "task_notification",
                    "task_id": task_id,
                    "status": status,
                    "summary": summary,
                    "output_file": output_file,
                    "usage": usage,
                    "tool_use_id": tool_use_id,
                    "timestamp": now,
                })
            if not headless and trace_renderer:
                trace_renderer.render_task_notification(task_id, status, summary, output_file)
            if runner is not None and hasattr(runner, "_notify_terminal_task_completion"):
                try:
                    runner._notify_terminal_task_completion(task_id, status, summary, output_file)
                except Exception:
                    logger.debug("Failed terminal notification emit", exc_info=True)
            if on_activity:
                on_activity(f"✓ background {status}")
            continue

        # --- AssistantMessage ---
        if isinstance(message, AssistantMessage):
            for block in (message.content or []):
                if isinstance(block, TextBlock):
                    # Stop the spinner when showing complete text block
                    if thinking_status is not None:
                        thinking_status.stop()
                        thinking_status = None
                        if runner is not None:
                            runner._active_spinner = None
                        
                    text = block.text or ""
                    full_text.append(text)
                    # Trace capture
                    if trace_events is not None and text.strip():
                        trace_events.append({
                            "type": "text",
                            "content": text,
                            "timestamp": time.time(),
                        })
                    # Render as markdown (streamed deltas are tracked but not printed)
                    if not headless and trace_renderer:
                        streamed_len = 0  # reset for next turn
                        trace_renderer.render_reasoning(text)
                    # Activity callback — show snippet of reasoning
                    if on_activity and text.strip():
                        snippet = text.strip().replace("\n", " ")[:40]
                        on_activity(snippet)

                elif isinstance(block, ToolUseBlock):
                    # Restart spinner while waiting for tool result
                    if thinking_status is None and not headless and trace_renderer:
                        try:
                            from ct.ui.status import ThinkingStatus
                            thinking_status = ThinkingStatus(trace_renderer.console, phase="evaluating")
                            thinking_status.__enter__()
                            thinking_status.start_async_refresh()
                            if runner is not None:
                                runner._active_spinner = thinking_status
                        except ImportError:
                            pass
                        
                    block_id = getattr(block, "id", "") or ""
                    now = time.time()
                    inflight[block_id] = {
                        "name": block.name,
                        "input": block.input,
                        "start_time": now,
                    }
                    tool_calls.append({
                        "name": block.name,
                        "input": block.input,
                    })
                    # Trace capture
                    if trace_events is not None:
                        trace_events.append({
                            "type": "tool_start",
                            "tool": block.name.replace("mcp__ct-tools__", ""),
                            "input": block.input,
                            "tool_use_id": block_id,
                            "timestamp": now,
                        })
                    if not headless and trace_renderer:
                        trace_renderer.render_tool_start(block.name, block.input)
                    # Activity callback — show tool name
                    if on_activity:
                        clean = block.name.replace("mcp__ct-tools__", "")
                        on_activity(f"\u25b8 {clean}")

                elif ToolResultBlock is not None and isinstance(block, ToolResultBlock):
                    tool_use_id = getattr(block, "tool_use_id", "") or ""
                    is_error = getattr(block, "is_error", False)

                    # Extract result text from content
                    content = getattr(block, "content", None)
                    result_text = ""
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                result_text += item.get("text", "")
                    elif isinstance(content, str):
                        result_text = content

                    # Match to inflight tracker
                    tracked = inflight.pop(tool_use_id, None)
                    duration = 0.0
                    tool_name = ""
                    tool_input = {}
                    if tracked:
                        duration = time.time() - tracked["start_time"]
                        tool_name = tracked["name"]
                        tool_input = tracked["input"]
                    else:
                        logger.warning(
                            "Orphan ToolResultBlock with tool_use_id=%s",
                            tool_use_id,
                        )

                    # Update the matching tool_calls entry with result
                    for tc in reversed(tool_calls):
                        if tc["name"] == tool_name and "result_text" not in tc:
                            tc["result_text"] = result_text
                            tc["duration_s"] = duration
                            break

                    # Trace capture
                    if trace_events is not None:
                        clean_tool = tool_name.replace("mcp__ct-tools__", "")
                        trace_events.append({
                            "type": "tool_result",
                            "tool": clean_tool,
                            "tool_use_id": tool_use_id,
                            "result_text": result_text,
                            "is_error": is_error,
                            "duration_s": duration,
                            "timestamp": time.time(),
                        })

                    if not headless and trace_renderer:
                        if is_error:
                            trace_renderer.render_tool_error(
                                tool_name or "unknown", result_text
                            )
                        else:
                            trace_renderer.render_tool_complete(
                                tool_name or "unknown",
                                tool_input,
                                result_text,
                                duration,
                            )

        # --- ResultMessage ---
        elif isinstance(message, ResultMessage):
            # Final message, make sure animation is stopped
            if thinking_status is not None:
                thinking_status.stop()
                thinking_status = None
                if runner is not None:
                    runner._active_spinner = None
                
            result_msg = message

    return {
        "full_text": full_text,
        "tool_calls": tool_calls,
        "result_msg": result_msg,
        "streamed_len": streamed_len,
        "pending_background_tasks": list(background_tasks.values()),
        "completed_background_tasks": completed_background_tasks,
    }


class AgentRunner:
    """Run queries via the Claude Agent SDK agentic loop.

    All 192 domain tools are exposed as MCP tools.  Claude handles planning,
    execution, error recovery, and synthesis in one conversation.
    """

    def __init__(
        self,
        session,
        trajectory=None,
        headless: bool = False,
        trace_store=None,
    ):
        self.session = session
        self.trajectory = trajectory
        self._headless = headless
        self.trace_store = trace_store
        self._bg_watch_lock = threading.Lock()
        self._bg_watchers: dict[str, threading.Thread] = {}
        self._bg_watch_state: dict[str, dict[str, Any]] = {}
        cfg = getattr(self.session, "config", None)
        timeout_raw = cfg.get("agent.background_watch_timeout_s", 7200) if cfg else 7200
        try:
            timeout_s = int(timeout_raw)
        except (TypeError, ValueError):
            timeout_s = 7200
        # Guardrails: minimum 1 minute, maximum 24 hours.
        self._bg_watch_timeout_s = max(60, min(timeout_s, 24 * 60 * 60))
        self._bg_watch_retry_min_s = 5.0
        self._bg_watch_retry_max_s = 60.0
        self._bg_watch_probe_interval_s = 30.0
        self._bg_watch_ui_probe_interval_s = 20.0
        # IMPORTANT: keep TaskOutput probes isolated from user conversation state.
        # Probes run in a short-lived session (no resume) to avoid stale-task
        # follow-up contamination while still reconciling completed tasks.
        self._bg_watch_enable_taskoutput_probe = True

    def _notify_terminal_task_completion(
        self,
        task_id: str,
        status: str,
        summary: str = "",
        output_file: str = "",
    ) -> None:
        """Emit a desktop notification in Warp on task completion."""
        if self._headless:
            return
        status_norm = str(status or "").strip().lower()
        if status_norm not in {"completed", "failed", "stopped"}:
            return
        if not _is_warp_terminal_env():
            return

        title = _sanitize_notification_text(f"Fastfold task {status_norm}")
        body_parts = []
        if task_id:
            body_parts.append(task_id)
        if summary:
            body_parts.append(summary)
        if output_file:
            body_parts.append(output_file)
        body = _sanitize_notification_text(" - ".join(body_parts) if body_parts else status_norm)
        if not body:
            return

        # Warp desktop notification via OSC 777.
        sys.stdout.write(f"\033]777;notify;{title};{body}\007")
        sys.stdout.flush()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        context: dict | None = None,
        progress_callback=None,
    ) -> ExecutionResult:
        """Execute a query synchronously (blocking wrapper around async)."""
        return self._run_coro_sync(
            self._run_async(query, context, progress_callback)
        )

    @staticmethod
    def _run_coro_sync(coro):
        """Run async coroutine with safer KeyboardInterrupt cleanup."""
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        except KeyboardInterrupt:
            pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                with suppress(Exception):
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
            with suppress(Exception):
                loop.run_until_complete(loop.shutdown_asyncgens())
            with suppress(Exception):
                loop.run_until_complete(loop.shutdown_default_executor())
            raise
        finally:
            asyncio.set_event_loop(None)
            with suppress(Exception):
                loop.close()

    async def _run_async(
        self,
        query: str,
        context: dict | None = None,
        progress_callback=None,
    ) -> ExecutionResult:
        """Execute a query using the Agent SDK agentic loop.

        Uses ``ClaudeSDKClient`` (bidirectional client) which supports custom
        MCP tools, unlike ``query()`` which does not.
        """
        from claude_agent_sdk import (
            ClaudeSDKClient,
            ClaudeAgentOptions,
        )

        # Start spinner immediately — user should see feedback the moment they hit Enter
        thinking_status = None
        if not self._headless:
            from ct.ui.status import ThinkingStatus
            thinking_status = ThinkingStatus(self.session.console, phase="planning")
            thinking_status.__enter__()
            thinking_status.start_async_refresh()
            self._active_spinner = thinking_status
        from ct.agent.mcp_server import create_ct_mcp_server
        from ct.agent.system_prompt import build_system_prompt
        from ct.ui.traces import TraceRenderer

        t0 = time.time()
        config = self.session.config
        ctx = context or {}

        # ----- Build MCP server -----
        exclude_cats = set()
        if not config.get("agent.enable_experimental_tools", False):
            from ct.tools import EXPERIMENTAL_CATEGORIES
            exclude_cats = set(EXPERIMENTAL_CATEGORIES)

        server, sandbox, tool_names, code_trace_buffer = create_ct_mcp_server(
            self.session,
            exclude_categories=exclude_cats,
        )

        # ----- Build system prompt -----
        data_context = None
        data_dir = ctx.get("data_dir")
        if data_dir:
            data_context = f"Data directory: {data_dir}\n"
            config.set("sandbox.extra_read_dirs", str(data_dir))

        history = None
        if self.trajectory and self.trajectory.turns:
            history = self.trajectory.context_for_planner()

        system_prompt = build_system_prompt(
            self.session,
            tool_names=tool_names,
            data_context=data_context,
            history=history,
        )

        # ----- Configure Agent SDK -----
        model = config.get("llm.model") or "claude-sonnet-4-5-20250929"
        max_turns = int(config.get("agent.max_sdk_turns", 30))

        allowed_tools = [f"mcp__ct-tools__{name}" for name in tool_names]

        _STRIP_VARS = {
            "CLAUDECODE",
            "CLAUDE_CODE_SESSION_ID",
            "CLAUDE_CODE_PARENT_SESSION_ID",
        }
        clean_env = {
            k: v for k, v in os.environ.items()
            if k not in _STRIP_VARS
        }
        api_key = config.llm_api_key("anthropic")
        if api_key:
            clean_env["ANTHROPIC_API_KEY"] = api_key
        fastfold_cloud_key = config.get("api.fastfold_cloud_key")
        if fastfold_cloud_key:
            clean_env["FASTFOLD_API_KEY"] = str(fastfold_cloud_key)
        # Suppress warnings in SDK subprocess (matplotlib, pydeseq2, numpy, etc.)
        clean_env["PYTHONWARNINGS"] = "ignore"

        # Enable Foundry mode for Agent SDK subprocess if Foundry env vars present
        if any(clean_env.get(v) for v in (
            "ANTHROPIC_FOUNDRY_API_KEY",
            "ANTHROPIC_FOUNDRY_RESOURCE",
            "ANTHROPIC_FOUNDRY_BASE_URL",
        )):
            clean_env["CLAUDE_CODE_USE_FOUNDRY"] = "1"
            # Pin model names for Foundry deployments
            clean_env.setdefault(
                "ANTHROPIC_DEFAULT_SONNET_MODEL", model
            )
            clean_env.setdefault(
                "ANTHROPIC_DEFAULT_OPUS_MODEL", model
            )
            clean_env.setdefault(
                "ANTHROPIC_DEFAULT_HAIKU_MODEL", model
            )

        # Plan mode: use SDK's built-in plan permission mode.
        # In plan mode, Claude outputs a plan then calls ExitPlanMode.
        # We intercept that to show the plan and ask for approval.
        plan_preview = bool(config.get("agent.plan_preview", False))
        permission_mode = "plan" if (plan_preview and not self._headless) else "bypassPermissions"

        # Enable streaming for real-time output
        options_kwargs = dict(
            system_prompt=system_prompt,
            model=model,
            max_turns=max_turns,
            mcp_servers={"ct-tools": server},
            allowed_tools=allowed_tools,
            permission_mode=permission_mode,
            env=clean_env,
            hooks={},  # Disable inherited hooks (e.g. from Claude Code)
        )

        if plan_preview and not self._headless:
            options_kwargs["can_use_tool"] = self._plan_approval_hook()

        # Try to enable partial message streaming (graceful fallback)
        try:
            options = ClaudeAgentOptions(
                include_partial_messages=True,
                **options_kwargs,
            )
        except TypeError:
            # SDK version doesn't support include_partial_messages
            logger.info("SDK does not support include_partial_messages, using non-streaming")
            options = ClaudeAgentOptions(**options_kwargs)

        # ----- Build user prompt -----
        user_prompt = query
        context_parts = []
        if ctx.get("compound_smiles"):
            context_parts.append(f"Compound SMILES: {ctx['compound_smiles']}")
        if ctx.get("target"):
            context_parts.append(f"Target: {ctx['target']}")
        if ctx.get("indication"):
            context_parts.append(f"Indication: {ctx['indication']}")
        # Inject mention context if present
        if ctx.get("mention_context"):
            context_parts.append(ctx["mention_context"])
        if context_parts:
            user_prompt = query + "\n\nContext:\n" + "\n".join(context_parts)

        # ----- Create trace renderer -----
        trace_renderer = TraceRenderer(self.session.console)

        # ----- Prepare trace capture -----
        trace_events: list[dict] | None = None
        if self.trace_store is not None:
            trace_events = []

        # ----- Run the agentic loop via ClaudeSDKClient -----
        try:
            async with ClaudeSDKClient(options=options) as client:
                await client.query(user_prompt)
                result = await process_messages(
                    client.receive_response(),
                    trace_renderer=trace_renderer,
                    headless=self._headless,
                    trace_events=trace_events,
                    thinking_status=thinking_status,
                    runner=self,
                    on_activity=progress_callback,
                )
        except Exception as e:
            logger.error("Agent SDK query failed: %s\n%s", e, traceback.format_exc())
            duration = time.time() - t0
            return self._make_error_result(query, str(e), duration)
        finally:
            # Ensure animation is cleaned up even on error
            if thinking_status is not None:
                thinking_status.stop()

        duration = time.time() - t0

        full_text = result["full_text"]
        tool_calls = result["tool_calls"]
        result_msg = result["result_msg"]
        pending_background_tasks = result.get("pending_background_tasks", [])
        completed_background_tasks = result.get("completed_background_tasks", [])
        output_path_map = _extract_task_output_paths_from_text(full_text)
        if output_path_map:
            for task in pending_background_tasks:
                task_id = str(task.get("task_id") or "").strip()
                if task_id and task_id in output_path_map:
                    task["output_file"] = output_path_map[task_id]

        # ----- Build ExecutionResult -----
        summary = "\n".join(full_text).strip()
        if not summary:
            summary = "(Agent produced no text output)"

        answer = None
        if sandbox:
            result_var = sandbox.get_variable("result")
            if isinstance(result_var, dict):
                answer = result_var.get("answer")

        steps = []
        for i, tc in enumerate(tool_calls, 1):
            step = Step(
                id=i,
                tool=tc["name"].replace("mcp__ct-tools__", ""),
                description=f"Called {tc['name']}",
                tool_args=tc.get("input", {}),
            )
            step.status = "completed"
            steps.append(step)

        plan = Plan(query=query, steps=steps)

        cost_usd = 0.0
        if result_msg:
            cost_usd = getattr(result_msg, "total_cost_usd", 0.0) or 0.0

        exec_result = ExecutionResult(
            plan=plan,
            summary=summary,
            raw_results={
                "tool_calls": tool_calls,
                "answer": answer,
                "pending_background_tasks": pending_background_tasks,
                "completed_background_tasks": completed_background_tasks,
            },
            duration_s=duration,
            iterations=1,
            metadata={
                "sdk_cost_usd": cost_usd,
                "sdk_turns": getattr(result_msg, "num_turns", 0) if result_msg else 0,
                "sdk_duration_ms": getattr(result_msg, "duration_ms", 0) if result_msg else 0,
                "tool_call_count": len(tool_calls),
                "pending_background_task_count": len(pending_background_tasks),
                "completed_background_task_count": len(completed_background_tasks),
            },
        )

        # ----- Inject tool_result events from code_trace_buffer -----
        # The SDK stream typically does NOT include ToolResultBlock messages,
        # so process_messages() only produces tool_start events for code tools.
        # MCP handlers write structured results (code, stdout, plots) to
        # code_trace_buffer. We match buffer entries to tool_start events
        # by tool name in sequential order, and insert tool_result events
        # immediately after each tool_start.
        if trace_events is not None and trace_events:
            buffer_iter = iter(code_trace_buffer)
            # Also create tool_result events for non-code tools from tool_calls
            non_code_results = {}
            for tc in tool_calls:
                name = tc["name"].replace("mcp__ct-tools__", "")
                if name not in ("run_python", "run_r") and "result_text" in tc:
                    key = name + ":" + str(tc.get("input", {}))
                    non_code_results[key] = tc

            enriched: list[dict] = []
            non_code_iter_idx = {}  # track which non-code tool_calls we've used
            for event in trace_events:
                enriched.append(event)
                if event.get("type") != "tool_start":
                    continue

                tool = event.get("tool", "")
                tool_use_id = event.get("tool_use_id", "")

                if tool in ("run_python", "run_r"):
                    meta = next(buffer_iter, None)
                    if meta:
                        enriched.append({
                            "type": "tool_result",
                            "tool": tool,
                            "tool_use_id": tool_use_id,
                            "result_text": meta.get("stdout", ""),
                            "is_error": bool(meta.get("error")),
                            "duration_s": 0.0,
                            "code": meta.get("code", ""),
                            "stdout": meta.get("stdout", ""),
                            "plots": meta.get("plots", []),
                            "exports": meta.get("exports", []),
                            "error": meta.get("error"),
                            "timestamp": time.time(),
                        })
                else:
                    # For non-code tools, find matching result from tool_calls
                    for tc in tool_calls:
                        tc_name = tc["name"].replace("mcp__ct-tools__", "")
                        if tc_name == tool and "result_text" in tc and not tc.get("_used"):
                            tc["_used"] = True
                            enriched.append({
                                "type": "tool_result",
                                "tool": tool,
                                "tool_use_id": tool_use_id,
                                "result_text": tc["result_text"],
                                "is_error": False,
                                "duration_s": tc.get("duration_s", 0.0),
                                "timestamp": time.time(),
                            })
                            break

            trace_events = enriched

        # ----- Flush trace events -----
        if self.trace_store is not None and trace_events:
            try:
                self.trace_store.add_events(
                    trace_events,
                    query=query,
                    model=model,
                    duration_s=duration,
                    cost_usd=cost_usd,
                )
                self.trace_store.flush()
            except Exception as e:
                logger.warning("Failed to flush trace: %s", e)

        if not self._headless and result_msg:
            self._print_usage(result_msg, duration)
            if pending_background_tasks:
                self.session.console.print(
                    "\n  [yellow]Background task(s) still running.[/yellow]"
                )
                for task in pending_background_tasks:
                    task_id = task.get("task_id", "")
                    description = task.get("description", "")
                    task_type = task.get("task_type")
                    line = f"  [dim]- {task_id}[/dim]"
                    if task_type:
                        line += f" [dim]({task_type})[/dim]"
                    if description:
                        line += f" [dim]· {description}[/dim]"
                    self.session.console.print(line)
                self.session.console.print(
                    "  [dim]This turn ended before those tasks completed. "
                    "Follow up to check status/results.[/dim]"
                )
                session_id = getattr(result_msg, "session_id", None)
                self._start_background_task_watcher(
                    session_id=session_id,
                    pending_background_tasks=pending_background_tasks,
                    model=model,
                    env=clean_env,
                )

        return exec_result

    def _start_background_task_watcher(
        self,
        session_id: str | None,
        pending_background_tasks: list[dict],
        model: str,
        env: dict[str, str],
    ) -> None:
        """Start a detached watcher that listens for post-turn task notifications."""
        if not session_id:
            return
        task_ids = sorted({
            str(task.get("task_id") or "").strip()
            for task in pending_background_tasks
            if isinstance(task, dict)
        })
        task_ids = [task_id for task_id in task_ids if task_id]
        if not task_ids:
            return

        now = time.time()
        with self._bg_watch_lock:
            existing = self._bg_watchers.get(session_id)
            if existing and existing.is_alive():
                return

            output_files = {}
            for task in pending_background_tasks:
                task_id = str(task.get("task_id") or "").strip()
                output_file = str(task.get("output_file") or "").strip()
                if not output_file and task_id:
                    output_file = _default_local_task_output_path(task_id)
                if task_id and output_file:
                    output_files[task_id] = output_file

            state = self._bg_watch_state.get(session_id, {})
            self._bg_watch_state[session_id] = {
                "started_at": state.get("started_at", now),
                "last_update_at": now,
                "model": model,
                "pending_task_ids": sorted(task_ids),
                "completed_task_ids": list(state.get("completed_task_ids", [])),
                "status": "running",
                "error": None,
                "connection_attempts": int(state.get("connection_attempts", 0)),
                "last_disconnect_reason": None,
                "output_files": output_files,
                "last_probe_at": state.get("last_probe_at"),
                "probe_attempts": int(state.get("probe_attempts", 0)),
            }

            thread = threading.Thread(
                target=self._run_background_task_watcher,
                args=(session_id, task_ids, model, dict(env)),
                daemon=True,
                name=f"ct-bg-watch-{session_id[:8]}",
            )
            self._bg_watchers[session_id] = thread
            thread.start()

        if not self._headless:
            self.session.console.print(
                "  [dim]Background notification watcher armed for "
                f"{len(task_ids)} task(s).[/dim]"
            )

    def _run_background_task_watcher(
        self,
        session_id: str,
        task_ids: list[str],
        model: str,
        env: dict[str, str],
    ) -> None:
        """Thread entrypoint for async background task notification watcher."""
        try:
            self._run_coro_sync(
                self._watch_background_tasks_async(
                    session_id=session_id,
                    task_ids=task_ids,
                    model=model,
                    env=env,
                )
            )
        except Exception:
            with self._bg_watch_lock:
                state = self._bg_watch_state.get(session_id, {})
                state["status"] = "error"
                state["last_update_at"] = time.time()
                state["error"] = "watcher_crash"
                self._bg_watch_state[session_id] = state
            logger.warning(
                "Background task watcher crashed for session %s:\n%s",
                session_id,
                traceback.format_exc(),
            )
        finally:
            with self._bg_watch_lock:
                self._bg_watchers.pop(session_id, None)

    async def _watch_background_tasks_async(
        self,
        session_id: str,
        task_ids: list[str],
        model: str,
        env: dict[str, str],
    ) -> None:
        """Watch a resumed SDK session for task completion notifications."""
        from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

        from ct.ui.traces import TraceRenderer

        remaining = set(task_ids)
        if not remaining:
            return

        trace_renderer = TraceRenderer(self.session.console)
        deadline = time.time() + self._bg_watch_timeout_s
        attempt = 0
        while remaining and time.time() < deadline:
            self._probe_local_task_outputs(session_id, remaining, trace_renderer)
            if not remaining:
                break

            await self._maybe_probe_pending_tasks(
                session_id=session_id,
                remaining=remaining,
                model=model,
                env=env,
                trace_renderer=trace_renderer,
            )
            if not remaining:
                break

            attempt += 1
            with self._bg_watch_lock:
                state = self._bg_watch_state.get(session_id, {})
                state["status"] = "listening"
                state["connection_attempts"] = attempt
                state["last_update_at"] = time.time()
                self._bg_watch_state[session_id] = state

            options = ClaudeAgentOptions(
                resume=session_id,
                continue_conversation=True,
                model=model,
                env=env,
                hooks={},
            )

            disconnect_reason = "unknown"
            try:
                async with ClaudeSDKClient(options=options) as client:
                    await client.connect()
                    iterator = client.receive_messages().__aiter__()

                    while remaining and time.time() < deadline:
                        self._probe_local_task_outputs(session_id, remaining, trace_renderer)
                        if not remaining:
                            break

                        if self._bg_watch_enable_taskoutput_probe:
                            await self._maybe_probe_pending_tasks(
                                session_id=session_id,
                                remaining=remaining,
                                model=model,
                                env=env,
                                trace_renderer=trace_renderer,
                            )
                        if not remaining:
                            break

                        timeout_s = max(1.0, min(30.0, deadline - time.time()))
                        if time.time() >= deadline:
                            break

                        try:
                            message = await asyncio.wait_for(iterator.__anext__(), timeout=timeout_s)
                        except asyncio.TimeoutError:
                            self._probe_local_task_outputs(session_id, remaining, trace_renderer)
                            if self._bg_watch_enable_taskoutput_probe:
                                await self._maybe_probe_pending_tasks(
                                    session_id=session_id,
                                    remaining=remaining,
                                    model=model,
                                    env=env,
                                    trace_renderer=trace_renderer,
                                )
                            continue
                        except StopAsyncIteration:
                            disconnect_reason = "stream_closed"
                            break
                        except Exception:
                            disconnect_reason = "stream_error"
                            logger.warning(
                                "Background watcher read error for session %s:\n%s",
                                session_id,
                                traceback.format_exc(),
                            )
                            break

                        event = _extract_task_event(message)
                        if not event:
                            continue

                        if event["type"] == "task_notification":
                            task_id = event.get("task_id", "")
                            if task_id in remaining:
                                remaining.remove(task_id)
                                with self._bg_watch_lock:
                                    state = self._bg_watch_state.get(session_id, {})
                                    completed = list(state.get("completed_task_ids", []))
                                    if task_id not in completed:
                                        completed.append(task_id)
                                    pending = set(state.get("pending_task_ids", []))
                                    pending.discard(task_id)
                                    state["pending_task_ids"] = sorted(pending)
                                    state["completed_task_ids"] = completed
                                    state["last_update_at"] = time.time()
                                    self._bg_watch_state[session_id] = state
                                if not self._headless:
                                    trace_renderer.render_task_notification(
                                        task_id,
                                        event.get("status", ""),
                                        event.get("summary", ""),
                                        event.get("output_file", ""),
                                    )
                                self._notify_terminal_task_completion(
                                    task_id,
                                    event.get("status", ""),
                                    event.get("summary", ""),
                                    event.get("output_file", ""),
                                )
            except Exception:
                disconnect_reason = "connect_error"
                logger.warning(
                    "Background watcher connect error for session %s:\n%s",
                    session_id,
                    traceback.format_exc(),
                )

            with self._bg_watch_lock:
                state = self._bg_watch_state.get(session_id, {})
                state["last_disconnect_reason"] = disconnect_reason
                state["last_update_at"] = time.time()
                if remaining:
                    state["status"] = "retrying"
                self._bg_watch_state[session_id] = state

            if remaining and time.time() < deadline:
                retry_sleep = min(
                    self._bg_watch_retry_max_s,
                    self._bg_watch_retry_min_s * (2 ** min(attempt - 1, 4)),
                )
                await asyncio.sleep(max(1.0, retry_sleep))

        with self._bg_watch_lock:
            state = self._bg_watch_state.get(session_id, {})
            state["pending_task_ids"] = sorted(remaining)
            if remaining:
                state["status"] = "timeout"
            else:
                state["status"] = "completed"
            state["last_update_at"] = time.time()
            self._bg_watch_state[session_id] = state

        if remaining and not self._headless:
            left = ", ".join(sorted(remaining))
            self.session.console.print(
                "  [dim]Background watcher reached timeout before all tasks reported "
                f"(session {session_id[:8]}..., pending: {left}).[/dim]"
            )

    async def _maybe_probe_pending_tasks(
        self,
        session_id: str,
        remaining: set[str],
        model: str,
        env: dict[str, str],
        trace_renderer,
    ) -> None:
        """Rate-limited TaskOutput probe for pending tasks."""
        if not self._bg_watch_enable_taskoutput_probe:
            return
        if not remaining:
            return
        with self._bg_watch_lock:
            state = self._bg_watch_state.get(session_id, {})
            last_probe_at = float(state.get("last_probe_at") or 0.0)
        now = time.time()
        if now - last_probe_at < self._bg_watch_probe_interval_s:
            return
        await self._probe_pending_tasks_via_taskoutput(
            session_id=session_id,
            remaining=remaining,
            model=model,
            env=env,
            trace_renderer=trace_renderer,
        )

    async def _probe_pending_tasks_via_taskoutput(
        self,
        session_id: str,
        remaining: set[str],
        model: str,
        env: dict[str, str],
        trace_renderer,
    ) -> None:
        """Active fallback probe using TaskOutput in an isolated short session."""
        from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, AssistantMessage, TextBlock

        task_ids = sorted(remaining)
        if not task_ids:
            return

        prompt = (
            "Check these Claude background task ids using TaskOutput with block=false and timeout=1000 ms:\n"
            + ", ".join(task_ids)
            + "\nReturn ONLY JSON mapping task id to one of "
            + "\"running\", \"completed\", \"failed\", \"stopped\", or \"unknown\". "
            + "If task output says missing/not found/no longer in system, use \"completed\"."
        )

        options = ClaudeAgentOptions(
            resume=session_id,
            continue_conversation=False,
            model=model,
            env=env,
            hooks={},
            max_turns=2,
            permission_mode="bypassPermissions",
        )

        response_text_parts: list[str] = []
        probe_status: dict[str, str] = {}
        try:
            async with ClaudeSDKClient(options=options) as client:
                await client.query(prompt)
                async for message in client.receive_response():
                    if isinstance(message, AssistantMessage):
                        for block in (message.content or []):
                            if isinstance(block, TextBlock):
                                response_text_parts.append(block.text or "")
            probe_status = _parse_task_probe_json("\n".join(response_text_parts))
        except Exception:
            logger.warning(
                "TaskOutput probe failed for session %s:\n%s",
                session_id,
                traceback.format_exc(),
            )
        finally:
            with self._bg_watch_lock:
                state = self._bg_watch_state.get(session_id, {})
                state["last_probe_at"] = time.time()
                state["probe_attempts"] = int(state.get("probe_attempts", 0)) + 1
                state["last_update_at"] = time.time()
                self._bg_watch_state[session_id] = state

        if not probe_status:
            # Heuristic fallback when model didn't return JSON.
            raw_lower = "\n".join(response_text_parts).lower()
            for task_id in task_ids:
                if task_id.lower() in raw_lower and (
                    "no longer in the system" in raw_lower
                    or "not found" in raw_lower
                    or "missing" in raw_lower
                ):
                    probe_status[task_id] = "completed"

        if not probe_status:
            return

        completed_now: list[tuple[str, str, str]] = []
        for task_id, status in probe_status.items():
            if task_id not in remaining:
                continue
            if status in {"completed", "failed", "stopped"}:
                summary = f"Detected via TaskOutput probe ({status})."
                completed_now.append((task_id, status, summary))

        if not completed_now:
            return

        with self._bg_watch_lock:
            state = self._bg_watch_state.get(session_id, {})
            pending = set(state.get("pending_task_ids", []))
            completed = list(state.get("completed_task_ids", []))
            output_files = dict(state.get("output_files", {}))
            for task_id, status, summary in completed_now:
                remaining.discard(task_id)
                pending.discard(task_id)
                if task_id not in completed:
                    completed.append(task_id)
                if not self._headless:
                    trace_renderer.render_task_notification(
                        task_id,
                        status,
                        summary,
                        str(output_files.get(task_id) or ""),
                    )
                self._notify_terminal_task_completion(
                    task_id,
                    status,
                    summary,
                    str(output_files.get(task_id) or ""),
                )
            state["pending_task_ids"] = sorted(pending)
            state["completed_task_ids"] = completed
            state["last_update_at"] = time.time()
            self._bg_watch_state[session_id] = state

    def refresh_background_watch_status(
        self,
        force: bool = False,
        include_taskoutput: bool = False,
    ) -> None:
        """Reconcile pending tasks for /tasks view.

        Fast path (default): local output-file probe only.
        Slow path (include_taskoutput=True): also probes TaskOutput via SDK.
        """
        candidates: list[tuple[str, set[str], str, float]] = []
        with self._bg_watch_lock:
            for session_id, state in self._bg_watch_state.items():
                pending = {
                    str(t).strip() for t in (state.get("pending_task_ids") or [])
                    if str(t).strip()
                }
                if not pending:
                    continue
                model = str(state.get("model") or "").strip() or (
                    self.session.config.get("llm.model") or "claude-sonnet-4-5-20250929"
                )
                last_probe_at = float(state.get("last_probe_at") or 0.0)
                candidates.append((session_id, pending, model, last_probe_at))

        if not candidates:
            return

        from ct.ui.traces import TraceRenderer
        now = time.time()
        # Cheap local reconciliation first (always).
        for session_id, pending, _, _ in candidates:
            if not pending:
                continue
            trace_renderer = TraceRenderer(self.session.console)
            self._probe_local_task_outputs(session_id, pending, trace_renderer)

        if not include_taskoutput:
            return

        async def _refresh_all():
            for session_id, pending, model, last_probe_at in candidates:
                if not pending:
                    continue
                probe_due = (now - last_probe_at) >= self._bg_watch_ui_probe_interval_s
                if not force and not probe_due:
                    continue
                env = _clean_sdk_env(self.session, model)
                trace_renderer = TraceRenderer(self.session.console)
                await self._probe_pending_tasks_via_taskoutput(
                    session_id=session_id,
                    remaining=pending,
                    model=model,
                    env=env,
                    trace_renderer=trace_renderer,
                )

        try:
            self._run_coro_sync(_refresh_all())
        except KeyboardInterrupt:
            # Keep Ctrl+C behavior predictable in interactive terminal.
            raise
        except Exception:
            logger.warning("refresh_background_watch_status failed:\n%s", traceback.format_exc())

    def _probe_local_task_outputs(self, session_id: str, remaining: set[str], trace_renderer) -> None:
        """Fallback completion detector by reading known local task output files."""
        with self._bg_watch_lock:
            state = self._bg_watch_state.get(session_id, {})
            output_files = dict(state.get("output_files", {}))

        if not output_files:
            return

        completed_now: list[tuple[str, str, str]] = []
        for task_id in list(remaining):
            output_file = str(output_files.get(task_id) or "").strip()
            if not output_file:
                output_file = _discover_local_task_output_file(task_id)
                if output_file:
                    with self._bg_watch_lock:
                        state = self._bg_watch_state.get(session_id, {})
                        state_output_files = dict(state.get("output_files", {}))
                        state_output_files[task_id] = output_file
                        state["output_files"] = state_output_files
                        state["last_update_at"] = time.time()
                        self._bg_watch_state[session_id] = state
            if not output_file:
                continue
            path = Path(output_file)
            if not path.exists():
                discovered = _discover_local_task_output_file(task_id)
                if discovered and discovered != output_file:
                    path = Path(discovered)
                    output_file = discovered
                    with self._bg_watch_lock:
                        state = self._bg_watch_state.get(session_id, {})
                        state_output_files = dict(state.get("output_files", {}))
                        state_output_files[task_id] = output_file
                        state["output_files"] = state_output_files
                        state["last_update_at"] = time.time()
                        self._bg_watch_state[session_id] = state
            if not path.exists():
                continue
            try:
                text = path.read_text(errors="replace")
            except Exception:
                continue

            # SDK output files include terminal metadata/footer when completed.
            exit_code_match = re.search(r"exit_code:\s*(-?\d+)", text)
            if not exit_code_match:
                continue
            exit_code = exit_code_match.group(1)
            status = "completed" if exit_code == "0" else "failed"
            summary = (
                f"Detected via local output probe (exit_code={exit_code})."
            )
            completed_now.append((task_id, status, summary))

        if not completed_now:
            return

        with self._bg_watch_lock:
            state = self._bg_watch_state.get(session_id, {})
            pending = set(state.get("pending_task_ids", []))
            completed = list(state.get("completed_task_ids", []))
            for task_id, status, summary in completed_now:
                if task_id in remaining:
                    remaining.remove(task_id)
                pending.discard(task_id)
                if task_id not in completed:
                    completed.append(task_id)
                if not self._headless:
                    trace_renderer.render_task_notification(
                        task_id,
                        status,
                        summary,
                        str(output_files.get(task_id) or ""),
                    )
                self._notify_terminal_task_completion(
                    task_id,
                    status,
                    summary,
                    str(output_files.get(task_id) or ""),
                )
            state["pending_task_ids"] = sorted(pending)
            state["completed_task_ids"] = completed
            state["last_update_at"] = time.time()
            self._bg_watch_state[session_id] = state

    def get_background_watch_status(self, include_inactive: bool = True) -> list[dict]:
        """Return snapshot status for background task watchers."""
        with self._bg_watch_lock:
            rows: list[dict] = []
            for session_id, state in self._bg_watch_state.items():
                thread = self._bg_watchers.get(session_id)
                watcher_alive = bool(thread and thread.is_alive())
                if not include_inactive and not watcher_alive:
                    continue
                row = dict(state)
                row["session_id"] = session_id
                row["watcher_alive"] = watcher_alive
                rows.append(row)
        rows.sort(key=lambda r: float(r.get("last_update_at") or 0.0), reverse=True)
        return rows

    # ------------------------------------------------------------------
    # Plan mode
    # ------------------------------------------------------------------

    def _plan_approval_hook(self):
        """Return a can_use_tool callback for SDK plan mode.

        Intercepts the ExitPlanMode call to show Claude's plan and ask
        for user approval. All other tool calls are auto-allowed.
        """
        console = self.session.console
        # Shared ref so process_messages can keep it in sync
        self._active_spinner = None

        async def _hook(tool_name, input_data, context):
            if tool_name == "ExitPlanMode":
                # Stop the spinner so it doesn't interfere with input()
                if self._active_spinner is not None:
                    self._active_spinner.stop()
                    self._active_spinner = None

                # Claude is requesting to exit plan mode and start executing
                console.print("\n  [bold cyan]Proposed Plan[/bold cyan]")
                # The plan text may be in the input data or in Claude's
                # preceding text output (which the user already saw streamed).
                if isinstance(input_data, dict):
                    for key in ("plan", "description", "summary"):
                        if key in input_data and input_data[key]:
                            console.print(f"  {input_data[key]}")
                            break
                console.print()

                try:
                    answer = input("  Execute this plan? [Y/n] ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    answer = "n"

                if answer in ("", "y", "yes"):
                    return {"allow": True, "updated_input": input_data}
                else:
                    # Ask what to change so Claude can revise the plan
                    try:
                        feedback = input("  What would you change? ").strip()
                    except (EOFError, KeyboardInterrupt):
                        feedback = ""

                    msg = f"User rejected the plan. Feedback: {feedback}" if feedback else "User rejected the plan."
                    return {"allow": False, "message": msg}

            # All other tools: allow
            return {"allow": True, "updated_input": input_data}

        return _hook

    # ------------------------------------------------------------------
    # Console output helpers
    # ------------------------------------------------------------------

    def _print_usage(self, result_msg, duration: float):
        """Print cost and usage summary."""
        turns = getattr(result_msg, "num_turns", 0)
        parts = []
        if turns:
            parts.append(f"{turns} turns")
        if duration >= 60:
            mins = int(duration // 60)
            secs = int(duration % 60)
            parts.append(f"{mins}m {secs}s")
        else:
            parts.append(f"{duration:.1f}s")
        self.session.console.print(f"\n  [dim]{' | '.join(parts)}[/dim]")

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    @staticmethod
    def _make_error_result(query: str, error: str, duration: float) -> ExecutionResult:
        """Build an ExecutionResult representing a failed query."""
        plan = Plan(query=query, steps=[])
        return ExecutionResult(
            plan=plan,
            summary=f"Agent SDK error: {error}",
            raw_results={"error": error},
            duration_s=duration,
            iterations=1,
        )
