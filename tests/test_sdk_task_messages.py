"""Tests for SDK background task message handling."""

import asyncio
from io import StringIO

from claude_agent_sdk import ResultMessage, SystemMessage
from rich.console import Console

from ct.agent.runner import (
    AgentRunner,
    _extract_task_event,
    _extract_task_output_paths_from_text,
    _default_local_task_output_path,
    _is_warp_terminal_env,
    _parse_task_probe_json,
    _sanitize_notification_text,
    process_messages,
)
from ct.ui.traces import TraceRenderer


async def _aiter(items):
    for item in items:
        yield item


def _renderer():
    buf = StringIO()
    console = Console(file=buf, no_color=True, width=120)
    return TraceRenderer(console), buf


def _success_result() -> ResultMessage:
    return ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=1,
        session_id="session-1",
        result="ok",
    )


def test_task_started_is_tracked_as_pending():
    task_started = SystemMessage(
        subtype="task_started",
        data={
            "task_id": "task-123",
            "description": "Waiting for workflow completion",
            "uuid": "uuid-1",
            "session_id": "session-1",
            "task_type": "local_bash",
        },
    )
    messages = [task_started, _success_result()]

    renderer, _ = _renderer()
    result = asyncio.run(process_messages(_aiter(messages), trace_renderer=renderer))

    assert len(result["pending_background_tasks"]) == 1
    pending = result["pending_background_tasks"][0]
    assert pending["task_id"] == "task-123"
    assert pending["task_type"] == "local_bash"
    assert result["completed_background_tasks"] == []


def test_task_notification_completes_pending_task():
    usage = {"total_tokens": 42, "tool_uses": 2, "duration_ms": 1200}
    messages = [
        SystemMessage(
            subtype="task_started",
            data={
                "task_id": "task-999",
                "description": "Running background wait",
                "uuid": "uuid-1",
                "session_id": "session-1",
                "task_type": "local_bash",
            },
        ),
        SystemMessage(
            subtype="task_progress",
            data={
                "task_id": "task-999",
                "description": "Still running",
                "usage": usage,
                "uuid": "uuid-2",
                "session_id": "session-1",
                "last_tool_name": "Bash",
            },
        ),
        SystemMessage(
            subtype="task_notification",
            data={
                "task_id": "task-999",
                "status": "completed",
                "output_file": "/tmp/task-999.txt",
                "summary": "Workflow completed successfully",
                "uuid": "uuid-3",
                "session_id": "session-1",
                "usage": usage,
            },
        ),
        _success_result(),
    ]

    renderer, buf = _renderer()
    result = asyncio.run(process_messages(_aiter(messages), trace_renderer=renderer))

    assert result["pending_background_tasks"] == []
    assert len(result["completed_background_tasks"]) == 1
    completed = result["completed_background_tasks"][0]
    assert completed["task_id"] == "task-999"
    assert completed["status"] == "completed"
    assert completed["output_file"] == "/tmp/task-999.txt"

    output = buf.getvalue()
    assert "background task started" in output
    assert "background task" in output
    assert "completed" in output


def test_extract_task_event_parses_system_message_payload():
    msg = SystemMessage(
        subtype="task_notification",
        data={
            "task_id": "task-abc",
            "status": "completed",
            "summary": "done",
            "output_file": "/tmp/task-abc.txt",
        },
    )
    event = _extract_task_event(msg)
    assert event is not None
    assert event["type"] == "task_notification"
    assert event["task_id"] == "task-abc"
    assert event["status"] == "completed"


def test_background_watch_status_snapshot():
    class _DummySession:
        def __init__(self):
            self.console = Console(file=StringIO(), no_color=True, width=120)

    runner = AgentRunner(session=_DummySession(), headless=True)
    runner._bg_watch_state["session-xyz"] = {
        "started_at": 100.0,
        "last_update_at": 120.0,
        "model": "claude-sonnet",
        "pending_task_ids": ["task-1"],
        "completed_task_ids": ["task-0"],
        "status": "running",
        "error": None,
    }

    rows = runner.get_background_watch_status(include_inactive=True)
    assert len(rows) == 1
    assert rows[0]["session_id"] == "session-xyz"
    assert rows[0]["pending_task_ids"] == ["task-1"]
    assert rows[0]["completed_task_ids"] == ["task-0"]
    assert rows[0]["watcher_alive"] is False

    # With no live watcher thread, active-only view should be empty.
    assert runner.get_background_watch_status(include_inactive=False) == []


def test_extract_task_output_paths_from_text():
    full_text = [
        "Background Task ID: b409d04",
        "Output file path: /private/tmp/claude-501/-Users-juliocesar-Projects-fastfold-agent-cli/tasks/b409d04.output",
    ]
    mapping = _extract_task_output_paths_from_text(full_text)
    assert mapping["b409d04"].endswith("/tasks/b409d04.output")


def test_default_local_task_output_path_shape():
    path = _default_local_task_output_path("abc123")
    if path:
        assert path.endswith("/tasks/abc123.output")
        assert "/claude-" in path


def test_parse_task_probe_json_strict_mapping():
    payload = _parse_task_probe_json('{"a":"running","b":"completed","c":"weird"}')
    assert payload["a"] == "running"
    assert payload["b"] == "completed"
    assert payload["c"] == "unknown"


def test_is_warp_terminal_env_detection():
    assert _is_warp_terminal_env({"TERM_PROGRAM": "WarpTerminal"})
    assert _is_warp_terminal_env({"WARP_IS_LOCAL_SHELL_SESSION": "1"})
    assert _is_warp_terminal_env({"WARP_SESSION_ID": "abc"})
    assert not _is_warp_terminal_env({"TERM_PROGRAM": "iTerm.app"})


def test_sanitize_notification_text():
    raw = "Line1;\nLine2\rLine3"
    cleaned = _sanitize_notification_text(raw)
    assert ";" not in cleaned
    assert "\n" not in cleaned
    assert "\r" not in cleaned


def test_runner_interrupt_timeout_config_clamped():
    class _DummyConfig:
        def __init__(self, values):
            self._values = dict(values)

        def get(self, key, default=None):
            return self._values.get(key, default)

    class _DummySession:
        def __init__(self, values):
            self.console = Console(file=StringIO(), no_color=True, width=120)
            self.config = _DummyConfig(values)

    runner_low = AgentRunner(
        session=_DummySession({"agent.interrupt_drain_timeout_s": 0}),
        headless=True,
    )
    assert runner_low._interrupt_drain_timeout_s == 1

    runner_high = AgentRunner(
        session=_DummySession({"agent.interrupt_drain_timeout_s": 999}),
        headless=True,
    )
    assert runner_high._interrupt_drain_timeout_s == 120


def test_interrupt_active_query_helper():
    class _DummySession:
        def __init__(self):
            self.console = Console(file=StringIO(), no_color=True, width=120)

    class _OkClient:
        def __init__(self):
            self.interrupted = False

        async def interrupt(self):
            self.interrupted = True

    class _FailClient:
        async def interrupt(self):
            raise RuntimeError("boom")

    runner = AgentRunner(session=_DummySession(), headless=True)

    # No active client.
    assert asyncio.run(runner._interrupt_active_query()) is False

    # Successful interrupt.
    ok = _OkClient()
    with runner._active_client_lock:
        runner._active_client = ok
    assert asyncio.run(runner._interrupt_active_query()) is True
    assert ok.interrupted is True

    # Failing interrupt.
    with runner._active_client_lock:
        runner._active_client = _FailClient()
    assert asyncio.run(runner._interrupt_active_query()) is False
