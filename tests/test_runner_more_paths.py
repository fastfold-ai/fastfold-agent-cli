"""Additional branch-heavy unit tests for AgentRunner helpers."""

from __future__ import annotations

import asyncio
import importlib
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.runner import AgentRunner


def _mk_runner(headless: bool = True):
    cfg = SimpleNamespace(
        get=lambda key, default=None: default,
        llm_api_key=lambda provider: None,
    )
    console = MagicMock()
    console.width = 50
    session = SimpleNamespace(config=cfg, console=console)
    return AgentRunner(session=session, trajectory=None, headless=headless), session


class _FakeFuture:
    def result(self, timeout=None):
        return None


class _AssistantMessage:
    def __init__(self, content):
        self.content = content


class _TextBlock:
    def __init__(self, text):
        self.text = text


class _FakeClient:
    def __init__(self, options, text):
        self.options = options
        self._text = text

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def query(self, prompt):
        return None

    async def receive_response(self):
        yield _AssistantMessage([_TextBlock(self._text)])


class TestWatcherStartAndInterrupt:
    def test_start_background_task_watcher_skips_when_existing_alive(self):
        runner, _ = _mk_runner(headless=False)
        alive_thread = MagicMock()
        alive_thread.is_alive.return_value = True
        runner._bg_watchers["sess-1"] = alive_thread
        runner._bg_watch_state["sess-1"] = {"status": "running", "pending_task_ids": ["a"]}
        runner._start_background_task_watcher(
            session_id="sess-1",
            pending_background_tasks=[{"task_id": "b", "output_file": "/tmp/b"}],
            model="claude-sonnet-4-5-20250929",
            env={},
        )
        assert runner._bg_watch_state["sess-1"]["pending_task_ids"] == ["a"]

    def test_run_background_task_watcher_success_removes_thread_entry(self):
        runner, _ = _mk_runner(headless=True)
        runner._bg_watchers["sess-1"] = MagicMock()

        def _run_and_close(coro):
            coro.close()
            return None

        with patch.object(runner, "_run_coro_sync", side_effect=_run_and_close):
            runner._run_background_task_watcher("sess-1", ["task-a"], "model", {})
        assert "sess-1" not in runner._bg_watchers

    def test_request_interrupt_client_success_path(self):
        runner, _ = _mk_runner(headless=True)
        loop = MagicMock()
        loop.is_running.return_value = True
        task = MagicMock()
        task.done.return_value = False
        runner._active_loop = loop
        runner._active_task = task
        runner._active_client = MagicMock()
        runner._active_client.interrupt.return_value = "coro"

        with patch("agent.runner.asyncio.run_coroutine_threadsafe", return_value=_FakeFuture()) as call_mock:
            assert runner.request_interrupt(force=False) is True
        call_mock.assert_called_once_with("coro", loop)


class TestProbePaths:
    def test_probe_local_task_outputs_marks_failed_when_exit_code_nonzero(self, tmp_path):
        runner, _ = _mk_runner(headless=False)
        output = tmp_path / "task_fail.output"
        output.write_text("hello\nexit_code: 2\n", encoding="utf-8")
        runner._bg_watch_state = {
            "sess-1": {
                "pending_task_ids": ["task_fail"],
                "completed_task_ids": [],
                "output_files": {"task_fail": str(output)},
            }
        }
        remaining = {"task_fail"}
        trace_renderer = MagicMock()
        with patch.object(runner, "_notify_terminal_task_completion") as notify_mock:
            runner._probe_local_task_outputs("sess-1", remaining, trace_renderer)
        assert "task_fail" not in remaining
        trace_renderer.render_task_notification.assert_called_once()
        notify_mock.assert_called_once()
        assert notify_mock.call_args.args[1] == "failed"

    def test_maybe_probe_pending_tasks_respects_rate_limit(self):
        runner, _ = _mk_runner(headless=True)
        runner._bg_watch_state = {"sess-1": {"last_probe_at": 100.0}}
        remaining = {"task-1"}

        async def _run():
            with patch("time.time", return_value=110.0), patch.object(
                runner, "_probe_pending_tasks_via_taskoutput"
            ) as probe_mock:
                await runner._maybe_probe_pending_tasks(
                    session_id="sess-1",
                    remaining=remaining,
                    model="m",
                    env={},
                    trace_renderer=MagicMock(),
                )
                probe_mock.assert_not_called()

            with patch("time.time", return_value=200.0), patch.object(
                runner, "_probe_pending_tasks_via_taskoutput"
            ) as probe_mock:
                await runner._maybe_probe_pending_tasks(
                    session_id="sess-1",
                    remaining=remaining,
                    model="m",
                    env={},
                    trace_renderer=MagicMock(),
                )
                probe_mock.assert_called_once()

        asyncio.run(_run())

    def test_probe_pending_tasks_via_taskoutput_json_parse_and_complete(self):
        runner, _ = _mk_runner(headless=False)
        runner._bg_watch_state = {
            "sess-1": {
                "pending_task_ids": ["task-1", "task-2"],
                "completed_task_ids": [],
                "output_files": {"task-1": "/tmp/task1.out", "task-2": "/tmp/task2.out"},
            }
        }
        remaining = {"task-1", "task-2"}
        trace_renderer = MagicMock()

        fake_sdk = types.ModuleType("claude_agent_sdk")
        fake_sdk.ClaudeAgentOptions = lambda **kwargs: kwargs
        fake_sdk.AssistantMessage = _AssistantMessage
        fake_sdk.TextBlock = _TextBlock
        fake_sdk.ClaudeSDKClient = lambda options: _FakeClient(options, '{"task-1":"completed","task-2":"running"}')

        with patch.dict("sys.modules", {"claude_agent_sdk": fake_sdk}), patch.object(
            runner, "_notify_terminal_task_completion"
        ) as notify_mock:
            asyncio.run(
                runner._probe_pending_tasks_via_taskoutput(
                    session_id="sess-1",
                    remaining=remaining,
                    model="m",
                    env={},
                    trace_renderer=trace_renderer,
                )
            )

        assert "task-1" not in remaining
        assert "task-2" in remaining
        notify_mock.assert_called_once()
        assert notify_mock.call_args.args[0] == "task-1"
        assert notify_mock.call_args.args[1] == "completed"

    def test_probe_pending_tasks_via_taskoutput_heuristic_fallback(self):
        runner, _ = _mk_runner(headless=False)
        runner._bg_watch_state = {
            "sess-1": {
                "pending_task_ids": ["task-x"],
                "completed_task_ids": [],
                "output_files": {"task-x": "/tmp/taskx.out"},
            }
        }
        remaining = {"task-x"}
        trace_renderer = MagicMock()

        fake_sdk = types.ModuleType("claude_agent_sdk")
        fake_sdk.ClaudeAgentOptions = lambda **kwargs: kwargs
        fake_sdk.AssistantMessage = _AssistantMessage
        fake_sdk.TextBlock = _TextBlock
        fake_sdk.ClaudeSDKClient = lambda options: _FakeClient(
            options, "task-x is no longer in the system"
        )

        with patch.dict("sys.modules", {"claude_agent_sdk": fake_sdk}), patch.object(
            runner, "_notify_terminal_task_completion"
        ) as notify_mock:
            asyncio.run(
                runner._probe_pending_tasks_via_taskoutput(
                    session_id="sess-1",
                    remaining=remaining,
                    model="m",
                    env={},
                    trace_renderer=trace_renderer,
                )
            )

        assert "task-x" not in remaining
        notify_mock.assert_called_once()
        assert notify_mock.call_args.args[1] == "completed"


class TestNotificationAndUsageFormatting:
    def test_notify_terminal_task_completion_ignores_non_terminal_status(self):
        runner, _ = _mk_runner(headless=False)
        with patch("agent.runner._is_warp_terminal_env", return_value=True), patch(
            "agent.runner.sys.stdout.write"
        ) as write_mock:
            runner._notify_terminal_task_completion("task-1", "running", "still going", "")
        write_mock.assert_not_called()

    def test_print_usage_long_duration_wraps_line(self):
        runner, session = _mk_runner(headless=True)
        session.console.width = 28
        with patch.object(runner, "_random_usage_word", return_value="Tested"):
            runner._print_usage(
                result_msg=None,
                duration=65.0,
                input_tokens=123456789012345,
                output_tokens=987654321,
            )
        rendered = " ".join(str(c) for c in session.console.print.call_args_list)
        assert "Tested for 1m 5s" in rendered
        assert "·" in rendered

    def test_make_error_result_contains_error_and_query(self):
        result = AgentRunner._make_error_result("check target", "boom", 3.5)
        assert result.plan.query == "check target"
        assert result.summary == "Agent SDK error: boom"
        assert result.raw_results["error"] == "boom"
        assert result.duration_s == 3.5


class TestRunAsyncAnthropicPath:
    def test_run_async_anthropic_happy_path(self):
        runner, session = _mk_runner(headless=False)

        def _cfg_get(key, default=None):
            values = {
                "llm.provider": "anthropic",
                "agent.runtime": "sdk",
                "agent.enable_experimental_tools": False,
                "llm.model": "claude-sonnet-4-5-20250929",
                "agent.max_sdk_turns": 5,
                "agent.plan_preview": False,
                "api.fastfold_cloud_key": "ff-key",
            }
            return values.get(key, default)

        session.config.get = _cfg_get
        session.config.set = MagicMock()
        session.console = MagicMock()

        fake_trace_store = MagicMock()
        runner.trace_store = fake_trace_store
        runner._print_usage = MagicMock()
        runner._start_background_task_watcher = MagicMock()

        sandbox = MagicMock()
        sandbox.get_variable.return_value = {"answer": "Structured answer"}
        code_trace_buffer = [{"stdout": "ok", "error": "", "code": "print(1)", "plots": [], "exports": []}]

        class _FakeStatus:
            def __init__(self, *_a, **_k):
                self.started = False
                self.stopped = False

            def __enter__(self):
                return self

            def start_async_refresh(self):
                self.started = True

            def stop(self):
                self.stopped = True

        class _FakeOptions:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class _FakeClient:
            def __init__(self, options=None):
                self.options = options

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def query(self, _prompt):
                return None

            async def receive_response(self):
                if False:
                    yield None

        result_msg = SimpleNamespace(
            total_cost_usd=1.4,
            usage={"input_tokens": 12, "output_tokens": 6},
            model_usage={
                "input_tokens": 14,
                "output_tokens": 7,
                "cache_creation_input_tokens": 2,
                "cache_read_input_tokens": 1,
                "models": ["claude-sonnet-4-5-20250929"],
                "cost_usd": 1.0,
            },
            num_turns=2,
            duration_ms=1234,
            session_id="sess-123",
        )

        async def _fake_process_messages(*_a, **_k):
            return {
                "full_text": ["All done."],
                "tool_calls": [
                    {
                        "name": "mcp__ct-tools__run_python",
                        "input": {"code": "print(1)"},
                        "result_text": "ok",
                        "duration_s": 0.1,
                    },
                    {
                        "name": "mcp__ct-tools__target.search",
                        "input": {"query": "crbn"},
                        "result_text": "found",
                        "duration_s": 0.2,
                    },
                ],
                "result_msg": result_msg,
                "token_usage": {"input_tokens": 10, "output_tokens": 5},
                "pending_background_tasks": [
                    {"task_id": "task-1", "description": "bg task", "task_type": "shell"}
                ],
                "completed_background_tasks": [{"task_id": "task-0"}],
            }

        fake_sdk = types.ModuleType("claude_agent_sdk")
        fake_sdk.ClaudeSDKClient = _FakeClient
        fake_sdk.ClaudeAgentOptions = _FakeOptions

        mcp_server_module = importlib.import_module("agent.mcp_server")
        system_prompt_module = importlib.import_module("agent.system_prompt")

        with patch.dict("sys.modules", {"claude_agent_sdk": fake_sdk}), patch.object(
            mcp_server_module,
            "create_ct_mcp_server",
            return_value=("server", sandbox, ["run_python", "target.search"], code_trace_buffer),
        ), patch.object(system_prompt_module, "build_system_prompt", return_value="system"), patch(
            "agent.runner.process_messages", side_effect=_fake_process_messages
        ), patch("ui.status.ThinkingStatus", _FakeStatus):
            result = asyncio.run(runner._run_async("investigate", {"target": "CRBN"}))

        assert "All done." in result.summary
        assert result.raw_results["answer"] == "Structured answer"
        assert result.metadata["tool_call_count"] == 2
        runner._print_usage.assert_called_once()
        runner._start_background_task_watcher.assert_called_once()


class TestBackgroundWatcherAsync:
    def test_watch_background_tasks_async_completes_from_notification(self):
        runner, session = _mk_runner(headless=False)
        runner._bg_watch_timeout_s = 30
        runner._bg_watch_enable_taskoutput_probe = False
        runner._probe_local_task_outputs = MagicMock()
        runner._notify_terminal_task_completion = MagicMock()

        class _FakeOptions:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class _FakeClient:
            def __init__(self, options=None):
                self.options = options

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def connect(self):
                return None

            def receive_messages(self):
                async def _iter():
                    yield SimpleNamespace(
                        subtype="task_notification",
                        data={
                            "task_id": "task-1",
                            "status": "completed",
                            "summary": "done",
                            "output_file": "/tmp/task-1.output",
                        },
                    )

                return _iter()

        fake_sdk = types.ModuleType("claude_agent_sdk")
        fake_sdk.ClaudeAgentOptions = _FakeOptions
        fake_sdk.ClaudeSDKClient = _FakeClient

        runner._bg_watch_state["sess-1"] = {
            "pending_task_ids": ["task-1"],
            "completed_task_ids": [],
            "status": "running",
            "last_update_at": 0.0,
        }

        with patch.dict("sys.modules", {"claude_agent_sdk": fake_sdk}), patch(
            "ui.traces.TraceRenderer", return_value=MagicMock()
        ):
            asyncio.run(
                runner._watch_background_tasks_async(
                    session_id="sess-1",
                    task_ids=["task-1"],
                    model="claude-sonnet-4-5-20250929",
                    env={},
                )
            )

        state = runner._bg_watch_state["sess-1"]
        assert state["status"] == "completed"
        assert state["pending_task_ids"] == []
        assert "task-1" in state["completed_task_ids"]
        runner._notify_terminal_task_completion.assert_called_once()
