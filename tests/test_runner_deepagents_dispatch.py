"""Tests for the AgentRunner deepagents dispatch and result assembly."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.runner import AgentRunner


class _Cfg:
    def __init__(self, data=None):
        self.data = dict(data or {})

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value

    def llm_api_key(self, _provider=None):
        return "k"


def _mk_runner(provider="anthropic", *, headless=True, trace_store=None, extra=None):
    data = {"llm.provider": provider}
    data.update(extra or {})
    session = SimpleNamespace(config=_Cfg(data), console=MagicMock())
    return AgentRunner(
        session=session, trajectory=None, headless=headless, trace_store=trace_store
    )


def _run(coro):
    return asyncio.run(coro)


class TestRunAsyncDispatch:
    def test_anthropic_routes_to_deepagents(self):
        runner = _mk_runner("anthropic")
        sentinel = object()

        async def fake(query, context, progress_callback):
            return sentinel

        runner._run_async_deepagents = fake
        assert _run(runner._run_async("q")) is sentinel

    def test_openai_routes_to_deepagents(self):
        runner = _mk_runner("openai")
        sentinel = object()

        async def fake(query, context, progress_callback):
            return sentinel

        runner._run_async_deepagents = fake
        assert _run(runner._run_async("q")) is sentinel

    def test_unsupported_provider_returns_error(self):
        runner = _mk_runner("cohere")
        result = _run(runner._run_async("q"))
        assert "Unsupported llm.provider 'cohere'" in result.summary


class TestPlanPreview:
    def _model(self, content):
        class _M:
            async def ainvoke(self, _messages):
                return SimpleNamespace(content=content)

        return _M()

    def test_plan_accepted(self):
        runner = _mk_runner()
        with patch("builtins.input", return_value="y"):
            assert _run(runner._deepagents_plan_preview(self._model("plan"), "x")) is True

    def test_plan_accepted_on_empty_enter(self):
        runner = _mk_runner()
        with patch("builtins.input", return_value=""):
            assert _run(runner._deepagents_plan_preview(self._model("plan"), "x")) is True

    def test_plan_rejected(self):
        runner = _mk_runner()
        with patch("builtins.input", return_value="n"):
            assert _run(runner._deepagents_plan_preview(self._model("plan"), "x")) is False

    def test_plan_preview_model_error_defaults_true(self):
        runner = _mk_runner()

        class _Bad:
            async def ainvoke(self, _messages):
                raise RuntimeError("boom")

        assert _run(runner._deepagents_plan_preview(_Bad(), "x")) is True


class TestRunAsyncDeepagentsHappyPath:
    def test_assembles_execution_result(self):
        runner = _mk_runner(
            "anthropic",
            extra={"llm.model": "claude-sonnet-4-5-20250929"},
        )

        fake_sandbox = SimpleNamespace(
            get_variable=lambda name: {"answer": "the answer"} if name == "result" else None
        )

        process_result = {
            "full_text": ["This is the synthesized answer."],
            "tool_calls": [{"name": "target.neo", "input": {"gene": "TP53"}}],
            "token_usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "cache_creation_input_tokens": 5,
                "cache_read_input_tokens": 10,
            },
            "model_call_count": 2,
        }

        async def fake_process_events(events, **kwargs):
            return process_result

        fake_agent = SimpleNamespace(astream_events=lambda *a, **k: MagicMock())

        with patch("agent.deepagents_runtime.build_chat_model", return_value=MagicMock()), \
            patch(
                "agent.deepagents_runtime.create_ct_langchain_tools",
                return_value=([MagicMock(name="t")], fake_sandbox, [], {}),
            ), \
            patch("agent.deepagents_runtime.skill_source_dirs", return_value=[]), \
            patch("agent.deepagents_runtime.process_events", side_effect=fake_process_events), \
            patch("agent.system_prompt.build_system_prompt", return_value="SYS"), \
            patch("ui.traces.TraceRenderer", return_value=MagicMock()), \
            patch("deepagents.create_deep_agent", return_value=fake_agent), \
            patch("deepagents.backends.FilesystemBackend", return_value=MagicMock()):
            result = _run(runner._run_async_deepagents("find TP53 degraders"))

        assert result.summary == "This is the synthesized answer."
        assert result.metadata["runtime"] == "deepagents"
        assert result.metadata["sdk_input_tokens"] == 100
        assert result.metadata["sdk_output_tokens"] == 20
        assert result.metadata["sdk_cache_read_input_tokens"] == 10
        assert result.raw_results["answer"] == "the answer"
        assert len(result.raw_results["tool_calls"]) == 1

    def test_build_chat_model_value_error_returns_error_result(self):
        runner = _mk_runner("anthropic")
        with patch(
            "agent.deepagents_runtime.build_chat_model",
            side_effect=ValueError("bad provider config"),
        ):
            result = _run(runner._run_async_deepagents("q"))
        assert "bad provider config" in result.summary
