"""Tests for agent.mcp_server tool formatting and runtime execution."""

import asyncio
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from agent.mcp_server import (
    RuntimeToolExecutor,
    _format_tool_result,
    _make_tool_handler,
    _params_to_json_schema,
    create_ct_tool_runtime,
)


class TestFormatToolResult:
    def test_non_dict_truncates(self):
        long_text = "x" * 9000
        assert len(_format_tool_result(long_text)) == 8000

    def test_dict_with_summary_and_compact_keys(self):
        result = {
            "summary": "Found 3 hits",
            "top_hits": [{"a": 1}, {"b": 2}],
            "gene": "TP53",
        }
        text = _format_tool_result(result)
        assert "Found 3 hits" in text
        assert "top_hits: list with 2 entries" in text
        assert "gene: TP53" in text

    def test_long_field_value_truncated(self):
        result = {"summary": "ok", "data": "z" * 2000}
        text = _format_tool_result(result)
        assert "chars total" in text


class TestParamsToJsonSchema:
    def test_empty_parameters(self):
        schema = _params_to_json_schema({})
        assert schema == {"type": "object", "properties": {}}

    def test_maps_descriptions(self):
        schema = _params_to_json_schema({"gene": "Gene symbol", "limit": "Max rows"})
        assert schema["properties"]["gene"]["description"] == "Gene symbol"
        assert schema["properties"]["limit"]["type"] == "string"


@dataclass
class _FakeTool:
    name: str = "target.druggability"
    description: str = "Score druggability"
    parameters: dict = None
    category: str = "target"

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {"gene": "Gene symbol"}

    def run(self, **kwargs):
        return {"summary": f"Druggability for {kwargs.get('gene')}"}


class TestRuntimeToolExecutor:
    def test_unknown_tool_returns_error(self):
        executor = RuntimeToolExecutor({})

        async def _run():
            return await executor.run("missing.tool", {})

        result = asyncio.run(_run())
        assert result["is_error"] is True
        assert "unknown tool" in result["content"][0]["text"]


class TestMakeToolHandler:
    def test_coerces_numeric_and_bool_strings(self):
        session = MagicMock()
        tool = _FakeTool()
        calls = []

        def _record(**kwargs):
            calls.append(kwargs)
            return {"summary": "ok"}

        tool.run = _record
        handler = _make_tool_handler(tool, session)
        result = asyncio.run(handler({"gene": "TP53", "limit": "10", "active": "true"}))
        assert result["content"][0]["text"] == "ok"
        assert calls[0]["limit"] == 10
        assert calls[0]["active"] is True

    def test_tool_exception_returns_is_error(self):
        session = MagicMock()
        tool = _FakeTool()

        def _boom(**kwargs):
            raise RuntimeError("tool failed")

        tool.run = _boom
        handler = _make_tool_handler(tool, session)
        result = asyncio.run(handler({"gene": "TP53"}))
        assert result["is_error"] is True
        assert "tool failed" in result["content"][0]["text"]


class TestCreateCtToolRuntime:
    def test_builds_specs_and_handlers(self):
        session = MagicMock()
        fake_tool = _FakeTool()

        with patch("tools.registry.list_tools", return_value=[fake_tool]):
            with patch("tools.ensure_loaded"):
                with patch("tools.EXPERIMENTAL_CATEGORIES", set()):
                    specs, executor, sandbox, names, _buffer = create_ct_tool_runtime(
                        session, include_run_python=False
                    )

        assert len(specs) == 1
        assert specs[0].name == "target.druggability"
        assert "target.druggability" in executor._handlers
        assert "run_python" not in names
        assert sandbox is None

    def test_excludes_categories_and_tools(self):
        session = MagicMock()
        fake_tool = _FakeTool(category="experimental")

        with patch("tools.registry.list_tools", return_value=[fake_tool]):
            with patch("tools.ensure_loaded"):
                with patch("tools.EXPERIMENTAL_CATEGORIES", {"experimental"}):
                    _, _, _, names, _ = create_ct_tool_runtime(session, include_run_python=False)

        assert names == []


class TestRunPythonHandler:
    def test_empty_code_returns_error(self):
        from agent.mcp_server import _make_run_python_handler

        session = MagicMock()
        session.config = MagicMock()
        session.config.get.side_effect = lambda key, default=None: {
            "sandbox.timeout": 30,
            "sandbox.max_retries": 1,
        }.get(key, default)

        with patch("agent.sandbox.Sandbox") as mock_sandbox_cls:
            mock_sandbox_cls.return_value.load_datasets.return_value = None
            handler, _sandbox = _make_run_python_handler(session)

        result = asyncio.run(handler({"code": "   "}))
        assert result["is_error"] is True
        assert "no code" in result["content"][0]["text"].lower()

    def test_successful_execution_with_trace_buffer(self):
        from agent.mcp_server import _make_run_python_handler

        session = MagicMock()
        session.config = MagicMock()
        session.config.get.side_effect = lambda key, default=None: {
            "sandbox.timeout": 30,
            "sandbox.max_retries": 1,
        }.get(key, default)

        buffer = []
        mock_sandbox = MagicMock()
        mock_sandbox.execute.return_value = {
            "stdout": "hello world",
            "plots": ["/tmp/plot.png"],
            "exports": [],
            "error": None,
        }
        mock_sandbox.get_variable.return_value = {
            "summary": "done",
            "answer": "42",
        }

        with patch("agent.sandbox.Sandbox", return_value=mock_sandbox):
            handler, sandbox = _make_run_python_handler(session, buffer)

        result = asyncio.run(handler({"code": "print('hello world')"}))
        assert result["is_error"] is False
        text = result["content"][0]["text"]
        assert "hello world" in text
        assert "Result summary: done" in text
        assert len(buffer) == 1
        assert buffer[0]["tool"] == "run_python"
        assert sandbox is mock_sandbox


class TestCreateCtMcpServer:
    def test_creates_server_with_run_python(self):
        from agent.mcp_server import create_ct_mcp_server

        session = MagicMock()
        session.config = MagicMock()
        session.config.get.side_effect = lambda key, default=None: {
            "sandbox.timeout": 30,
            "sandbox.max_retries": 1,
        }.get(key, default)
        fake_tool = _FakeTool()

        mock_sandbox = MagicMock()
        mock_sandbox.load_datasets.return_value = None

        with patch("tools.registry.list_tools", return_value=[fake_tool]):
            with patch("tools.ensure_loaded"):
                with patch("tools.EXPERIMENTAL_CATEGORIES", set()):
                    with patch("agent.sandbox.Sandbox", return_value=mock_sandbox):
                        with patch("agent.mcp_server.create_sdk_mcp_server") as mock_create:
                            mock_create.return_value = MagicMock(name="server")
                            server, sandbox, names, buffer = create_ct_mcp_server(
                                session, include_run_python=True
                            )

        assert server is mock_create.return_value
        assert sandbox is mock_sandbox
        assert "target.druggability" in names
        assert "run_python" in names
        assert isinstance(buffer, list)
        mock_create.assert_called_once()
        sdk_tools = mock_create.call_args.kwargs["tools"]
        tool_names = {t.name for t in sdk_tools}
        assert "run_python" in tool_names


class TestRuntimeToolExecutorSuccess:
    def test_known_tool_invokes_handler(self):
        async def _handler(args):
            return {"content": [{"type": "text", "text": "ok"}], "is_error": False}

        executor = RuntimeToolExecutor({"demo.tool": _handler})
        result = asyncio.run(executor.run("demo.tool", {"x": 1}))
        assert result["is_error"] is False