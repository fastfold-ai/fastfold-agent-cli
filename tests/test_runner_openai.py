"""Tests for OpenAI runtime path in AgentRunner."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ct.agent.mcp_server import RuntimeToolSpec
from ct.agent.runner import (
    AgentRunner,
    _format_openai_error,
    _openai_temperature_kwargs,
    _openai_token_limit_kwargs,
)


class _FakeToolCall:
    def __init__(self, call_id: str, name: str, arguments: str):
        self.id = call_id
        self.function = SimpleNamespace(name=name, arguments=arguments)

    def model_dump(self):
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.function.name,
                "arguments": self.function.arguments,
            },
        }


class _FakeExecutor:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    async def run(self, tool_name: str, args: dict):
        self.calls.append((tool_name, args))
        return {
            "content": [{"type": "text", "text": f"ran {tool_name} with {args}"}],
            "is_error": False,
        }


def _fake_response(content: str, tool_calls: list | None = None, in_tokens: int = 10, out_tokens: int = 5):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    choice = SimpleNamespace(message=msg)
    usage = SimpleNamespace(prompt_tokens=in_tokens, completion_tokens=out_tokens)
    return SimpleNamespace(choices=[choice], usage=usage)


def test_runner_openai_tool_loop_and_summary():
    cfg = MagicMock()
    cfg.get.side_effect = lambda key, default=None: {
        "llm.provider": "openai",
        "llm.model": "gpt-4o",
        "llm.openai_base_url": "http://localhost:11434/v1",
        "llm.temperature": 0.1,
        "agent.max_sdk_turns": 5,
        "agent.synthesis_max_tokens": 512,
        "agent.plan_preview": False,
        "agent.enable_experimental_tools": False,
    }.get(key, default)
    cfg.llm_api_key.return_value = "sk-test"
    cfg.llm_openai_base_url.return_value = "http://localhost:11434/v1"

    session = MagicMock()
    session.config = cfg
    session.console = MagicMock()
    runner = AgentRunner(session, headless=True)

    tool_specs = [
        RuntimeToolSpec(
            name="chemistry.pubchem_lookup",
            description="lookup",
            input_schema={"type": "object", "properties": {"compound": {"type": "string"}}},
        )
    ]
    tool_call = _FakeToolCall("call_1", "chemistry_pubchem_lookup", '{"compound":"aspirin"}')
    executor = _FakeExecutor()

    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _fake_response("Planning tool call", tool_calls=[tool_call], in_tokens=40, out_tokens=20),
        _fake_response("Final synthesized answer", tool_calls=[], in_tokens=50, out_tokens=30),
    ]

    with patch("openai.OpenAI", return_value=client) as openai_ctor, patch(
        "ct.agent.mcp_server.create_ct_tool_runtime",
        return_value=(tool_specs, executor, None, ["chemistry.pubchem_lookup"], []),
    ), patch("ct.agent.system_prompt.build_system_prompt", return_value="sys prompt"), patch(
        "ct.ui.traces.TraceRenderer"
    ):
        result = runner.run("test query", context={})

    assert "Final synthesized answer" in result.summary
    assert result.plan.steps
    assert result.plan.steps[0].tool == "chemistry.pubchem_lookup"
    assert executor.calls[0][0] == "chemistry.pubchem_lookup"
    assert result.metadata["sdk_input_tokens"] == 50
    assert result.metadata["sdk_output_tokens"] == 30
    tools_payload = client.chat.completions.create.call_args_list[0].kwargs["tools"]
    assert tools_payload[0]["function"]["name"] == "chemistry_pubchem_lookup"
    openai_kwargs = openai_ctor.call_args.kwargs
    assert openai_kwargs["base_url"] == "http://localhost:11434/v1"


def test_runner_openai_plan_preview_rejection_returns_error():
    cfg = MagicMock()
    cfg.get.side_effect = lambda key, default=None: {
        "llm.provider": "openai",
        "llm.model": "gpt-4o",
        "llm.temperature": 0.1,
        "agent.max_sdk_turns": 3,
        "agent.synthesis_max_tokens": 512,
        "agent.plan_preview": True,
        "agent.enable_experimental_tools": False,
    }.get(key, default)
    cfg.llm_api_key.return_value = "sk-test"
    cfg.llm_openai_base_url.return_value = None

    session = MagicMock()
    session.config = cfg
    session.console = MagicMock()
    runner = AgentRunner(session, headless=False)

    client = MagicMock()
    client.chat.completions.create.return_value = _fake_response("Plan: 1) do X 2) do Y", tool_calls=[])

    with patch("openai.OpenAI", return_value=client), patch(
        "ct.agent.mcp_server.create_ct_tool_runtime",
        return_value=([], _FakeExecutor(), None, [], []),
    ), patch("ct.agent.system_prompt.build_system_prompt", return_value="sys prompt"), patch(
        "ct.ui.traces.TraceRenderer"
    ), patch("builtins.input", return_value="n"):
        result = runner.run("test query", context={}, progress_callback=lambda *_a, **_k: None)

    assert "User rejected plan preview" in result.summary


def test_format_openai_error_includes_model_hint():
    exc = Exception("Error code: 400")
    exc.status_code = 400
    exc.body = {
        "error": {
            "message": "The model `gpt-5.5` does not exist or you do not have access.",
            "code": "model_not_found",
        }
    }
    text = _format_openai_error(exc)
    assert "request failed" in text.lower()
    assert "Use `/model`" in text


def test_format_openai_error_reads_message_from_response_json():
    class _Resp:
        def json(self):
            return {
                "error": {
                    "message": "The model `gpt-5.4-nano` is not supported for this endpoint.",
                    "code": "unsupported_model",
                }
            }

    exc = Exception("Error code: 400")
    exc.status_code = 400
    exc.response = _Resp()
    text = _format_openai_error(exc)
    assert "not supported for this endpoint" in text
    assert "Use `/model`" in text


def test_runner_openai_caps_tool_payload_to_provider_limit():
    cfg = MagicMock()
    cfg.get.side_effect = lambda key, default=None: {
        "llm.provider": "openai",
        "llm.model": "gpt-5.5",
        "llm.temperature": 0.1,
        "agent.max_sdk_turns": 2,
        "agent.synthesis_max_tokens": 256,
        "agent.plan_preview": False,
        "agent.enable_experimental_tools": False,
    }.get(key, default)
    cfg.llm_api_key.return_value = "sk-test"
    cfg.llm_openai_base_url.return_value = None

    session = MagicMock()
    session.config = cfg
    session.console = MagicMock()
    runner = AgentRunner(session, headless=True)

    tool_specs = [
        RuntimeToolSpec(
            name=f"chemistry.tool_{i}",
            description="tool",
            input_schema={"type": "object", "properties": {}},
        )
        for i in range(183)
    ]
    tool_specs.append(
        RuntimeToolSpec(
            name="run_python",
            description="python",
            input_schema={"type": "object", "properties": {"code": {"type": "string"}}},
        )
    )
    tool_specs.append(
        RuntimeToolSpec(
            name="shell.run",
            description="shell",
            input_schema={"type": "object", "properties": {"command": {"type": "string"}}},
        )
    )

    client = MagicMock()
    client.chat.completions.create.return_value = _fake_response(
        "Final synthesized answer", tool_calls=[], in_tokens=20, out_tokens=10
    )

    with patch("openai.OpenAI", return_value=client), patch(
        "ct.agent.mcp_server.create_ct_tool_runtime",
        return_value=(tool_specs, _FakeExecutor(), None, [s.name for s in tool_specs], []),
    ), patch("ct.agent.system_prompt.build_system_prompt", return_value="sys prompt"), patch(
        "ct.ui.traces.TraceRenderer"
    ):
        result = runner.run("test query", context={})

    assert "Final synthesized answer" in result.summary
    create_calls = client.chat.completions.create.call_args_list
    assert create_calls
    tools_payload = create_calls[0].kwargs["tools"]
    assert len(tools_payload) == 128
    tool_names = [t["function"]["name"] for t in tools_payload]
    assert "run_python" in tool_names
    assert "shell_run" in tool_names


def test_openai_token_limit_kwargs_for_gpt5_family():
    assert _openai_token_limit_kwargs("gpt-5.5", 1234) == {"max_completion_tokens": 1234}
    assert _openai_token_limit_kwargs("gpt-5-mini", 99) == {"max_completion_tokens": 99}


def test_openai_token_limit_kwargs_for_legacy_models():
    assert _openai_token_limit_kwargs("gpt-4o", 512) == {"max_tokens": 512}


def test_openai_temperature_kwargs_for_gpt5_family():
    assert _openai_temperature_kwargs("gpt-5.5", 0.1) == {}
    assert _openai_temperature_kwargs("gpt-5-mini", 0.3) == {}


def test_openai_temperature_kwargs_for_legacy_models():
    assert _openai_temperature_kwargs("gpt-4o", 0.1) == {"temperature": 0.1}
