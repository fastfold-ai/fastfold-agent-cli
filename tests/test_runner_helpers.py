"""Tests for agent.runner pure helper functions."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.runner import (
    OPENAI_MAX_TOOLS,
    _as_dict,
    _build_openai_tool_name_maps,
    _extract_model_usage_totals,
    _extract_task_event,
    _extract_task_output_paths_from_text,
    _extract_usage_totals,
    _format_openai_error,
    _limit_openai_tool_specs,
    _looks_like_unverified_execution_claim,
    _openai_temperature_kwargs,
    _openai_token_limit_kwargs,
    _parse_task_probe_json,
    _safe_float,
    _safe_int_token,
    _sdk_msg_field,
    _sdk_msg_subtype,
)


class TestOpenAIHelpers:
    def test_limit_openai_tool_specs_under_cap(self):
        specs = [SimpleNamespace(name=f"tool.{i}") for i in range(5)]
        assert len(_limit_openai_tool_specs(specs, max_tools=10)) == 5

    def test_limit_openai_pins_sandbox_tools(self):
        specs = [SimpleNamespace(name=f"t{i}") for i in range(OPENAI_MAX_TOOLS + 1)]
        specs.append(SimpleNamespace(name="run_python"))
        limited = _limit_openai_tool_specs(specs, max_tools=OPENAI_MAX_TOOLS)
        names = {s.name for s in limited}
        assert "run_python" in names
        assert len(limited) == OPENAI_MAX_TOOLS

    def test_build_openai_tool_name_maps_sanitizes_dots(self):
        specs = [
            SimpleNamespace(name="target.druggability"),
            SimpleNamespace(name="target.druggability"),
        ]
        fwd, rev = _build_openai_tool_name_maps(specs)
        assert rev["target.druggability"] in fwd
        assert fwd[rev["target.druggability"]] == "target.druggability"

    def test_gpt5_token_and_temperature_kwargs(self):
        assert _openai_token_limit_kwargs("gpt-5-preview", 1024) == {
            "max_completion_tokens": 1024
        }
        assert _openai_temperature_kwargs("gpt-5-preview", 0.7) == {}
        assert _openai_token_limit_kwargs("gpt-4o", 1024) == {"max_tokens": 1024}
        assert _openai_temperature_kwargs("gpt-4o", 0.7) == {"temperature": 0.7}


class TestSdkMessageHelpers:
    def test_sdk_msg_field_from_attribute_and_data(self):
        msg = SimpleNamespace(task_id="abc", data={"description": "from data"})
        assert _sdk_msg_field(msg, "task_id") == "abc"
        assert _sdk_msg_field(msg, "description") == "from data"
        assert _sdk_msg_field(msg, "missing", "default") == "default"

    def test_sdk_msg_subtype(self):
        msg = SimpleNamespace(subtype="task_started")
        assert _sdk_msg_subtype(msg) == "task_started"
        msg2 = SimpleNamespace(data={"subtype": "task_progress"})
        assert _sdk_msg_subtype(msg2) == "task_progress"

    def test_extract_task_event_started(self):
        msg = SimpleNamespace(
            subtype="task_started",
            task_id="t1",
            description="Running",
            task_type="local",
        )
        event = _extract_task_event(msg)
        assert event["type"] == "task_started"
        assert event["task_id"] == "t1"

    def test_extract_task_event_ignores_unknown(self):
        msg = SimpleNamespace(subtype="other")
        assert _extract_task_event(msg) is None


class TestSafeCoercion:
    @pytest.mark.parametrize("value,expected", [
        (None, 0),
        (42, 42),
        ("15", 15),
        ("bad", 0),
        (True, 1),
    ])
    def test_safe_int_token(self, value, expected):
        assert _safe_int_token(value) == expected

    @pytest.mark.parametrize("value,expected", [
        (None, 0.0),
        (1.5, 1.5),
        ("2.5", 2.5),
        ("", 0.0),
    ])
    def test_safe_float(self, value, expected):
        assert _safe_float(value) == expected

    def test_as_dict_from_dict_model_and_vars(self):
        assert _as_dict({"a": 1}) == {"a": 1}
        assert _as_dict(None) is None

        class _Model:
            def model_dump(self):
                return {"x": 2}

        assert _as_dict(_Model()) == {"x": 2}


class TestUsageExtraction:
    def test_extract_usage_totals(self):
        usage = {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_creation_input_tokens": 10,
            "cache_read_input_tokens": 5,
        }
        totals = _extract_usage_totals(usage)
        assert totals["input_tokens"] == 100
        assert totals["output_tokens"] == 50

    def test_extract_model_usage_totals(self):
        payload = {
            "claude-sonnet": {
                "inputTokens": 200,
                "outputTokens": 80,
                "costUSD": 0.05,
            }
        }
        totals = _extract_model_usage_totals(payload)
        assert totals["input_tokens"] == 200
        assert totals["cost_usd"] == 0.05
        assert "claude-sonnet" in totals["models"]


class TestExecutionClaimDetection:
    def test_no_tools_with_submit_claim(self):
        summary = "I successfully submitted fold job fold_abc123def456"
        assert _looks_like_unverified_execution_claim(summary, []) is True

    def test_with_tool_calls_not_flagged(self):
        summary = "Job fold_abc123 is running"
        assert _looks_like_unverified_execution_claim(summary, [{"tool": "fold"}]) is False

    def test_benign_summary(self):
        assert _looks_like_unverified_execution_claim("TP53 is a tumor suppressor.", []) is False


class TestOpenAIErrorFormatting:
    def test_auth_error_includes_hint(self):
        exc = Exception("invalid api key")
        exc.status_code = 401
        text = _format_openai_error(exc)
        assert "authentication failed" in text.lower()

    def test_model_error_includes_hint(self):
        exc = Exception("model not found")
        exc.status_code = 400
        text = _format_openai_error(exc)
        assert "/model" in text


class TestTaskHelpers:
    def test_extract_task_output_paths(self):
        text = ["See output at /home/user/tasks/task_abc.output for details."]
        paths = _extract_task_output_paths_from_text(text)
        assert paths["task_abc"] == "/home/user/tasks/task_abc.output"

    def test_parse_task_probe_json(self):
        raw = '{"task_abc": "completed", "task_def": "running"}'
        parsed = _parse_task_probe_json(raw)
        assert parsed["task_abc"] == "completed"
        assert parsed["task_def"] == "running"
