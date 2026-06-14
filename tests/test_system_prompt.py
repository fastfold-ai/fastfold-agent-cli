"""Tests for agent.system_prompt builder."""

from unittest.mock import patch

from agent.system_prompt import build_system_prompt


class TestSystemPrompt:
    @patch("agent.system_prompt._load_installed_skills", return_value="")
    def test_build_system_prompt_includes_identity(self, mock_skills):
        session = type("Session", (), {"config": type("Cfg", (), {
            "get": lambda self, key, default=None: default,
        })()})()

        prompt = build_system_prompt(session, tool_names=["target.druggability", "run_python"])
        assert "fastfold-agent-cli" in prompt or "Fastfold Agent" in prompt
        assert "Available Tools" in prompt
        assert "run_python" in prompt

    @patch("agent.system_prompt._load_installed_skills", return_value="## Skills\n- fold")
    def test_build_includes_skills_section(self, mock_skills):
        session = type("Session", (), {"config": type("Cfg", (), {
            "get": lambda self, key, default=None: default,
        })()})()

        prompt = build_system_prompt(session)
        assert "fold" in prompt

    @patch("agent.system_prompt._load_installed_skills", return_value="")
    def test_build_includes_data_and_history(self, mock_skills):
        session = type("Session", (), {"config": type("Cfg", (), {
            "get": lambda self, key, default=None: default,
        })()})()

        prompt = build_system_prompt(
            session,
            data_context="L1000 data available",
            history="User: prior question",
        )
        assert "L1000 data available" in prompt
        assert "prior question" in prompt
