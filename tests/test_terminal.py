"""Tests for contextual ghost-text suggestions and terminal features."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path
from prompt_toolkit.document import Document
from prompt_toolkit.completion import CompleteEvent
from rich.markdown import Markdown
from rich.table import Table
from ct.ui.terminal import (  # type: ignore[import-untyped]
    _extract_llm_suggestions,
    extract_mentions,
    build_mention_context,
    MentionCompleter,
    MergedCompleter,
    SlashCompleter,
    DATASET_CANDIDATES,
    KNOWN_DATASETS,
    DEFAULT_SUGGESTIONS,
    SLASH_COMMANDS,
    InteractiveTerminal,
)



class TestSlashCommands:
    """Verify all expected slash commands are registered."""

    def test_core_commands_registered(self):
        expected = ["/help", "/tools", "/model", "/settings", "/usage", "/copy",
                    "/export", "/compact", "/new", "/sessions", "/resume", "/upgrade",
                    "/clear", "/exit", "/config", "/keys", "/doctor"]
        for cmd in expected:
            assert cmd in SLASH_COMMANDS, f"{cmd} not in SLASH_COMMANDS"

    def test_slash_commands_have_descriptions(self):
        for cmd, desc in SLASH_COMMANDS.items():
            assert isinstance(desc, str) and len(desc) > 5, f"{cmd} has no description"



class TestTerminalMethods:
    """Unit tests for InteractiveTerminal methods (no actual REPL)."""

    @pytest.fixture
    def terminal(self):
        """Create a terminal with mocked session."""
        with patch("ct.ui.terminal.InteractiveTerminal.__init__", return_value=None):
            t = InteractiveTerminal.__new__(InteractiveTerminal)
            t.session = MagicMock()
            t.session.verbose = False
            t.session.current_model = "claude-sonnet-4-5-20250929"
            t.console = MagicMock()
            t._last_response = None
            t._verbose_hint = None
            t._show_exit_hint = False
            t._suggestions = list(DEFAULT_SUGGESTIONS)
            t._suggestion_idx = 0
            t._run_lock = MagicMock()
            t._run_lock.__enter__ = MagicMock(return_value=None)
            t._run_lock.__exit__ = MagicMock(return_value=False)
            return t

    def test_model_display_name(self, terminal):
        assert terminal._model_display_name("claude-sonnet-4-5-20250929") == "Sonnet 4.5"
        assert terminal._model_display_name("claude-opus-4-6") == "Opus 4.6"
        assert terminal._model_display_name("gpt-5.5") == "GPT-5.5"
        assert terminal._model_display_name("gpt-5.4-mini") == "GPT-5.4 Mini"
        assert terminal._model_display_name("unknown-model") == "unknown-model"

    def test_switch_model_can_change_provider(self, terminal):
        terminal.session.config.get.side_effect = (
            lambda key, default=None: "anthropic" if key == "llm.provider" else default
        )
        terminal._prompt_session = MagicMock()
        # 4th option in AVAILABLE_MODELS order is gpt-5.5 (openai)
        terminal._prompt_session.prompt.return_value = "4"

        terminal._switch_model()

        terminal.session.set_model.assert_called_once_with("gpt-5.5", provider="openai")
        terminal.session.config.save.assert_called_once()

    def test_switch_model_no_change_when_same_model_and_provider(self, terminal):
        terminal.session.config.get.side_effect = (
            lambda key, default=None: "anthropic" if key == "llm.provider" else default
        )
        terminal._prompt_session = MagicMock()
        # 1st option is current model (claude-sonnet-4-5-20250929, anthropic)
        terminal._prompt_session.prompt.return_value = "1"

        terminal._switch_model()

        terminal.session.set_model.assert_not_called()
        terminal.session.config.save.assert_not_called()

    def test_ensure_llm_ready_prompts_and_saves_openai_key(self, terminal):
        terminal.session.config.llm_preflight_issue.side_effect = [
            "OpenAI API key not configured",
            None,
        ]
        terminal.session.config.get.side_effect = (
            lambda key, default=None: "openai" if key == "llm.provider" else default
        )
        terminal._secret_prompt_session = MagicMock()
        terminal._secret_prompt_session.prompt.return_value = "sk-openai-123"
        assert terminal._ensure_llm_ready_for_query() is True
        terminal.session.config.set.assert_called_once_with("llm.openai_api_key", "sk-openai-123")
        terminal.session.config.save.assert_called_once()

    def test_ensure_llm_ready_cancelled_returns_false(self, terminal):
        terminal.session.config.llm_preflight_issue.return_value = "OpenAI API key not configured"
        terminal.session.config.get.side_effect = (
            lambda key, default=None: "openai" if key == "llm.provider" else default
        )
        terminal._secret_prompt_session = MagicMock()
        terminal._secret_prompt_session.prompt.return_value = ""
        assert terminal._ensure_llm_ready_for_query() is False

    def test_ensure_llm_ready_prompts_and_saves_anthropic_key(self, terminal):
        terminal.session.config.llm_preflight_issue.side_effect = [
            "Anthropic API key not configured",
            None,
        ]
        terminal.session.config.get.side_effect = (
            lambda key, default=None: "anthropic" if key == "llm.provider" else default
        )
        terminal._secret_prompt_session = MagicMock()
        terminal._secret_prompt_session.prompt.return_value = "sk-ant-api03-test"
        assert terminal._ensure_llm_ready_for_query() is True
        terminal.session.config.set.assert_called_once_with(
            "llm.anthropic_api_key", "sk-ant-api03-test"
        )
        terminal.session.config.save.assert_called_once()

    def test_ensure_llm_ready_invalid_key_returns_false(self, terminal):
        terminal.session.config.llm_preflight_issue.return_value = "OpenAI API key not configured"
        terminal.session.config.get.side_effect = (
            lambda key, default=None: "openai" if key == "llm.provider" else default
        )
        terminal.session.config.set.side_effect = ValueError("Invalid OpenAI API key format")
        terminal._secret_prompt_session = MagicMock()
        terminal._secret_prompt_session.prompt.return_value = "bad-key"
        assert terminal._ensure_llm_ready_for_query() is False

    def test_copy_no_response(self, terminal):
        terminal._copy_last_response()
        terminal.console.print.assert_called_once()
        assert "No response" in str(terminal.console.print.call_args)

    def test_copy_with_response(self, terminal):
        terminal._last_response = "GSPT1 is a promising target."
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            terminal._copy_last_response()
            mock_run.assert_called_once()
            assert mock_run.call_args[1]["input"] == b"GSPT1 is a promising target."

    def test_copy_clipboard_unavailable(self, terminal):
        terminal._last_response = "Some text"
        with patch("subprocess.run", side_effect=FileNotFoundError):
            terminal._copy_last_response()
            assert "not available" in str(terminal.console.print.call_args).lower()

    def test_export_no_session(self, terminal):
        terminal._export_session()
        terminal.console.print.assert_called_once()
        assert "No session" in str(terminal.console.print.call_args)

    def test_export_with_trajectory(self, terminal, tmp_path):
        """Export should write a markdown file."""
        from ct.agent.trajectory import Turn, Trajectory  # type: ignore[import-untyped]
        terminal.agent = MagicMock()
        terminal.agent.trajectory = Trajectory()
        terminal.agent.trajectory.turns = [
            Turn(query="test query", answer="test answer",
                 entities=["TP53"], tools_used=["target.coessentiality"],
                 timestamp=0),
        ]
        with patch("pathlib.Path.home", return_value=tmp_path), patch(
            "ct.ui.terminal.Path.cwd", return_value=tmp_path
        ):
            terminal._export_session()
            exports = list((tmp_path / "exports").glob("*.md"))
            assert len(exports) == 1
            content = exports[0].read_text()
            assert "test query" in content
            assert "test answer" in content
            assert "TP53" in content
            exports[0].unlink(missing_ok=True)

    def test_run_shell_basic(self, terminal):
        terminal._run_shell("echo hello")
        # Should have printed the output
        terminal.console.print.assert_called()
        output = str(terminal.console.print.call_args_list[0])
        assert "hello" in output

    def test_run_shell_empty(self, terminal):
        terminal._run_shell("")
        terminal.console.print.assert_called_once()
        assert "Usage" in str(terminal.console.print.call_args)

    def test_run_shell_timeout(self, terminal):
        with patch("subprocess.run", side_effect=__import__("subprocess").TimeoutExpired("cmd", 30)):
            terminal._run_shell("sleep 100")
            assert "timed out" in str(terminal.console.print.call_args).lower()

    def test_run_shell_blocked(self, terminal):
        terminal._run_shell("rm -rf /tmp/x")
        terminal.console.print.assert_called_once()
        assert "blocked" in str(terminal.console.print.call_args).lower()

    def test_advance_suggestion(self, terminal):
        assert terminal._suggestion_idx == 0
        terminal._advance_suggestion()
        assert terminal._suggestion_idx == 1

    def test_update_suggestions_with_llm_result(self, terminal):
        """_update_suggestions extracts LLM suggestions from result.summary."""
        result = MagicMock()
        result.summary = (
            "## Suggested Next Steps\n"
            "- **\"Check co-essential partners of BRCA1\"**\n"
            "- **\"Run pathway enrichment on BRCA1 signature\"**\n"
        )
        terminal._update_suggestions("analyze BRCA1", result=result)
        assert len(terminal._suggestions) == 2
        assert any("BRCA1" in s for s in terminal._suggestions)

    def test_compact_context_calls_llm_chat_with_messages(self, terminal):
        from ct.agent.trajectory import Trajectory, Turn

        terminal.agent = MagicMock()
        trajectory = Trajectory()
        trajectory.turns = [
            Turn(query="q1", answer="a1"),
            Turn(query="q2", answer="a2"),
            Turn(query="q3", answer="a3"),
        ]
        terminal.agent.trajectory = trajectory

        mock_llm = MagicMock()
        mock_llm.chat.return_value = MagicMock(content="session summary text")
        terminal.session.get_llm.return_value = mock_llm

        terminal._compact_context("focus on actionable outcomes")

        kwargs = mock_llm.chat.call_args.kwargs
        assert kwargs["system"].startswith("You are a research session summarizer")
        assert kwargs["messages"][0]["role"] == "user"
        assert "focus on actionable outcomes" in kwargs["messages"][0]["content"]

        assert len(terminal.agent.trajectory.turns) == 2
        assert terminal.agent.trajectory.turns[0].query == "[session summary]"
        assert terminal.agent.trajectory.turns[0].answer == "session summary text"

    def test_run_orchestrated_updates_suggestions_from_result(self, terminal):
        terminal.agent = MagicMock()
        terminal._update_suggestions = MagicMock()

        result = MagicMock()
        result.summary = "summary"
        result.merged_plan = MagicMock()

        with patch("ct.agent.orchestrator.ResearchOrchestrator") as mock_orchestrator:
            mock_orchestrator.return_value.run.return_value = result
            terminal._run_orchestrated("query", {}, 2)

        terminal._update_suggestions.assert_called_once_with("query", result.merged_plan, result)

    def test_case_study_updates_suggestions_from_result(self, terminal):
        terminal._update_suggestions = MagicMock()

        result = MagicMock()
        result.summary = "summary"
        result.merged_plan = MagicMock()

        case = MagicMock()
        case.name = "Demo Case"
        case.description = "desc"
        case.compound = "lenalidomide"

        with patch("ct.agent.case_studies.CASE_STUDIES", {"demo": case}), patch(
            "ct.agent.case_studies.run_case_study",
            return_value=result,
        ):
            terminal._handle_case_study_command("/case-study demo", {})

        terminal._update_suggestions.assert_called_once_with(case.compound, result.merged_plan, result)

    def test_list_sessions_renders_threads_style_table(self, terminal):
        terminal.agent = MagicMock()
        terminal.agent.trajectory.session_id = "ab123456"
        sessions = [
            {
                "session_id": "ab123456",
                "title": "Current session title",
                "n_turns": 4,
                "model": "gpt-5.5",
                "created_at": 1000,
                "updated_at": 2000,
            },
            {
                "session_id": "cd789012",
                "title": "Other session title",
                "n_turns": 2,
                "model": "claude-sonnet-4-5-20250929",
                "created_at": 900,
                "updated_at": 1500,
            },
        ]
        with patch("ct.agent.trajectory.Trajectory.list_sessions", return_value=sessions), patch(
            "time.time", return_value=2002
        ):
            terminal._list_sessions()
        table_calls = [
            c for c in terminal.console.print.call_args_list if c.args and isinstance(c.args[0], Table)
        ]
        assert table_calls
        table = table_calls[-1].args[0]
        headers = [col.header for col in table.columns]
        assert headers == ["", "ID", "Preview", "Messages", "Model", "Last Used"]

    def test_list_sessions_preview_collapses_multiline_titles(self, terminal):
        terminal.agent = MagicMock()
        terminal.agent.trajectory.session_id = "ab123456"
        sessions = [
            {
                "session_id": "ab123456",
                "title": "Use esm1b in Fastfold to run a fold job.\n\nUse these sequences:\nSequence 1",
                "n_turns": 3,
                "model": "claude-sonnet-4-5-20250929",
                "created_at": 1000,
                "updated_at": 2000,
            },
        ]
        with patch("ct.agent.trajectory.Trajectory.list_sessions", return_value=sessions), patch(
            "time.time", return_value=2002
        ):
            terminal._list_sessions()

        table_calls = [
            c for c in terminal.console.print.call_args_list if c.args and isinstance(c.args[0], Table)
        ]
        table = table_calls[-1].args[0]
        previews = list(getattr(table.columns[2], "_cells", []))
        assert previews
        assert "\n" not in previews[0]
        assert "Use these sequences:" in previews[0]

    def test_resume_session_accepts_session_prefix(self, terminal):
        sessions = [
            {"session_id": "abc12345"},
            {"session_id": "xyz98765"},
        ]
        with patch("ct.agent.trajectory.Trajectory.list_sessions", return_value=sessions), patch(
            "ct.agent.loop.AgentLoop.resume"
        ) as mock_resume:
            terminal._resume_session("abc")
        mock_resume.assert_called_once()
        assert mock_resume.call_args.args[1] == "abc12345"

    def test_delete_session_accepts_prefix(self, terminal):
        sessions = [
            {"session_id": "abc12345"},
            {"session_id": "xyz98765"},
        ]
        terminal.agent = MagicMock()
        terminal.agent.trajectory.session_id = "other999"
        with patch("ct.agent.trajectory.Trajectory.list_sessions", return_value=sessions), patch(
            "ct.agent.trajectory.Trajectory.delete_session",
            return_value={"session_id": "abc12345", "session_deleted": True, "trace_deleted": True},
        ) as mock_delete:
            terminal._delete_session("abc")
        mock_delete.assert_called_once_with("abc12345")

    def test_delete_current_session_switches_to_new_loop(self, terminal):
        sessions = [{"session_id": "abc12345"}]
        terminal.agent = MagicMock()
        terminal.agent.trajectory.session_id = "abc12345"
        new_loop = MagicMock()
        new_loop.trajectory.session_id = "new11111"
        with patch("ct.agent.trajectory.Trajectory.list_sessions", return_value=sessions), patch(
            "ct.agent.trajectory.Trajectory.delete_session",
            return_value={"session_id": "abc12345", "session_deleted": True, "trace_deleted": False},
        ), patch("ct.agent.loop.AgentLoop", return_value=new_loop):
            terminal._delete_session("abc12345")
        assert terminal.agent is new_loop

    def test_new_session_creates_fresh_loop_and_resets_usage(self, terminal):
        terminal.agent = MagicMock()
        terminal._last_response = "old response"
        terminal._session_sdk_calls = 5
        terminal._session_sdk_input_tokens = 100
        terminal._session_sdk_output_tokens = 50
        terminal._session_sdk_cache_read_tokens = 10
        terminal._session_sdk_cache_creation_tokens = 2
        terminal._session_sdk_cost_usd = 1.2
        terminal._session_sdk_total_cost_usd = 1.3
        terminal._session_sdk_extra_server_tool_cost_usd = 0.1
        terminal._session_sdk_models = {"gpt-5.5"}
        terminal._session_sdk_turn_rows = [{"turn": 1}]

        new_loop = MagicMock()
        new_loop.trajectory.session_id = "new12345"
        with patch("ct.agent.loop.AgentLoop", return_value=new_loop), patch(
            "ct.cli.print_banner"
        ) as mock_print_banner:
            terminal._new_session()

        terminal.console.clear.assert_called_once()
        mock_print_banner.assert_called_once()
        assert terminal.agent is new_loop
        assert terminal._last_response is None
        assert terminal._session_sdk_calls == 0
        assert terminal._session_sdk_input_tokens == 0
        assert terminal._session_sdk_output_tokens == 0
        assert terminal._session_sdk_cache_read_tokens == 0
        assert terminal._session_sdk_cache_creation_tokens == 0
        assert terminal._session_sdk_cost_usd == 0.0
        assert terminal._session_sdk_total_cost_usd == 0.0
        assert terminal._session_sdk_extra_server_tool_cost_usd == 0.0
        assert terminal._session_sdk_models == set()
        assert terminal._session_sdk_turn_rows == []

    def test_print_exit_with_resume_hint(self, terminal):
        terminal.agent = MagicMock()
        terminal.agent.trajectory.session_id = "d0b0571d"
        terminal.agent.trajectory.turns = [MagicMock()]
        terminal._print_exit_with_resume_hint()
        printed = " ".join(str(c) for c in terminal.console.print.call_args_list)
        assert "Resume this session with:" in printed
        assert "fastfold --resume d0b0571d" in printed

    def test_print_exit_without_messages_hides_resume_hint(self, terminal):
        terminal.agent = MagicMock()
        terminal.agent.trajectory.session_id = "d0b0571d"
        terminal.agent.trajectory.turns = []
        terminal._print_exit_with_resume_hint()
        printed = " ".join(str(c) for c in terminal.console.print.call_args_list)
        assert "Goodbye!" in printed
        assert "Resume this session with:" not in printed

    def test_render_resumed_history_shows_query_and_answer(self, terminal):
        turn = MagicMock()
        turn.query = "what can you do?"
        turn.answer = "I can help with target discovery."
        terminal._render_resumed_history(40, [turn])
        printed = " ".join(str(c) for c in terminal.console.print.call_args_list)
        assert "Session History" in printed
        assert "what can you do?" in printed
        markdown_calls = [
            c.args[0]
            for c in terminal.console.print.call_args_list
            if c.args and isinstance(c.args[0], Markdown)
        ]
        assert markdown_calls
        assert markdown_calls[0].markup == "I can help with target discovery."

    def test_render_resumed_history_prefers_trace_replay(self, terminal):
        turn = MagicMock()
        turn.query = "run fold"
        turn.answer = "I can run fold."
        with patch.object(terminal, "_load_trace_blocks", return_value=[{"events": [{"type": "text", "content": "Trace text"}]}]), patch.object(
            terminal, "_replay_trace_events", return_value=True
        ) as mock_replay:
            terminal._render_resumed_history(40, [turn])
        mock_replay.assert_called_once()
        markdown_calls = [
            c.args[0]
            for c in terminal.console.print.call_args_list
            if c.args and isinstance(c.args[0], Markdown)
        ]
        assert not markdown_calls

    def test_render_resumed_history_trace_without_text_falls_back_to_answer(self, terminal):
        turn = MagicMock()
        turn.query = "run fold"
        turn.answer = "I can run fold."
        with patch.object(terminal, "_load_trace_blocks", return_value=[{"events": [{"type": "tool_start", "tool": "x"}]}]), patch.object(
            terminal, "_replay_trace_events", return_value=False
        ) as mock_replay:
            terminal._render_resumed_history(40, [turn])
        mock_replay.assert_called_once()
        markdown_calls = [
            c.args[0]
            for c in terminal.console.print.call_args_list
            if c.args and isinstance(c.args[0], Markdown)
        ]
        assert markdown_calls
        assert markdown_calls[0].markup == "I can run fold."

    def test_render_resumed_history_shows_generated_duration_from_trace_end(self, terminal):
        turn = MagicMock()
        turn.query = "run fold"
        turn.answer = "done"
        terminal._run_lock = MagicMock()
        terminal._run_lock.__enter__ = MagicMock(return_value=None)
        terminal._run_lock.__exit__ = MagicMock(return_value=False)
        terminal._session_sdk_turn_rows = [{"input_tokens": 49652, "output_tokens": 524}]
        with patch.object(
            terminal,
            "_load_trace_blocks",
            return_value=[{"events": [{"type": "text", "content": "ok"}], "end": {"duration_s": 24.0}}],
        ), patch.object(terminal, "_replay_trace_events", return_value=True):
            terminal._render_resumed_history(40, [turn])
        printed = " ".join(str(c) for c in terminal.console.print.call_args_list)
        assert "Generated for 24s" in printed
        assert "↑ 49,652 ↓ 524" in printed

    def test_restore_usage_from_trajectory(self, terminal):
        terminal.agent = MagicMock()
        terminal.agent.trajectory.get_usage_data.return_value = {
            "sdk_calls": 3,
            "sdk_input_tokens": 1000,
            "sdk_output_tokens": 400,
            "sdk_cache_read_tokens": 20,
            "sdk_cache_creation_tokens": 10,
            "sdk_cost_usd": 0.12,
            "sdk_total_cost_usd": 0.15,
            "sdk_extra_server_tool_cost_usd": 0.03,
            "sdk_models": ["gpt-5.5"],
            "sdk_turn_rows": [{"turn": 1, "input_tokens": 200}],
        }
        terminal._run_lock = MagicMock()
        terminal._run_lock.__enter__ = MagicMock(return_value=None)
        terminal._run_lock.__exit__ = MagicMock(return_value=False)
        terminal._restore_usage_from_trajectory()
        assert terminal._session_sdk_calls == 3
        assert terminal._session_sdk_input_tokens == 1000
        assert terminal._session_sdk_output_tokens == 400
        assert terminal._session_sdk_models == {"gpt-5.5"}

    def test_run_upgrade_uses_shared_cli_upgrade_flow(self, terminal):
        with patch("ct.cli.execute_upgrade", return_value=True) as mock_exec:
            terminal._run_upgrade()
        mock_exec.assert_called_once_with(console_obj=terminal.console, cfg=terminal.session.config)


class TestExtractMentions:
    """Tests for extract_mentions() — parsing @ tokens from query text."""

    def test_tool_mention(self):
        query, tools, datasets, workflows = extract_mentions("analyze CRBN with @target.coessentiality")
        assert "target.coessentiality" in tools
        assert datasets == []
        assert workflows == []
        assert "@" not in query

    def test_dataset_mention(self):
        query, tools, datasets, workflows = extract_mentions("check sensitivity in @depmap @prism")
        assert set(datasets) == {"depmap", "prism"}
        assert "@" not in query

    def test_mixed_mentions(self):
        query, tools, datasets, workflows = extract_mentions(
            "@depmap find @target.coessentiality for CRBN"
        )
        assert "target.coessentiality" in tools
        assert "depmap" in datasets
        assert "CRBN" in query

    def test_no_mentions(self):
        query, tools, datasets, workflows = extract_mentions("analyze CRBN degradation")
        assert query == "analyze CRBN degradation"
        assert tools == []
        assert datasets == []
        assert workflows == []

    def test_workflow_mention(self):
        query, tools, datasets, workflows = extract_mentions("run @target_validation on KRAS")
        assert "target_validation" in workflows
        assert tools == []
        assert datasets == []
        assert "KRAS" in query

    def test_mixed_with_workflow(self):
        query, tools, datasets, workflows = extract_mentions(
            "@depmap @target_validation check CRBN"
        )
        assert "depmap" in datasets
        assert "target_validation" in workflows
        assert "CRBN" in query


class TestBuildMentionContext:
    """Tests for build_mention_context() — formatting planner instructions."""

    def test_tools_only(self):
        ctx = build_mention_context(["target.coessentiality"], [])
        assert "target.coessentiality" in ctx
        assert "tool" in ctx.lower()

    def test_datasets_only(self):
        ctx = build_mention_context([], ["depmap", "prism"])
        assert "depmap" in ctx
        assert "prism" in ctx

    def test_empty(self):
        ctx = build_mention_context([], [])
        assert ctx == ""

    def test_workflow(self):
        ctx = build_mention_context([], [], ["target_validation"])
        assert "target_validation" in ctx
        assert "workflow" in ctx.lower()

    def test_mixed_with_workflow(self):
        ctx = build_mention_context(["target.coessentiality"], ["depmap"], ["compound_safety"])
        assert "target.coessentiality" in ctx
        assert "depmap" in ctx
        assert "compound_safety" in ctx


class TestMentionCompleterTabs:
    """Tests for MentionCompleter tabbed filtering."""

    @pytest.fixture
    def candidates(self):
        return [
            ("target.coessentiality", "target", "Co-essential gene networks", "tool"),
            ("expression.pathway_enrichment", "expression", "Pathway enrichment analysis", "tool"),
            ("depmap", "dataset", "DepMap CRISPR/model data", "dataset"),
            ("prism", "dataset", "PRISM drug sensitivity", "dataset"),
            ("CRISPRGeneEffect.csv", "file", "depmap/CRISPRGeneEffect.csv", "file"),
            ("target_validation", "workflow", "Validate a potential drug target (5 steps)", "workflow"),
        ]

    @pytest.fixture
    def completer(self, candidates):
        return MentionCompleter(candidates)

    def _get_completions(self, completer, text):
        doc = Document(text, len(text))
        return list(completer.get_completions(doc, CompleteEvent()))

    def test_all_tab_shows_everything(self, completer):
        completer._active_tab = 0
        completions = self._get_completions(completer, "@")
        assert len(completions) == 6

    def test_tools_tab_filters(self, completer):
        completer._active_tab = 1
        completions = self._get_completions(completer, "@")
        assert len(completions) == 2
        assert all("tool" in c.style for c in completions)

    def test_db_tab_filters(self, completer):
        completer._active_tab = 2
        completions = self._get_completions(completer, "@")
        assert len(completions) == 2
        names = {c.text for c in completions}
        assert names == {"@depmap", "@prism"}

    def test_files_tab_filters(self, completer):
        completer._active_tab = 3
        completions = self._get_completions(completer, "@")
        assert len(completions) == 1
        assert completions[0].text == "@CRISPRGeneEffect.csv"

    def test_flows_tab_filters(self, completer):
        completer._active_tab = 4  # Flows
        completions = self._get_completions(completer, "@")
        assert len(completions) == 1
        assert completions[0].text == "@target_validation"
        assert "workflow" in completions[0].style

    def test_substring_match(self, completer):
        completions = self._get_completions(completer, "@coess")
        assert len(completions) == 1
        assert completions[0].text == "@target.coessentiality"

    def test_no_at_returns_nothing(self, completer):
        completions = self._get_completions(completer, "hello world")
        assert len(completions) == 0


class TestMergedCompleterDispatch:
    """Tests for MergedCompleter dispatch logic."""

    @pytest.fixture
    def merged(self):
        candidates = [("depmap", "dataset", "DepMap data", "dataset")]
        return MergedCompleter(SlashCompleter(), MentionCompleter(candidates))

    def _get_completions(self, completer, text):
        doc = Document(text, len(text))
        return list(completer.get_completions(doc, CompleteEvent()))

    def test_slash_routes_to_slash(self, merged):
        completions = self._get_completions(merged, "/he")
        assert any(c.text == "/help" for c in completions)

    def test_at_routes_to_mention(self, merged):
        completions = self._get_completions(merged, "query @dep")
        assert any("depmap" in c.display_text for c in completions)

    def test_no_trigger_yields_nothing(self, merged):
        completions = self._get_completions(merged, "analyze compound")
        assert len(completions) == 0
