"""Tests for ui.markdown and ui.status display helpers."""

from io import StringIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rich.console import Console
from rich.markdown import Markdown

from ui.markdown import (
    LeftMarkdown,
    _split_mermaid_fences,
    print_markdown_with_mermaid,
)
from ui.status import SPINNERS, ThinkingStatus, apply_gradient


class TestLeftMarkdown:
    def test_renders_heading(self):
        console = Console(file=StringIO(), force_terminal=True, width=80)
        md = LeftMarkdown("# Title\n\nBody text")
        console.print(md)
        output = console.file.getvalue()
        assert "Title" in output


class TestMermaidMarkdown:
    def test_split_mermaid_fences(self):
        sections = _split_mermaid_fences(
            "Intro\n```mermaid\ngraph LR\nA-->B\n```\nOutro\n"
        )
        assert sections[0] == ("markdown", "Intro\n")
        assert sections[1][0] == "mermaid"
        assert "graph LR" in sections[1][1]
        assert sections[2] == ("markdown", "Outro\n")

    def test_split_mermaid_fence_unclosed_falls_back_to_markdown(self):
        sections = _split_mermaid_fences("Before\n```mermaid\ngraph LR\nA-->B\n")
        assert len(sections) == 1
        assert sections[0][0] == "markdown"
        assert sections[0][1].startswith("Before\n```mermaid")

    def test_print_markdown_with_mermaid_renders_diagram(self):
        console = MagicMock()
        with patch("termaid.render_rich", return_value="diagram-output"):
            print_markdown_with_mermaid(
                console,
                "Intro\n```mermaid\ngraph LR\nA-->B\n```\n",
            )
        assert any(call.args and call.args[0] == "diagram-output" for call in console.print.call_args_list)

    def test_print_markdown_with_mermaid_fallbacks_to_fence(self):
        console = MagicMock()
        with patch("termaid.render_rich", side_effect=RuntimeError("bad rich")), patch(
            "termaid.render", side_effect=RuntimeError("bad plain")
        ):
            print_markdown_with_mermaid(
                console,
                "```mermaid\ngraph LR\nA-->B\n```\n",
            )
        markdown_calls = [
            call.args[0]
            for call in console.print.call_args_list
            if call.args and isinstance(call.args[0], Markdown)
        ]
        assert markdown_calls
        assert "```mermaid" in markdown_calls[0].markup

    def test_print_markdown_with_mermaid_disabled(self):
        console = MagicMock()
        cfg = SimpleNamespace(
            get=lambda key, default=None: (
                False if key == "ui.mermaid.enabled" else default
            )
        )
        print_markdown_with_mermaid(
            console,
            "```mermaid\ngraph LR\nA-->B\n```\n",
            config=cfg,
        )
        markdown_calls = [
            call.args[0]
            for call in console.print.call_args_list
            if call.args and isinstance(call.args[0], Markdown)
        ]
        assert markdown_calls
        assert "```mermaid" in markdown_calls[0].markup


class TestStatusHelpers:
    def test_apply_gradient_empty(self):
        text = apply_gradient("")
        assert str(text) == ""

    def test_apply_gradient_colored(self):
        text = apply_gradient("Analyzing", elapsed_s=0.5)
        assert "Analyzing" in str(text)

    def test_spinners_defined(self):
        assert "dna_helix" in SPINNERS
        assert len(SPINNERS["dna_helix"]["frames"]) > 0

    def test_thinking_status_context_manager(self, captured_console):
        console, buf = captured_console
        with ThinkingStatus(console, "planning"):
            pass
        # Context manager should complete without error
        assert True
