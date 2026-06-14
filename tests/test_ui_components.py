"""Tests for ui.markdown and ui.status display helpers."""

from io import StringIO

from rich.console import Console

from ui.markdown import LeftMarkdown
from ui.status import SPINNERS, ThinkingStatus, apply_gradient


class TestLeftMarkdown:
    def test_renders_heading(self):
        console = Console(file=StringIO(), force_terminal=True, width=80)
        md = LeftMarkdown("# Title\n\nBody text")
        console.print(md)
        output = console.file.getvalue()
        assert "Title" in output


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
