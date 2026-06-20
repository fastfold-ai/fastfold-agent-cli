from __future__ import annotations

import sys

from agent import runner


def test_windows_should_inline_system_prompt_auto_threshold(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("FASTFOLD_INLINE_SYSTEM_PROMPT", raising=False)
    monkeypatch.delenv("FASTFOLD_INLINE_SYSTEM_PROMPT_THRESHOLD", raising=False)
    monkeypatch.delenv("FASTFOLD_WINDOWS_INLINE_SYSTEM_PROMPT", raising=False)
    monkeypatch.setenv("FASTFOLD_WINDOWS_INLINE_SYSTEM_PROMPT_THRESHOLD", "10")
    assert runner._should_inline_system_prompt("x" * 11) is True
    assert runner._should_inline_system_prompt("x" * 9) is False


def test_inline_system_prompt_into_user_prompt_shapes_payload():
    system_prompt = "SYSTEM RULES"
    user_prompt = "What is ALK resistance?"
    effective, bridged = runner._inline_system_prompt_into_user_prompt(
        system_prompt, user_prompt
    )
    assert "Follow the SYSTEM INSTRUCTIONS block" in effective
    assert "SYSTEM INSTRUCTIONS" in bridged
    assert "SYSTEM RULES" in bridged
    assert "What is ALK resistance?" in bridged
