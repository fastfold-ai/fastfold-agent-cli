"""Runtime settings shared by terminal and web presentation layers."""

from agent.config import Config
from agent_server.models import RuntimeSettings, UpdateRuntimeSettingsRequest


class SettingsService:
    @staticmethod
    def get() -> RuntimeSettings:
        config = Config.load()
        return RuntimeSettings(
            provider=str(config.get("llm.provider") or "anthropic"),
            model=str(config.get("llm.model") or "claude-sonnet-4-5-20250929"),
            openai_base_url=str(config.get("llm.openai_base_url") or "").strip() or None,
            agent_profile=str(config.get("agent.profile") or "research"),
            tool_mode=str(config.get("agent.tool_mode") or "ptc"),
        )

    @staticmethod
    def update(payload: UpdateRuntimeSettingsRequest) -> RuntimeSettings:
        config = Config.load()
        updates = {
            "llm.provider": payload.provider,
            "llm.model": payload.model,
            "llm.openai_base_url": payload.openai_base_url,
            "agent.profile": payload.agent_profile,
            "agent.tool_mode": payload.tool_mode,
        }
        for key, value in updates.items():
            if value is not None:
                config.set(key, value)
        config.save()
        return SettingsService.get()
