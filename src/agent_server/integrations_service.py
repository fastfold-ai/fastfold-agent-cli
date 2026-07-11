"""Shared implementation behind CLI keys and web integrations."""

from __future__ import annotations

import os
import threading

from agent.config import API_KEYS, Config
from agent_server.models import (
    IntegrationField,
    IntegrationProvider,
    ValidateIntegrationResponse,
)

KEY_ALIASES = {
    "llm.anthropic_api_key": "anthropic",
    "llm.openai_api_key": "openai",
    "llm.openai_compatible_api_key": "openai-compatible",
    "api.fastfold_cloud_key": "fastfold-cloud",
    "api.boltz_api_key": "boltz",
    "api.ibm_rxn_key": "ibm-rxn",
    "api.lens_key": "lens",
    "notification.sendgrid_api_key": "sendgrid",
    "compute.lambda_api_key": "lambda-labs",
    "compute.runpod_api_key": "runpod",
}

PROVIDER_METADATA = {
    "modal": {
        "name": "Modal",
        "category": "Agent Runtime",
        "description": "Modal runtime credentials and on-demand GPU compute.",
    },
    "langsmith": {
        "name": "LangSmith",
        "category": "Observability",
        "description": "LangChain tracing API keys and project overrides.",
    },
    "nvidia": {
        "name": "NVIDIA",
        "category": "AI Models",
        "description": "NVIDIA API key overrides for model inference.",
    },
    "tavily": {
        "name": "Tavily",
        "category": "Search",
        "description": "Tavily search API key used by research tools.",
    },
    "slack": {
        "name": "Slack",
        "category": "Communication",
        "description": "Slack reports and agent notifications.",
    },
    "custom-webhook": {
        "name": "Custom Webhook",
        "category": "Automation",
        "description": "Completion event webhook URL and signing secret.",
    },
}


def category_for(config_key: str) -> str:
    if config_key.startswith("llm."):
        return "AI Models"
    if config_key.startswith("compute."):
        return "Compute"
    if config_key.startswith("notification."):
        return "Notifications"
    if config_key.startswith("api."):
        return "Scientific Services"
    return "Other"


class IntegrationsService:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._providers: dict[str, list[tuple[str, dict]]] = {}
        for config_key, info in API_KEYS.items():
            key = str(
                info.get("provider_key")
                or KEY_ALIASES.get(config_key)
                or info["env_var"].lower().replace("_", "-")
            )
            self._providers.setdefault(key, []).append((config_key, info))

    @staticmethod
    def _masked(value: str, *, is_secret: bool = True) -> str | None:
        normalized = value.strip()
        if not normalized:
            return None
        if not is_secret:
            return normalized
        if len(normalized) <= 8:
            return "••••••••"
        return f"{normalized[:4]}…{normalized[-4:]}"

    def _provider(self, key: str, entries: list[tuple[str, dict]]) -> IntegrationProvider:
        config = Config.load()
        fields: list[IntegrationField] = []
        for config_key, info in entries:
            env_value = str(os.environ.get(info["env_var"]) or "").strip()
            config_value = str(config.get(config_key) or "").strip()
            value = env_value or config_value
            source = "environment" if env_value else "config" if config_value else "none"
            fields.append(
                IntegrationField(
                    env_var=str(info["env_var"]),
                    label=str(info["name"]),
                    is_secret=bool(info.get("secret", True)),
                    configured=bool(value),
                    source=source,
                    masked_preview=self._masked(
                        value,
                        is_secret=bool(info.get("secret", True)),
                    ),
                )
            )

        first_config_key, first_info = entries[0]
        metadata = PROVIDER_METADATA.get(key, {})
        configured_fields = [field for field in fields if field.configured]
        primary = configured_fields[0] if configured_fields else fields[0]
        return IntegrationProvider(
            key=key,
            name=str(metadata.get("name") or first_info["name"]),
            category=str(metadata.get("category") or category_for(first_config_key)),
            description=str(metadata.get("description") or first_info["description"]),
            env_var=fields[0].env_var,
            configured=bool(configured_fields),
            source=primary.source,
            masked_preview=primary.masked_preview,
            setup_url=str(first_info.get("url") or "") or None,
            free=all(bool(info.get("free")) for _, info in entries),
            fields=fields,
        )

    def list(self) -> list[IntegrationProvider]:
        return sorted(
            (
                self._provider(key, entries)
                for key, entries in self._providers.items()
            ),
            key=lambda provider: (provider.category, provider.name.lower()),
        )

    def get(self, key: str) -> IntegrationProvider | None:
        entries = self._providers.get(key)
        return self._provider(key, entries) if entries else None

    def update(self, key: str, values: dict[str, str]) -> IntegrationProvider:
        entries = self._providers.get(key)
        if entries is None:
            raise KeyError(key)
        by_env = {str(info["env_var"]): config_key for config_key, info in entries}
        normalized = {
            env_name: str(value).strip()
            for env_name, value in values.items()
            if env_name in by_env and str(value).strip()
        }
        if not normalized:
            raise ValueError("At least one integration value is required.")
        with self._lock:
            config = Config.load()
            for env_name, value in normalized.items():
                config.set(by_env[env_name], value)
            config.save()
        return self._provider(key, entries)

    def remove(self, key: str) -> IntegrationProvider:
        entries = self._providers.get(key)
        if entries is None:
            raise KeyError(key)
        with self._lock:
            config = Config.load()
            for config_key, _ in entries:
                config.unset(config_key)
            config.save()
        return self._provider(key, entries)

    def validate(self, key: str) -> ValidateIntegrationResponse:
        provider = self.get(key)
        if provider is None:
            raise KeyError(key)
        return ValidateIntegrationResponse(
            ok=provider.configured,
            configured=provider.configured,
            source=provider.source,
            message=(
                f"{provider.name} is configured."
                if provider.configured
                else f"{provider.name} is not configured."
            ),
        )
