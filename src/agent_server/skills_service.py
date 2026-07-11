"""Web-facing adapter around the CLI's canonical skill manager."""

from __future__ import annotations

from agent.skills import install_skill, list_skills, remove_skill, skill_info, upgrade_skills
from agent_server.models import SkillDetail, SkillMutationResponse, SkillSummary


class SkillsService:
    @staticmethod
    def _summary(info) -> SkillSummary:
        return SkillSummary(
            name=info.name,
            description=info.description or "",
            tags=list(info.tags or []),
            source=info.source or "",
            author=info.author or ("fastfold-ai" if info.source == "bundled" else ""),
            version=info.version,
            updated_at=info.updated_at or None,
        )

    def list(self) -> list[SkillSummary]:
        return [self._summary(info) for info in list_skills()]

    def get(self, name: str) -> SkillDetail | None:
        info = skill_info(name)
        if info is None or info.path is None:
            return None
        try:
            content = info.path.read_text(encoding="utf-8")
        except OSError:
            content = ""
        summary = self._summary(info)
        return SkillDetail(
            **summary.model_dump(),
            content=content,
            directory=str(info.directory) if info.directory else None,
        )

    def install(self, source: str) -> SkillMutationResponse:
        result = install_skill(source)
        return SkillMutationResponse(
            ok=bool(result.get("ok")),
            summary=str(result.get("summary") or ""),
            installed=[str(item) for item in result.get("installed") or []],
        )

    def remove(self, name: str) -> SkillMutationResponse:
        result = remove_skill(name)
        return SkillMutationResponse(
            ok=bool(result.get("ok")),
            summary=str(result.get("summary") or ""),
        )

    def upgrade(self) -> SkillMutationResponse:
        result = upgrade_skills()
        installed = [
            *[str(item) for item in result.get("added") or []],
            *[str(item) for item in result.get("updated") or []],
        ]
        return SkillMutationResponse(
            ok=not bool(result.get("failed")),
            summary=str(result.get("summary") or ""),
            installed=installed,
        )
