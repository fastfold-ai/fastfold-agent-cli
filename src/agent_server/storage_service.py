"""Measure on-disk usage under ~/.fastfold-cli for the Storage settings page."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from agent.config import CONFIG_DIR, Config
from agent_server.models import (
    StorageCategory,
    StorageLocation,
    StorageReport,
)


@dataclass(frozen=True)
class _CategorySpec:
    id: str
    label: str
    color: str
    paths: tuple[Path, ...]
    children_from: Path | None = None


def _path_size_bytes(path: Path) -> int:
    """Return total bytes for a file or directory. Missing paths count as 0."""
    if not path.exists():
        return 0
    if path.is_file() or path.is_symlink():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    # Prefer `du` — much faster than walking multi‑GB dataset trees.
    try:
        result = subprocess.run(
            ["du", "-sk", str(path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0 and result.stdout.strip():
            kib = int(result.stdout.split()[0])
            return max(0, kib * 1024)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass

    total = 0
    try:
        for child in path.rglob("*"):
            try:
                if child.is_file() and not child.is_symlink():
                    total += child.stat().st_size
            except OSError:
                continue
    except OSError:
        return total
    return total


def _category_specs(home: Path, cfg: Config) -> list[_CategorySpec]:
    data_base = Path(
        str(cfg.get("data.base", str(home / "data")) or home / "data")
    ).expanduser()
    return [
        _CategorySpec(
            id="artifacts",
            label="Artifacts",
            color="#3b82f6",
            paths=(home / "workspaces",),
        ),
        _CategorySpec(
            id="datasets",
            label="Datasets",
            color="#22c55e",
            paths=(data_base,),
            children_from=data_base,
        ),
        _CategorySpec(
            id="sessions",
            label="Sessions & chats",
            color="#f97316",
            paths=(
                home / "sessions",
                home / "agent-server.db",
                home / "history",
            ),
        ),
        _CategorySpec(
            id="skills",
            label="Skills",
            color="#a855f7",
            paths=(
                home / "skills",
                home / ".agents" / "skills",
                home / ".claude" / "skills",
            ),
        ),
        _CategorySpec(
            id="downloads",
            label="Downloads",
            color="#06b6d4",
            paths=(home / "downloads",),
        ),
        _CategorySpec(
            id="logs",
            label="Logs",
            color="#94a3b8",
            paths=(home / "logs",),
        ),
        _CategorySpec(
            id="cache",
            label="Cache",
            color="#eab308",
            paths=(home / "cache",),
        ),
    ]


class StorageService:
    def get_report(self, *, home: Path | None = None) -> StorageReport:
        cfg = Config.load()
        root = (home or CONFIG_DIR).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)

        claimed_names: set[str] = set()
        categories: list[StorageCategory] = []
        for spec in _category_specs(root, cfg):
            bytes_used = 0
            for path in spec.paths:
                resolved = path.expanduser()
                bytes_used += _path_size_bytes(resolved)
                try:
                    abs_path = resolved.resolve()
                except OSError:
                    abs_path = resolved
                if abs_path.parent == root or abs_path == root:
                    claimed_names.add(abs_path.name)
                elif root in abs_path.parents:
                    claimed_names.add(abs_path.relative_to(root).parts[0])

            children: list[StorageCategory] = []
            if spec.children_from and spec.children_from.is_dir():
                try:
                    entries = sorted(
                        [
                            child
                            for child in spec.children_from.iterdir()
                            if child.name not in {".DS_Store"}
                        ],
                        key=lambda p: p.name.lower(),
                    )
                except OSError:
                    entries = []
                for child in entries:
                    child_bytes = _path_size_bytes(child)
                    children.append(
                        StorageCategory(
                            id=f"{spec.id}:{child.name}",
                            label=child.name,
                            color=spec.color,
                            bytes=child_bytes,
                            path=str(child),
                        )
                    )

            categories.append(
                StorageCategory(
                    id=spec.id,
                    label=spec.label,
                    color=spec.color,
                    bytes=bytes_used,
                    path=str(spec.paths[0]) if len(spec.paths) == 1 else None,
                    children=children,
                )
            )

        # Catch remaining top-level entries under the home root as "Other".
        other_bytes = 0
        try:
            for entry in root.iterdir():
                if entry.name in {".DS_Store", *claimed_names}:
                    continue
                other_bytes += _path_size_bytes(entry)
        except OSError:
            other_bytes = 0

        if other_bytes > 0:
            categories.append(
                StorageCategory(
                    id="other",
                    label="Other",
                    color="#71717a",
                    bytes=other_bytes,
                    path=str(root),
                )
            )

        total_bytes = sum(item.bytes for item in categories)
        try:
            disk = shutil.disk_usage(root)
            available_bytes = int(disk.free)
        except OSError:
            available_bytes = 0

        return StorageReport(
            location=StorageLocation(
                path=str(root),
                label="default location",
                bytes=total_bytes,
            ),
            categories=categories,
            total_bytes=total_bytes,
            available_bytes=available_bytes,
            checked_at=datetime.now(timezone.utc).isoformat(),
        )
