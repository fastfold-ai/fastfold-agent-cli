"""
Skill management for fastfold-agent-cli.

Single source of truth for discovering, loading, installing, and removing
agent skills. A *skill* is a directory containing a ``SKILL.md`` file (plus
optional ``scripts/``, ``references/``, ``assets/``).

Skills are resolved from three tiers (highest priority first):

1. **global**  — ``~/.fastfold-cli/skills/`` (where ``skill add`` installs).
2. **project** — ``<cwd>/.claude/skills/`` gated by ``<cwd>/skills-lock.json``
   (legacy/back-compat; read-only here).
3. **bundled** — ``src/skills/`` shipped inside the package.

Higher tiers override lower tiers on a name collision.

Installation is native (``git clone``) with a fallback to the external
``npx skills`` CLI when git is unavailable or the clone fails.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from agent.config import CONFIG_DIR

logger = logging.getLogger("skills")

# ─── Locations ────────────────────────────────────────────────────────────
GLOBAL_SKILLS_DIR = CONFIG_DIR / "skills"
BUNDLED_SKILLS_DIR = Path(__file__).parent.parent / "skills"
MANIFEST_FILENAME = ".installed.json"

# Fastfold-controlled directory for `npx skills add` installs. We pin the Skills
# CLI's working directory to NPX_INSTALL_ROOT and target the `claude-code` agent in
# project scope, so files land in a single fastfold-owned location (not the user's
# global ~/.claude/skills). The loader reads NPX_SKILLS_DIR as its own tier.
NPX_INSTALL_ROOT = CONFIG_DIR
NPX_SKILLS_DIR = CONFIG_DIR / ".claude" / "skills"

# Default discovery catalog (GitHub ``owner/repo``); skills live under ``skills/``.
DEFAULT_CATALOG = "fastfold-ai/skills"

# Curated third-party skill collections, grouped by provider. Each ``source`` is a
# GitHub shorthand installable via ``install_skill`` (native git or npx). Installing
# a whole-repo source batch-installs all skills it contains.
SUGGESTED_SKILL_SOURCES = [
    {
        "provider": "K-Dense-AI",
        "source": "K-Dense-AI/scientific-agent-skills",
        "url": "https://github.com/K-Dense-AI/scientific-agent-skills",
        "description": "Scientific agent skills collection",
    },
    {
        "provider": "Anthropic",
        "source": "anthropics/life-sciences",
        "url": "https://github.com/anthropics/life-sciences#skills",
        "description": "Anthropic life-sciences skills",
    },
    {
        "provider": "DeepMind",
        "source": "google-deepmind/science-skills",
        "url": "https://github.com/google-deepmind/science-skills",
        "description": "Google DeepMind science skills",
    },
]

_GITHUB_SHORTHAND_RE = re.compile(r"^(?P<owner>[\w.-]+)/(?P<repo>[\w.-]+?)(?:@(?P<sub>.+))?$")
_CLONE_TIMEOUT_S = 120


# ─── Data model ─────────────────────────────────────────────────────────────
@dataclass
class SkillInfo:
    """A discovered skill."""

    name: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    path: Optional[Path] = None  # path to SKILL.md
    source: str = ""  # tier: global | project | bundled

    @property
    def directory(self) -> Optional[Path]:
        return self.path.parent if self.path else None


# ─── SKILL.md parsing ─────────────────────────────────────────────────────
def parse_skill_md(skill_md: Path) -> SkillInfo:
    """Parse name/description/tags from a SKILL.md frontmatter block.

    Uses a lightweight line parser (no YAML dependency) consistent with the
    rest of the codebase. Falls back to the directory name for ``name``.
    """
    name = skill_md.parent.name
    description = ""
    tags: list[str] = []

    try:
        content = skill_md.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not read %s: %s", skill_md, exc)
        return SkillInfo(name=name, path=skill_md)

    # Restrict parsing to the frontmatter block when present.
    block = content
    if content.lstrip().startswith("---"):
        stripped = content.lstrip()
        end = stripped.find("\n---", 3)
        if end != -1:
            block = stripped[3:end]

    for line in block.splitlines():
        line = line.strip()
        if line.startswith("name:"):
            value = line.split(":", 1)[1].strip().strip('"').strip("'")
            if value:
                name = value
        elif line.startswith("description:"):
            description = line.split(":", 1)[1].strip().strip('"').strip("'")
        elif line.startswith("tags:"):
            raw = line.split(":", 1)[1].strip().strip("[]")
            tags = [t.strip().strip('"').strip("'") for t in raw.split(",") if t.strip()]

    return SkillInfo(name=name, description=description, tags=tags, path=skill_md)


# ─── Discovery / loading ───────────────────────────────────────────────────
def _scan_dir(base: Path, source: str) -> dict[str, SkillInfo]:
    out: dict[str, SkillInfo] = {}
    if not base.exists():
        return out
    for child in sorted(base.iterdir()):
        skill_md = child / "SKILL.md"
        if child.is_dir() and skill_md.exists():
            info = parse_skill_md(skill_md)
            info.source = source
            # Key by directory name (canonical identifier everywhere else).
            out[child.name] = info
    return out


def _scan_project(project_root: Path) -> dict[str, SkillInfo]:
    """Scan project-local skills gated by skills-lock.json (back-compat)."""
    out: dict[str, SkillInfo] = {}
    lock_file = project_root / "skills-lock.json"
    claude_dir = project_root / ".claude" / "skills"
    if not (lock_file.exists() and claude_dir.exists()):
        return out
    try:
        lock = json.loads(lock_file.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not read %s: %s", lock_file, exc)
        return out
    for name, meta in lock.get("skills", {}).items():
        skill_md = claude_dir / name / "SKILL.md"
        if skill_md.exists():
            info = parse_skill_md(skill_md)
            label = "project"
            if isinstance(meta, dict) and meta.get("source"):
                label = f"project ({meta['source']})"
            info.source = label
            out[name] = info
    return out


def iter_skills(project_root: Optional[Path] = None) -> dict[str, SkillInfo]:
    """Return merged skills keyed by directory name.

    Priority (high → low): global > project (lock-gated) > bundled.
    """
    if project_root is None:
        project_root = Path.cwd()

    merged: dict[str, SkillInfo] = {}
    merged.update(_scan_dir(BUNDLED_SKILLS_DIR, "bundled"))
    merged.update(_scan_project(project_root))
    merged.update(_scan_dir(NPX_SKILLS_DIR, "npx"))
    merged.update(_scan_dir(GLOBAL_SKILLS_DIR, "global"))
    return merged


def list_skills(project_root: Optional[Path] = None) -> list[SkillInfo]:
    return [s for _, s in sorted(iter_skills(project_root).items())]


def skill_info(name: str, project_root: Optional[Path] = None) -> Optional[SkillInfo]:
    return iter_skills(project_root).get(name)


def installed_skill_names(project_root: Optional[Path] = None) -> list[str]:
    return sorted(iter_skills(project_root).keys())


def build_skills_prompt(project_root: Optional[Path] = None) -> str:
    """Build the 'Installed Agent Skills' system-prompt section (full SKILL.md)."""
    skills = iter_skills(project_root)
    sections: list[str] = []
    for name, info in sorted(skills.items()):
        if not info.path:
            continue
        try:
            content = info.path.read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not read skill %s: %s", name, exc)
            continue
        directory = info.directory
        header = f"### Skill: `{name}`"
        if directory is not None:
            dir_str = str(directory)
            scripts_dir = directory / "scripts"
            has_scripts = scripts_dir.is_dir() and any(scripts_dir.glob("*.py"))
            if has_scripts:
                # Rewrite the skill's own script references to absolute paths so the agent
                # never resolves `scripts/<name>.py` against the wrong (repo) directory.
                content = content.replace("python scripts/", f"python {dir_str}/scripts/")
                content = content.replace("`scripts/", f"`{dir_str}/scripts/")
                names = sorted(p.name for p in scripts_dir.glob("*.py") if not p.name.startswith("_"))
                header += (
                    f"\n\n**Skill directory:** `{dir_str}`. "
                    f"This skill's scripts live in `{dir_str}/scripts/` (available: {', '.join(names)}). "
                    f"Always invoke them by their absolute path, e.g. `python {dir_str}/scripts/<name>.py ...`. "
                    f"Do NOT use repo-relative paths like `src/skills/...`, a bare `scripts/...`, "
                    f"or any script name not listed above."
                )
            else:
                header += (
                    f"\n\n**Skill directory:** `{dir_str}`. This skill has **no scripts** — "
                    f"follow its instructions using the available tools (e.g. the `skills.manage` tool). "
                    f"Do NOT run shell scripts for it (there is no `scripts/` directory)."
                )
        sections.append(f"{header}\n\n{content}")

    if not sections:
        return ""

    return (
        "## Installed Agent Skills\n\n"
        "You have the following agent skills installed. Follow their instructions exactly "
        "when the user's request matches a skill's use-case. Each skill is self-contained: "
        "when a skill references a script as `scripts/<name>.py`, run it from that skill's "
        "directory shown below (for example, `python <skill directory>/scripts/<name>.py ...`).\n\n"
        + "\n\n---\n\n".join(sections)
    )


# ─── Source parsing ─────────────────────────────────────────────────────────
def detect_source_type(source: str) -> str:
    """Classify a skill source string: 'local', 'github', 'shorthand', or 'name'."""
    s = (source or "").strip()
    if not s:
        return "name"
    if s.startswith(("./", "../", "/", "~")) or Path(s).expanduser().exists():
        return "local"
    if s.startswith("http://") or s.startswith("https://") or s.startswith("git@"):
        return "github"
    if "/" in s and _GITHUB_SHORTHAND_RE.match(s):
        return "shorthand"
    return "name"


def _parse_github(source: str) -> tuple[str, Optional[str], Optional[str]]:
    """Return (clone_url, ref, subpath) for a GitHub URL or shorthand."""
    s = source.strip()

    # Shorthand: owner/repo[@subpath]
    m = _GITHUB_SHORTHAND_RE.match(s)
    if m and not s.startswith(("http", "git@")):
        owner, repo, sub = m.group("owner"), m.group("repo"), m.group("sub")
        repo = repo[:-4] if repo.endswith(".git") else repo
        return f"https://github.com/{owner}/{repo}.git", None, sub

    # Full URL, optionally with /tree/<ref>/<subpath>
    url = s
    ref: Optional[str] = None
    sub: Optional[str] = None
    tree = re.search(r"github\.com[:/](?P<owner>[\w.-]+)/(?P<repo>[\w.-]+?)(?:\.git)?(?:/tree/(?P<ref>[^/]+)(?:/(?P<sub>.+))?)?/?$", s)
    if tree:
        owner = tree.group("owner")
        repo = tree.group("repo")
        ref = tree.group("ref")
        sub = tree.group("sub")
        url = f"https://github.com/{owner}/{repo}.git"
    return url, ref, sub


# ─── Install ────────────────────────────────────────────────────────────────
def _git_available() -> bool:
    return shutil.which("git") is not None


def _npx_available() -> bool:
    return shutil.which("npx") is not None


def _git_clone(url: str, ref: Optional[str], dest: Path) -> Optional[str]:
    """Shallow clone ``url`` into ``dest``. Returns the commit SHA or None."""
    cmd = ["git", "clone", "--depth", "1"]
    if ref:
        cmd += ["--branch", ref]
    cmd += [url, str(dest)]
    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=_CLONE_TIMEOUT_S)
    try:
        out = subprocess.run(
            ["git", "-C", str(dest), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=30,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _resolve_skill_dirs(root: Path, subpath: Optional[str]) -> list[Path]:
    """Find skill directories (containing SKILL.md) within a cloned repo."""
    base = (root / subpath) if subpath else root
    if not base.exists():
        # Common convention: skills live under a 'skills/' folder.
        if (root / "skills").exists():
            base = root / "skills"
        else:
            return []

    # Single skill directly at base.
    if (base / "SKILL.md").exists():
        return [base]

    # Otherwise treat children (depth 1) with SKILL.md as a pack.
    found = [c for c in sorted(base.iterdir()) if c.is_dir() and (c / "SKILL.md").exists()]
    return found


def _read_manifest(dest: Path) -> dict:
    path = dest / MANIFEST_FILENAME
    if not path.exists():
        return {"version": 1, "skills": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if "skills" not in data:
            data["skills"] = {}
        return data
    except Exception:  # noqa: BLE001
        return {"version": 1, "skills": {}}


def _write_manifest(dest: Path, manifest: dict) -> None:
    path = dest / MANIFEST_FILENAME
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp.replace(path)


def _record_install(dest: Path, name: str, source: str, commit: Optional[str]) -> None:
    manifest = _read_manifest(dest)
    manifest["skills"][name] = {
        "source": source,
        "commit": commit,
        "installed_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_manifest(dest, manifest)


def _npx_target(source: str) -> tuple[str, Optional[str], bool]:
    """Map a skill source to (npx_target, skill_name, whole_repo) for `npx skills add`.

    - ``owner/repo@subpath`` -> (``owner/repo``, ``<last path segment>``, False)
    - GitHub ``/tree/<ref>/<path>`` URL -> (url, None, False)  # URL targets one skill
    - whole-repo shorthand/URL -> (source, None, True)
    """
    s = (source or "").strip()
    if s.startswith(("http://", "https://", "git@")):
        return s, None, ("/tree/" not in s)
    if "@" in s:
        repo, sub = s.split("@", 1)
        name = sub.rstrip("/").split("/")[-1] or None
        return repo, name, False
    return s, None, True


def _run_npx_add(target: str, skill_names: Optional[list[str]] = None, whole: bool = False) -> None:
    """Run `npx skills add` non-interactively, pinned to the fastfold-owned dir."""
    NPX_INSTALL_ROOT.mkdir(parents=True, exist_ok=True)
    cmd = ["npx", "-y", "skills", "add", target, "-a", "claude-code", "--copy", "-y"]
    if skill_names:
        for name in skill_names:
            cmd += ["--skill", name]
    elif whole:
        cmd += ["--skill", "*"]
    subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        timeout=_CLONE_TIMEOUT_S,
        stdin=subprocess.DEVNULL,
        cwd=str(NPX_INSTALL_ROOT),
    )


def npx_add(target: str, skill_names: Optional[list[str]] = None, whole: bool = False) -> dict:
    """Install one or more skills from a single repo/URL via `npx skills add`."""
    if not _npx_available():
        return {"ok": False, "summary": "npx is not available.", "via": "npx"}
    try:
        _run_npx_add(target, skill_names, whole)
        what = f"{target}" + (f" (skills: {', '.join(skill_names)})" if skill_names else " (all)" if whole else "")
        return {
            "ok": True,
            "installed": list(skill_names or []),
            "summary": f"Installed {what} via npx into {NPX_SKILLS_DIR}.",
            "via": "npx",
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "summary": f"npx skills add timed out for {target}.", "via": "npx"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "summary": f"npx skills add failed: {exc}", "via": "npx"}


def _npx_install(source: str) -> dict:
    """Install a single source via npx (handles owner/repo@subpath via --skill)."""
    target, skill, whole = _npx_target(source)
    return npx_add(target, [skill] if skill else None, whole)


def _npx_update() -> dict:
    """Update all npx-installed skills in the fastfold-owned dir (`npx skills update`)."""
    if not _npx_available():
        return {"ok": False, "summary": "npx is not available."}
    try:
        subprocess.run(
            ["npx", "-y", "skills", "update", "-y"],
            check=True, capture_output=True, text=True,
            timeout=_CLONE_TIMEOUT_S, stdin=subprocess.DEVNULL, cwd=str(NPX_INSTALL_ROOT),
        )
        return {"ok": True, "summary": "Updated npx-installed skills."}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "summary": f"npx skills update failed: {exc}"}


def install_skill(source: str, *, dest: Optional[Path] = None, prefer_npx: bool = False) -> dict:
    """Install a skill from a GitHub URL/shorthand, local path, or catalog name.

    By default uses a native ``git clone`` and falls back to ``npx skills add``.
    When ``prefer_npx`` is True and ``npx`` is available, tries ``npx skills add``
    first and falls back to the native clone if it fails.

    Returns a dict with keys: ok, installed (list of names), summary.
    """
    source = (source or "").strip()
    if not source:
        return {"ok": False, "summary": "No skill source provided."}

    dest = dest or GLOBAL_SKILLS_DIR
    dest.mkdir(parents=True, exist_ok=True)

    src_type = detect_source_type(source)

    # Resolve a bare name against the catalog before installing.
    if src_type == "name":
        match = _resolve_name_to_source(source)
        if not match:
            return {"ok": False, "summary": f"Could not find a skill named '{source}' in the catalog."}
        source = match
        src_type = detect_source_type(source)

    if src_type == "local":
        return _install_local(source, dest)

    url, ref, subpath = _parse_github(source)

    # Prefer npx when requested. npx handles both whole-repo and specific-skill
    # installs (the latter via --skill, derived from our owner/repo@subpath shorthand).
    if prefer_npx and _npx_available():
        result = _npx_install(source)
        if result.get("ok"):
            return result
        logger.debug("npx skills add failed, falling back to git: %s", result.get("summary"))

    # GitHub (URL or shorthand) — native clone, fall back to npx.
    if not _git_available():
        return _npx_install(source)
    try:
        with tempfile.TemporaryDirectory(prefix="fastfold-skill-") as tmp:
            clone_root = Path(tmp) / "repo"
            commit = _git_clone(url, ref, clone_root)
            skill_dirs = _resolve_skill_dirs(clone_root, subpath)
            if not skill_dirs:
                return {"ok": False, "summary": f"No SKILL.md found in {source}."}
            installed = _copy_skill_dirs(skill_dirs, dest, source, commit)
        return {
            "ok": True,
            "installed": installed,
            "summary": f"Installed {len(installed)} skill(s) to {dest}: {', '.join(installed)}.",
            "via": "git",
        }
    except subprocess.CalledProcessError as exc:
        logger.debug("git clone failed: %s", exc)
        return _npx_install(source)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "summary": f"Install failed: {exc}"}


def _install_local(source: str, dest: Path) -> dict:
    base = Path(source).expanduser().resolve()
    if not base.exists():
        return {"ok": False, "summary": f"Local path does not exist: {source}"}
    skill_dirs = _resolve_skill_dirs(base, None)
    if not skill_dirs:
        return {"ok": False, "summary": f"No SKILL.md found at {source}."}
    installed = _copy_skill_dirs(skill_dirs, dest, str(base), None)
    return {
        "ok": True,
        "installed": installed,
        "summary": f"Installed {len(installed)} skill(s) to {dest}: {', '.join(installed)}.",
        "via": "local",
    }


def _copy_skill_dirs(skill_dirs: list[Path], dest: Path, source: str, commit: Optional[str]) -> list[str]:
    installed: list[str] = []
    for sd in skill_dirs:
        name = sd.name
        target = dest / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(sd, target, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"))
        _record_install(dest, name, source, commit)
        installed.append(name)
    return installed


def _project_lock_sources(project_root: Path) -> list[str]:
    """Return distinct skill sources recorded in a project's skills-lock.json (npx installs)."""
    lock_file = project_root / "skills-lock.json"
    if not lock_file.exists():
        return []
    try:
        lock = json.loads(lock_file.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []
    sources: list[str] = []
    for meta in (lock.get("skills") or {}).values():
        if isinstance(meta, dict) and meta.get("source"):
            src = str(meta["source"]).strip()
            if src and src not in sources:
                sources.append(src)
    return sources


def upgrade_skills(
    *,
    include_catalog: bool = True,
    include_npx: bool = True,
    catalog: str = DEFAULT_CATALOG,
    project_root: Optional[Path] = None,
) -> dict:
    """Re-sync installed skills to their latest versions.

    1. When ``include_catalog`` is True, installs/refreshes the entire Fastfold
       catalog (adds new skills + overrides existing ones).
    2. Re-installs every other skill recorded in the global manifest from its
       recorded source (overwrites in place).
    3. When ``include_npx`` is True and ``npx`` is available, re-runs
       ``npx skills add`` for each source recorded in the project-local
       ``skills-lock.json`` (skills the Skills CLI installed into .claude/skills).

    Returns a dict: added (new names), updated (refreshed names), failed (list of
    (source, reason)), and a human-readable summary.
    """
    dest = GLOBAL_SKILLS_DIR
    if project_root is None:
        project_root = Path.cwd()
    before = set(iter_skills(project_root).keys())
    installed_total: list[str] = []
    failed: list[tuple[str, str]] = []

    # 1) Sync the whole Fastfold catalog (one clone: adds new + overrides existing).
    catalog_owner_repo = catalog.split("@", 1)[0]
    if include_catalog:
        if _git_available() or _npx_available():
            result = install_skill(catalog_owner_repo, dest=dest)
            if result.get("ok"):
                installed_total += result.get("installed", [])
            else:
                failed.append((catalog_owner_repo, result.get("summary", "failed")))
        else:
            failed.append((catalog_owner_repo, "git and npx are both unavailable"))

    # 2) Update other manifest-tracked skills from their recorded sources.
    manifest = _read_manifest(dest)
    for name, meta in sorted(manifest.get("skills", {}).items()):
        source = (meta or {}).get("source")
        if not source:
            continue
        # Skip skills already refreshed by the catalog sync above.
        if include_catalog and source.split("@", 1)[0] == catalog_owner_repo:
            continue
        result = install_skill(source, dest=dest)
        if result.get("ok"):
            installed_total += result.get("installed", []) or [name]
        else:
            failed.append((source, result.get("summary", "failed")))

    # 3) Update npx-installed skills (fastfold-owned dir) via `npx skills update`.
    npx_synced = 0
    if include_npx and _npx_available() and NPX_SKILLS_DIR.exists():
        result = _npx_update()
        if result.get("ok"):
            npx_synced = len(_scan_dir(NPX_SKILLS_DIR, "npx"))
        else:
            failed.append(("npx skills update", result.get("summary", "failed")))

    after = set(iter_skills(project_root).keys())
    added = sorted(after - before)
    updated = sorted(set(installed_total) - set(added))
    summary = (
        f"Synced skills: {len(updated)} updated, {len(added)} added, "
        f"{npx_synced} npx-synced, {len(failed)} failed."
    )
    return {
        "added": added,
        "updated": updated,
        "npx_synced": npx_synced,
        "failed": failed,
        "summary": summary,
    }


def user_installed_skill_names(project_root: Optional[Path] = None) -> list[str]:
    """Return names of user-installed skills (global + project tiers; excludes bundled)."""
    if project_root is None:
        project_root = Path.cwd()
    names = set(_scan_dir(GLOBAL_SKILLS_DIR, "global").keys())
    names |= set(_scan_dir(NPX_SKILLS_DIR, "npx").keys())
    names |= set(_scan_project(project_root).keys())
    return sorted(names)


def remove_all_skills(*, project_root: Optional[Path] = None) -> dict:
    """Remove ALL user-installed skills (global install dir + project lock-gated).

    Bundled skills (shipped in the package) are never touched.
    """
    if project_root is None:
        project_root = Path.cwd()
    removed: list[str] = []

    # 1) Global install dir (~/.fastfold-cli/skills).
    if GLOBAL_SKILLS_DIR.exists():
        for child in sorted(GLOBAL_SKILLS_DIR.iterdir()):
            if child.is_dir() and (child / "SKILL.md").exists():
                try:
                    shutil.rmtree(child)
                    removed.append(child.name)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Could not remove %s: %s", child, exc)
        _write_manifest(GLOBAL_SKILLS_DIR, {"version": 1, "skills": {}})

    # 1b) npx-installed skills (fastfold-owned ~/.fastfold-cli/.claude/skills).
    if NPX_SKILLS_DIR.exists():
        for child in sorted(NPX_SKILLS_DIR.iterdir()):
            if child.is_dir() and (child / "SKILL.md").exists():
                try:
                    shutil.rmtree(child)
                    removed.append(child.name)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Could not remove %s: %s", child, exc)
        npx_lock = NPX_INSTALL_ROOT / "skills-lock.json"
        if npx_lock.exists():
            try:
                npx_lock.unlink()
            except Exception:  # noqa: BLE001
                pass

    # 2) Project-local lock-gated skills (npx installs under .claude/skills).
    lock_file = project_root / "skills-lock.json"
    claude_dir = project_root / ".claude" / "skills"
    if lock_file.exists() and claude_dir.exists():
        try:
            lock = json.loads(lock_file.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            lock = {"version": 1, "skills": {}}
        for name in list((lock.get("skills") or {}).keys()):
            skill_dir = claude_dir / name
            if (skill_dir / "SKILL.md").exists():
                try:
                    shutil.rmtree(skill_dir)
                    removed.append(name)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Could not remove %s: %s", skill_dir, exc)
            lock.get("skills", {}).pop(name, None)
        try:
            lock_file.write_text(json.dumps(lock, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not rewrite %s: %s", lock_file, exc)

    removed = sorted(set(removed))
    return {"ok": True, "removed": removed, "summary": f"Removed {len(removed)} skill(s)."}


def remove_skill(name: str, *, dest: Optional[Path] = None) -> dict:
    """Remove a globally-installed skill by name."""
    dest = dest or GLOBAL_SKILLS_DIR
    target = dest / name
    if not target.exists():
        return {"ok": False, "summary": f"Skill '{name}' is not installed in {dest}."}
    try:
        shutil.rmtree(target)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "summary": f"Could not remove '{name}': {exc}"}

    manifest = _read_manifest(dest)
    if name in manifest.get("skills", {}):
        del manifest["skills"][name]
        _write_manifest(dest, manifest)
    return {"ok": True, "summary": f"Removed skill '{name}'."}


# ─── Discovery (catalog + ecosystem) ────────────────────────────────────────
def discover_skills(query: Optional[str] = None, *, catalog: str = DEFAULT_CATALOG) -> list[dict]:
    """Discover skills from the catalog repo (shallow clone) filtered by query.

    Returns dicts with: name, description, tags, install_source.
    """
    results: list[dict] = []
    if not _git_available():
        return results

    owner_repo = catalog.split("@", 1)[0]
    url = f"https://github.com/{owner_repo}.git"
    try:
        with tempfile.TemporaryDirectory(prefix="fastfold-catalog-") as tmp:
            clone_root = Path(tmp) / "catalog"
            _git_clone(url, None, clone_root)
            skills_dir = clone_root / "skills"
            base = skills_dir if skills_dir.exists() else clone_root
            for child in sorted(base.iterdir()):
                skill_md = child / "SKILL.md"
                if child.is_dir() and skill_md.exists():
                    info = parse_skill_md(skill_md)
                    results.append({
                        "name": info.name,
                        "description": info.description,
                        "tags": info.tags,
                        "install_source": f"{owner_repo}@skills/{child.name}",
                    })
    except Exception as exc:  # noqa: BLE001
        logger.debug("Catalog discovery failed: %s", exc)
        return results

    if query:
        q = query.lower()
        results = [
            r for r in results
            if q in r["name"].lower()
            or q in r["description"].lower()
            or any(q in t.lower() for t in r["tags"])
        ]
    return results


def _resolve_name_to_source(name: str, *, catalog: str = DEFAULT_CATALOG) -> Optional[str]:
    for entry in discover_skills(name, catalog=catalog):
        if entry["name"] == name:
            return entry["install_source"]
    # Loose fallback: first match.
    matches = discover_skills(name, catalog=catalog)
    return matches[0]["install_source"] if matches else None
