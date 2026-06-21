"""
Deployment readiness checks for fastfold.

Used by `fastfold doctor` and interactive `/doctor` to surface actionable setup issues.
"""

from dataclasses import dataclass
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from rich.table import Table

from agent.config import CONFIG_FILE, Config
from tools import EXPERIMENTAL_CATEGORIES, ensure_loaded, tool_load_errors

logger = logging.getLogger("doctor")


@dataclass
class DoctorCheck:
    name: str
    status: str  # "ok" | "warn" | "error"
    detail: str


def _status_markup(status: str) -> str:
    if status == "ok":
        return "[green]ok[/green]"
    if status == "warn":
        return "[yellow]warn[/yellow]"
    return "[red]error[/red]"


def run_checks(config: Config | None = None, session=None) -> list[DoctorCheck]:
    """Run production-readiness checks and return structured results.

    Args:
        config: Optional Config instance. Loaded from disk if not provided.
        session: Optional Session instance. When provided, runtime tool health
            data (suppressed tools, failure counts) is included in the report.
    """
    cfg = config or Config.load()
    checks: list[DoctorCheck] = []

    # 1) Config file readability (best-effort: load already handled parse errors)
    if CONFIG_FILE.exists():
        checks.append(
            DoctorCheck(
                name="config_file",
                status="ok",
                detail=f"Using {CONFIG_FILE}",
            )
        )
    else:
        checks.append(
            DoctorCheck(
                name="config_file",
                status="warn",
                detail=f"No config file yet at {CONFIG_FILE} (defaults/env vars are used)",
            )
        )

    # 2) LLM configuration readiness
    llm_issue = cfg.llm_preflight_issue()
    provider = cfg.get("llm.provider", "anthropic")
    model = cfg.get("llm.model")
    if llm_issue:
        checks.append(DoctorCheck(name="llm", status="error", detail=llm_issue))
    else:
        if os.environ.get("ANTHROPIC_FOUNDRY_API_KEY") or os.environ.get("ANTHROPIC_FOUNDRY_RESOURCE"):
            detail = f"provider=anthropic (Azure Foundry), model={model}"
        else:
            detail = f"provider={provider}, model={model}"
        checks.append(
            DoctorCheck(name="llm", status="ok", detail=detail)
        )

    # 3) Output directory availability
    out_dir = Path(cfg.get("sandbox.output_dir", str(Path.cwd() / "outputs")))
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        checks.append(
            DoctorCheck(name="output_dir", status="ok", detail=f"Writable: {out_dir}")
        )
    except OSError as exc:
        checks.append(
            DoctorCheck(name="output_dir", status="error", detail=f"{out_dir}: {exc}")
        )

    # 4) Data base directory availability
    data_base = Path(cfg.get("data.base", str(Path.home() / ".fastfold-cli" / "data")))
    try:
        data_base.mkdir(parents=True, exist_ok=True)
        checks.append(
            DoctorCheck(name="data_base", status="ok", detail=f"Writable: {data_base}")
        )
    except OSError as exc:
        checks.append(
            DoctorCheck(name="data_base", status="warn", detail=f"{data_base}: {exc}")
        )

    # 5) Tool module import health
    ensure_loaded()
    load_errors = tool_load_errors()
    if load_errors:
        sample = ", ".join(sorted(load_errors.keys())[:8])
        extra = "" if len(load_errors) <= 8 else f" (+{len(load_errors) - 8} more)"
        checks.append(
            DoctorCheck(
                name="tool_modules",
                status="warn",
                detail=f"{len(load_errors)} module(s) failed to load: {sample}{extra}",
            )
        )
    else:
        checks.append(
            DoctorCheck(name="tool_modules", status="ok", detail="All tool modules loaded")
        )

    # 6) Experimental categories planning status
    if cfg.get("agent.enable_experimental_tools", False):
        checks.append(
            DoctorCheck(
                name="experimental_tools",
                status="warn",
                detail=(
                    f"Experimental categories enabled for planning: "
                    f"{', '.join(sorted(EXPERIMENTAL_CATEGORIES))}"
                ),
            )
        )
    else:
        checks.append(
            DoctorCheck(
                name="experimental_tools",
                status="ok",
                detail=(
                    f"Experimental categories hidden from planning by default: "
                    f"{', '.join(sorted(EXPERIMENTAL_CATEGORIES))}"
                ),
            )
        )

    # 7) Grounding guardrail status
    if cfg.get("agent.enforce_grounded_synthesis", True):
        checks.append(
            DoctorCheck(
                name="grounding_guard",
                status="ok",
                detail="Grounded synthesis enforcement is enabled",
            )
        )
    else:
        checks.append(
            DoctorCheck(
                name="grounding_guard",
                status="warn",
                detail="Grounded synthesis enforcement is disabled",
            )
        )

    # 8) Runtime profile
    profile = str(cfg.get("agent.profile", "research"))
    if profile not in {"research", "pharma", "enterprise"}:
        checks.append(
            DoctorCheck(
                name="runtime_profile",
                status="warn",
                detail=f"Unknown agent.profile '{profile}'",
            )
        )
    else:
        checks.append(
            DoctorCheck(
                name="runtime_profile",
                status="ok",
                detail=f"{profile}",
            )
        )

    synthesis_style = str(cfg.get("agent.synthesis_style", "standard")).strip().lower()
    if synthesis_style not in {"standard", "pharma"}:
        checks.append(
            DoctorCheck(
                name="synthesis_style",
                status="warn",
                detail=f"Unknown agent.synthesis_style '{synthesis_style}'",
            )
        )
    elif profile == "pharma" and synthesis_style != "pharma":
        checks.append(
            DoctorCheck(
                name="synthesis_style",
                status="warn",
                detail="agent.profile=pharma but synthesis style is not pharma",
            )
        )
    else:
        checks.append(
            DoctorCheck(
                name="synthesis_style",
                status="ok",
                detail=synthesis_style,
            )
        )

    # 9) Quality gate policy
    if cfg.get("agent.quality_gate_enabled", True):
        strict = bool(cfg.get("agent.quality_gate_strict", False))
        checks.append(
            DoctorCheck(
                name="quality_gate",
                status="ok" if strict else "warn",
                detail=(
                    "Strict quality gate enabled (must pass citation/actionability checks)"
                    if strict
                    else "Quality gate is warn-only (strict mode disabled)"
                ),
            )
        )
    else:
        checks.append(
            DoctorCheck(
                name="quality_gate",
                status="warn",
                detail="Quality gate is disabled",
            )
        )

    # 10) Enterprise policy layer
    enforce_policy = bool(cfg.get("enterprise.enforce_policy", False))
    checks.append(
        DoctorCheck(
            name="enterprise_policy",
            status="ok" if enforce_policy else "warn",
            detail=(
                "Policy enforcement enabled"
                if enforce_policy
                else "Policy enforcement disabled (research mode)"
            ),
        )
    )

    # 11) Knowledge substrate path
    substrate_path = Path(
        cfg.get("knowledge.substrate_path", str(Path.home() / ".fastfold-cli" / "knowledge" / "substrate.json"))
    )
    try:
        substrate_path.parent.mkdir(parents=True, exist_ok=True)
        checks.append(
            DoctorCheck(
                name="knowledge_substrate",
                status="ok",
                detail=f"Writable substrate path: {substrate_path}",
            )
        )
    except OSError as exc:
        checks.append(
            DoctorCheck(
                name="knowledge_substrate",
                status="warn",
                detail=f"Could not prepare substrate path {substrate_path}: {exc}",
            )
        )

    # 12) Schema monitor readiness
    if cfg.get("knowledge.schema_monitor_enabled", False):
        baseline = Path.home() / ".fastfold-cli" / "knowledge" / "schema_baselines.json"
        if baseline.exists():
            checks.append(
                DoctorCheck(
                    name="schema_monitor",
                    status="ok",
                    detail=f"Baseline present: {baseline}",
                )
            )
        else:
            checks.append(
                DoctorCheck(
                    name="schema_monitor",
                    status="warn",
                    detail="Schema monitor enabled but no baseline found. Run: fastfold knowledge schema-update",
                )
            )
    else:
        checks.append(
            DoctorCheck(
                name="schema_monitor",
                status="warn",
                detail="Schema monitor disabled",
            )
        )

    # 13) Claude Code delegation policy
    if cfg.get("agent.enable_claude_code_tool", False):
        checks.append(
            DoctorCheck(
                name="claude_code_policy",
                status="warn",
                detail="claude.code is enabled for autonomous use (high privilege)",
            )
        )
    else:
        checks.append(
            DoctorCheck(
                name="claude_code_policy",
                status="ok",
                detail="claude.code is disabled by default (opt-in)",
            )
        )

    # 14) Data availability — verify key datasets can be found
    checks.append(_check_data_availability(cfg))

    # 15) Downloads directory
    checks.append(_check_downloads_dir())

    # 16) API connectivity (lightweight HEAD probes)
    checks.extend(_check_api_connectivity())

    # 17) Runtime tool health
    checks.append(_check_tool_health(session))

    # 18) Preflight validation config
    if cfg.get("agent.preflight_validation_enabled", True):
        checks.append(
            DoctorCheck(
                name="preflight_validation",
                status="ok",
                detail="Pre-query API key validation is enabled",
            )
        )
    else:
        checks.append(
            DoctorCheck(
                name="preflight_validation",
                status="warn",
                detail="Pre-query API key validation is disabled",
            )
        )

    # 19) Boltz CLI + key readiness
    checks.append(_check_boltz_cli(cfg))

    return checks


def has_errors(checks: list[DoctorCheck]) -> bool:
    """Return True if any check has error status."""
    return any(c.status == "error" for c in checks)


def to_table(checks: list[DoctorCheck]) -> Table:
    """Render doctor checks as a rich table."""
    table = Table(title="Fastfold Doctor")
    table.add_column("Check", style="cyan")
    table.add_column("Status")
    table.add_column("Details")

    for check in checks:
        table.add_row(check.name, _status_markup(check.status), check.detail)

    return table


# ---------------------------------------------------------------------------
# Runtime health check helpers
# ---------------------------------------------------------------------------


# Key datasets and the file patterns used by loaders
_KEY_DATASETS = {
    "depmap": ("CRISPRGeneEffect.csv", ["", "depmap"]),
    "prism": ("prism_LFC_COLLAPSED.csv", ["", "prism"]),
    "l1000": ("l1000_landmark_only.parquet", ["", "l1000"]),
}


def _check_data_availability(cfg: Config) -> DoctorCheck:
    """Check whether key datasets can be found on disk."""
    data_base = Path(cfg.get("data.base", str(Path.home() / ".fastfold-cli" / "data")))
    search_dirs = [data_base]
    # Also check ct-data sister project
    ct_data = Path.home() / "Projects" / "CellType" / "ct-data"
    if ct_data.exists():
        search_dirs.append(ct_data)

    found = []
    missing = []
    for name, (filename, subdirs) in _KEY_DATASETS.items():
        located = False
        stem = Path(filename).stem
        for base_dir in search_dirs:
            for sub in subdirs:
                d = base_dir / sub if sub else base_dir
                if (d / filename).exists():
                    located = True
                    break
                parquet = d / f"{stem}.parquet"
                if parquet.exists():
                    located = True
                    break
            if located:
                break
        if located:
            found.append(name)
        else:
            missing.append(name)

    if not missing:
        return DoctorCheck(
            name="data_availability",
            status="ok",
            detail=f"Key datasets found: {', '.join(sorted(found))}",
        )
    if found:
        return DoctorCheck(
            name="data_availability",
            status="warn",
            detail=f"Missing datasets: {', '.join(sorted(missing))} (found: {', '.join(sorted(found))}). Run: fastfold data pull <name>",
        )
    return DoctorCheck(
        name="data_availability",
        status="warn",
        detail=f"No key datasets found ({', '.join(sorted(missing))}). Run: fastfold data pull depmap",
    )


def _check_downloads_dir() -> DoctorCheck:
    """Verify ~/.fastfold-cli/downloads/ exists and is writable."""
    downloads = Path.home() / ".fastfold-cli" / "downloads"
    try:
        downloads.mkdir(parents=True, exist_ok=True)
        return DoctorCheck(
            name="downloads_dir",
            status="ok",
            detail=f"Writable: {downloads}",
        )
    except OSError as exc:
        return DoctorCheck(
            name="downloads_dir",
            status="warn",
            detail=f"{downloads}: {exc}",
        )


# APIs to probe with HEAD requests (short timeout, best-effort)
_API_PROBES = [
    ("PubMed eutils", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/einfo.fcgi"),
    ("Enrichr", "https://maayanlab.cloud/Enrichr/"),
    ("GDC API", "https://api.gdc.cancer.gov/status"),
]


def _check_api_connectivity() -> list[DoctorCheck]:
    """Quick HEAD/GET probe against key public APIs."""
    checks = []
    try:
        import httpx
    except ImportError:
        checks.append(
            DoctorCheck(
                name="api_connectivity",
                status="warn",
                detail="httpx not installed — skipping API connectivity probes",
            )
        )
        return checks

    reachable = []
    unreachable = []
    for label, url in _API_PROBES:
        try:
            resp = httpx.head(url, timeout=5, follow_redirects=True)
            if resp.status_code < 500:
                reachable.append(label)
            else:
                unreachable.append(f"{label} (HTTP {resp.status_code})")
        except Exception:
            unreachable.append(label)

    if not unreachable:
        checks.append(
            DoctorCheck(
                name="api_connectivity",
                status="ok",
                detail=f"All probes passed: {', '.join(reachable)}",
            )
        )
    elif reachable:
        checks.append(
            DoctorCheck(
                name="api_connectivity",
                status="warn",
                detail=f"Unreachable: {', '.join(unreachable)} (reachable: {', '.join(reachable)})",
            )
        )
    else:
        checks.append(
            DoctorCheck(
                name="api_connectivity",
                status="warn",
                detail=f"All API probes failed: {', '.join(unreachable)}. Check network connectivity.",
            )
        )
    return checks


def _check_tool_health(session) -> DoctorCheck:
    """Report runtime tool suppression state from session."""
    if session is None:
        return DoctorCheck(
            name="tool_health",
            status="warn",
            detail="No active session context; run /doctor in interactive mode for runtime tool-health diagnostics",
        )

    suppressed = set()
    failure_counts: dict[str, int] = {}
    if hasattr(session, "tool_health_suppressed_tools"):
        suppressed = session.tool_health_suppressed_tools()
    if hasattr(session, "_tool_health_failures"):
        failure_counts = {
            name: len(timestamps)
            for name, timestamps in session._tool_health_failures.items()
            if timestamps
        }

    if not suppressed and not failure_counts:
        return DoctorCheck(
            name="tool_health",
            status="ok",
            detail="No tool failures or suppressions in this session",
        )

    parts = []
    if suppressed:
        parts.append(f"Suppressed: {', '.join(sorted(suppressed))}")
    if failure_counts:
        failing = [f"{n}({c})" for n, c in sorted(failure_counts.items()) if n not in suppressed]
        if failing:
            parts.append(f"Recent failures: {', '.join(failing)}")

    return DoctorCheck(
        name="tool_health",
        status="warn",
        detail="; ".join(parts),
    )


def _locate_boltz_cli_path() -> Path | None:
    """Locate the boltz-api executable across common install locations."""
    candidates: list[str] = []
    in_path = shutil.which("boltz-api")
    if in_path:
        candidates.append(in_path)
    candidates.extend(
        [
            str(Path.home() / ".local" / "bin" / "boltz-api"),
            str(Path.home() / ".boltz" / "bin" / "boltz-api"),
        ]
    )
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.exists() and path.is_file() and os.access(str(path), os.X_OK):
            return path
    return None


def _boltz_cli_version(path: Path) -> str:
    """Return boltz-api version text (best effort)."""
    try:
        proc = subprocess.run(
            [str(path), "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:  # noqa: BLE001
        return "unknown version"
    output = (proc.stdout or proc.stderr or "").strip()
    if not output:
        return "unknown version"
    return output.splitlines()[0].strip()


def _check_boltz_cli(cfg: Config) -> DoctorCheck:
    """Report Boltz integration readiness (key + CLI)."""
    boltz_key = str(cfg.get("api.boltz_api_key") or os.environ.get("BOLTZ_API_KEY") or "").strip()
    cli_path = _locate_boltz_cli_path()

    if cli_path is None and not boltz_key:
        return DoctorCheck(
            name="boltz_cli",
            status="ok",
            detail="Boltz integration not configured (optional).",
        )

    if cli_path is None and boltz_key:
        return DoctorCheck(
            name="boltz_cli",
            status="warn",
            detail=(
                "BOLTZ_API_KEY is configured but `boltz-api` CLI is missing. "
                "Install with: curl -fsSL https://install.boltz.bio/boltz-api/install.sh | sh"
            ),
        )

    version = _boltz_cli_version(cli_path)
    if boltz_key:
        return DoctorCheck(
            name="boltz_cli",
            status="ok",
            detail=f"boltz-api ready at {cli_path} ({version}); BOLTZ_API_KEY configured.",
        )
    return DoctorCheck(
        name="boltz_cli",
        status="warn",
        detail=(
            f"boltz-api found at {cli_path} ({version}) but BOLTZ_API_KEY is not configured. "
            "Set with: fastfold config set api.boltz_api_key <key> or /keys set-boltz"
        ),
    )
