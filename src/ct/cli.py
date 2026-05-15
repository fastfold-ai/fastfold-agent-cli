"""
Fastfold Agent CLI entry point.

Usage:
    fastfold                              # Interactive mode
    fastfold "your question"              # Single query
    fastfold --smiles "CCO" "Profile"     # With compound context
    fastfold config set key value         # Configuration
    fastfold data pull depmap             # Data management
"""

import os
import json
import random
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import typer
from typing import Optional
from pathlib import Path
from datetime import datetime, timezone
from typer.models import OptionInfo
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ct import __version__
from ct.agent.session import Session
from ct.ui.terminal import InteractiveTerminal, SLASH_COMMANDS


# ─── Startup banner ─────────────────────────────────────────
BANNER = """
[bold #2D7BEA]     ▃▃[/]
[bold #2D7BEA]    ▃▃▃▃[/]
[bold #25C19F]  ▅▅ ▃▃▃[/]
[bold #25C19F] ▅▅▅▅▅[/]
[bold #F5A623]  ▅▅▅ ▆▆[/]
[bold #F5A623]    ▆▆▆▆▆[/]
[bold #D4148E]   ▃ ▆▆▆▆[/]
[bold #D4148E]  ▃▃▃ ▆[/]
[bold #D4148E]   ▃▃▃[/]
"""

app = typer.Typer(
    name="fastfold",
    help=(
        "Fastfold Agent CLI — Where scientists and AI agents work together doing real science.\n\n"
        "Common usage:\n"
        '  fastfold "your research question"\n'
        '  fastfold --smiles "CCO" "Profile this compound"\n'
        "  fastfold config show\n"
        "  fastfold tool list"
    ),
    no_args_is_help=False,
)
console = Console()
FASTFOLD_CLOUD_API_KEYS_URL = "https://cloud.fastfold.ai/api-keys"
SETUP_PROVIDER_ORDER = ("anthropic", "openai")
UV_INSTALL_FLAVORS = frozenset({"all", "win_build"})
PYPI_PROJECT_JSON_URL = "https://pypi.org/pypi/fastfold-agent-cli/json"
_SEMVER_TRIPLET_PATTERN = re.compile(r"^\s*v?(\d+)\.(\d+)\.(\d+)")


def _installed_claude_skill_names() -> list[str]:
    """Return installed skill names: bundled catalog + user-installed lockfile."""
    import json

    skill_names: set[str] = set()

    # 1. Bundled skills shipped with the package
    bundled_dir = Path(__file__).parent / "skills"
    if bundled_dir.exists():
        for d in bundled_dir.iterdir():
            if d.is_dir() and (d / "SKILL.md").exists():
                skill_names.add(d.name)

    # 2. User-installed skills listed in skills-lock.json
    lock_file = Path.cwd() / "skills-lock.json"
    claude_skills_dir = Path.cwd() / ".claude" / "skills"
    if lock_file.exists() and claude_skills_dir.exists():
        try:
            lock = json.loads(lock_file.read_text())
            for name in lock.get("skills", {}):
                if (claude_skills_dir / name / "SKILL.md").exists():
                    skill_names.add(name)
        except Exception:
            pass

    return sorted(skill_names)


def _count_installed_claude_skills() -> int:
    """Count installed agent skills."""
    return len(_installed_claude_skill_names())


def _resolve_fastfold_subscription_tier(cfg) -> str:
    """Resolve Fastfold subscription tier from API, with local fallback."""
    fallback_raw = str(
        os.environ.get("FASTFOLD_SUBSCRIPTION_TIER")
        or cfg.get("fastfold.subscription_tier", "")
        or ""
    ).strip().lower()

    def _normalize_plan_code(raw: str) -> str:
        value = str(raw or "").strip().lower()
        if value in {"pro+", "pro-plus", "pro plus"}:
            return "pro_plus"
        return value

    def _plan_rank(plan_code: str) -> int:
        ranks = {
            "free": 0,
            "pro": 1,
            "pro_plus": 2,
            "ultra": 3,
        }
        return ranks.get(plan_code, -1)

    fallback = _normalize_plan_code(fallback_raw)
    if _plan_rank(fallback) < 0:
        fallback = ""

    api_key = str(
        os.environ.get("FASTFOLD_API_KEY")
        or cfg.get("api.fastfold_cloud_key")
        or ""
    ).strip()
    if not api_key:
        return fallback

    base_url = (
        os.environ.get("FASTFOLD_API_BASE_URL", "https://api.fastfold.ai").strip()
        or "https://api.fastfold.ai"
    )
    url = f"{base_url.rstrip('/')}/v1/billing/workspaces/plans"
    req = urllib.request.Request(
        url=url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "X-API-Key": api_key,
            "Accept": "application/json",
        },
        method="GET",
    )

    try:
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        # Billing endpoints are currently cookie-auth only. For API-key callers,
        # avoid showing a misleading default plan when backend returns 401.
        if exc.code == 401:
            return fallback
        return fallback
    except (urllib.error.URLError, TimeoutError):
        return fallback

    try:
        payload = json.loads(text) if text else {}
    except Exception:
        return fallback
    if not isinstance(payload, dict):
        return fallback

    items = payload.get("items")
    if not isinstance(items, list):
        return fallback

    best_plan = None
    for item in items:
        if not isinstance(item, dict):
            continue
        raw_plan = _normalize_plan_code(str(item.get("plan_code") or ""))
        if not raw_plan:
            continue
        if best_plan is None or _plan_rank(raw_plan) > _plan_rank(best_plan):
            best_plan = raw_plan

    resolved = best_plan or fallback
    return resolved if _plan_rank(resolved) >= 0 else fallback


def _format_plan_label(plan_code: str) -> str:
    """Convert normalized plan code to friendly display label."""
    normalized = str(plan_code or "").strip().lower()
    if normalized == "pro_plus":
        return "Pro+"
    if not normalized:
        return ""
    return normalized.replace("_", " ").title()


def _random_command_tip_markup() -> str:
    """Return a random slash-command tip from actual terminal commands."""
    if not SLASH_COMMANDS:
        return "[dim]Tip: use slash commands in interactive mode.[/dim]"

    command, description = random.choice(list(SLASH_COMMANDS.items()))
    return f"[dim]Tip: try [/][bold #D4148E]{command}[/][dim] — {description}[/dim]"


def _random_news_item_markup() -> str:
    """Return the BoltzGen-only news/action line."""
    installed = set(_installed_claude_skill_names())
    if "protein_design_boltzgen" in installed:
        return "[bold #25C19F]BoltzGen universal protein design now available![/] [#7A7A7A]· Try: Show me Boltzgen protein design examples[/]"
    return "[bold #25C19F]BoltzGen universal protein design now available[/] [#7A7A7A]· Try: Show me Boltzgen protein design examples[/]"


def _normalize_upgrade_flavor(value: object) -> Optional[str]:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in UV_INSTALL_FLAVORS else None


def _default_upgrade_flavor() -> str:
    return "win_build" if os.name == "nt" else "all"


def resolve_upgrade_flavor(cfg=None, persist: bool = True) -> str:
    """Resolve uv install flavor from config, with OS fallback."""
    if cfg is None:
        from ct.agent.config import Config
        cfg = Config.load()

    configured = _normalize_upgrade_flavor(cfg.get("install.uv_flavor"))
    if configured:
        return configured

    fallback = _default_upgrade_flavor()
    if persist:
        try:
            cfg.set("install.uv_flavor", fallback)
            cfg.save()
        except Exception:
            # Upgrade flow should still continue even if config write fails.
            pass
    return fallback


def build_upgrade_command(flavor: str) -> list[str]:
    normalized = _normalize_upgrade_flavor(flavor) or _default_upgrade_flavor()
    package_ref = f"fastfold-agent-cli[{normalized}]"
    return ["uv", "tool", "install", package_ref, "--python", "3.10", "--upgrade"]


def _parse_semver_triplet(version: str) -> Optional[tuple[int, int, int]]:
    match = _SEMVER_TRIPLET_PATTERN.match(str(version or "").strip())
    if not match:
        return None
    return tuple(int(part) for part in match.groups())


def is_newer_version(latest: str, current: str) -> bool:
    latest_triplet = _parse_semver_triplet(latest)
    current_triplet = _parse_semver_triplet(current)
    if not latest_triplet or not current_triplet:
        return False
    return latest_triplet > current_triplet


def fetch_pypi_latest_version(timeout_s: float = 2.5) -> Optional[str]:
    req = urllib.request.Request(
        url=PYPI_PROJECT_JSON_URL,
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return None

    try:
        payload = json.loads(text) if text else {}
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None
    info = payload.get("info")
    if not isinstance(info, dict):
        return None
    version = str(info.get("version") or "").strip()
    return version or None


def get_upgrade_available_version(current_version: str = __version__) -> Optional[str]:
    latest = fetch_pypi_latest_version()
    if not latest:
        return None
    return latest if is_newer_version(latest, current_version) else None


def execute_upgrade(console_obj: Optional[Console] = None, cfg=None) -> bool:
    """Run uv tool upgrade with persisted (or fallback) install flavor."""
    ui = console_obj or console
    flavor = resolve_upgrade_flavor(cfg=cfg, persist=True)
    cmd = build_upgrade_command(flavor)
    ui.print("\n[bold cyan]Upgrading fastfold-agent-cli[/bold cyan]")
    ui.print(f"[dim]Flavor:[/dim] {flavor}")
    ui.print(f"[dim]$ {' '.join(cmd)}[/dim]")

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        ui.print("[red]`uv` command not found. Install uv first: https://docs.astral.sh/uv/[/red]")
        return False
    except Exception as exc:
        ui.print(f"[red]Upgrade failed to start:[/red] {exc}")
        return False

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    if stdout:
        ui.print(stdout)
    if stderr:
        ui.print(stderr, style="yellow" if proc.returncode == 0 else "red")

    if proc.returncode != 0:
        ui.print(f"[red]Upgrade failed[/red] (exit={proc.returncode}).")
        return False

    ui.print("[green]Upgrade complete.[/green] Restart `fastfold` to use the new version.")
    return True


# ─── Config subcommand ────────────────────────────────────────

config_app = typer.Typer(help="Manage fastfold configuration")
app.add_typer(config_app, name="config")


@config_app.command("set")
def config_set(key: str, value: str):
    """Set a configuration value."""
    from ct.agent.config import Config

    cfg = Config.load()
    try:
        cfg.set(key, value)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2)
    cfg.save()
    if key == "agent.profile":
        console.print(
            f"  [green]Set[/green] {key} = {cfg.get('agent.profile')} (applied preset settings)"
        )
    else:
        console.print(f"  [green]Set[/green] {key} = {value}")


@config_app.command("unset")
def config_unset(key: str):
    """Unset a configuration value (revert to default/env fallback)."""
    from ct.agent.config import Config

    cfg = Config.load()
    cfg.unset(key)
    cfg.save()
    console.print(f"  [green]Unset[/green] {key}")


@config_app.command("get")
def config_get(key: str):
    """Get a configuration value."""
    from ct.agent.config import Config

    cfg = Config.load()
    val = cfg.get(key)
    console.print(f"  {key} = {val}")


@config_app.command("show")
def config_show():
    """Show all configuration."""
    from ct.agent.config import Config

    cfg = Config.load()
    console.print(cfg.to_table())


@config_app.command("validate")
def config_validate():
    """Validate configuration and report issues."""
    from ct.agent.config import Config

    cfg = Config.load()
    issues = cfg.validate()
    if not issues:
        console.print("[green]Configuration is valid. No issues found.[/green]")
        return
    console.print(f"[yellow]Found {len(issues)} issue(s):[/yellow]")
    for issue in issues:
        console.print(f"  - {issue}")
    raise typer.Exit(code=2)


# ─── Keys command ────────────────────────────────────────────


@app.command("keys")
def keys_cmd():
    """Show status of optional API keys and what they unlock."""
    from ct.agent.config import Config

    cfg = Config.load()
    console.print(cfg.keys_table())


@app.command("upgrade")
def upgrade_cmd():
    """Upgrade fastfold-agent-cli with uv tool install --upgrade."""
    from ct.agent.config import Config

    cfg = Config.load()
    ok = execute_upgrade(console_obj=console, cfg=cfg)
    if not ok:
        raise typer.Exit(code=1)


@app.command("setup")
def setup_cmd(
    api_key: Optional[str] = typer.Option(
        None, "--api-key", help="Anthropic API key (non-interactive mode)"
    ),
    openai_api_key: Optional[str] = typer.Option(
        None, "--openai-api-key", help="OpenAI API key (non-interactive mode)"
    ),
    fastfold_api_key: Optional[str] = typer.Option(
        None, "--fastfold-api-key", help="Fastfold AI Cloud API key (non-interactive mode)"
    ),
    provider: Optional[str] = typer.Option(
        None, "--provider", help="LLM provider: anthropic or openai"
    ),
):
    """Interactive setup wizard — configure fastfold for first use."""
    from ct.agent.config import Config

    # setup_cmd is also called directly from Python paths (first-run flows),
    # where Typer option defaults remain OptionInfo objects.
    if isinstance(api_key, OptionInfo):
        api_key = None
    if isinstance(openai_api_key, OptionInfo):
        openai_api_key = None
    if isinstance(fastfold_api_key, OptionInfo):
        fastfold_api_key = None
    if isinstance(provider, OptionInfo):
        provider = None

    cfg = Config.load()

    default_provider = str(provider or cfg.get("llm.provider", "anthropic") or "anthropic").strip().lower()
    if default_provider not in {"anthropic", "openai"}:
        default_provider = "anthropic"

    # Azure AI Foundry: skip interactive key prompt when Foundry is configured
    if (
        default_provider == "anthropic"
        and (os.environ.get("ANTHROPIC_FOUNDRY_API_KEY") or os.environ.get("ANTHROPIC_FOUNDRY_RESOURCE"))
    ):
        console.print("\n  [green]Azure AI Foundry detected. No API key needed.[/green]")
        cfg.set("llm.provider", "anthropic")
        cfg.save()
        return

    console.print()
    console.print(
        Panel(
            "[bold]This wizard will help you configure Fastfold Agent CLI.[/bold]\n"
            "Press Ctrl+C at any time to cancel.",
            title="[cyan]Fastfold Setup Wizard[/cyan]",
            border_style="#00bcd4",
        )
    )
    console.print()

    if provider:
        selected_providers = _parse_provider_list(provider)
    else:
        selected_providers = _prompt_setup_providers(default_provider)

    resolved_keys: dict[str, str] = {}
    for prov in selected_providers:
        chosen_key = _resolve_provider_key(
            cfg,
            provider=prov,
            cli_key=(openai_api_key if prov == "openai" else api_key),
        )
        if prov == "anthropic" and (not chosen_key or not chosen_key.startswith("sk-ant-")):
            console.print(
                "\n  [yellow]Warning:[/yellow] Key doesn't start with 'sk-ant-'. "
                "Anthropic API keys typically begin with 'sk-ant-api03-'."
            )
            try:
                proceed = input("  Continue anyway? [y/N] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                console.print("\n  [dim]Setup cancelled.[/dim]")
                raise typer.Exit()
            if proceed not in ("y", "yes"):
                console.print("  [dim]Setup cancelled.[/dim]")
                raise typer.Exit()
        resolved_keys[prov] = chosen_key

    # Determine Fastfold AI Cloud API key (optional, but recommended for cloud-integrated skills)
    cloud_key = _prompt_fastfold_cloud_api_key(cfg, fastfold_api_key)

    # Save
    for prov, key in resolved_keys.items():
        if prov == "openai":
            cfg.set("llm.openai_api_key", key)
        elif prov == "anthropic":
            cfg.set("llm.anthropic_api_key", key)
    active_provider = default_provider if default_provider in selected_providers else selected_providers[0]
    cfg.set("llm.provider", active_provider)
    if cloud_key:
        cfg.set("api.fastfold_cloud_key", cloud_key)
        # Make available immediately to subprocess tools/skills in this run.
        os.environ["FASTFOLD_API_KEY"] = cloud_key
    cfg.save()
    provider_labels = ", ".join(
        "OpenAI" if p == "openai" else "Anthropic" for p in selected_providers
    )
    console.print(
        f"\n  [green]{provider_labels} API key(s) saved to ~/.fastfold-cli/config.json[/green]"
    )
    if cloud_key:
        console.print("  [green]Fastfold AI Cloud API key saved.[/green]")
    else:
        console.print(
            "  [yellow]Fastfold AI Cloud API key skipped.[/yellow] "
            f"Set later with `fastfold config set api.fastfold_cloud_key <key>` "
            f"or visit {FASTFOLD_CLOUD_API_KEYS_URL}"
        )

    if sys.platform == "win32":
        from ct.agent.claude_code_cli import run_windows_autofix

        fix = run_windows_autofix()
        if fix.get("ok"):
            console.print(f"  [green]{fix.get('summary')}[/green]")
        else:
            console.print(f"  [yellow]{fix.get('summary')}[/yellow]")

    # Quick health check
    console.print()
    console.print("  [cyan]Running health check...[/cyan]")
    from ct.agent.doctor import run_checks, to_table, has_errors

    checks = run_checks(cfg)
    console.print(to_table(checks))

    if has_errors(checks):
        console.print(
            "\n  [yellow]Some issues detected.[/yellow] Run `fastfold doctor` for details."
        )
    else:
        console.print("\n  [green]All checks passed.[/green]")

    # Done
    console.print()
    console.print(
        Panel(
            "[bold green]You're all set![/bold green]\n\n"
            "  [cyan]fastfold[/cyan]                      Interactive mode\n"
            '  [cyan]fastfold "your question"[/cyan]      Single query\n'
            "  [cyan]fastfold doctor[/cyan]               Full health check\n"
            "  [cyan]fastfold keys[/cyan]                 Optional API keys",
            title="[green]Quick Start[/green]",
            border_style="green",
        )
    )


def _prompt_api_key() -> str:
    """Prompt user for API key with masked input."""
    console.print(
        "  Get your key at: [link=https://console.anthropic.com/settings/keys]https://console.anthropic.com/settings/keys[/link]"
    )
    console.print()
    try:
        key = _prompt_masked_secret("  Enter your Anthropic API key: ")
    except (EOFError, KeyboardInterrupt):
        console.print("\n  [dim]Setup cancelled.[/dim]")
        raise typer.Exit()
    return key.strip()


def _prompt_openai_api_key() -> str:
    """Prompt user for OpenAI API key with masked input."""
    console.print(
        "  Get your key at: [link=https://platform.openai.com/api-keys]https://platform.openai.com/api-keys[/link]"
    )
    console.print()
    try:
        key = _prompt_masked_secret("  Enter your OpenAI API key: ")
    except (EOFError, KeyboardInterrupt):
        console.print("\n  [dim]Setup cancelled.[/dim]")
        raise typer.Exit()
    return key.strip()


def _prompt_masked_secret(message: str) -> str:
    """Prompt for secrets with visible masked characters while typing."""
    # Prefer questionary in interactive terminals (shows mask while typing).
    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            import questionary
            from prompt_toolkit.styles import Style

            secret_style = Style.from_dict(
                {
                    "qmark": "fg:#00bcd4 bold",
                    "question": "bold",
                    "answer": "fg:#4caf50 bold",
                    "pointer": "fg:#4caf50",
                    "instruction": "fg:#858585",
                    "text": "fg:#858585",
                }
            )
            value = questionary.password(
                message.strip(),
                instruction="(input is masked)",
                qmark="❯",
                style=secret_style,
            ).ask()
            if value is None:
                raise KeyboardInterrupt
            return str(value)
        except (EOFError, KeyboardInterrupt):
            raise
        except Exception:
            pass

    # Fallback for non-interactive terminals.
    import getpass

    return getpass.getpass(message)


def _parse_provider_list(raw: str) -> list[str]:
    """Parse provider list from CLI option, preserving canonical order."""
    requested = {
        part.strip().lower()
        for part in str(raw or "").split(",")
        if part.strip()
    }
    selected = [p for p in SETUP_PROVIDER_ORDER if p in requested]
    if not selected:
        valid = ", ".join(SETUP_PROVIDER_ORDER)
        console.print(f"[red]Invalid --provider value '{raw}'. Valid providers: {valid}[/red]")
        raise typer.Exit(code=2)
    return selected


def _prompt_setup_providers(default_provider: str) -> list[str]:
    """Interactive provider selector (arrow keys + space toggles)."""
    options = list(SETUP_PROVIDER_ORDER)
    labels = {"anthropic": "Anthropic", "openai": "OpenAI"}
    default = default_provider if default_provider in options else options[0]

    # Preferred UX in real terminals: questionary inline checklist.
    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            import questionary
            from prompt_toolkit.styles import Style

            selector_style = Style.from_dict(
                {
                    "qmark": "fg:#00bcd4 bold",
                    "question": "bold",
                    "answer": "fg:#4caf50 bold",
                    "pointer": "fg:#4caf50",
                    "highlighted": "noreverse bold",
                    "selected": "fg:#4caf50 bold",
                    "separator": "fg:#6c6c6c",
                    "disabled": "fg:#858585",
                    "instruction": "fg:#858585",
                    "text": "fg:#858585",
                }
            )

            choices = [
                questionary.Choice(
                    title=f"{labels.get(prov, prov.title())}"
                    + (" (default)" if prov == default else ""),
                    value=prov,
                    checked=(prov == default),
                )
                for prov in options
            ]
            selected = questionary.checkbox(
                "Select provider(s) to configure",
                choices=choices,
                instruction="(↑/↓ move, space toggle, enter confirm)",
                validate=lambda answer: True if answer else "Select at least one provider.",
                style=selector_style,
                qmark="❯",
                pointer="▸",
            ).ask()
            if selected is None:
                console.print("  [dim]Setup cancelled.[/dim]")
                raise typer.Exit()
            normalized = {str(v).strip().lower() for v in selected}
            return [p for p in options if p in normalized]
        except typer.Exit:
            raise
        except Exception:
            # Fall back to text input when terminal capabilities are limited.
            pass

    # Fallback selector for non-interactive or limited environments.
    alias_map = {
        "a": "anthropic",
        "anthropic": "anthropic",
        "o": "openai",
        "openai": "openai",
        "all": "all",
        "both": "all",
    }

    while True:
        console.print("  [cyan]Select provider(s) to configure[/cyan]")
        console.print(
            "  [dim]Options:[/dim] anthropic (a), openai (o), both/all"
        )
        console.print(
            f"  [dim]Press Enter to keep default:[/dim] {labels[default]}"
        )
        try:
            raw = input("  Providers: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print("\n  [dim]Setup cancelled.[/dim]")
            raise typer.Exit()

        if not raw:
            return [default]

        tokens = [t for t in raw.replace(",", " ").split() if t]
        selected: set[str] = set()
        bad: list[str] = []
        for token in tokens:
            mapped = alias_map.get(token)
            if mapped == "all":
                selected.update(options)
            elif mapped in options:
                selected.add(mapped)
            else:
                bad.append(token)

        if bad:
            console.print(f"  [yellow]Invalid selection:[/yellow] {' '.join(bad)}")
            continue
        if not selected:
            console.print("  [yellow]Select at least one provider.[/yellow]")
            continue
        return [p for p in options if p in selected]


def _resolve_provider_key(cfg, provider: str, cli_key: Optional[str] = None) -> str:
    """Resolve provider key from cli arg, existing config, env var, or prompt."""
    provider = str(provider).strip().lower()
    if provider == "openai":
        existing_key = cfg.llm_api_key("openai")
        env_var_name = "OPENAI_API_KEY"
        prompt_fn = _prompt_openai_api_key
        label = "OpenAI"
        config_key = "llm.openai_api_key"
    else:
        existing_key = cfg.llm_api_key("anthropic")
        env_var_name = "ANTHROPIC_API_KEY"
        prompt_fn = _prompt_api_key
        label = "Anthropic"
        config_key = "llm.anthropic_api_key"

    console.print(f"\n  [cyan]{label} setup[/cyan]")
    if cli_key:
        candidate = cli_key.strip()
        issue = cfg.validate_llm_api_key(config_key, candidate)
        if issue:
            console.print(f"  [red]{issue}[/red]")
            raise typer.Exit(code=2)
        return candidate
    if existing_key:
        existing_issue = cfg.validate_llm_api_key(config_key, existing_key)
        if existing_issue:
            console.print(
                f"  [yellow]Existing {label} key is invalid and will be replaced:[/yellow] {existing_issue}"
            )
        else:
            masked = existing_key[:7] + "..." + existing_key[-4:] if len(existing_key) > 11 else "***"
            console.print(f"  API key already configured: [green]{masked}[/green]")
            try:
                keep = input("  Keep existing key? [Y/n] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                console.print("\n  [dim]Setup cancelled.[/dim]")
                raise typer.Exit()
            if keep in ("", "y", "yes"):
                console.print("  [green]Keeping existing key.[/green]")
                return existing_key

    env_key = os.environ.get(env_var_name)
    if env_key:
        env_issue = cfg.validate_llm_api_key(config_key, env_key)
        if env_issue:
            console.print(
                f"  [yellow]Ignoring invalid {env_var_name} value:[/yellow] {env_issue}"
            )
        else:
            masked = env_key[:7] + "..." + env_key[-4:] if len(env_key) > 11 else "***"
            console.print(f"  Found {env_var_name} in environment: [green]{masked}[/green]")
            try:
                save_it = input("  Save to fastfold config? [Y/n] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                console.print("\n  [dim]Setup cancelled.[/dim]")
                raise typer.Exit()
            if save_it in ("", "y", "yes"):
                return env_key

    while True:
        candidate = prompt_fn()
        issue = cfg.validate_llm_api_key(config_key, candidate)
        if not issue:
            return candidate
        console.print(f"  [yellow]{issue}[/yellow]")


def _prompt_fastfold_cloud_api_key(cfg, cli_value: Optional[str] = None) -> Optional[str]:
    """Prompt for Fastfold AI Cloud API key, allowing users to keep/skip."""
    existing = cfg.get("api.fastfold_cloud_key") or os.environ.get("FASTFOLD_API_KEY")
    if isinstance(cli_value, str) and cli_value:
        return cli_value.strip() or None

    console.print()
    console.print(
        "  Fastfold AI Cloud API keys: "
        f"[link={FASTFOLD_CLOUD_API_KEYS_URL}]{FASTFOLD_CLOUD_API_KEYS_URL}[/link]"
    )
    if existing:
        masked = existing[:7] + "..." + existing[-4:] if len(existing) > 11 else "***"
        console.print(f"  Existing Fastfold key detected: [green]{masked}[/green]")
        try:
            keep = input("  Keep existing Fastfold key? [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print("\n  [dim]Setup cancelled.[/dim]")
            raise typer.Exit()
        if keep in ("", "y", "yes"):
            return existing

    console.print("  Press Enter to skip this step.")
    try:
        key = _prompt_masked_secret("  Enter your Fastfold AI Cloud API key: ").strip()
    except (EOFError, KeyboardInterrupt):
        console.print("\n  [dim]Setup cancelled.[/dim]")
        raise typer.Exit()
    return key or None


@app.command("doctor")
def doctor_cmd():
    """Run environment and configuration health checks."""
    from ct.agent.config import Config
    from ct.agent.doctor import run_checks, to_table, has_errors

    cfg = Config.load()
    checks = run_checks(cfg, session=Session(config=cfg, mode="batch"))
    console.print(to_table(checks))

    if has_errors(checks):
        console.print(
            "\n[red]Blocking issues found.[/red] Fix errors above, then rerun `fastfold doctor`."
        )
        raise typer.Exit(code=1)

    console.print("\n[green]No blocking issues found.[/green]")


@app.command("autofix")
def autofix_cmd():
    """Apply automatic local fixes for common install/runtime issues."""
    if sys.platform != "win32":
        console.print("[green]No autofix needed on this platform.[/green]")
        return

    from ct.agent.claude_code_cli import run_windows_autofix

    console.print("[cyan]Running Windows autofix...[/cyan]")
    result = run_windows_autofix()
    if result.get("ok"):
        console.print(f"[green]{result.get('summary')}[/green]")
        if result.get("path"):
            console.print(f"[dim]Using launcher:[/dim] {result.get('path')}")
    else:
        console.print(f"[red]{result.get('summary')}[/red]")
        raise typer.Exit(code=2)


# ─── Data subcommand ──────────────────────────────────────────

data_app = typer.Typer(help="Manage local datasets")
app.add_typer(data_app, name="data")


@data_app.command("pull")
def data_pull(
    dataset: str = typer.Argument(help="Dataset to download (depmap, prism, msigdb, alphafold)"),
    output: Optional[Path] = typer.Option(None, help="Output directory"),
):
    """Download a dataset for local use."""
    from ct.data.downloader import download_dataset

    download_dataset(dataset, output)


@data_app.command("status")
def data_status():
    """Show status of local datasets."""
    from ct.data.downloader import dataset_status

    console.print(dataset_status())


# ─── Tool subcommands (direct tool access) ────────────────────

tool_app = typer.Typer(help="Run individual tools directly")
app.add_typer(tool_app, name="tool")


@tool_app.command("list")
def tool_list():
    """List all available tools."""
    from ct.tools import registry, ensure_loaded, tool_load_errors

    ensure_loaded()
    console.print(registry.list_tools_table())
    errors = tool_load_errors()
    if errors:
        names = ", ".join(sorted(errors.keys())[:8])
        extra = "" if len(errors) <= 8 else f" (+{len(errors) - 8} more)"
        console.print(
            f"[yellow]Warning:[/yellow] {len(errors)} tool module(s) failed to load: {names}{extra}"
        )


# ─── Skill subcommands ────────────────────────────────────────

skill_app = typer.Typer(help="Manage and inspect agent skills")
app.add_typer(skill_app, name="skill")


@skill_app.command("list")
def skill_list():
    """List all loaded agent skills."""
    import json
    from rich.table import Table

    skill_names: dict[str, str] = {}  # name → source label

    # Bundled skills
    bundled_dir = Path(__file__).parent / "skills"
    if bundled_dir.exists():
        for d in sorted(bundled_dir.iterdir()):
            if d.is_dir() and (d / "SKILL.md").exists():
                skill_names[d.name] = "bundled"

    # User-installed skills (skills-lock.json)
    lock_file = Path.cwd() / "skills-lock.json"
    claude_skills_dir = Path.cwd() / ".claude" / "skills"
    if lock_file.exists() and claude_skills_dir.exists():
        try:
            lock = json.loads(lock_file.read_text())
            for name, meta in lock.get("skills", {}).items():
                if (claude_skills_dir / name / "SKILL.md").exists():
                    source = meta.get("source", "user")
                    skill_names[name] = f"installed ({source})"
        except Exception:
            pass

    if not skill_names:
        console.print("[yellow]No skills loaded.[/yellow]")
        raise typer.Exit()

    table = Table(title=f"Agent Skills ({len(skill_names)} loaded)", show_lines=False)
    table.add_column("Skill", style="bold cyan", no_wrap=True)
    table.add_column("Source", style="dim")
    table.add_column("Description", style="white")

    for name, source in sorted(skill_names.items()):
        # Read description from SKILL.md frontmatter
        description = ""
        skill_md_path = (
            (bundled_dir / name / "SKILL.md")
            if source == "bundled"
            else (claude_skills_dir / name / "SKILL.md")
        )
        try:
            content = skill_md_path.read_text()
            for line in content.splitlines():
                if line.startswith("description:"):
                    description = line.split(":", 1)[1].strip()
                    break
        except Exception:
            pass
        table.add_row(name, source, description)

    console.print(table)


# ─── Knowledge subcommands ────────────────────────────────────

knowledge_app = typer.Typer(help="Manage knowledge substrate, ingestion, and quality gates")
app.add_typer(knowledge_app, name="knowledge")


# ─── Trace subcommands ────────────────────────────────────────

trace_app = typer.Typer(help="Inspect and diagnose execution traces")
app.add_typer(trace_app, name="trace")


def _latest_trace_path() -> Optional[Path]:
    from ct.agent.trace import TraceLogger

    traces_dir = TraceLogger.traces_dir()
    traces = list(traces_dir.glob("*.trace.jsonl"))
    if not traces:
        return None
    return max(traces, key=lambda p: p.stat().st_mtime)


def _resolve_trace_path(path: Optional[Path], session_id: Optional[str]) -> Optional[Path]:
    from ct.agent.trace import TraceLogger

    if path is not None and session_id is not None:
        console.print("[red]Use either --path or --session-id, not both.[/red]")
        raise typer.Exit(code=2)

    if path is not None:
        return path
    if session_id:
        return TraceLogger.traces_dir() / f"{session_id}.trace.jsonl"
    return _latest_trace_path()


def _latest_report_path(output_base: Optional[str] = None) -> Optional[Path]:
    reports_dir = (
        Path(output_base) / "reports" if output_base else Path.cwd() / "outputs" / "reports"
    )
    if not reports_dir.exists():
        return None
    reports = list(reports_dir.glob("*.md"))
    if not reports:
        return None
    return max(reports, key=lambda p: p.stat().st_mtime)


def _trace_has_issues(diag: dict) -> bool:
    return any(
        (
            diag.get("unclosed_queries"),
            diag.get("queries_with_no_plan"),
            diag.get("queries_with_no_completion"),
            diag.get("queries_with_synthesis_mismatch"),
        )
    )


def _print_trace_diagnostics_table(diag: dict, title: str):
    table = Table(title=title)
    table.add_column("Metric", style="cyan")
    table.add_column("Value")
    table.add_row("Session", diag.get("session_id", "(unknown)") or "(unknown)")
    table.add_row("Events", str(diag.get("event_count", 0)))
    table.add_row("Queries", str(diag.get("query_count", 0)))
    table.add_row(
        "Query starts / ends",
        f"{diag.get('query_start_count', 0)} / {diag.get('query_end_count', 0)}",
    )
    table.add_row("Step starts", str(diag.get("total_step_start_count", 0)))
    table.add_row("Step completes", str(diag.get("total_step_complete_count", 0)))
    table.add_row("Step fails", str(diag.get("total_step_fail_count", 0)))
    table.add_row("Step retries", str(diag.get("total_step_retry_count", 0)))
    table.add_row("Unclosed queries", str(diag.get("unclosed_queries", [])))
    table.add_row("Queries with failures", str(diag.get("queries_with_failures", [])))
    table.add_row("Queries with no plan", str(diag.get("queries_with_no_plan", [])))
    table.add_row(
        "Queries with no completion",
        str(diag.get("queries_with_no_completion", [])),
    )
    table.add_row(
        "Synthesis mismatches",
        str(diag.get("queries_with_synthesis_mismatch", [])),
    )
    console.print(table)


def _run_step_command(label: str, cmd: list[str], env: Optional[dict] = None) -> bool:
    console.print(f"\n[bold cyan]{label}[/bold cyan]")
    console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    if stdout:
        console.print(stdout)
    if stderr:
        style = "yellow" if proc.returncode == 0 else "red"
        console.print(stderr, style=style)
    if proc.returncode == 0:
        console.print(f"[green]PASS[/green] {label}")
        return True
    console.print(f"[red]FAIL[/red] {label} (exit={proc.returncode})")
    return False


@trace_app.command("diagnose")
def trace_diagnose(
    path: Optional[Path] = typer.Option(None, "--path", "-p", help="Path to a trace JSONL file"),
    session_id: Optional[str] = typer.Option(
        None,
        "--session-id",
        "-s",
        help="Session ID (looks up ~/.fastfold-cli/traces/<id>.trace.jsonl)",
    ),
    as_json: bool = typer.Option(False, "--json", help="Print diagnostics as JSON"),
    show_queries: bool = typer.Option(
        False, "--show-queries", help="Show per-query diagnostics table"
    ),
    strict: bool = typer.Option(
        False, "--strict", help="Exit non-zero if health issues are detected"
    ),
):
    """Diagnose trace health (query integrity, failures, synthesis lifecycle)."""
    from ct.agent.trace import TraceLogger

    trace_path = _resolve_trace_path(path, session_id)
    if trace_path is None:
        console.print("[yellow]No trace files found in ~/.fastfold-cli/traces[/yellow]")
        raise typer.Exit(code=2)
    if not trace_path.exists():
        console.print(f"[red]Trace file not found:[/red] {trace_path}")
        raise typer.Exit(code=2)

    trace = TraceLogger.load(trace_path)
    diag = trace.diagnostics()

    if as_json:
        console.print_json(data=diag)
    else:
        _print_trace_diagnostics_table(diag, title=f"Trace Diagnostics: {trace_path.name}")

        if show_queries:
            q_table = Table(title="Per-Query Diagnostics")
            q_table.add_column("#", style="cyan")
            q_table.add_column("Closed")
            q_table.add_column("Plans")
            q_table.add_column("Step OK")
            q_table.add_column("Step Fail")
            q_table.add_column("Retries")
            q_table.add_column("Synth start/end")
            q_table.add_column("Query")
            for q in diag["queries"]:
                q_table.add_row(
                    str(q["query_number"]),
                    "yes" if q["closed"] else "no",
                    str(q["plan_count"]),
                    str(q["step_complete_count"]),
                    str(q["step_fail_count"]),
                    str(q["step_retry_count"]),
                    f"{q['synthesize_start_count']}/{q['synthesize_end_count']}",
                    (q["query"] or "")[:80],
                )
            console.print(q_table)

    has_issues = _trace_has_issues(diag)
    if strict and has_issues:
        raise typer.Exit(code=2)


@trace_app.command("export")
def trace_export(
    path: Optional[Path] = typer.Option(None, "--path", "-p", help="Path to a trace JSONL file"),
    session_id: Optional[str] = typer.Option(
        None,
        "--session-id",
        "-s",
        help="Session ID (looks up ~/.fastfold-cli/traces/<id>.trace.jsonl)",
    ),
    report: Optional[Path] = typer.Option(
        None, "--report", "-r", help="Optional markdown report to include"
    ),
    out_dir: Optional[Path] = typer.Option(
        None, "--out-dir", help="Bundle output directory (default: ~/.fastfold-cli/exports)"
    ),
    zip_bundle: bool = typer.Option(True, "--zip/--no-zip", help="Also produce a zip archive"),
):
    """Export a reproducible run bundle (trace, diagnostics, report, metadata)."""
    from ct.agent.config import Config
    from ct.agent.trace import TraceLogger
    from ct.agent.trajectory import Trajectory

    trace_path = _resolve_trace_path(path, session_id)
    if trace_path is None:
        console.print("[yellow]No trace files found in ~/.fastfold-cli/traces[/yellow]")
        raise typer.Exit(code=2)
    if not trace_path.exists():
        console.print(f"[red]Trace file not found:[/red] {trace_path}")
        raise typer.Exit(code=2)

    trace = TraceLogger.load(trace_path)
    diag = trace.diagnostics()

    cfg = Config.load()
    resolved_report = report
    if resolved_report is None:
        resolved_report = _latest_report_path(cfg.get("sandbox.output_dir"))
    if resolved_report is not None and not resolved_report.exists():
        console.print(f"[red]Report file not found:[/red] {resolved_report}")
        raise typer.Exit(code=2)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = out_dir or (Path.home() / ".fastfold-cli" / "exports")
    bundle_dir = base / f"ct_run_bundle_{trace.session_id or 'session'}_{ts}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    trace_copy = bundle_dir / "trace.jsonl"
    shutil.copy2(trace_path, trace_copy)
    (bundle_dir / "trace.txt").write_text(trace.to_text(), encoding="utf-8")
    (bundle_dir / "trace_diagnostics.json").write_text(
        json.dumps(diag, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (bundle_dir / "query_summaries.json").write_text(
        json.dumps(trace.query_summaries(), indent=2),
        encoding="utf-8",
    )

    copied_report = None
    if resolved_report is not None:
        copied_report = bundle_dir / "report.md"
        shutil.copy2(resolved_report, copied_report)

    copied_session = None
    session_file = None
    if trace.session_id:
        session_file = Trajectory.sessions_dir() / f"{trace.session_id}.jsonl"
    if session_file is not None and session_file.exists():
        copied_session = bundle_dir / "session.jsonl"
        shutil.copy2(session_file, copied_session)

    manifest = {
        "generated_at_utc": ts,
        "session_id": trace.session_id,
        "source_trace": str(trace_path),
        "included_files": {
            "trace_jsonl": str(trace_copy),
            "trace_txt": str(bundle_dir / "trace.txt"),
            "trace_diagnostics_json": str(bundle_dir / "trace_diagnostics.json"),
            "query_summaries_json": str(bundle_dir / "query_summaries.json"),
            "report_md": str(copied_report) if copied_report else None,
            "session_jsonl": str(copied_session) if copied_session else None,
        },
        "note": (
            "If report was auto-selected, it is the latest markdown report by mtime "
            "from sandbox.output_dir/reports."
            if report is None
            else "Report path explicitly provided."
        ),
    }
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    console.print(f"[green]Bundle exported:[/green] {bundle_dir}")
    if copied_report:
        console.print(f"[dim]Included report:[/dim] {resolved_report}")
    else:
        console.print("[yellow]No report included (none found/provided).[/yellow]")

    if zip_bundle:
        archive = shutil.make_archive(str(bundle_dir), "zip", root_dir=bundle_dir)
        console.print(f"[green]Zip archive:[/green] {archive}")


@app.command("release-check")
def release_check_cmd(
    run_tests: bool = typer.Option(
        True, "--tests/--no-tests", help="Run local pytest regression suite"
    ),
    run_benchmark: bool = typer.Option(
        True, "--benchmark/--no-benchmark", help="Run strict knowledge benchmark gate"
    ),
    run_trace: bool = typer.Option(
        True, "--trace/--no-trace", help="Run strict diagnostics on latest trace"
    ),
    trace_path: Optional[Path] = typer.Option(
        None, "--trace-path", help="Trace path for diagnostics"
    ),
    trace_required: bool = typer.Option(
        False, "--trace-required", help="Fail if no trace file is found"
    ),
    include_live: bool = typer.Option(
        False, "--live", help="Also run live API smoke + live E2E prompt matrix"
    ),
    matrix_limit: int = typer.Option(10, "--matrix-limit", help="Prompt limit for live E2E matrix"),
    matrix_strict: bool = typer.Option(
        True,
        "--matrix-strict/--no-matrix-strict",
        help="Enable strict assertions in live E2E matrix",
    ),
    matrix_max_failed: int = typer.Option(
        1, "--matrix-max-failed", help="Max failed prompts allowed in strict matrix mode"
    ),
    require_profile: Optional[str] = typer.Option(
        None, "--require-profile", help="Require agent.profile to match (e.g. pharma)"
    ),
    pharma: bool = typer.Option(False, "--pharma", help="Enforce pharma deployment policy checks"),
):
    """Run a production release gate: doctor + tests + benchmark + trace diagnostics."""
    from ct.agent.config import Config
    from ct.agent.doctor import has_errors, run_checks, to_table
    from ct.agent.trace import TraceLogger
    from ct.kb.benchmarks import BenchmarkSuite

    failed = False

    console.print("\n[bold]Release Check[/bold]")

    cfg = Config.load()
    if pharma and not require_profile:
        require_profile = "pharma"

    if require_profile:
        expected = require_profile.strip().lower()
        actual = str(cfg.get("agent.profile", "research")).strip().lower()
        if actual != expected:
            console.print(f"[red]Profile mismatch:[/red] expected '{expected}', got '{actual}'.")
            failed = True

    if pharma:
        policy_issues = []
        if str(cfg.get("agent.synthesis_style", "standard")).strip().lower() != "pharma":
            policy_issues.append("agent.synthesis_style must be 'pharma'")
        if not bool(cfg.get("agent.quality_gate_strict", False)):
            policy_issues.append("agent.quality_gate_strict must be true")
        if bool(cfg.get("agent.enable_experimental_tools", False)):
            policy_issues.append("agent.enable_experimental_tools must be false")
        if bool(cfg.get("agent.enable_claude_code_tool", False)):
            policy_issues.append("agent.enable_claude_code_tool must be false")
        if policy_issues:
            console.print("[red]Pharma policy checks failed:[/red]")
            for issue in policy_issues:
                console.print(f"- {issue}")
            failed = True

    checks = run_checks(cfg)
    console.print(to_table(checks))
    if has_errors(checks):
        console.print("[red]Doctor checks have blocking errors.[/red]")
        failed = True

    if run_tests:
        ok = _run_step_command(
            "Local test suite",
            ["pytest", "-q", "tests", "-m", "not api_smoke and not e2e and not e2e_matrix"],
        )
        failed = failed or (not ok)

    if run_benchmark:
        suite = BenchmarkSuite.load()
        summary = suite.run()
        gate = suite.gate(summary, min_pass_rate=0.9)

        table = Table(title="Release Benchmark Gate")
        table.add_column("Metric", style="cyan")
        table.add_column("Value")
        table.add_row("Total cases", str(summary["total_cases"]))
        table.add_row("Expected behavior matches", str(summary["expected_behavior_matches"]))
        table.add_row("Pass rate", str(summary["pass_rate"]))
        table.add_row("Gate", gate["message"])
        console.print(table)

        if not gate["ok"]:
            console.print("[red]Benchmark release gate failed.[/red]")
            failed = True

    if run_trace:
        resolved_trace = trace_path or _latest_trace_path()
        if resolved_trace is None or not resolved_trace.exists():
            msg = "No trace file found for diagnostics."
            if trace_required:
                console.print(f"[red]{msg}[/red]")
                failed = True
            else:
                console.print(f"[yellow]{msg}[/yellow]")
        else:
            trace = TraceLogger.load(resolved_trace)
            diag = trace.diagnostics()
            _print_trace_diagnostics_table(diag, title=f"Trace Diagnostics: {resolved_trace.name}")
            if _trace_has_issues(diag):
                console.print("[red]Trace diagnostics detected integrity issues.[/red]")
                failed = True

    if include_live:
        smoke_env = dict(os.environ)
        smoke_env["CT_RUN_API_SMOKE"] = "1"
        smoke_env.setdefault("CT_API_SMOKE_STRICT", "1")
        smoke_ok = _run_step_command(
            "Live API smoke checks",
            ["pytest", "-q", "tests/test_api_smoke.py"],
            env=smoke_env,
        )
        failed = failed or (not smoke_ok)

        matrix_env = dict(os.environ)
        matrix_env["CT_RUN_E2E_MATRIX"] = "1"
        matrix_env["CT_E2E_MATRIX_LIMIT"] = str(max(1, matrix_limit))
        matrix_env["CT_E2E_MATRIX_STRICT"] = "1" if matrix_strict else "0"
        matrix_env["CT_E2E_MATRIX_MAX_FAILED_QUERIES"] = str(max(0, matrix_max_failed))
        matrix_ok = _run_step_command(
            "Live E2E prompt matrix",
            ["pytest", "-q", "tests/test_e2e_matrix.py", "--run-e2e"],
            env=matrix_env,
        )
        failed = failed or (not matrix_ok)

    if failed:
        console.print("\n[red]Release check failed.[/red]")
        raise typer.Exit(code=2)

    console.print("\n[green]Release check passed.[/green]")


@knowledge_app.command("status")
def knowledge_status():
    """Show knowledge substrate status."""
    from ct.kb.substrate import KnowledgeSubstrate

    substrate = KnowledgeSubstrate()
    summary = substrate.summary()
    table = Table(title="Knowledge Substrate")
    table.add_column("Metric", style="cyan")
    table.add_column("Value")
    table.add_row("Path", summary["path"])
    table.add_row("Schema Version", str(summary["schema_version"]))
    table.add_row("Entities", str(summary["n_entities"]))
    table.add_row("Relations", str(summary["n_relations"]))
    table.add_row("Evidence", str(summary["n_evidence"]))
    for et, count in sorted(summary.get("entity_types", {}).items()):
        table.add_row(f"entity_type:{et}", str(count))
    console.print(table)


@knowledge_app.command("ingest")
def knowledge_ingest(
    source: str = typer.Argument(
        ..., help="Source: evidence_store | pubmed | openalex | opentargets"
    ),
    query: Optional[str] = typer.Option(None, "--query", "-q", help="Query for API sources"),
    max_results: int = typer.Option(10, "--max-results", help="Max records for API sources"),
    scan_limit: int = typer.Option(1000, "--scan-limit", help="Max local evidence rows to scan"),
):
    """Ingest knowledge into canonical substrate."""
    from ct.kb.ingest import KnowledgeIngestionPipeline

    pipeline = KnowledgeIngestionPipeline()
    result = pipeline.ingest(
        source=source,
        query=query,
        max_results=max_results,
        scan_limit=scan_limit,
    )
    if result.get("error"):
        console.print(f"[red]{result['error']}[/red]")
        raise typer.Exit(code=2)
    console.print(result.get("summary", "Ingestion completed."))


@knowledge_app.command("search")
def knowledge_search(
    query: str = typer.Argument(..., help="Search text"),
    limit: int = typer.Option(20, "--limit", help="Maximum entities to return"),
):
    """Search canonical entities."""
    from ct.kb.substrate import KnowledgeSubstrate

    substrate = KnowledgeSubstrate()
    entities = substrate.search_entities(query, limit=limit)
    table = Table(title=f"Knowledge Search: {query}")
    table.add_column("Entity ID", style="cyan")
    table.add_column("Type")
    table.add_column("Name")
    table.add_column("Synonyms", style="dim")
    for entity in entities:
        table.add_row(entity.id, entity.entity_type, entity.name, ", ".join(entity.synonyms[:4]))
    console.print(table)


@knowledge_app.command("related")
def knowledge_related(
    entity_id: str = typer.Argument(..., help="Canonical entity id (e.g., gene:TP53)"),
    predicate: Optional[str] = typer.Option(None, "--predicate", help="Filter predicate"),
    limit: int = typer.Option(20, "--limit", help="Maximum relations"),
):
    """Show related entities for an entity."""
    from ct.kb.substrate import KnowledgeSubstrate

    substrate = KnowledgeSubstrate()
    rows = substrate.related_entities(entity_id, predicate=predicate, limit=limit)
    table = Table(title=f"Related Entities: {entity_id}")
    table.add_column("Predicate", style="cyan")
    table.add_column("Other Entity")
    table.add_column("Support")
    table.add_column("Contradict")
    table.add_column("Avg Score")
    for row in rows:
        table.add_row(
            row["predicate"],
            row["other_entity_id"],
            str(row["support_claims"]),
            str(row["contradict_claims"]),
            str(row["average_claim_score"]),
        )
    console.print(table)


@knowledge_app.command("rank")
def knowledge_rank(
    entity_id: Optional[str] = typer.Option(None, "--entity-id", help="Entity id filter"),
    predicate: Optional[str] = typer.Option(None, "--predicate", help="Predicate filter"),
    limit: int = typer.Option(20, "--limit", help="Maximum relations"),
):
    """Rank relations by evidence strength."""
    from ct.kb.reasoning import EvidenceReasoner
    from ct.kb.substrate import KnowledgeSubstrate

    reasoner = EvidenceReasoner(KnowledgeSubstrate())
    rows = reasoner.rank_relations(entity_id=entity_id, predicate=predicate, limit=limit)
    table = Table(title="Ranked Relations")
    table.add_column("Relation", style="cyan")
    table.add_column("Score")
    table.add_column("Claims")
    for row in rows:
        relation = f"{row['subject_id']} --{row['predicate']}--> {row['object_id']}"
        table.add_row(relation, str(row["score"]), str(row["n_claims"]))
    console.print(table)


@knowledge_app.command("contradictions")
def knowledge_contradictions(
    entity_id: Optional[str] = typer.Option(None, "--entity-id", help="Entity id filter"),
    predicate: Optional[str] = typer.Option(None, "--predicate", help="Predicate filter"),
):
    """Detect contradictory evidence clusters."""
    from ct.kb.reasoning import EvidenceReasoner
    from ct.kb.substrate import KnowledgeSubstrate

    reasoner = EvidenceReasoner(KnowledgeSubstrate())
    rows = reasoner.detect_contradictions(entity_id=entity_id, predicate=predicate)
    table = Table(title="Contradictions")
    table.add_column("Relation", style="cyan")
    table.add_column("Support")
    table.add_column("Contradict")
    table.add_column("Support Score")
    table.add_column("Contradict Score")
    for row in rows:
        relation = f"{row['subject_id']} --{row['predicate']}--> {row['object_id']}"
        table.add_row(
            relation,
            str(row["support_claims"]),
            str(row["contradict_claims"]),
            str(row["support_score"]),
            str(row["contradict_score"]),
        )
    console.print(table)


@knowledge_app.command("schema-check")
def knowledge_schema_check():
    """Run schema drift checks against external integration baselines."""
    from ct.kb.schema_monitor import SchemaMonitor

    monitor = SchemaMonitor()
    results = monitor.check()
    summary = monitor.summarize(results)
    table = Table(title="Schema Drift Monitor")
    table.add_column("Monitor", style="cyan")
    table.add_column("Status")
    table.add_column("Added")
    table.add_column("Removed")
    table.add_column("Error")
    for row in summary["results"]:
        table.add_row(
            row["monitor"],
            row["status"],
            str(len(row["added_paths"])),
            str(len(row["removed_paths"])),
            row.get("error", ""),
        )
    console.print(table)
    if summary["counts"].get("drift", 0) > 0 or summary["counts"].get("error", 0) > 0:
        raise typer.Exit(code=2)


@knowledge_app.command("schema-update")
def knowledge_schema_update(
    monitor: Optional[str] = typer.Option(None, "--monitor", help="Single monitor to update"),
):
    """Update schema drift baselines from current responses."""
    from ct.kb.schema_monitor import SchemaMonitor

    mon = SchemaMonitor()
    results = mon.update_baseline(monitor=monitor)
    summary = mon.summarize(results)
    console.print(f"Updated schema baseline for {summary['total']} monitor(s).")


@knowledge_app.command("benchmark")
def knowledge_benchmark(
    min_pass_rate: float = typer.Option(0.9, "--min-pass-rate", help="Release gate threshold"),
    strict: bool = typer.Option(False, "--strict", help="Exit non-zero if gate fails"),
):
    """Run benchmark suite and evaluate release gate."""
    from ct.kb.benchmarks import BenchmarkSuite

    suite = BenchmarkSuite.load()
    summary = suite.run()
    gate = suite.gate(summary, min_pass_rate=min_pass_rate)

    table = Table(title="Knowledge Benchmarks")
    table.add_column("Metric", style="cyan")
    table.add_column("Value")
    table.add_row("Total cases", str(summary["total_cases"]))
    table.add_row("Expected behavior matches", str(summary["expected_behavior_matches"]))
    table.add_row("Pass rate", str(summary["pass_rate"]))
    table.add_row("Gate", gate["message"])
    console.print(table)
    if strict and not gate["ok"]:
        raise typer.Exit(code=2)


# ─── Report subcommands ──────────────────────────────────────

report_app = typer.Typer(help="Generate and publish reports")
app.add_typer(report_app, name="report")


@report_app.command("list")
def report_list():
    """List available markdown reports."""
    from ct.agent.config import Config

    cfg = Config.load()
    reports_dir = (
        Path(cfg.get("sandbox.output_dir")) / "reports"
        if cfg.get("sandbox.output_dir")
        else Path.cwd() / "outputs" / "reports"
    )
    if not reports_dir.exists():
        console.print("[dim]No reports directory found.[/dim]")
        raise typer.Exit()

    reports = sorted(reports_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not reports:
        console.print("[dim]No reports found.[/dim]")
        raise typer.Exit()

    table = Table(title="Reports")
    table.add_column("#", style="dim")
    table.add_column("File", style="cyan")
    table.add_column("Size", style="dim")
    table.add_column("Modified")
    for i, r in enumerate(reports[:20], 1):
        size = r.stat().st_size
        size_str = f"{size / 1024:.1f}K" if size > 1024 else f"{size}B"
        mtime = datetime.fromtimestamp(r.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        table.add_row(str(i), r.name, size_str, mtime)
    console.print(table)


@report_app.command("publish")
def report_publish(
    path: Optional[Path] = typer.Option(None, "--path", "-p", help="Markdown report to convert"),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Output HTML path"),
):
    """Convert a markdown report to a shareable HTML page."""
    from ct.agent.config import Config
    from ct.reports.html import publish_report

    if path is None:
        cfg = Config.load()
        path = _latest_report_path(cfg.get("sandbox.output_dir"))
        if path is None:
            console.print("[yellow]No reports found. Run a query first.[/yellow]")
            raise typer.Exit(code=2)

    if not path.exists():
        console.print(f"[red]File not found:[/red] {path}")
        raise typer.Exit(code=2)

    result = publish_report(path, out_path=out)
    console.print(f"[green]Published:[/green] {result}")


@report_app.command("show")
def report_show(
    path: Optional[Path] = typer.Option(None, "--path", "-p", help="HTML report to open"),
):
    """Open an HTML report in the default browser."""
    import webbrowser

    from ct.agent.config import Config
    from ct.reports.html import publish_report

    if path is None:
        cfg = Config.load()
        md_path = _latest_report_path(cfg.get("sandbox.output_dir"))
        if md_path is None:
            console.print("[yellow]No reports found.[/yellow]")
            raise typer.Exit(code=2)
        html_path = md_path.with_suffix(".html")
        if not html_path.exists():
            html_path = publish_report(md_path)
            console.print(f"[dim]Auto-published: {html_path}[/dim]")
        path = html_path

    if not path.exists():
        console.print(f"[red]File not found:[/red] {path}")
        raise typer.Exit(code=2)

    webbrowser.open(f"file://{path.resolve()}")
    console.print(f"[green]Opened in browser:[/green] {path}")


@report_app.command("notebook")
def report_notebook(
    session: Optional[str] = typer.Option(
        None, "--session", "-s", help="Session ID (prefix or full). Default: most recent"
    ),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Output notebook path"),
    html: bool = typer.Option(False, "--html", help="Also export as HTML"),
):
    """Export an agent trace as a Jupyter notebook (.ipynb)."""
    import re
    from ct.agent.trace_store import TraceStore

    # Find trace file
    trace_path = TraceStore.find_trace(session)
    if trace_path is None:
        console.print(
            "[yellow]No trace files found. Run a query first to generate a trace.[/yellow]"
        )
        raise typer.Exit(code=2)

    console.print(f"  [dim]Trace:[/dim] {trace_path.name}")

    # Lazy import nbformat
    try:
        from ct.reports.notebook import trace_to_notebook, save_notebook
    except ImportError:
        console.print("[red]nbformat is required. Install with:[/red] pip install nbformat")
        raise typer.Exit(code=2)

    # Convert trace to notebook
    nb = trace_to_notebook(trace_path)

    # Determine output path
    if out is None:
        from ct.agent.config import Config

        cfg = Config.load()
        reports_dir = (
            Path(cfg.get("sandbox.output_dir")) / "reports"
            if cfg.get("sandbox.output_dir")
            else Path.cwd() / "outputs" / "reports"
        )
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", trace_path.stem.replace(".trace", "")).strip("_")
        out = reports_dir / f"{slug}.ipynb"

    out_path = save_notebook(nb, out)
    console.print(f"  [green]Notebook:[/green] {out_path}")

    # Optional HTML export
    if html:
        try:
            import nbconvert
            from nbconvert import HTMLExporter

            exporter = HTMLExporter()
            html_body, _ = exporter.from_notebook_node(nb)
            html_path = out_path.with_suffix(".html")
            html_path.write_text(html_body, encoding="utf-8")
            console.print(f"  [green]HTML:[/green] {html_path}")
        except ImportError:
            console.print(
                "[yellow]nbconvert not installed. Falling back to markdown-based HTML.[/yellow]\n"
                "  [dim]Install with: pip install nbconvert[/dim]"
            )
            # Fall back to existing HTML renderer on markdown cells
            from ct.reports.html import render_html_report

            md_parts = [c.source for c in nb.cells if c.cell_type == "markdown"]
            md_text = "\n\n".join(md_parts)
            html_content = render_html_report(md_text, title="Fastfold Notebook Export")
            html_path = out_path.with_suffix(".html")
            html_path.write_text(html_content, encoding="utf-8")
            console.print(f"  [green]HTML (markdown only):[/green] {html_path}")


# ─── Case study subcommands ─────────────────────────────────

case_study_app = typer.Typer(help="Run curated drug case studies")
app.add_typer(case_study_app, name="case-study")


@case_study_app.command("list")
def case_study_list():
    """List available curated case studies."""
    from ct.agent.case_studies import CASE_STUDIES

    table = Table(title="Case Studies")
    table.add_column("ID", style="cyan")
    table.add_column("Drug")
    table.add_column("Threads", style="dim")
    table.add_column("Description")
    for case_id, case in CASE_STUDIES.items():
        table.add_row(
            case_id,
            case.name,
            str(len(case.thread_goals)),
            case.description[:80] + ("..." if len(case.description) > 80 else ""),
        )
    console.print(table)


@case_study_app.command("run")
def case_study_run(
    case_id: str = typer.Argument(..., help="Case study ID (e.g., revlimid, gleevec)"),
    threads: Optional[int] = typer.Option(
        None, "--threads", "-t", help="Number of parallel threads"
    ),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="LLM model to use"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Run a curated drug case study with multi-agent analysis."""
    from ct.agent.case_studies import CASE_STUDIES, run_case_study
    from ct.agent.config import Config
    from ct.reports.html import publish_report

    if case_id not in CASE_STUDIES:
        available = ", ".join(sorted(CASE_STUDIES.keys()))
        console.print(f"[red]Unknown case study '{case_id}'.[/red] Available: {available}")
        raise typer.Exit(code=2)

    cfg = Config.load()
    if model:
        cfg.set("llm.model", model)

    llm_issue = cfg.llm_preflight_issue()
    if llm_issue:
        console.print("\n  [yellow]First-time setup required.[/yellow]\n")
        setup_cmd(api_key=None, openai_api_key=None, fastfold_api_key=None, provider=None)
        cfg = Config.load()
        if model:
            cfg.set("llm.model", model)
        llm_issue = cfg.llm_preflight_issue()
        if llm_issue:
            console.print(f"\n  [red]Setup incomplete:[/red] {llm_issue}")
            raise typer.Exit(code=2)

    session = Session(config=cfg, verbose=verbose)
    case = CASE_STUDIES[case_id]

    print_banner()
    console.print(
        Panel(
            f"[bold]{case.name}[/bold]\n[dim]{case.description}[/dim]",
            title="[cyan]Case Study[/cyan]",
            border_style="cyan",
        )
    )
    console.print()

    result = run_case_study(session, case_id, n_threads=threads)

    # Auto-publish HTML
    md_path = _latest_report_path(cfg.get("sandbox.output_dir"))
    if md_path:
        html_path = publish_report(md_path)
        console.print(f"\n  [green]HTML report:[/green] {html_path}")

    console.print()


# ─── Main entry point ─────────────────────────────────────────


@app.command("run", hidden=True)
def run_cmd(
    query_parts: list[str] = typer.Argument(None, help="Research question to investigate"),
    smiles: Optional[str] = typer.Option(None, "--smiles", "-s", help="Compound SMILES string"),
    target: Optional[str] = typer.Option(
        None, "--target", "-t", help="Target protein (UniProt ID or gene symbol)"
    ),
    indication: Optional[str] = typer.Option(
        None, "--indication", "-i", help="Cancer type / indication"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output directory for reports"
    ),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="LLM model to use"),
    agents: Optional[int] = typer.Option(
        None, "--agents", "-a", help="Run with N parallel research agents"
    ),
    resume: Optional[str] = typer.Option(
        None, "--resume", "-r", help="Resume a previous session (ID or 'last')"
    ),
    continue_last: bool = typer.Option(
        False, "--continue", "-c", help="Continue the most recent session"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    version: bool = typer.Option(False, "--version", "-V", help="Show version"),
):
    """
    Fastfold Agent CLI — Where scientists and AI agents work together doing real science.

    Run without arguments for interactive mode.
    Pass a question for single-query mode.
    """
    if version:
        console.print(f"fastfold v{__version__}")
        raise typer.Exit()

    query = " ".join(query_parts).strip() if query_parts else None

    # Build context from flags
    context = {}
    if smiles:
        context["compound_smiles"] = smiles
    if target:
        context["target"] = target
    if indication:
        context["indication"] = indication

    # Determine session resume
    resume_id = None
    if continue_last:
        resume_id = "last"
    elif resume:
        resume_id = resume

    if query:
        # Single query mode
        run_query(query, context, output, model, verbose, agents=agents)
    else:
        # Interactive mode
        run_interactive(context, output, model, verbose, resume_id=resume_id)


def run_query(
    query: str,
    context: dict,
    output: Optional[Path],
    model: Optional[str],
    verbose: bool,
    agents: Optional[int] = None,
):
    """Execute a single research query."""
    from ct.agent.config import Config

    cfg = Config.load()
    if model:
        cfg.set("llm.model", model)

    llm_issue = cfg.llm_preflight_issue()
    if llm_issue:
        console.print("\n  [yellow]First-time setup required.[/yellow]\n")
        setup_cmd(api_key=None, openai_api_key=None, fastfold_api_key=None, provider=None)
        cfg = Config.load()
        if model:
            cfg.set("llm.model", model)
        llm_issue = cfg.llm_preflight_issue()
        if llm_issue:
            console.print(f"\n  [red]Setup incomplete:[/red] {llm_issue}")
            raise typer.Exit(code=2)

    session = Session(config=cfg, verbose=verbose)

    print_banner()
    console.print(
        Panel(
            f"[bold]{query}[/bold]",
            title="[cyan]fastfold[/cyan]",
            border_style="cyan",
        )
    )
    console.print()

    # Multi-agent mode
    if agents is not None and agents > 1:
        from ct.agent.orchestrator import ResearchOrchestrator

        orchestrator = ResearchOrchestrator(session, n_threads=agents)
        result = orchestrator.run(query, context)

        if output:
            output.mkdir(parents=True, exist_ok=True)
            report_path = output / "report.md"
            report_path.write_text(result.to_markdown())
            console.print(f"\n  Report saved to {report_path}")

        console.print()
        return

    # Execute via Agent SDK runner (default) or legacy AgentLoop (fallback)
    use_sdk = cfg.get("agent.use_sdk", True)

    if use_sdk:
        from ct.agent.runner import AgentRunner

        agent = AgentRunner(session)
        result = agent.run(query, context)
    else:
        from ct.agent.loop import AgentLoop, ClarificationNeeded

        agent = AgentLoop(session)
        try:
            result = agent.run(query, context)
        except ClarificationNeeded as e:
            console.print(f"\n  [cyan]{e.clarification.question}[/cyan]")
            if e.clarification.suggestions:
                console.print(f"  [dim]e.g. {', '.join(e.clarification.suggestions[:3])}[/dim]")
            console.print(
                f"\n  [dim]Tip: provide context with --smiles, --target, or --indication flags.[/dim]"
            )
            return

    # Output
    if output:
        output.mkdir(parents=True, exist_ok=True)
        report_path = output / "report.md"
        report_path.write_text(result.to_markdown())
        console.print(f"\n  Report saved to {report_path}")

    # Summary already streamed to stdout during synthesis
    console.print()


@app.command("bench")
def bench(
    question: Optional[str] = typer.Option(
        None, "--question", "-q", help="Run a single question by ID"
    ),
    parallel: int = typer.Option(10, "--parallel", "-p", help="Number of parallel workers"),
    timeout: int = typer.Option(300, "--timeout", help="Timeout per question in seconds"),
    max_turns: int = typer.Option(15, "--max-turns", help="Max agentic loop turns"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="LLM model override"),
    eval_model: str = typer.Option(
        "claude-sonnet-4-5-20250929", "--eval-model", help="Model for LLM-as-judge evaluation"
    ),
    manifest: str = typer.Option(
        "/mnt/bixbench/manifest.json", "--manifest", help="Path to manifest JSON"
    ),
    output: str = typer.Option("/mnt/bixbench/outputs", "--output", "-o", help="Output directory"),
    only_failed: bool = typer.Option(False, "--only-failed", help="Re-run only failed questions"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview questions without executing"),
    no_eval: bool = typer.Option(False, "--no-eval", help="Skip inline LLM evaluation"),
    force: bool = typer.Option(
        False, "--force", "-f", help="Clear previous results and re-run everything"
    ),
    max_questions: Optional[int] = typer.Option(
        None, "--max-questions", "-n", help="Limit to first N questions"
    ),
):
    """Run the BixBench-50 benchmark suite."""
    import shutil as _shutil
    from ct.bench.runner import BenchRunner

    if force:
        out = Path(output)
        for sub in ("results", "evals", ".preview_cache"):
            d = out / sub
            if d.exists():
                _shutil.rmtree(d)
        for f in ("all_results.json", "llm_eval.json"):
            p = out / f
            if p.exists():
                p.unlink()
        console.print(f"  [dim]Cleared {out}[/dim]")

    if dry_run:
        import json as _json

        with open(manifest) as f:
            questions = _json.load(f)
        if question:
            questions = [q for q in questions if q["question_id"] == question]
        if max_questions:
            questions = questions[:max_questions]

        table = Table(title=f"BixBench Dry Run — {len(questions)} questions")
        table.add_column("#", width=4)
        table.add_column("Question ID", style="cyan", width=14)
        table.add_column("Data", width=5)
        table.add_column("Question", max_width=60)
        table.add_column("Ideal", max_width=30)

        for i, q in enumerate(questions, 1):
            has_data = "Y" if q.get("data_dir") and Path(q["data_dir"]).exists() else "N"
            table.add_row(
                str(i),
                q["question_id"],
                has_data,
                q["question"][:60],
                q["ideal"][:30],
            )
        console.print(table)
        return

    runner = BenchRunner(
        manifest_path=manifest,
        output_dir=output,
        parallel=parallel,
        timeout=timeout,
        max_turns=max_turns,
        model=model,
        eval_model=eval_model,
        no_eval=no_eval,
        only_failed=only_failed,
        question_id=question,
        max_questions=max_questions,
    )

    summary = runner.run()
    if summary.get("total"):
        console.print(
            f"\n[bold]Score: {summary['passed']}/{summary['total']} "
            f"({summary['accuracy']:.1%})[/bold]"
        )


def print_banner():
    """Print the startup banner with molecule illustration."""
    from ct.agent.config import Config
    from rich.padding import Padding
    from ct.tools import registry, ensure_loaded
    from rich.table import Table
    from rich.text import Text

    ensure_loaded()
    cfg = Config.load()
    n_tools = len(registry.list_tools())
    n_skills = _count_installed_claude_skills()
    model_raw = str(cfg.get("llm.model") or "").strip()
    model_names = {
        "claude-sonnet-4-5-20250929": "Sonnet 4.5",
        "claude-haiku-4-5-20251001": "Haiku 4.5",
        "claude-opus-4-6": "Opus 4.6",
        "gpt-4o": "GPT-4o",
        "gpt-4o-mini": "GPT-4o Mini",
    }
    model_name = model_names.get(model_raw, model_raw)

    tier = _resolve_fastfold_subscription_tier(cfg)
    tier_display = _format_plan_label(tier)
    status_parts: list[str] = []
    if model_name:
        status_parts.append(model_name)
    if tier_display:
        status_parts.append(tier_display)
    status_line = " · ".join(status_parts)
    upgrade_version = get_upgrade_available_version(__version__)

    logo_text = Text.from_markup(BANNER.strip("\n"))
    meta_lines = [
        f"[bold white]Fastfold Agent CLI[/] [dim]v{__version__}[/]",
    ]
    if status_line:
        meta_lines.append(f"[dim]{status_line}[/dim]")
    if upgrade_version:
        meta_lines.append(
            f"[bold yellow]Upgrade available:[/] [dim]v{__version__} -> v{upgrade_version}[/dim] "
            "[dim]Run [/dim][bold #D4148E]/upgrade[/]"
        )
    meta_lines.append(f"[dim]{n_tools} tools · {n_skills} skills[/dim]")
    meta_lines.append(_random_command_tip_markup())
    meta_lines.append(_random_news_item_markup())
    meta_text = Text.from_markup("\n".join(meta_lines))

    header = Table.grid(padding=(0, 2))
    header.add_column(no_wrap=True)
    header.add_column()
    header.add_row(logo_text, Padding(meta_text, (2, 0, 0, 1)))
    console.print(header)


def run_interactive(
    context: dict,
    output: Optional[Path],
    model: Optional[str],
    verbose: bool,
    resume_id: str = None,
):
    """Run interactive session."""
    from ct.agent.config import Config

    def _pin_startup_to_bottom() -> None:
        """Place startup banner/prompt near terminal bottom like Claude Code."""
        try:
            rows = int(getattr(console.size, "height", 0) or 0)
        except Exception:
            rows = 0
        # Approximate visible startup footprint:
        # banner (~11 lines) + spacing + first prompt/separator.
        startup_lines = 15
        spacer = max(0, rows - startup_lines)
        console.clear()
        if spacer:
            console.print("\n" * spacer, end="")

    cfg = Config.load()
    if model:
        cfg.set("llm.model", model)

    llm_issue = cfg.llm_preflight_issue()
    if llm_issue:
        console.print("\n  [yellow]First-time setup required.[/yellow]\n")
        setup_cmd(api_key=None, openai_api_key=None, fastfold_api_key=None, provider=None)
        # Reload config after setup
        cfg = Config.load()
        llm_issue = cfg.llm_preflight_issue()
        if llm_issue:
            console.print(f"\n  [red]Setup incomplete:[/red] {llm_issue}")
            return

    _pin_startup_to_bottom()
    print_banner()

    console.print()

    terminal = InteractiveTerminal(config=cfg, verbose=verbose)
    terminal.run(initial_context=context, resume_id=resume_id)


def entry():
    """Package entry point."""
    argv = list(sys.argv[1:])
    passthrough = {
        "config",
        "data",
        "tool",
        "skill",
        "trace",
        "knowledge",
        "keys",
        "doctor",
        "setup",
        "release-check",
        "upgrade",
        "report",
        "case-study",
        "bench",
        "run",
        "--help",
        "-h",
        "--install-completion",
        "--show-completion",
    }

    # Support `fastfold help` and `fastfold help <subcommand>` as aliases.
    if argv and argv[0] == "help":
        argv = ["--help"] if len(argv) == 1 else [*argv[1:], "--help"]

    # Route plain invocations to hidden `run` command so:
    #   fastfold                       -> interactive mode
    #   fastfold "question"            -> single-query mode
    #   fastfold --smiles ... "q"      -> single-query with context
    # while preserving explicit subcommands like `fastfold config ...`.
    if not argv or argv[0] not in passthrough:
        argv = ["run", *argv]

    app(args=argv, prog_name="fastfold")


if __name__ == "__main__":
    entry()
