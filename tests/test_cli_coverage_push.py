"""Extensive mocked CliRunner tests to push cli.py coverage on remaining gaps."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from agent.config import Config
from agent.skills import SkillInfo
from agent.trace import TraceLogger
from cli import (
    _prompt_install_skills,
    _random_command_tip_markup,
    _random_news_item_markup,
    _resolve_provider_key,
    _select_suggested_sources,
    add_skills_cmd,
    app,
    is_newer_version,
    resolve_upgrade_flavor,
    run_query,
    skill_delete,
    skill_info_cmd,
    skill_remove,
)

runner = CliRunner()


def _mock_cfg(monkeypatch, data=None):
    cfg = Config(data=dict(data or {}))
    monkeypatch.setattr("agent.config.Config.load", lambda: cfg)
    return cfg


class TestAutofixCmd:
    def test_autofix_non_windows(self):
        result = runner.invoke(app, ["autofix"])
        assert result.exit_code == 0
        assert "No autofix needed" in result.stdout

    @patch("agent.claude_code_cli.run_windows_autofix")
    def test_autofix_windows_success(self, mock_autofix):
        mock_autofix.return_value = {
            "ok": True,
            "summary": "Fixed launcher",
            "path": "C:\\fastfold\\launcher.cmd",
        }
        with patch("cli.sys.platform", "win32"):
            result = runner.invoke(app, ["autofix"])
        assert result.exit_code == 0
        assert "Fixed launcher" in result.stdout
        assert "launcher.cmd" in result.stdout

    @patch("agent.claude_code_cli.run_windows_autofix")
    def test_autofix_windows_failure_exits(self, mock_autofix):
        mock_autofix.return_value = {"ok": False, "summary": "Could not repair"}
        with patch("cli.sys.platform", "win32"):
            result = runner.invoke(app, ["autofix"])
        assert result.exit_code == 2
        assert "Could not repair" in result.stdout


class TestDataPull:
    def test_data_pull_invokes_downloader(self, tmp_path):
        with patch("data.downloader.download_dataset") as mock_download:
            result = runner.invoke(app, ["data", "pull", "depmap", "--output", str(tmp_path)])
        assert result.exit_code == 0
        mock_download.assert_called_once_with("depmap", tmp_path)


class TestSkillUpgradeRemoveDelete:
    @patch("agent.skills.upgrade_skills")
    def test_skill_upgrade_reports_added_updated_failed(self, mock_upgrade):
        mock_upgrade.return_value = {
            "added": ["fold"],
            "updated": ["md"],
            "npx_synced": 2,
            "failed": [("bad/source", "network error")],
            "summary": "Sync complete",
        }
        result = runner.invoke(app, ["skills", "upgrade"])
        assert result.exit_code == 0
        assert "Added:" in result.stdout
        assert "Updated:" in result.stdout
        assert "npx-synced:" in result.stdout
        assert "Failed:" in result.stdout

    @patch("agent.skills.install_skill")
    def test_skill_upgrade_catalog_only_success(self, mock_install):
        mock_install.return_value = {"ok": True, "summary": "Catalog synced"}
        result = runner.invoke(app, ["skills", "upgrade", "--catalog-only"])
        assert result.exit_code == 0
        assert "Catalog synced" in result.stdout

    @patch("agent.skills.install_skill")
    def test_skill_upgrade_catalog_only_failure(self, mock_install):
        mock_install.return_value = {"ok": False, "summary": "Catalog sync failed"}
        result = runner.invoke(app, ["skills", "upgrade", "--catalog-only"])
        assert result.exit_code == 1

    def test_skill_upgrade_mutually_exclusive_flags(self):
        result = runner.invoke(app, ["skills", "upgrade", "--catalog-only", "--no-catalog"])
        assert result.exit_code == 2
        assert "mutually exclusive" in result.stdout.lower()

    @patch("agent.skills.remove_skill")
    def test_skill_remove_success(self, mock_remove):
        mock_remove.return_value = {"ok": True, "summary": "Removed fold"}
        result = runner.invoke(app, ["skills", "remove", "fold"])
        assert result.exit_code == 0
        assert "Removed fold" in result.stdout

    @patch("agent.skills.remove_skill")
    def test_skill_remove_not_found(self, mock_remove):
        mock_remove.return_value = {"ok": False, "summary": "Skill fold not installed"}
        result = runner.invoke(app, ["skills", "remove", "fold"])
        assert result.exit_code == 1

    @patch("agent.skills.remove_skill")
    def test_skill_delete_single_name(self, mock_remove):
        mock_remove.return_value = {"ok": True, "summary": "Deleted fold"}
        result = runner.invoke(app, ["skills", "delete", "fold"])
        assert result.exit_code == 0
        assert "Deleted fold" in result.stdout

    def test_skill_delete_requires_name_without_all(self):
        result = runner.invoke(app, ["skills", "delete"])
        assert result.exit_code == 2
        assert "Specify a skill name" in result.stdout

    @patch("agent.skills.user_installed_skill_names", return_value=[])
    def test_skill_delete_all_when_none_installed(self, _mock_names):
        result = runner.invoke(app, ["skills", "delete", "--all", "--yes"])
        assert result.exit_code == 0
        assert "No user-installed skills" in result.stdout

    @patch("agent.skills.remove_all_skills")
    @patch("agent.skills.user_installed_skill_names", return_value=["fold"])
    def test_skill_delete_all_with_yes(self, _mock_names, mock_remove_all):
        mock_remove_all.return_value = {
            "ok": True,
            "removed": ["fold"],
            "summary": "Removed 1 skill(s).",
        }
        result = runner.invoke(app, ["skills", "delete", "--all", "--yes"])
        assert result.exit_code == 0
        mock_remove_all.assert_called_once()

    @patch("agent.skills.discover_skills")
    def test_skill_find_with_results(self, mock_discover):
        mock_discover.return_value = [
            {
                "name": "fold",
                "install_source": "fastfold-ai/skills@skills/fold",
                "description": "Protein folding",
            }
        ]
        result = runner.invoke(app, ["skills", "find", "fold"])
        assert result.exit_code == 0
        assert "fold" in result.stdout

    @patch("agent.skills.skill_info")
    def test_skill_info_cmd_success(self, mock_info):
        mock_info.return_value = SkillInfo(
            name="fold",
            description="Fold proteins",
            tags=["folding"],
            path=Path("/tmp/fold/SKILL.md"),
            source="global",
        )
        result = runner.invoke(app, ["skills", "info", "fold"])
        assert result.exit_code == 0
        assert "Fold proteins" in result.stdout

    @patch("cli.skill_add")
    def test_add_skills_cmd_delegates(self, mock_add):
        result = runner.invoke(app, ["add", "skills", "owner/repo@skills/foo"])
        assert result.exit_code == 0
        mock_add.assert_called_once_with("owner/repo@skills/foo")

    @patch("cli.skill_add")
    def test_add_skills_cmd_direct(self, mock_add):
        add_skills_cmd("local/path")
        mock_add.assert_called_once_with("local/path")

    @patch("agent.skills.install_skill")
    def test_add_skill_hidden_alias(self, mock_install):
        mock_install.return_value = {"ok": True, "summary": "ok", "via": "git"}
        result = runner.invoke(app, ["add", "skill", "fold"])
        assert result.exit_code == 0
        mock_install.assert_called_once()


class TestKnowledgeSubcommands:
    def test_knowledge_ingest_success(self):
        with patch("kb.ingest.KnowledgeIngestionPipeline") as mock_cls:
            mock_cls.return_value.ingest.return_value = {"summary": "Ingested 5 records"}
            result = runner.invoke(
                app,
                ["knowledge", "ingest", "pubmed", "--query", "TP53", "--max-results", "3"],
            )
        assert result.exit_code == 0
        assert "Ingested 5 records" in result.stdout

    def test_knowledge_ingest_error(self):
        with patch("kb.ingest.KnowledgeIngestionPipeline") as mock_cls:
            mock_cls.return_value.ingest.return_value = {"error": "bad source"}
            result = runner.invoke(app, ["knowledge", "ingest", "evidence_store"])
        assert result.exit_code == 2

    def test_knowledge_status(self):
        summary = {
            "path": "/tmp/substrate.json",
            "schema_version": 1,
            "n_entities": 10,
            "n_relations": 5,
            "n_evidence": 20,
            "entity_types": {"gene": 7},
        }
        with patch("kb.substrate.KnowledgeSubstrate") as mock_cls:
            mock_cls.return_value.summary.return_value = summary
            result = runner.invoke(app, ["knowledge", "status"])
        assert result.exit_code == 0
        assert "Knowledge Substrate" in result.stdout

    def test_knowledge_search(self):
        entity = MagicMock()
        entity.id = "gene:TP53"
        entity.entity_type = "gene"
        entity.name = "TP53"
        entity.synonyms = ["P53", "tumor protein"]
        with patch("kb.substrate.KnowledgeSubstrate") as mock_cls:
            mock_cls.return_value.search_entities.return_value = [entity]
            result = runner.invoke(app, ["knowledge", "search", "TP53", "--limit", "5"])
        assert result.exit_code == 0
        assert "gene:TP53" in result.stdout

    def test_knowledge_related(self):
        rows = [
            {
                "predicate": "targets",
                "other_entity_id": "gene:MDM2",
                "support_claims": 3,
                "contradict_claims": 0,
                "average_claim_score": 0.9,
            }
        ]
        with patch("kb.substrate.KnowledgeSubstrate") as mock_cls:
            mock_cls.return_value.related_entities.return_value = rows
            result = runner.invoke(
                app,
                ["knowledge", "related", "gene:TP53", "--predicate", "targets"],
            )
        assert result.exit_code == 0
        assert "gene:MDM2" in result.stdout

    def test_knowledge_rank(self):
        rows = [
            {
                "subject_id": "gene:TP53",
                "predicate": "targets",
                "object_id": "disease:cancer",
                "score": 0.95,
                "n_claims": 4,
            }
        ]
        with patch("kb.reasoning.EvidenceReasoner") as mock_reasoner_cls:
            mock_reasoner_cls.return_value.rank_relations.return_value = rows
            result = runner.invoke(app, ["knowledge", "rank", "--entity-id", "gene:TP53"])
        assert result.exit_code == 0
        assert "gene:TP53" in result.stdout

    def test_knowledge_contradictions(self):
        rows = [
            {
                "subject_id": "gene:TP53",
                "predicate": "associated_with",
                "object_id": "disease:aml",
                "support_claims": 2,
                "contradict_claims": 1,
                "support_score": 0.8,
                "contradict_score": 0.6,
            }
        ]
        with patch("kb.reasoning.EvidenceReasoner") as mock_reasoner_cls:
            mock_reasoner_cls.return_value.detect_contradictions.return_value = rows
            result = runner.invoke(app, ["knowledge", "contradictions"])
        assert result.exit_code == 0
        assert "Contradictions" in result.stdout

    def test_knowledge_schema_check_passes(self):
        with patch("kb.schema_monitor.SchemaMonitor") as mock_cls:
            monitor = mock_cls.return_value
            monitor.check.return_value = []
            monitor.summarize.return_value = {
                "results": [
                    {
                        "monitor": "pubmed",
                        "status": "ok",
                        "added_paths": [],
                        "removed_paths": [],
                        "error": "",
                    }
                ],
                "counts": {"drift": 0, "error": 0},
            }
            result = runner.invoke(app, ["knowledge", "schema-check"])
        assert result.exit_code == 0
        assert "Schema Drift Monitor" in result.stdout

    def test_knowledge_schema_check_fails_on_drift(self):
        with patch("kb.schema_monitor.SchemaMonitor") as mock_cls:
            monitor = mock_cls.return_value
            monitor.check.return_value = [{"status": "drift"}]
            monitor.summarize.return_value = {
                "results": [
                    {
                        "monitor": "pubmed",
                        "status": "drift",
                        "added_paths": ["$.new"],
                        "removed_paths": [],
                        "error": "",
                    }
                ],
                "counts": {"drift": 1, "error": 0},
            }
            result = runner.invoke(app, ["knowledge", "schema-check"])
        assert result.exit_code == 2

    def test_knowledge_schema_update(self):
        with patch("kb.schema_monitor.SchemaMonitor") as mock_cls:
            monitor = mock_cls.return_value
            monitor.update_baseline.return_value = [{"monitor": "pubmed"}]
            monitor.summarize.return_value = {"total": 1}
            result = runner.invoke(app, ["knowledge", "schema-update", "--monitor", "pubmed"])
        assert result.exit_code == 0
        assert "Updated schema baseline" in result.stdout

    def test_knowledge_benchmark_success(self):
        class FakeSuite:
            def run(self):
                return {
                    "total_cases": 3,
                    "expected_behavior_matches": 3,
                    "pass_rate": 1.0,
                }

            def gate(self, summary, min_pass_rate=0.9):
                return {"ok": True, "message": "passed"}

        with patch("kb.benchmarks.BenchmarkSuite.load", return_value=FakeSuite()):
            result = runner.invoke(app, ["knowledge", "benchmark"])
        assert result.exit_code == 0
        assert "Knowledge Benchmarks" in result.stdout

    def test_knowledge_benchmark_strict_failure(self):
        class FakeSuite:
            def run(self):
                return {
                    "total_cases": 2,
                    "expected_behavior_matches": 1,
                    "pass_rate": 0.5,
                }

            def gate(self, summary, min_pass_rate=0.9):
                return {"ok": False, "message": "failed gate"}

        with patch("kb.benchmarks.BenchmarkSuite.load", return_value=FakeSuite()):
            result = runner.invoke(app, ["knowledge", "benchmark", "--strict"])
        assert result.exit_code == 2


class TestTraceSubcommands:
    def test_trace_diagnose_show_queries(self, tmp_path):
        trace = TraceLogger("show-q")
        trace.query_start("analyze EGFR")
        trace.plan([], query="analyze EGFR")
        trace.step_start(1, "target.druggability", {"target": "EGFR"})
        trace.step_complete(1, "target.druggability", {"summary": "ok"}, duration_s=0.1)
        trace.synthesize_start()
        trace.synthesize_end(token_count=20, duration_s=0.1)
        trace.query_end(iterations=1, total_steps=1, completed_steps=1, failed_steps=0)
        path = tmp_path / "show-q.trace.jsonl"
        trace.save(path)

        result = runner.invoke(
            app,
            ["trace", "diagnose", "--path", str(path), "--show-queries"],
        )
        assert result.exit_code == 0
        assert "Per-Query Diagnostics" in result.stdout

    def test_trace_diagnose_session_id_not_found(self, tmp_path, monkeypatch):
        traces = tmp_path / "traces"
        traces.mkdir()
        monkeypatch.setattr("agent.trace.TraceLogger.traces_dir", staticmethod(lambda: traces))
        result = runner.invoke(app, ["trace", "diagnose", "--session-id", "missing-session"])
        assert result.exit_code == 2
        assert "Trace file not found" in result.stdout

    def test_trace_diagnose_both_path_and_session_exits(self, tmp_path):
        path = tmp_path / "x.trace.jsonl"
        path.write_text("{}\n")
        result = runner.invoke(
            app,
            ["trace", "diagnose", "--path", str(path), "--session-id", "abc"],
        )
        assert result.exit_code == 2
        assert "not both" in result.stdout.lower()

    def test_trace_export_with_zip_and_report(self, tmp_path):
        trace = TraceLogger("zip-trace")
        trace.query_start("export me")
        trace.query_end(iterations=1, total_steps=0, completed_steps=0, failed_steps=0)
        trace_path = tmp_path / "zip-trace.trace.jsonl"
        trace.save(trace_path)
        report = tmp_path / "report.md"
        report.write_text("# Report")
        out_dir = tmp_path / "exports"
        cfg = Config(data={"sandbox.output_dir": str(tmp_path)})

        with patch("agent.config.Config.load", return_value=cfg), patch(
            "cli.shutil.make_archive", return_value=str(out_dir / "bundle.zip")
        ) as mock_zip:
            result = runner.invoke(
                app,
                [
                    "trace",
                    "export",
                    "--path",
                    str(trace_path),
                    "--report",
                    str(report),
                    "--out-dir",
                    str(out_dir),
                ],
            )

        assert result.exit_code == 0
        assert "Bundle exported:" in result.stdout
        assert "Included report:" in result.stdout
        assert "Zip archive:" in result.stdout
        mock_zip.assert_called_once()

    def test_trace_export_no_report_message(self, tmp_path):
        trace = TraceLogger("no-report")
        trace.query_start("q")
        trace.query_end(iterations=1, total_steps=0, completed_steps=0, failed_steps=0)
        trace_path = tmp_path / "no-report.trace.jsonl"
        trace.save(trace_path)
        out_dir = tmp_path / "exports"
        cfg = Config(data={"sandbox.output_dir": str(tmp_path / "missing_reports")})

        with patch("agent.config.Config.load", return_value=cfg):
            result = runner.invoke(
                app,
                ["trace", "export", "--path", str(trace_path), "--out-dir", str(out_dir), "--no-zip"],
            )

        assert result.exit_code == 0
        assert "No report included" in result.stdout


class TestReportNotebook:
    def test_report_notebook_html_fallback(self, tmp_path):
        trace_path = tmp_path / "t.trace.jsonl"
        trace_path.write_text("{}\n")
        cfg = Config(data={"sandbox.output_dir": str(tmp_path)})
        fake_nb = MagicMock()
        fake_nb.cells = [MagicMock(cell_type="markdown", source="# Title")]

        import builtins

        real_import = builtins.__import__

        def import_without_nbconvert(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "nbconvert":
                raise ImportError("no nbconvert")
            return real_import(name, globals, locals, fromlist, level)

        with patch("agent.config.Config.load", return_value=cfg), patch(
            "agent.trace_store.TraceStore.find_trace", return_value=trace_path
        ), patch("reports.notebook.trace_to_notebook", return_value=fake_nb), patch(
            "reports.notebook.save_notebook", return_value=tmp_path / "out.ipynb"
        ), patch("builtins.__import__", side_effect=import_without_nbconvert), patch(
            "reports.html.render_html_report", return_value="<html>ok</html>"
        ):
            result = runner.invoke(app, ["report", "notebook", "--html"])

        assert result.exit_code == 0
        assert (tmp_path / "out.html").exists()
        assert "markdown only" in result.stdout.lower()

    def test_report_notebook_nbformat_missing(self, tmp_path):
        trace_path = tmp_path / "t.trace.jsonl"
        trace_path.write_text("{}\n")

        import builtins

        real_import = builtins.__import__

        def import_without_nbformat(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "reports.notebook":
                raise ImportError("no nbformat")
            return real_import(name, globals, locals, fromlist, level)

        with patch("agent.trace_store.TraceStore.find_trace", return_value=trace_path), patch(
            "builtins.__import__", side_effect=import_without_nbformat
        ):
            result = runner.invoke(app, ["report", "notebook"])

        assert result.exit_code == 2
        assert "nbformat is required" in result.stdout


class TestUpgradeAndConfig:
    def test_upgrade_cmd_failure_exits_one(self):
        cfg = Config(data={})
        with patch("agent.config.Config.load", return_value=cfg), patch(
            "cli.execute_upgrade", return_value=False
        ):
            result = runner.invoke(app, ["upgrade"])
        assert result.exit_code == 1

    def test_config_get(self, monkeypatch):
        cfg = Config(data={"llm.provider": "anthropic", "llm.model": "claude-sonnet"})
        monkeypatch.setattr("agent.config.Config.load", lambda: cfg)
        result = runner.invoke(app, ["config", "get", "llm.provider"])
        assert result.exit_code == 0
        assert "anthropic" in result.stdout

    def test_config_show(self, monkeypatch):
        cfg = Config(data={"llm.provider": "openai"})
        monkeypatch.setattr("agent.config.Config.load", lambda: cfg)
        with patch.object(cfg, "to_table", return_value="llm.provider = openai"):
            result = runner.invoke(app, ["config", "show"])
        assert result.exit_code == 0
        assert "openai" in result.stdout

    def test_config_validate_clean(self, monkeypatch):
        cfg = MagicMock()
        cfg.validate.return_value = []
        monkeypatch.setattr("agent.config.Config.load", lambda: cfg)
        result = runner.invoke(app, ["config", "validate"])
        assert result.exit_code == 0
        assert "valid" in result.stdout.lower()

    def test_config_validate_with_issues(self, monkeypatch):
        cfg = MagicMock()
        cfg.validate.return_value = ["Missing llm.api_key", "Invalid model"]
        monkeypatch.setattr("agent.config.Config.load", lambda: cfg)
        result = runner.invoke(app, ["config", "validate"])
        assert result.exit_code == 2
        assert "2 issue" in result.stdout


class TestCaseStudyCommands:
    def test_case_study_list(self):
        result = runner.invoke(app, ["case-study", "list"])
        assert result.exit_code == 0
        assert "Case Studies" in result.stdout

    def test_case_study_unknown_id(self):
        result = runner.invoke(app, ["case-study", "run", "not-a-case"])
        assert result.exit_code == 2
        assert "Unknown case study" in result.stdout

    @patch("reports.html.publish_report")
    @patch("agent.case_studies.run_case_study")
    @patch("cli.Session")
    @patch("cli.setup_cmd")
    def test_case_study_run_success(self, mock_setup, mock_session_cls, mock_run, mock_publish, tmp_path):
        cfg = Config(data={"llm.api_key": "sk-test", "sandbox.output_dir": str(tmp_path)})
        mock_session_cls.return_value = MagicMock()
        mock_run.return_value = MagicMock(summary="done", duration_s=1.0)
        reports = tmp_path / "reports"
        reports.mkdir()
        (reports / "case.md").write_text("# Case")

        with patch("agent.config.Config.load", return_value=cfg), patch.object(
            cfg, "llm_preflight_issue", return_value=None
        ), patch("cli.print_banner"), patch(
            "cli._latest_report_path", return_value=reports / "case.md"
        ):
            mock_publish.return_value = reports / "case.html"
            result = runner.invoke(app, ["case-study", "run", "revlimid", "--threads", "2"])

        assert result.exit_code == 0
        mock_run.assert_called_once()
        mock_publish.assert_called_once()


class TestRunCommand:
    def test_run_version_flag(self):
        result = runner.invoke(app, ["run", "--version"])
        assert result.exit_code == 0
        assert "fastfold v" in result.stdout

    @patch("cli.run_query")
    def test_run_query_with_context_flags(self, mock_run_query):
        result = runner.invoke(
            app,
            [
                "run",
                "--smiles",
                "CCO",
                "--target",
                "EGFR",
                "--indication",
                "NSCLC",
                "Profile compound",
            ],
        )
        assert result.exit_code == 0
        mock_run_query.assert_called_once()
        context = mock_run_query.call_args[0][1]
        assert context["compound_smiles"] == "CCO"
        assert context["target"] == "EGFR"
        assert context["indication"] == "NSCLC"

    @patch("cli.run_interactive")
    def test_run_continue_flag(self, mock_interactive):
        result = runner.invoke(app, ["run", "--continue"])
        assert result.exit_code == 0
        mock_interactive.assert_called_once()
        assert mock_interactive.call_args.kwargs["resume_id"] == "last"

    @patch("cli.run_interactive")
    def test_run_resume_flag(self, mock_interactive):
        result = runner.invoke(app, ["run", "--resume", "abc123"])
        assert result.exit_code == 0
        assert mock_interactive.call_args.kwargs["resume_id"] == "abc123"

    def test_run_query_legacy_clarification(self, captured_console, monkeypatch):
        from agent.loop import ClarificationNeeded

        console, buf = captured_console
        monkeypatch.setattr("cli.console", console)
        cfg = Config(data={"llm.api_key": "sk-ant-test", "agent.use_sdk": False})
        monkeypatch.setattr("agent.config.Config.load", lambda: cfg)

        clarification = MagicMock(question="Which target?", suggestions=["EGFR", "TP53"])
        fake_agent = MagicMock()
        fake_agent.run.side_effect = ClarificationNeeded(clarification)

        with patch("cli.print_banner"), patch("cli.setup_cmd"), patch.object(
            cfg, "llm_preflight_issue", return_value=None
        ), patch("agent.loop.AgentLoop", return_value=fake_agent):
            run_query("ambiguous query", {}, None, None, False)

        assert "Which target?" in buf.getvalue()

    def test_run_query_writes_output_file(self, captured_console, monkeypatch, tmp_path):
        console, _ = captured_console
        monkeypatch.setattr("cli.console", console)
        cfg = Config(data={"llm.api_key": "sk-ant-test", "agent.use_sdk": True})
        monkeypatch.setattr("agent.config.Config.load", lambda: cfg)

        fake_result = MagicMock()
        fake_result.to_markdown.return_value = "# Saved report"
        fake_agent = MagicMock()
        fake_agent.run.return_value = fake_result

        with patch("cli.print_banner"), patch.object(
            cfg, "llm_preflight_issue", return_value=None
        ), patch("agent.runner.AgentRunner", return_value=fake_agent):
            run_query("analyze TP53", {}, tmp_path / "out", None, False)

        assert (tmp_path / "out" / "report.md").read_text() == "# Saved report"


class TestPromptInstallSkillsAndSuggestedSources:
    def test_prompt_install_skills_with_skills_arg(self, monkeypatch):
        called = []

        def _install(sources):
            called.extend(sources)

        monkeypatch.setattr("cli._install_skill_sources", _install)
        _prompt_install_skills(skills_arg="fold,fastfold-ai/skills@skills/md")
        assert called == ["fold", "fastfold-ai/skills@skills/md"]

    def test_prompt_install_skills_skip(self):
        called = {"ran": False}

        def _install(sources):
            called["ran"] = True

        with patch("cli._install_skill_sources", _install):
            _prompt_install_skills(skip=True)
        assert called["ran"] is False

    def test_select_suggested_sources_empty_catalog(self, monkeypatch):
        monkeypatch.setattr("agent.skills.SUGGESTED_SKILL_SOURCES", [])
        assert _select_suggested_sources() == []

    def test_select_suggested_sources_inline_all(self, monkeypatch):
        monkeypatch.setattr("cli.sys.stdin.isatty", lambda: False)
        monkeypatch.setattr("cli.sys.stdout.isatty", lambda: False)
        monkeypatch.setattr("builtins.input", lambda _: "all")
        sources = _select_suggested_sources()
        assert sources
        assert any("anthropics" in s or "google-deepmind" in s or "K-Dense" in s for s in sources)

    def test_select_suggested_sources_inline_number(self, monkeypatch):
        monkeypatch.setattr("cli.sys.stdin.isatty", lambda: False)
        monkeypatch.setattr("cli.sys.stdout.isatty", lambda: False)
        monkeypatch.setattr("builtins.input", lambda _: "1")
        sources = _select_suggested_sources()
        assert len(sources) == 1

    def test_select_suggested_sources_enter_skips(self, monkeypatch):
        monkeypatch.setattr("cli.sys.stdin.isatty", lambda: False)
        monkeypatch.setattr("cli.sys.stdout.isatty", lambda: False)
        monkeypatch.setattr("builtins.input", lambda _: "")
        assert _select_suggested_sources() == []


class TestResolveProviderKeyKeepExisting:
    def test_resolve_provider_key_keeps_valid_existing_key(self, captured_console, monkeypatch):
        console, buf = captured_console
        monkeypatch.setattr("cli.console", console)
        cfg = Config(data={"llm.anthropic_api_key": "sk-ant-api03-existing-key-value"})
        monkeypatch.setattr("builtins.input", lambda _: "")
        key = _resolve_provider_key(cfg, "anthropic")
        assert key == "sk-ant-api03-existing-key-value"
        assert "Keeping existing key" in buf.getvalue()

    def test_resolve_provider_key_replaces_invalid_existing_key(self, captured_console, monkeypatch):
        console, _ = captured_console
        monkeypatch.setattr("cli.console", console)
        cfg = Config(data={"llm.anthropic_api_key": "bad-key"})
        monkeypatch.setattr("cli._prompt_api_key", lambda: "sk-ant-api03-replacement-key")
        key = _resolve_provider_key(cfg, "anthropic")
        assert key == "sk-ant-api03-replacement-key"


class TestMarkupAndVersionHelpers:
    def test_random_command_tip_empty_commands(self, monkeypatch):
        monkeypatch.setattr("cli.SLASH_COMMANDS", {})
        tip = _random_command_tip_markup()
        assert "slash commands" in tip.lower()

    def test_random_command_tip_from_commands(self, monkeypatch):
        monkeypatch.setattr("cli.SLASH_COMMANDS", {"/help": "Show help"})
        tip = _random_command_tip_markup()
        assert "/help" in tip

    def test_random_news_item_without_boltzgen(self, monkeypatch):
        monkeypatch.setattr("cli._installed_claude_skill_names", lambda: ["fold"])
        news = _random_news_item_markup()
        assert "BoltzGen" in news
        assert "!" not in news.split("BoltzGen")[1][:20]

    def test_random_news_item_with_boltzgen_installed(self, monkeypatch):
        monkeypatch.setattr(
            "cli._installed_claude_skill_names",
            lambda: ["fold", "protein_design_boltzgen"],
        )
        news = _random_news_item_markup()
        assert "BoltzGen universal protein design now available!" in news

    def test_resolve_upgrade_flavor_uses_configured_value(self, tmp_path, monkeypatch):
        cfg = Config(data={"install.uv_flavor": "win_build"})
        flavor = resolve_upgrade_flavor(cfg=cfg, persist=False)
        assert flavor == "win_build"

    def test_resolve_upgrade_flavor_persists_fallback(self, tmp_path, monkeypatch):
        cfg = Config(data={})
        monkeypatch.setattr("cli.os.name", "posix", raising=False)
        with patch.object(cfg, "save") as mock_save:
            flavor = resolve_upgrade_flavor(cfg=cfg, persist=True)
        assert flavor == "all"
        mock_save.assert_called_once()

    def test_is_newer_version_invalid_inputs(self):
        assert is_newer_version("not-a-version", "0.0.1") is False
        assert is_newer_version("1.0.0", "bad") is False


class TestDirectSkillFunctionCalls:
    @patch("agent.skills.upgrade_skills")
    def test_skill_upgrade_direct_no_catalog(self, mock_upgrade):
        mock_upgrade.return_value = {
            "added": [],
            "updated": ["fold"],
            "npx_synced": 0,
            "failed": [],
            "summary": "done",
        }
        result = runner.invoke(app, ["skills", "upgrade", "--no-catalog"])
        assert result.exit_code == 0
        mock_upgrade.assert_called_once_with(include_catalog=False, include_npx=True)

    @patch("agent.skills.remove_skill")
    def test_skill_remove_direct(self, mock_remove):
        mock_remove.return_value = {"ok": True, "summary": "removed"}
        skill_remove("fold")
        mock_remove.assert_called_once_with("fold")

    @patch("agent.skills.skill_info")
    def test_skill_info_cmd_direct_not_installed(self, mock_info):
        mock_info.return_value = None
        with pytest.raises(typer.Exit) as exc:
            skill_info_cmd("missing")
        assert exc.value.exit_code == 1
