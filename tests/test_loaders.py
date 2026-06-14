"""Tests for data.loaders path resolution and tabular loading."""

import json

import pandas as pd
import pytest

from agent.config import Config
import data.loaders as loaders


@pytest.fixture(autouse=True)
def clear_loader_caches():
    loaders.load_crispr.cache_clear()
    loaders.load_model_metadata.cache_clear()
    loaders.load_proteomics.cache_clear()
    loaders.load_l1000.cache_clear()
    loaders.load_prism.cache_clear()
    yield
    loaders.load_crispr.cache_clear()
    loaders.load_model_metadata.cache_clear()
    loaders.load_proteomics.cache_clear()
    loaders.load_l1000.cache_clear()
    loaders.load_prism.cache_clear()


@pytest.fixture
def mock_config(monkeypatch, tmp_path):
    cfg = Config(data={"data.base": str(tmp_path)})
    monkeypatch.setattr(
        loaders.Config,
        "load",
        classmethod(lambda cls: cfg),
    )
    monkeypatch.setattr(loaders, "_DATA_SEARCH_PATHS", [])
    return cfg, tmp_path


def test_data_path_uses_explicit_key(mock_config, tmp_path):
    cfg, _ = mock_config
    cfg.set("data.crispr", str(tmp_path / "custom.csv"))
    assert loaders._data_path("crispr") == tmp_path / "custom.csv"


def test_data_path_uses_base_dir(mock_config, tmp_path):
    assert loaders._data_path("depmap") == tmp_path / "depmap"


def test_find_file_exact_and_parquet_variant(mock_config, tmp_path):
    parquet = tmp_path / "CRISPRGeneEffect.parquet"
    pd.DataFrame({"A": [1]}).to_parquet(parquet)
    found = loaders._find_file("CRISPRGeneEffect.csv")
    assert found == parquet


def test_resolve_path_file_and_directory(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a\n1\n")
    assert loaders._resolve_path(csv_path, ["other.csv"]) == csv_path

    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "target.csv").write_text("x\n1\n")
    assert loaders._resolve_path(subdir, ["target.csv"]) == subdir / "target.csv"
    assert loaders._resolve_path(subdir, ["missing.csv"]) is None


def test_read_tabular_csv_and_parquet(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("gene,value\nTP53,1.0\n")
    df_csv = loaders._read_tabular(csv_path)
    assert list(df_csv.columns) == ["gene", "value"]

    parquet_path = tmp_path / "sample.parquet"
    pd.DataFrame({"gene": ["BRCA1"], "value": [2.0]}).to_parquet(parquet_path)
    df_parquet = loaders._read_tabular(parquet_path)
    assert df_parquet.iloc[0]["gene"] == "BRCA1"


def test_load_crispr_strips_gene_suffixes(mock_config, tmp_path, monkeypatch):
    csv_path = tmp_path / "CRISPRGeneEffect.csv"
    pd.DataFrame(
        {"TP53 (7157)": [0.2], "BRCA1 (672)": [0.1]},
        index=["ACH-000001"],
    ).to_csv(csv_path)
    monkeypatch.setattr(loaders, "_find_file", lambda name, subdirs=None: csv_path)
    df = loaders.load_crispr()
    assert "TP53" in df.columns
    assert "TP53 (7157)" not in df.columns


def test_load_crispr_missing_raises(mock_config, monkeypatch):
    monkeypatch.setattr(loaders, "_find_file", lambda name, subdirs=None: None)
    with pytest.raises(FileNotFoundError, match="DepMap CRISPR"):
        loaders.load_crispr()


def test_load_mutations_filters_defaults(mock_config, tmp_path, monkeypatch):
    csv_path = tmp_path / "OmicsSomaticMutationsMatrixDamaging.csv"
    pd.DataFrame({
        "ModelID": ["ACH-000001", "ACH-000002"],
        "IsDefaultEntryForModel": ["Yes", "No"],
        "TP53 (7157)": [1, 0],
    }).to_csv(csv_path, index=False)
    monkeypatch.setattr(loaders, "_find_file", lambda name, subdirs=None: csv_path)
    df = loaders.load_mutations()
    assert len(df) == 1
    assert "TP53" in df.columns


def test_load_msigdb_reads_json(mock_config, tmp_path, monkeypatch):
    msig_path = tmp_path / "h.all.v2024.1.Hs.json"
    payload = {"HALLMARK_APOPTOSIS": ["BAX", "BCL2"]}
    msig_path.write_text(json.dumps(payload))
    monkeypatch.setattr(loaders, "_find_file", lambda pattern, subdirs=None: msig_path)
    result = loaders.load_msigdb("h")
    assert result["HALLMARK_APOPTOSIS"] == ["BAX", "BCL2"]


def test_load_prism_from_explicit_path(mock_config, tmp_path):
    csv_path = tmp_path / "prism_LFC_COLLAPSED.csv"
    pd.DataFrame({"compound": ["A"], "value": [1.0]}).to_csv(csv_path, index=False)
    mock_config[0].set("data.prism", str(csv_path))
    df = loaders.load_prism()
    assert df.iloc[0]["compound"] == "A"
