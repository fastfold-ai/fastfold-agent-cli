# Changelog

All notable changes to `fastfold-agent-cli` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.55] - 2026-06-19

### Changed
- OpenAI-compatible key prompts now allow blank input to save a profile with no
  key (clear/unset key), with `:keep` to preserve an existing key and
  `:default` to use backend defaults like `ollama`.
- `/keys` now shows backend-specific install/reference URLs in the **Sign Up**
  column for each OpenAI-compatible profile row (Ollama, Unsloth, oMLX, DS4,
  llama.cpp, LM Studio, custom).
- `/model-manager` and setup endpoint-type selectors now print install links
  next to each compatible backend option.

## [0.0.54] - 2026-06-19

### Added
- Changelog tracking started for `fastfold-agent-cli`.
- Terminal Mermaid diagram rendering via `termaid` for assistant responses containing
  fenced `mermaid` blocks.
- New `/model-manager` interactive command for OpenAI-compatible profile
  diagnostics and CRUD operations (add/edit/delete).
- New OpenAI-compatible profile templates for `ds4`, `llama_cpp`, and
  `lm_studio`, available in both `/model-manager` and `fastfold setup`.

### Changed
- Markdown rendering now detects Mermaid fences in trace/live/resumed output paths
  and falls back to source fences when rendering is unavailable.
- `/model` is now selection-only; profile creation/editing moved to
  `/model-manager`.
- `fastfold setup` now prints a profile summary (label/template/endpoint)
  before compatible model selection.
- README compatibility docs now include local/self-hosted LLM engine install
  references for DS4, llama.cpp, LM Studio, Ollama, oMLX, and Unsloth.

### Fixed
- OpenAI-compatible profile key projection no longer mixes stale legacy key
  values across profiles (e.g., Unsloth key overwritten by oMLX key).

## [0.0.53] - 2026-06-18

### Notes
- Baseline release when changelog tracking was introduced.
