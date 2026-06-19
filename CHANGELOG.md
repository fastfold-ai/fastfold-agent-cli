# Changelog

All notable changes to `fastfold-agent-cli` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Changelog tracking started for `fastfold-agent-cli`.
- Terminal Mermaid diagram rendering via `termaid` for assistant responses containing
  fenced `mermaid` blocks.

### Changed
- Markdown rendering now detects Mermaid fences in trace/live/resumed output paths
  and falls back to source fences when rendering is unavailable.

## [0.0.53] - 2026-06-18

### Notes
- Baseline release when changelog tracking was introduced.
