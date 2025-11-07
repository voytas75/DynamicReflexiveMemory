# Changelog

All notable changes to this project will be documented in this file. The format follows [Keep a Changelog](https://keepachangelog.com/) and dates use ISO-8601 (`YYYY-MM-DD`).

## [0.1.2] - 2025-11-07

### Added
- Append-only memory revision logger with configurable log path (`DRM_MEMORY_LOG_PATH`).
- Integration-oriented tests covering the live task loop, drift controller biases, and revision history.
- GUI surfacing of recent memory revisions for quick inspection.
- Semantic graph utilities that link recent concepts and expose relation context in prompts.
- GUI accepts human review feedback and manual drift mitigation triggers; CLI exposes `--feedback` to capture reviewer notes.
- CLI smoke test harness and Chroma persistence integration coverage to validate full-loop execution.

### Changed
- Controller now emits informative logs when drift triggers workflow bias updates.
- Live loop applies automated drift mitigation, pruning stale working memory and decaying semantic relations.
- Memory manager emits telemetry spans and structured metrics around key operations.

### Fixed
- Prevented GUI startup from crashing on systems without a graphical backend by falling back to CLI mode.
- Suppressed false Azure embedding deployment warnings when the deployment is configured in `config.json`.

### Documentation
- Expanded README with revision logging guidance and clarified ChromaDB deployment model.

## [0.1.1] - 2025-11-07

### Added
- Automatic loading of environment variables from `.env` via `python-dotenv` at application startup.
- Tracked `config/config.example.json` template for bootstrapping new environments.

### Changed
- Normalised LiteLLM Ollama routing by prefixing model identifiers and supplying explicit provider hints.
- Expanded executor error logging to surface underlying LiteLLM failures during task runs.
- Logged automated review failures and ensured task loop surfaces them with stack traces.

### Dependencies
- Added `python-dotenv` to runtime requirements.
