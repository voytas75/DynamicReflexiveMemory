# Changelog

All notable changes to this project will be documented in this file. The format follows [Keep a Changelog](https://keepachangelog.com/) and dates use ISO-8601 (`YYYY-MM-DD`).

## [Unreleased]

### Changed
- Added an observational CodeQL workflow for Python source scanning on pushes and pull requests.
- Added isolated real Redis and ChromaDB acceptance coverage: Redis receives a dynamic loopback port and ChromaDB persists through a temporary local directory with a deterministic embedding function.
- Bound retrieved-memory prompt context to 6,000 characters and label it as untrusted reference data; the direct task instruction remains separate.
- Malformed, missing, or unavailable automated-review verdicts now produce `unverified` instead of a passing review; disabled review remains explicitly `skipped`.
- Documented and enabled `uv`-based local setup/workflow commands in README files and CI dependency installation.
- Bound Redis and Ollama Docker ports to loopback and pinned Ollama to `0.32.5`.
- Redacted CLI output and memory revision logs by default; raw revision payloads now require an explicit local opt-in.
- Pruned memory revision-log records older than 30 days during application initialization while preserving a valid hash chain for retained entries.
- Added a committed uv lockfile and frozen dependency synchronization for CI and developer quality gates.
- Updated CI actions to verified Node 24 runtimes and removed obsolete Compose metadata.
- Made CLI workflow failures return non-zero status and added the explicit `--show-result` output opt-in.
- Replaced mypy and Black with Ruff plus strict Pyright CI gates while retaining pytest for local regression checks.

## [0.1.9] - 2026-05-11

### Added
- Restored pinned runtime dependencies in `requirements.txt` and added pinned runtime/dev dependency metadata in `pyproject.toml`.
- Added coverage configuration and focused tests for executor, review, live-loop, telemetry, memory mitigation, and configuration helpers.

### Changed
- CI now installs `.[dev]` and enforces pytest coverage, mypy, Ruff, and Black checks.
- Formatted the codebase with Black and fixed Ruff/mypy issues in core and GUI modules.
- Updated developer documentation to match the implemented quality gate and mypy-based type checking workflow.

## [0.1.8] - 2025-11-08

### Added
- Controller drift analytics are now persisted to the Chroma analytics layer and the JSONL revision log for retrospective queries.
- PySide6 GUI gains a Drift Trends tab visualising latency averages, advisory counts, and latest workflow bias snapshots.

### Changed
- Memory metrics snapshots now include analytics record counts to feed telemetry consumers.

### Documentation
- Updated `README.md` and `drmblueprint.md` to describe the drift analytics pipeline and trend visualisation.

## [0.1.7] - 2025-11-08

### Added
- PySide6 telemetry dashboard streaming live memory metrics, drift advisories, and review history panes.
- In-process telemetry feed emitting controller drift signals and review records for GUI consumption.

### Changed
- Memory manager now snapshots and publishes memory metrics after mutations to keep the GUI in sync.

## [0.1.6] - 2025-11-08

### Added
- Docker-backed Redis integration tests validating TTL expiry, service reconnection, and fallback behaviour.

### Changed
- Hardened Redis working-memory store with automatic reconnection and graceful fallback to in-memory storage when the service is unavailable.

## [0.1.5] - 2025-11-07

### Changed
- Bumped PySide6 runtime dependency to 6.8.2 for latest Qt fixes and stability improvements.

## [0.1.4] - 2025-11-07

### Added
- Automatic detection of the Windows host IP for Ollama when running inside WSL2, removing the need for manual configuration in common setups.
- Regression tests covering Ollama base URL resolution logic.

### Documentation
- Documented the WSL Ollama workflow and manual override steps in `README.md`.

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

## [0.1.3] - 2025-11-07

### Added
- Persisted user settings (workflow preference and window geometry) in `data/user_settings.json`.
- CLI honours the last selected workflow when no override is provided.
- GUI restores the last workflow selection and window size on startup.
- GUI settings dialog for editing the active `config.json` without leaving the application.

### Changed
- Startup health checks now warn (rather than exit) when Redis or ChromaDB are unavailable, enabling seamless in-memory fallbacks.
- Automated review automatically adjusts sampling temperature for OpenAI O-series models to satisfy API constraints.
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
