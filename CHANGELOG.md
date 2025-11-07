# Changelog

All notable changes to this project will be documented in this file. The format follows [Keep a Changelog](https://keepachangelog.com/) and dates use ISO-8601 (`YYYY-MM-DD`).

## [0.1.1] - 2025-11-07

### Added
- Automatic loading of environment variables from `.env` via `python-dotenv` at application startup.
- Tracked `config/config.example.json` template for bootstrapping new environments.

### Changed
- Normalised LiteLLM Ollama routing by prefixing model identifiers and supplying explicit provider hints.

### Dependencies
- Added `python-dotenv` to runtime requirements.
