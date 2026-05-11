"""Tests for configuration loading, validation, and persistence.

Updates:
    v0.1 - 2025-11-06 - Added baseline configuration loader test.
    v0.2 - 2026-05-11 - Added failure-path and save coverage for config helpers.
"""

from __future__ import annotations

from pathlib import Path

import json

from config import settings
from core.exceptions import ConfigError

import pytest


def test_load_app_config(tmp_path: Path) -> None:
    sample_config = {
        "version": "0.1",
        "llm": {
            "default_workflow": "fast",
            "workflows": {
                "fast": {"provider": "azure", "model": "gpt-4.1", "temperature": 0.1}
            },
            "timeouts": {
                "request_seconds": 10,
                "retry_attempts": 1,
                "retry_backoff_seconds": 1,
            },
        },
        "memory": {
            "redis": {"host": "localhost", "port": 6379, "db": 0, "ttl_seconds": 120},
            "chromadb": {"persist_directory": "data/chromadb", "collection": "test"},
        },
        "review": {"enabled": True, "auto_reviewer_model": None},
        "embedding": {"provider": "azure", "model": "text-embedding-3-large"},
        "telemetry": {"log_level": "DEBUG"},
    }

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(sample_config), encoding="utf-8")

    loaded = settings.load_app_config(config_path)
    assert loaded.version == "0.1"
    assert loaded.llm.default_workflow == "fast"
    assert loaded.memory.redis.ttl_seconds == 120


def test_load_app_config_reports_missing_and_invalid_files(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Configuration file not found"):
        settings.load_app_config(tmp_path / "missing.json")

    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{", encoding="utf-8")
    with pytest.raises(ConfigError, match="Invalid JSON"):
        settings.load_app_config(invalid_json)

    invalid_payload = tmp_path / "invalid-payload.json"
    invalid_payload.write_text(json.dumps({"version": "0.1"}), encoding="utf-8")
    with pytest.raises(ConfigError, match="validation failed"):
        settings.load_app_config(invalid_payload)


def test_save_app_config_round_trips(tmp_path: Path) -> None:
    config_path = tmp_path / "saved" / "config.json"
    config = settings.AppConfig.model_validate(
        {
            "version": "0.1",
            "llm": {
                "default_workflow": "fast",
                "workflows": {
                    "fast": {
                        "provider": "azure",
                        "model": "gpt-4.1",
                        "temperature": 0.1,
                    }
                },
                "timeouts": {
                    "request_seconds": 10,
                    "retry_attempts": 1,
                    "retry_backoff_seconds": 1,
                },
            },
            "memory": {
                "redis": {
                    "host": "localhost",
                    "port": 6379,
                    "db": 0,
                    "ttl_seconds": 120,
                },
                "chromadb": {
                    "persist_directory": "data/chromadb",
                    "collection": "test",
                },
            },
            "review": {"enabled": True, "auto_reviewer_model": None},
            "embedding": {"provider": "azure", "model": "text-embedding-3-large"},
            "telemetry": {"log_level": "DEBUG"},
        }
    )

    settings.save_app_config(config, config_path)

    loaded = settings.load_app_config(config_path)
    assert loaded == config
