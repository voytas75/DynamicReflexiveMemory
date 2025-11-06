"""Tests for configuration loading and validation.

Updates: v0.1 - 2025-11-06 - Added baseline configuration loader test.
"""

from __future__ import annotations

from pathlib import Path

import json

from config import settings


def test_load_app_config(tmp_path: Path) -> None:
    sample_config = {
        "version": "0.1",
        "llm": {
            "default_workflow": "fast",
            "workflows": {
                "fast": {"provider": "azure", "model": "gpt-4.1", "temperature": 0.1}
            },
            "timeouts": {"request_seconds": 10, "retry_attempts": 1, "retry_backoff_seconds": 1},
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
