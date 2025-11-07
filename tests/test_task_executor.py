"""Tests for TaskExecutor provider configuration helpers.

Updates: v0.1 - 2025-11-07 - Added coverage for Ollama base URL resolution.
"""

from __future__ import annotations

import pytest

from config import settings
from core.task_executor import TaskExecutor


@pytest.fixture()
def task_executor() -> TaskExecutor:
    """Provide a TaskExecutor configured with a minimal workflow set."""

    config = settings.AppConfig.model_validate(
        {
            "version": "0.1",
            "llm": {
                "default_workflow": "local",
                "workflows": {
                    "local": {
                        "provider": "ollama",
                        "model": "gemma3:1b",
                        "temperature": 0.2,
                    }
                },
                "timeouts": {
                    "request_seconds": 10,
                    "retry_attempts": 1,
                    "retry_backoff_seconds": 1,
                },
                "enable_debug": False,
            },
            "memory": {
                "redis": {"host": "localhost", "port": 6379, "db": 0, "ttl_seconds": 120},
                "chromadb": {
                    "persist_directory": "data/chromadb",
                    "collection": "test",
                },
            },
            "review": {
                "enabled": False,
                "auto_reviewer_model": None,
                "auto_reviewer_provider": None,
            },
            "embedding": None,
            "telemetry": {"log_level": "INFO"},
        }
    )
    return TaskExecutor(config)


def test_resolve_ollama_base_url_prefers_env_override(
    task_executor: TaskExecutor, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://172.16.0.1:9000/")
    base_url = task_executor._resolve_ollama_base_url()
    assert base_url == "http://172.16.0.1:9000"


def test_resolve_ollama_base_url_detects_wsl_host(
    task_executor: TaskExecutor, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.setattr("core.task_executor._detect_windows_host_ip", lambda: "172.31.52.230")
    base_url = task_executor._resolve_ollama_base_url()
    assert base_url == "http://172.31.52.230:11434"


def test_resolve_ollama_base_url_falls_back_to_localhost(
    task_executor: TaskExecutor, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.setattr("core.task_executor._detect_windows_host_ip", lambda: None)
    base_url = task_executor._resolve_ollama_base_url()
    assert base_url == TaskExecutor.DEFAULT_OLLAMA_BASE
