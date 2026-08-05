"""Tests for TaskExecutor provider configuration helpers.

Updates:
    v0.1 - 2025-11-07 - Added coverage for Ollama base URL resolution.
    v0.2 - 2026-05-11 - Covered workflow selection, provider kwargs, and execution retry paths.
"""

from __future__ import annotations

import builtins
import logging
from io import StringIO
from types import SimpleNamespace
from typing import Any

import pytest

from config import settings
from core.exceptions import WorkflowError
from core.task_executor import TaskExecutor, _detect_windows_host_ip, _is_ipv4
from models.workflows import TaskRequest


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
    monkeypatch.setattr(
        "core.task_executor._detect_windows_host_ip", lambda: "172.31.52.230"
    )
    base_url = task_executor._resolve_ollama_base_url()
    assert base_url == "http://172.31.52.230:11434"


def test_resolve_ollama_base_url_falls_back_to_localhost(
    task_executor: TaskExecutor, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.setattr("core.task_executor._detect_windows_host_ip", lambda: None)
    base_url = task_executor._resolve_ollama_base_url()
    assert base_url == TaskExecutor.DEFAULT_OLLAMA_BASE


def test_select_workflow_falls_back_for_unknown_request(
    task_executor: TaskExecutor,
) -> None:
    selection = task_executor.select_workflow("missing")

    assert selection.workflow == "local"
    assert selection.rationale == "Fallback to default configuration"


def test_select_workflow_applies_controller_bias(task_executor: TaskExecutor) -> None:
    class Controller:
        @property
        def workflow_biases(self) -> dict[str, float]:
            return {"local": 0.1, "fast": 0.3}

    task_executor._config.llm.workflows["fast"] = settings.WorkflowModelConfig(
        provider="ollama",
        model="fast-model",
        temperature=0.1,
    )
    task_executor._controller = Controller()  # type: ignore[assignment]

    selection = task_executor.select_workflow()

    assert selection.workflow == "fast"
    assert selection.metadata["biases"] == {"local": 0.1, "fast": 0.3}


def test_apply_controller_bias_retains_current_choice(
    task_executor: TaskExecutor,
) -> None:
    choice, reason = task_executor._apply_controller_bias(
        "local",
        {"local": object()},
        {"local": 0.2},
    )

    assert choice == "local"
    assert reason == "Controller bias retained (0.20)"


def test_provider_kwargs_require_azure_credentials(
    task_executor: TaskExecutor, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)

    with pytest.raises(WorkflowError, match="credentials missing"):
        task_executor._build_provider_kwargs("azure")


def test_provider_kwargs_redact_azure_credentials(
    task_executor: TaskExecutor, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "secret")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2025-01-01")

    kwargs = task_executor._build_provider_kwargs("azure")
    redacted = task_executor._redact_sensitive(kwargs)

    assert kwargs["api_base"] == "https://example.openai.azure.com"
    assert kwargs["custom_llm_provider"] == "azure"
    assert redacted["api_key"] == "***redacted***"


def test_resolve_model_name_adds_provider_prefix(task_executor: TaskExecutor) -> None:
    azure_cfg = settings.WorkflowModelConfig(
        provider="azure",
        model="gpt-4.1",
        temperature=0.2,
    )
    ollama_cfg = settings.WorkflowModelConfig(
        provider="ollama",
        model="gemma3:1b",
        temperature=0.2,
    )

    assert task_executor._resolve_model_name(azure_cfg) == "azure/gpt-4.1"
    assert task_executor._resolve_model_name(ollama_cfg) == "ollama/gemma3:1b"
    assert (
        task_executor._resolve_model_name(
            settings.WorkflowModelConfig(
                provider="custom",
                model="custom-model",
                temperature=0.2,
            )
        )
        == "custom-model"
    )


def test_execute_returns_result_with_redacted_metadata(
    task_executor: TaskExecutor, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def _completion(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"total_tokens": 3},
        }

    dummy_litellm = SimpleNamespace(
        completion=_completion,
        Timeout=TimeoutError,
    )
    monkeypatch.setattr("core.task_executor.litellm", dummy_litellm)
    monkeypatch.setattr("core.task_executor.time.sleep", lambda _: None)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")

    request = TaskRequest(
        workflow="local",
        prompt="hello",
        context={"system": "system prompt"},
    )
    result = task_executor.execute(request)

    assert result.content == "ok"
    assert captured["model"] == "ollama/gemma3:1b"
    assert result.metadata["provider"] == "ollama"


def test_execute_retries_timeouts_then_raises(
    task_executor: TaskExecutor,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls = {"count": 0}
    marker = "provider-timeout-prompt-marker"

    class DummyTimeout(Exception):
        """Placeholder timeout exception."""

    def _completion(**_: object) -> dict[str, object]:
        calls["count"] += 1
        raise DummyTimeout(marker)

    dummy_litellm = SimpleNamespace(
        completion=_completion,
        Timeout=DummyTimeout,
    )
    monkeypatch.setattr("core.task_executor.litellm", dummy_litellm)
    monkeypatch.setattr("core.task_executor.time.sleep", lambda _: None)

    request = TaskRequest(workflow="local", prompt="hello")
    caplog.set_level(logging.WARNING, logger="drm.executor")
    with pytest.raises(WorkflowError, match="failed after 2 attempts") as error:
        task_executor.execute(request)

    assert calls["count"] == 2
    assert marker not in caplog.text
    assert marker not in str(error.value)
    assert "error_type=DummyTimeout" in caplog.text
    assert error.value.__cause__ is None


def test_execute_redacts_provider_failure_details(
    task_executor: TaskExecutor,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    marker = "provider-error-result-and-secret-marker"

    class ProviderFailure(Exception):
        """Synthetic provider exception carrying sensitive-looking detail."""

    def _completion(**_: object) -> dict[str, object]:
        raise ProviderFailure(marker)

    monkeypatch.setattr(
        "core.task_executor.litellm",
        SimpleNamespace(completion=_completion, Timeout=TimeoutError),
    )
    caplog.set_level(logging.ERROR, logger="drm.executor")

    with pytest.raises(WorkflowError, match="failed after 1 attempts") as error:
        task_executor.execute(TaskRequest(workflow="local", prompt="hello"))

    assert marker not in caplog.text
    assert marker not in str(error.value)
    assert "error_type=ProviderFailure" in caplog.text
    assert error.value.__cause__ is None


def test_execute_requires_configured_workflow(
    task_executor: TaskExecutor, monkeypatch: pytest.MonkeyPatch
) -> None:
    dummy_litellm = SimpleNamespace(completion=lambda **_: {}, Timeout=TimeoutError)
    monkeypatch.setattr("core.task_executor.litellm", dummy_litellm)

    with pytest.raises(WorkflowError, match="not configured"):
        task_executor.execute(TaskRequest(workflow="missing", prompt="hello"))


def test_execute_requires_litellm(
    task_executor: TaskExecutor, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("core.task_executor.litellm", None)

    with pytest.raises(WorkflowError, match="liteLLM is not installed"):
        task_executor.execute(TaskRequest(workflow="local", prompt="hello"))


def test_litellm_debug_hook_is_called(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"debug": False}

    def _turn_on_debug() -> None:
        called["debug"] = True

    config = task_config(enable_debug=True)
    monkeypatch.setattr(
        "core.task_executor.litellm",
        SimpleNamespace(_turn_on_debug=_turn_on_debug),
    )

    TaskExecutor(config)

    assert called["debug"]


def test_detect_windows_host_ip(monkeypatch: pytest.MonkeyPatch) -> None:
    _detect_windows_host_ip.cache_clear()

    def _open(path: str, *args: Any, **kwargs: Any) -> StringIO:
        assert path == "/proc/version"
        return StringIO("Linux version microsoft-standard-WSL2")

    monkeypatch.setattr(builtins, "open", _open)
    monkeypatch.setattr(
        "core.task_executor.subprocess.run",
        lambda *_, **__: SimpleNamespace(
            returncode=0,
            stdout="ignored line\ndefault via 172.20.80.1 dev eth0\n",
        ),
    )

    assert _detect_windows_host_ip() == "172.20.80.1"
    assert _is_ipv4("172.20.80.1")
    assert not _is_ipv4("not-an-ip")
    _detect_windows_host_ip.cache_clear()


def task_config(*, enable_debug: bool = False) -> settings.AppConfig:
    return settings.AppConfig.model_validate(
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
                "enable_debug": enable_debug,
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
            "review": {
                "enabled": False,
                "auto_reviewer_model": None,
                "auto_reviewer_provider": None,
            },
            "embedding": None,
            "telemetry": {"log_level": "INFO"},
        }
    )
