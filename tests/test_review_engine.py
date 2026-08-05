"""Tests for the review engine structured parsing and live loop integration.

Updates: v0.1 - 2025-11-07 - Added regression tests for automated review parsing
and LiveTaskLoop orchestration.
Updates: v0.2 - 2026-05-11 - Added review disabled, timeout, and human feedback coverage.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from config import settings
from core.exceptions import ReviewError
from core.live_loop import LiveTaskLoop
from core.review import ReviewEngine
from models.workflows import TaskRequest, TaskResult


def _entry_has_user_task(entry: Dict[str, object], expected: str) -> bool:
    metadata = entry.get("metadata")
    if isinstance(metadata, dict):
        user_task = metadata.get("user_task")
        return isinstance(user_task, str) and user_task == expected
    return False


def _load_sample_config(tmp_path: Path) -> settings.AppConfig:
    source = Path(__file__).resolve().parent.parent / "config" / "config.example.json"
    config_path = tmp_path / "config.json"
    config_path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return settings.load_app_config(config_path)


def test_parse_automated_review_structured_fields(tmp_path: Path) -> None:
    config = _load_sample_config(tmp_path)
    engine = ReviewEngine(config)
    payload = """
    VERDICT: FAIL
    REASONING: Output contradicted prior commitments.
    QUALITY_SCORE: 0.42
    SUGGESTIONS:
    - Correct the inconsistency.
    - Provide evidence for claims.
    """
    parsed = engine._parse_automated_review(payload)
    assert parsed.verdict == "FAIL"
    assert parsed.quality_score == pytest.approx(0.42)
    assert parsed.suggestions == [
        "Correct the inconsistency.",
        "Provide evidence for claims.",
    ]


def test_live_task_loop_persists_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _load_sample_config(tmp_path)
    config.review.auto_reviewer_model = "review-stub"
    config.review.auto_reviewer_provider = "ollama"
    config.llm.workflows["fast"].provider = "ollama"
    config.llm.workflows["fast"].model = "stub-fast-model"

    def _dummy_completion(
        model: str,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        system = messages[0]["content"] if messages else ""
        if "DRM audit agent" in system:
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "VERDICT: PASS\n"
                                "REASONING: Looks good.\n"
                                "QUALITY_SCORE: 0.95\n"
                                "SUGGESTIONS:\n"
                                "- Keep monitoring latency."
                            )
                        }
                    }
                ],
                "usage": {},
            }
        return {
            "choices": [{"message": {"content": "Task completed successfully."}}],
            "usage": {"total_tokens": 42},
        }

    class DummyTimeout(Exception):
        """Placeholder timeout exception."""

    dummy_litellm = SimpleNamespace(
        completion=_dummy_completion,
        Timeout=DummyTimeout,
    )
    monkeypatch.setattr("core.task_executor.litellm", dummy_litellm)
    monkeypatch.setattr("core.review.litellm", dummy_litellm)

    loop = LiveTaskLoop(config)
    outcome = loop.run_task("Draft integration plan for QA.")

    assert outcome.result.content == "Task completed successfully."
    assert outcome.review.quality_score == pytest.approx(0.95)
    assert outcome.review.suggestions == ["Keep monitoring latency."]
    assert outcome.drift_advisory is None

    episodic_entries = loop._memory_manager.list_layer("episodic")
    assert any(
        _entry_has_user_task(entry, "Draft integration plan for QA.")
        for entry in episodic_entries
    )

    review_records = loop._memory_manager.list_layer("review")
    assert review_records, "Expected persisted review records."
    stored_review = review_records[-1]
    quality_value = stored_review.get("quality_score")
    assert isinstance(quality_value, (int, float))
    assert quality_value == pytest.approx(0.95)
    suggestions_value = stored_review.get("suggestions")
    assert isinstance(suggestions_value, list)
    assert suggestions_value == ["Keep monitoring latency."]

    semantic_nodes = loop._memory_manager.list_layer("semantic")
    concept_ids = {str(node.get("id", "")) for node in semantic_nodes}
    assert any(id_.startswith("concept:") for id_ in concept_ids)

    working_items = loop._memory_manager.list_working_items()
    assert any(item.key.endswith(":result") for item in working_items)


def test_to_json_safe_serialises_usage_objects() -> None:
    payload = {
        "usage": SimpleNamespace(total_tokens=42, prompt_tokens=10),
        "sequence": [SimpleNamespace(value="a")],
        "primitive": "ok",
    }
    safe = ReviewEngine._to_json_safe(payload)
    assert safe == {
        "usage": {"total_tokens": 42, "prompt_tokens": 10},
        "sequence": [{"value": "a"}],
        "primitive": "ok",
    }


def test_resolve_model_configuration_uses_azure_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = settings.AppConfig.model_validate(
        {
            "version": "0.1",
            "llm": {
                "default_workflow": "fast",
                "workflows": {
                    "fast": {
                        "provider": "azure",
                        "model": "gpt-4.1",
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
                "enabled": True,
                "auto_reviewer_model": "gpt-4.1",
                "auto_reviewer_provider": None,
            },
            "embedding": None,
            "telemetry": {"log_level": "INFO"},
        }
    )

    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

    engine = ReviewEngine(config)
    model_name, kwargs = engine._resolve_model_configuration()
    assert model_name == "azure/gpt-4.1"
    assert kwargs["custom_llm_provider"] == "azure"
    assert kwargs["api_base"] == "https://example.openai.azure.com"


def test_resolve_model_configuration_uses_ollama_task_routing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _load_sample_config(tmp_path)
    config.review.auto_reviewer_model = "gemma3:1b"
    config.review.auto_reviewer_provider = "ollama"
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://172.16.0.1:11434/")

    model_name, kwargs = ReviewEngine(config)._resolve_model_configuration()

    assert model_name == "ollama/gemma3:1b"
    assert kwargs == {
        "base_url": "http://172.16.0.1:11434",
        "api_base": "http://172.16.0.1:11434",
        "custom_llm_provider": "ollama",
    }


def test_review_engine_sets_o_series_temperature(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _load_sample_config(tmp_path)
    config.review.auto_reviewer_model = "o3-mini"
    config.review.auto_reviewer_provider = None

    captured: Dict[str, Any] = {}

    def _fake_completion(
        *args: Any, temperature: float, **kwargs: Any
    ) -> Dict[str, Any]:
        captured["temperature"] = temperature
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            "VERDICT: PASS\n"
                            "REASONING: Compliant.\n"
                            "QUALITY_SCORE: 0.9\n"
                            "SUGGESTIONS:\n"
                            "- None."
                        )
                    }
                }
            ],
            "usage": {},
        }

    class DummyTimeout(Exception):
        """Placeholder timeout exception."""

    dummy_litellm = SimpleNamespace(
        completion=_fake_completion,
        Timeout=DummyTimeout,
    )
    monkeypatch.setattr("core.review.litellm", dummy_litellm)

    engine = ReviewEngine(config)
    monkeypatch.setattr(
        engine,
        "_resolve_model_configuration",
        lambda: ("o3-mini", {}),
    )

    request = TaskRequest(workflow="fast", prompt="demo")
    result = TaskResult(workflow="fast", content="ok", latency_seconds=0.1)

    engine.perform_review(request, result)

    assert captured.get("temperature") == pytest.approx(1.0)


def test_perform_review_skips_when_disabled(tmp_path: Path) -> None:
    config = _load_sample_config(tmp_path)
    config.review.enabled = False
    engine = ReviewEngine(config)

    review = engine.perform_review(
        TaskRequest(workflow="fast", prompt="demo"),
        TaskResult(workflow="fast", content="ok", latency_seconds=0.1),
    )

    assert review.verdict == "skipped"
    assert review.auto_verdict is None
    assert review.quality_score is None


def test_perform_review_applies_human_failure_feedback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _load_sample_config(tmp_path)
    engine = ReviewEngine(config)
    monkeypatch.setattr(engine, "_run_automated_review", lambda *_: None)

    review = engine.perform_review(
        TaskRequest(workflow="fast", prompt="demo"),
        TaskResult(workflow="fast", content="ok", latency_seconds=0.1),
        human_feedback="reject: missing constraint",
    )

    assert review.verdict == "fail-human"
    assert review.notes is not None
    assert "Human feedback" in review.notes


def test_automated_review_without_model_returns_none(tmp_path: Path) -> None:
    config = _load_sample_config(tmp_path)
    config.review.auto_reviewer_model = None
    engine = ReviewEngine(config)

    review = engine._run_automated_review(
        TaskRequest(workflow="fast", prompt="demo"),
        TaskResult(workflow="fast", content="ok", latency_seconds=0.1),
    )

    assert review is None


def test_perform_review_marks_missing_automated_review_unverified(
    tmp_path: Path,
) -> None:
    config = _load_sample_config(tmp_path)
    config.review.auto_reviewer_model = None
    engine = ReviewEngine(config)

    review = engine.perform_review(
        TaskRequest(workflow="fast", prompt="demo"),
        TaskResult(workflow="fast", content="ok", latency_seconds=0.1),
    )

    assert review.verdict == "unverified"
    assert review.auto_verdict is None


def test_automated_review_requires_litellm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _load_sample_config(tmp_path)
    config.review.auto_reviewer_model = "review-model"
    engine = ReviewEngine(config)
    monkeypatch.setattr("core.review.litellm", None)

    with pytest.raises(ReviewError, match="liteLLM is required"):
        engine._run_automated_review(
            TaskRequest(workflow="fast", prompt="demo"),
            TaskResult(workflow="fast", content="ok", latency_seconds=0.1),
        )


def test_automated_review_timeout_returns_timeout_review(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    marker = "review-timeout-prompt-marker"
    config = _load_sample_config(tmp_path)
    config.review.auto_reviewer_model = "review-model"
    config.review.auto_reviewer_provider = "ollama"

    class DummyTimeout(Exception):
        """Placeholder timeout exception."""

    def _completion(**_: object) -> dict[str, object]:
        raise DummyTimeout(marker)

    dummy_litellm = SimpleNamespace(completion=_completion, Timeout=DummyTimeout)
    monkeypatch.setattr("core.review.litellm", dummy_litellm)

    engine = ReviewEngine(config)
    caplog.set_level(logging.WARNING, logger="drm.review")
    automated = engine._run_automated_review(
        TaskRequest(workflow="fast", prompt="demo"),
        TaskResult(workflow="fast", content="ok", latency_seconds=0.1),
    )

    assert automated is not None
    assert automated.verdict == "timeout"
    assert automated.suggestions == ["Automated review timed out."]
    assert marker not in caplog.text
    assert "error_type=DummyTimeout" in caplog.text


def test_automated_review_redacts_provider_failure_details(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    marker = "review-provider-result-and-secret-marker"
    config = _load_sample_config(tmp_path)
    config.review.auto_reviewer_model = "review-model"
    config.review.auto_reviewer_provider = "ollama"

    class ProviderFailure(Exception):
        """Synthetic provider exception carrying sensitive-looking detail."""

    def _completion(**_: object) -> dict[str, object]:
        raise ProviderFailure(marker)

    monkeypatch.setattr(
        "core.review.litellm",
        SimpleNamespace(completion=_completion, Timeout=TimeoutError),
    )
    engine = ReviewEngine(config)
    caplog.set_level(logging.ERROR, logger="drm.review")

    with pytest.raises(ReviewError, match="Automated review failed") as error:
        engine._run_automated_review(
            TaskRequest(workflow="fast", prompt="demo"),
            TaskResult(workflow="fast", content="ok", latency_seconds=0.1),
        )

    assert marker not in caplog.text
    assert marker not in str(error.value)
    assert "error_type=ProviderFailure" in caplog.text
    assert error.value.__cause__ is None


def test_review_helpers_normalise_values() -> None:
    assert ReviewEngine._normalise_verdict("approve") == "pass"
    assert ReviewEngine._normalise_verdict("reject") == "fail-auto"
    assert ReviewEngine._normalise_verdict(None) == "unverified"
    assert ReviewEngine._normalise_verdict("needs-work") == "unverified"
    assert ReviewEngine._normalise_verdict("timeout") == "unverified"
    assert ReviewEngine._resolve_temperature("azure/gpt-4.1") == pytest.approx(0.0)
    assert ReviewEngine._extract_float("score: 0.82 / 1") == pytest.approx(0.82)
    assert ReviewEngine._normalise_bullet("1) Fix issue") == "Fix issue"
