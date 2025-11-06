"""Tests for the review engine structured parsing and live loop integration.

Updates: v0.1 - 2025-11-07 - Added regression tests for automated review parsing
and LiveTaskLoop orchestration.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from config import settings
from core.live_loop import LiveTaskLoop
from core.review import ReviewEngine


def _entry_has_user_task(entry: Dict[str, object], expected: str) -> bool:
    metadata = entry.get("metadata")
    if isinstance(metadata, dict):
        user_task = metadata.get("user_task")
        return isinstance(user_task, str) and user_task == expected
    return False


def _load_sample_config(tmp_path: Path) -> settings.AppConfig:
    source = Path(__file__).resolve().parent.parent / "config" / "config.json"
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


def test_live_task_loop_persists_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_resolve_model_configuration_uses_azure_provider(monkeypatch: pytest.MonkeyPatch) -> None:
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
                "redis": {"host": "localhost", "port": 6379, "db": 0, "ttl_seconds": 120},
                "chromadb": {"persist_directory": "data/chromadb", "collection": "test"},
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
