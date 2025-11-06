"""Tests for the review engine structured parsing and live loop integration.

Updates: v0.1 - 2025-11-07 - Added regression tests for automated review parsing
and LiveTaskLoop orchestration.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from config import settings
from core.live_loop import LiveTaskLoop
from core.review import ReviewEngine


def _load_sample_config(tmp_path: Path):
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
    config.llm.workflows["fast"].provider = "ollama"
    config.llm.workflows["fast"].model = "stub-fast-model"

    def _dummy_completion(model: str, messages, **kwargs):
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
    assert any(entry.get("metadata", {}).get("user_task") == "Draft integration plan for QA." for entry in episodic_entries)

    review_records = loop._memory_manager.list_layer("review")
    assert review_records, "Expected persisted review records."
    stored_review = review_records[-1]
    assert stored_review.get("quality_score") == pytest.approx(0.95)
    assert stored_review.get("suggestions") == ["Keep monitoring latency."]

    working_items = loop._memory_manager.list_working_items()
    assert any(item.key.endswith(":result") for item in working_items)
