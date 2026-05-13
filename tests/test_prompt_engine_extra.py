from __future__ import annotations

from datetime import datetime, timezone

from config import settings
from core.prompt_engine import AdaptivePromptEngine, PromptContext
from models.memory import ReviewRecord


def _engine() -> AdaptivePromptEngine:
    return AdaptivePromptEngine(settings.load_app_config())


def test_build_prompt_includes_drift_advisory() -> None:
    prompt = _engine().build_prompt(
        PromptContext(
            task="Investigate incident.",
            workflow="reasoning",
            working_memory={},
            episodic_memory=[],
            semantic_memory=[],
            recent_reviews=[],
            drift_indicator="Latency is trending up.",
        )
    )

    assert "### Drift Advisory" in prompt
    assert "Latency is trending up." in prompt


def test_semantic_relations_skips_empty_neighbour_sets() -> None:
    formatted = AdaptivePromptEngine._format_semantic_relations(
        {"concept-a": [], "concept-b": [{"id": "n1"}]}
    )

    assert "concept-a" not in formatted
    assert "concept-b: n1" in formatted


def test_format_reviews_includes_quality_and_suggestions() -> None:
    record = ReviewRecord(
        id="r1",
        task_reference="task-1",
        verdict="pass",
        notes="Looks solid",
        suggestions=["Ship it"],
        quality_score=0.95,
        auto_verdict="pass",
        created_at=datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
    )

    formatted = AdaptivePromptEngine._format_reviews([record])

    assert "2026-01-02T03:04:05+00:00" in formatted
    assert "quality=0.95" in formatted
    assert "suggestions: Ship it" in formatted


def test_format_reviews_returns_empty_for_no_reviews() -> None:
    assert AdaptivePromptEngine._format_reviews([]) == ""