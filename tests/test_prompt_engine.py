"""Tests for the adaptive prompt engine.

Updates: v0.1 - 2025-11-06 - Added sanity checks for prompt composition.
"""

from __future__ import annotations

import json
from pathlib import Path

from config import settings
from core.prompt_engine import AdaptivePromptEngine, PromptContext


def _load_sample_config(tmp_path: Path) -> settings.AppConfig:
    source = Path(__file__).resolve().parent.parent / "config" / "config.json"
    config_path = tmp_path / "config.json"
    config_path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return settings.load_app_config(config_path)


def test_prompt_includes_sections(tmp_path: Path) -> None:
    config = _load_sample_config(tmp_path)
    engine = AdaptivePromptEngine(config)
    context = PromptContext(
        task="Solve simple addition.",
        workflow="fast",
        working_memory={"a": 1, "b": 2},
        episodic_memory=[{"id": "ep1", "content": "Previous calculation."}],
        semantic_memory=[{"id": "concept1", "content": "Addition basics."}],
        semantic_relations={
            "concept1": [
                {"id": "concept2", "label": "Advanced addition", "weight": 0.85}
            ]
        },
        recent_reviews=[],
    )
    prompt = engine.build_prompt(context)
    assert "### DRM Adaptive Prompt" in prompt
    assert "### Working Memory" in prompt
    assert "### Recent Episodes" in prompt
    assert "### Semantic Relations" in prompt
