"""Tests for configuration persistence helpers."""

from __future__ import annotations

from pathlib import Path

from config import settings


def test_save_app_config_roundtrip(tmp_path: Path) -> None:
    source_path = Path(__file__).resolve().parent.parent / "config" / "config.json"
    config = settings.load_app_config(source_path)

    config.llm.enable_debug = True
    config.review.auto_reviewer_model = "o3-mini"

    target = tmp_path / "config.json"
    settings.save_app_config(config, target)

    reloaded = settings.load_app_config(target)
    assert reloaded.llm.enable_debug is True
    assert reloaded.review.auto_reviewer_model == "o3-mini"
