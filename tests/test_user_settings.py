"""Tests for the user settings persistence layer."""

from __future__ import annotations

from pathlib import Path

from core.user_settings import UserSettingsManager


def test_user_settings_roundtrip(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    manager = UserSettingsManager(settings_path)

    assert manager.settings.last_workflow is None

    manager.update(last_workflow="reasoning", window_width=1024, window_height=768)

    reloaded = UserSettingsManager(settings_path)
    assert reloaded.settings.last_workflow == "reasoning"
    assert reloaded.settings.window_width == 1024
    assert reloaded.settings.window_height == 768


def test_user_settings_handles_corruption(tmp_path: Path, caplog) -> None:
    settings_path = tmp_path / "settings.json"
    settings_path.write_text("{invalid json", encoding="utf-8")

    manager = UserSettingsManager(settings_path)
    assert manager.settings.last_workflow is None

    manager.update(last_workflow="fast")
    assert "fast" in settings_path.read_text(encoding="utf-8")
