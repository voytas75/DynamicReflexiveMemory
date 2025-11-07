"""Utilities for persisting lightweight user-specific application settings.

Updates:
    v0.1 - 2025-11-07 - Added JSON-backed storage for workflow preference and window size.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "data" / "user_settings.json"

LOGGER = logging.getLogger("drm.user_settings")


@dataclass(slots=True)
class UserSettings:
    """Represents persisted per-user DRM application settings."""

    last_workflow: Optional[str] = None
    window_width: Optional[int] = None
    window_height: Optional[int] = None


class UserSettingsManager:
    """Handles loading and storing user settings on disk."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = path or DEFAULT_SETTINGS_PATH
        self._lock = Lock()
        self._settings = UserSettings()
        self._load()

    @property
    def settings(self) -> UserSettings:
        """Return the current in-memory settings snapshot."""
        return self._settings

    def update(
        self,
        *,
        last_workflow: Optional[str] = None,
        window_width: Optional[int] = None,
        window_height: Optional[int] = None,
    ) -> UserSettings:
        """Update stored settings and persist them to disk."""
        changed = False
        if last_workflow is not None and last_workflow != self._settings.last_workflow:
            self._settings.last_workflow = last_workflow
            changed = True
        if window_width is not None and window_width > 0:
            if window_width != self._settings.window_width:
                self._settings.window_width = window_width
                changed = True
        if window_height is not None and window_height > 0:
            if window_height != self._settings.window_height:
                self._settings.window_height = window_height
                changed = True

        if changed:
            self._save()
        return self._settings

    def _load(self) -> None:
        """Load settings from disk, ignoring malformed payloads."""
        path = self._path
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.warning("Unable to load user settings (%s); using defaults.", exc)
            return

        last_workflow = payload.get("last_workflow")
        width = payload.get("window_width")
        height = payload.get("window_height")
        self._settings = UserSettings(
            last_workflow=str(last_workflow) if isinstance(last_workflow, str) else None,
            window_width=int(width) if isinstance(width, int) and width > 0 else None,
            window_height=int(height) if isinstance(height, int) and height > 0 else None,
        )

    def _save(self) -> None:
        """Persist settings to disk."""
        path = self._path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            LOGGER.error("Failed to create settings directory %s: %s", path.parent, exc)
            return

        with self._lock:
            try:
                path.write_text(
                    json.dumps(asdict(self._settings), indent=2, sort_keys=True),
                    encoding="utf-8",
                )
            except OSError as exc:
                LOGGER.error("Failed to store user settings %s: %s", path, exc)
