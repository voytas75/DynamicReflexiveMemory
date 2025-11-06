"""PySide6 GUI harness for the DRM application.

Updates: v0.1 - 2025-11-06 - Implemented minimal GUI window with runtime
diagnostics and graceful fallback when PySide6 is unavailable.
"""

from __future__ import annotations

import logging
from typing import Optional

from config.settings import AppConfig

try:  # pragma: no cover - optional dependency
    from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
except ImportError:  # pragma: no cover
    QApplication = None  # type: ignore[assignment]

LOGGER = logging.getLogger("drm.gui")


class DRMWindow(QWidget):  # pragma: no cover - requires GUI runtime
    """Simple status window exposing memory and workflow state."""

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self._config = config
        self.setWindowTitle("Dynamic Reflexive Memory")
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "<b>Dynamic Reflexive Memory</b><br>"
                f"Default Workflow: {config.llm.default_workflow}<br>"
                f"Review Enabled: {config.review.enabled}"
            )
        )


def launch_gui(config: AppConfig) -> Optional[int]:
    """Launch the PySide6 GUI; returns exit code or None if GUI unavailable."""
    if QApplication is None:
        LOGGER.error(
            "PySide6 is not installed; cannot launch GUI. "
            "Install PySide6 or use CLI mode."
        )
        return None

    app = QApplication([])
    window = DRMWindow(config)
    window.show()
    return app.exec()
