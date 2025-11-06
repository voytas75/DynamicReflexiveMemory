"""PySide6 GUI harness for the DRM application.

Updates:
    v0.1 - 2025-11-06 - Implemented minimal GUI window with runtime diagnostics and
        graceful fallback when PySide6 is unavailable.
    v0.2 - 2025-11-06 - Added memory snapshots and drift advisory display.
"""

from __future__ import annotations

import logging
from typing import Optional

from config.settings import AppConfig
from core.controller import SelfAdjustingController
from core.exceptions import MemoryError
from core.memory_manager import MemoryManager

try:  # pragma: no cover - optional dependency
    from PySide6.QtWidgets import (
        QApplication,
        QLabel,
        QPushButton,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError:  # pragma: no cover
    QApplication = None  # type: ignore[assignment]

LOGGER = logging.getLogger("drm.gui")


class DRMWindow(QWidget):  # pragma: no cover - requires GUI runtime
    """Status window exposing memory and workflow state."""

    def __init__(
        self,
        config: AppConfig,
        memory_manager: MemoryManager,
        controller: SelfAdjustingController,
    ) -> None:
        super().__init__()
        self._config = config
        self._memory_manager = memory_manager
        self._controller = controller
        self.setWindowTitle("Dynamic Reflexive Memory")
        layout = QVBoxLayout(self)

        header = QLabel(
            "<b>Dynamic Reflexive Memory</b><br>"
            f"Default Workflow: {config.llm.default_workflow}<br>"
            f"Review Enabled: {config.review.enabled}"
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        self._memory_view = QTextEdit()
        self._memory_view.setReadOnly(True)
        layout.addWidget(self._memory_view)

        self._drift_label = QLabel("No drift advisories yet.")
        self._drift_label.setWordWrap(True)
        layout.addWidget(self._drift_label)

        refresh_button = QPushButton("Refresh Memory Snapshot")
        refresh_button.clicked.connect(self._refresh_memory_snapshot)  # type: ignore[arg-type]
        layout.addWidget(refresh_button)

        self._refresh_memory_snapshot()

    def _refresh_memory_snapshot(self) -> None:
        """Load memory slices and update the GUI text area."""
        try:
            working_items = self._memory_manager.list_working_items()
            episodic = self._memory_manager.list_layer("episodic")
            semantic = self._memory_manager.list_layer("semantic")
            reviews = self._memory_manager.list_layer("review")
        except MemoryError as exc:
            LOGGER.error("Failed to load memory snapshot: %s", exc)
            self._memory_view.setPlainText(f"Unable to load memory snapshot: {exc}")
            return

        lines = [
            "=== Working Memory ===",
            *(self._format_working_item(item) for item in working_items[:5]),
            "=== Episodic Memory ===",
            *(self._format_generic_item(item) for item in episodic[:5]),
            "=== Semantic Concepts ===",
            *(self._format_generic_item(item) for item in semantic[:5]),
            "=== Review Records ===",
            *(self._format_generic_item(item) for item in reviews[:5]),
        ]
        self._memory_view.setPlainText("\n".join(lines))

        drift = self._controller.last_advisory or "No drift advisories yet."
        self._drift_label.setText(f"<b>Drift Advisory:</b> {drift}")

    @staticmethod
    def _format_working_item(item) -> str:
        return f"{item.key}: {item.payload} (ttl={item.ttl_seconds}s)"

    @staticmethod
    def _format_generic_item(item: dict) -> str:
        identifier = item.get("id", "unknown")
        content = item.get("content") or item.get("definition") or item
        return f"{identifier}: {content}"


def launch_gui(config: AppConfig) -> Optional[int]:
    """Launch the PySide6 GUI; returns exit code or None if GUI unavailable."""
    if QApplication is None:
        LOGGER.error(
            "PySide6 is not installed; cannot launch GUI. "
            "Install PySide6 or use CLI mode."
        )
        return None

    app = QApplication([])
    memory_manager = MemoryManager(config)
    controller = SelfAdjustingController(config)
    window = DRMWindow(config, memory_manager, controller)
    window.show()
    return app.exec()
