"""PySide6 GUI harness for the DRM application.

Updates:
    v0.1 - 2025-11-06 - Implemented minimal GUI window with runtime diagnostics and graceful fallback when PySide6 is unavailable.
    v0.2 - 2025-11-06 - Added memory snapshots and drift advisory display.
    v0.3 - 2025-11-07 - Integrated LiveTaskLoop with interactive task execution and drift advisory history.
"""

from __future__ import annotations

import logging
from typing import Optional

from config.settings import AppConfig
from core.controller import SelfAdjustingController
from core.exceptions import DRMError, MemoryError, WorkflowError
from core.live_loop import LiveTaskLoop
from core.memory_manager import MemoryManager

try:  # pragma: no cover - optional dependency
    from PySide6.QtWidgets import (
        QApplication,
        QComboBox,
        QHBoxLayout,
        QLabel,
        QMessageBox,
        QPlainTextEdit,
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
        task_loop: LiveTaskLoop,
    ) -> None:
        super().__init__()
        self._config = config
        self._memory_manager = memory_manager
        self._controller = controller
        self._task_loop = task_loop
        self.setWindowTitle("Dynamic Reflexive Memory")
        layout = QVBoxLayout(self)

        header = QLabel(
            "<b>Dynamic Reflexive Memory</b><br>"
            f"Default Workflow: {config.llm.default_workflow}<br>"
            f"Review Enabled: {config.review.enabled}"
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        controls_row = QHBoxLayout()
        self._workflow_selector = QComboBox()
        self._workflow_selector.setEditable(False)
        self._workflow_selector.addItem(
            f"Auto ({config.llm.default_workflow})", userData=None
        )
        for name in config.llm.workflows.keys():
            self._workflow_selector.addItem(name, userData=name)
        controls_row.addWidget(QLabel("Workflow:"))
        controls_row.addWidget(self._workflow_selector)

        self._task_input = QPlainTextEdit()
        self._task_input.setPlaceholderText("Enter task prompt...")
        self._task_input.setFixedHeight(80)

        run_button = QPushButton("Run Task")
        run_button.clicked.connect(self._handle_run_task)  # type: ignore[arg-type]
        controls_row.addWidget(run_button)

        layout.addLayout(controls_row)
        layout.addWidget(self._task_input)

        self._result_view = QTextEdit()
        self._result_view.setReadOnly(True)
        layout.addWidget(self._result_view)

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

    def _handle_run_task(self) -> None:
        """Execute a task using the live loop and render the outcome."""
        task_text = self._task_input.toPlainText().strip()
        if not task_text:
            QMessageBox.warning(self, "Task Required", "Enter a task prompt before running.")
            return

        workflow_data = self._workflow_selector.currentData()
        workflow: Optional[str] = workflow_data if isinstance(workflow_data, str) else None

        try:
            outcome = self._task_loop.run_task(
                task=task_text,
                workflow_override=workflow,
            )
        except WorkflowError as exc:
            QMessageBox.critical(self, "Workflow Error", str(exc))
            return
        except DRMError as exc:
            QMessageBox.critical(self, "Execution Error", str(exc))
            return

        self._task_input.clear()
        self._result_view.setPlainText(self._format_outcome(outcome))
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

        drift_items = [item for item in working_items if item.key.endswith(":drift")]
        drift_summary = self._format_drift_summary(drift_items)

        lines = [
            "=== Working Memory ===",
            *(self._format_working_item(item) for item in working_items[:5]),
            "=== Episodic Memory ===",
            *(self._format_generic_item(item) for item in episodic[:5]),
            "=== Semantic Concepts ===",
            *(self._format_generic_item(item) for item in semantic[:5]),
            "=== Review Records ===",
            *(self._format_generic_item(item) for item in reviews[:5]),
            "=== Drift Advisories ===",
            *(self._format_working_item(item) for item in drift_items[:5]),
        ]
        self._memory_view.setPlainText("\n".join(lines))

        self._drift_label.setText(drift_summary)

    @staticmethod
    def _format_working_item(item) -> str:
        return f"{item.key}: {item.payload} (ttl={item.ttl_seconds}s)"

    @staticmethod
    def _format_generic_item(item: dict) -> str:
        identifier = item.get("id", "unknown")
        content = item.get("content") or item.get("definition") or item
        return f"{identifier}: {content}"

    def _format_outcome(self, outcome) -> str:
        """Render the latest run outcome for display."""
        suggestions = "; ".join(outcome.review.suggestions) if outcome.review.suggestions else "None"
        quality = (
            f"{outcome.review.quality_score:.2f}"
            if outcome.review.quality_score is not None
            else "n/a"
        )
        drift = outcome.drift_advisory or "None"
        return (
            f"Task ID: {outcome.request.task_id}\n"
            f"Workflow: {outcome.selection.workflow} (reason: {outcome.selection.rationale}, score={outcome.selection.score:.2f})\n"
            f"Latency: {outcome.result.latency_seconds:.2f}s\n"
            f"Result:\n{outcome.result.content}\n\n"
            f"Review Verdict: {outcome.review.verdict} (auto={outcome.review.auto_verdict}, quality={quality})\n"
            f"Review Notes: {outcome.review.notes or 'None'}\n"
            f"Suggestions: {suggestions}\n"
            f"Drift Advisory: {drift}"
        )

    def _format_drift_summary(self, items) -> str:
        if not items:
            return "<b>Drift Advisories:</b> None recorded."
        latest = max(items, key=lambda entry: entry.created_at)
        return (
            f"<b>Drift Advisories:</b> {len(items)} recorded. "
            f"Latest ({latest.created_at.isoformat()}): {latest.payload.get('advisory')}"
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
    memory_manager = MemoryManager(config)
    controller = SelfAdjustingController(config)
    task_loop = LiveTaskLoop(
        config,
        memory_manager=memory_manager,
        controller=controller,
    )
    window = DRMWindow(config, memory_manager, controller, task_loop)
    window.show()
    return app.exec()
