"""PySide6 GUI harness for the DRM application.

Updates:
    v0.1 - 2025-11-06 - Implemented minimal GUI window with runtime diagnostics and graceful fallback when PySide6 is unavailable.
    v0.2 - 2025-11-06 - Added memory snapshots and drift advisory display.
    v0.3 - 2025-11-07 - Integrated LiveTaskLoop with interactive task execution and drift advisory history.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Optional

from config.settings import AppConfig
from core.controller import SelfAdjustingController
from core.exceptions import DRMError, MemoryError, WorkflowError
from core.live_loop import LiveTaskLoop
from core.memory_manager import MemoryManager
from models.workflows import TaskRunOutcome

try:  # pragma: no cover - optional dependency
    from PySide6.QtCore import QObject, QThread, Signal, Slot
    from PySide6.QtWidgets import (
        QApplication,
        QComboBox,
        QHBoxLayout,
        QLabel,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError:  # pragma: no cover
    QApplication = None  # type: ignore[assignment]

LOGGER = logging.getLogger("drm.gui")


class TaskExecutionWorker(QObject):  # pragma: no cover - requires GUI runtime
    """Executes a task in a background thread."""

    finished = Signal(object)
    failed = Signal(object)

    def __init__(
        self,
        task_loop: LiveTaskLoop,
        task_text: str,
        workflow_override: Optional[str],
    ) -> None:
        super().__init__()
        self._task_loop = task_loop
        self._task_text = task_text
        self._workflow_override = workflow_override

    @Slot()
    def run(self) -> None:
        """Execute the task and emit completion or failure."""
        try:
            outcome = self._task_loop.run_task(
                task=self._task_text,
                workflow_override=self._workflow_override,
            )
            self.finished.emit(outcome)
        except DRMError as exc:
            self.failed.emit(exc)
        except Exception as exc:  # pragma: no cover - defensive safeguard
            LOGGER.error("Unexpected error in task worker: %s", exc)
            self.failed.emit(exc)


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
        self._worker_thread: Optional[QThread] = None
        self._current_worker: Optional[TaskExecutionWorker] = None
        self._recent_outcomes: List[TaskRunOutcome] = []
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
        self._run_button = run_button
        controls_row.addWidget(run_button)

        layout.addLayout(controls_row)
        layout.addWidget(self._task_input)

        self._status_label = QLabel("Idle.")
        layout.addWidget(self._status_label)

        self._results_tab = QTabWidget()
        self._recent_output_view = QTextEdit()
        self._recent_output_view.setReadOnly(True)
        self._recent_output_view.setPlaceholderText("No task runs yet.")
        self._review_history_view = QTextEdit()
        self._review_history_view.setReadOnly(True)
        self._review_history_view.setPlaceholderText("No review records yet.")
        self._results_tab.addTab(self._recent_output_view, "Recent Outputs")
        self._results_tab.addTab(self._review_history_view, "Review History")
        layout.addWidget(self._results_tab)

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
        if self._worker_thread and self._worker_thread.isRunning():
            QMessageBox.information(
                self,
                "Task Running",
                "A task is already running. Please wait for it to complete.",
            )
            return

        task_text = self._task_input.toPlainText().strip()
        if not task_text:
            QMessageBox.warning(self, "Task Required", "Enter a task prompt before running.")
            return

        workflow_data = self._workflow_selector.currentData()
        workflow: Optional[str] = workflow_data if isinstance(workflow_data, str) else None

        self._set_interaction_enabled(False)
        self._status_label.setText("Running task…")

        worker = TaskExecutionWorker(self._task_loop, task_text, workflow)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)  # type: ignore[attr-defined]
        worker.finished.connect(self._on_task_finished)  # type: ignore[arg-type]
        worker.failed.connect(self._on_task_failed)  # type: ignore[arg-type]
        worker.finished.connect(thread.quit)  # type: ignore[attr-defined]
        worker.failed.connect(thread.quit)  # type: ignore[attr-defined]
        worker.finished.connect(worker.deleteLater)  # type: ignore[attr-defined]
        worker.failed.connect(worker.deleteLater)  # type: ignore[attr-defined]
        thread.finished.connect(thread.deleteLater)  # type: ignore[attr-defined]

        self._current_worker = worker
        self._worker_thread = thread
        thread.start()

    def _on_task_finished(self, outcome: object) -> None:
        """Handle successful task completion."""
        self._set_interaction_enabled(True)
        self._status_label.setText("Task completed.")
        self._task_input.clear()
        self._worker_thread = None
        self._current_worker = None

        if not isinstance(outcome, TaskRunOutcome):
            LOGGER.warning("Received unexpected outcome type: %s", type(outcome))
            return

        self._recent_outcomes.append(outcome)
        self._recent_outcomes = self._recent_outcomes[-5:]
        self._render_recent_outputs()
        self._refresh_memory_snapshot()

    def _on_task_failed(self, error: object) -> None:
        """Handle task failure from the worker thread."""
        self._set_interaction_enabled(True)
        self._status_label.setText("Task failed.")
        self._worker_thread = None
        self._current_worker = None

        message = str(error)
        if isinstance(error, WorkflowError):
            title = "Workflow Error"
        elif isinstance(error, DRMError):
            title = "Execution Error"
        elif isinstance(error, Exception):
            title = "Unexpected Error"
        else:
            title = "Task Error"
        QMessageBox.critical(self, title, message)

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
            self._review_history_view.setPlainText(f"Unable to load review history: {exc}")
            return

        drift_items = [item for item in working_items if item.key.endswith(":drift")]
        drift_summary = self._format_drift_summary(drift_items)
        self._update_review_history_view(reviews)

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

    def _format_drift_summary(self, items) -> str:
        if not items:
            return "<b>Drift Advisories:</b> None recorded."
        latest = max(items, key=lambda entry: entry.created_at)
        return (
            f"<b>Drift Advisories:</b> {len(items)} recorded. "
            f"Latest ({latest.created_at.isoformat()}): {latest.payload.get('advisory')}"
        )

    def _render_recent_outputs(self) -> None:
        """Render recent task outcomes in the dedicated view."""
        if not self._recent_outcomes:
            self._recent_output_view.setPlainText("No task runs yet.")
            return

        lines: List[str] = []
        for outcome in reversed(self._recent_outcomes):
            quality = (
                f"{outcome.review.quality_score:.2f}"
                if outcome.review.quality_score is not None
                else "n/a"
            )
            suggestions = "; ".join(outcome.review.suggestions) or "None"
            lines.append(
                (
                    f"[{outcome.request.created_at.isoformat()}] "
                    f"{outcome.selection.workflow} ({outcome.selection.rationale}, score={outcome.selection.score:.2f})\n"
                    f"Latency: {outcome.result.latency_seconds:.2f}s | Drift: {outcome.drift_advisory or 'None'}\n"
                    f"Output:\n{outcome.result.content}\n"
                    f"Review: verdict={outcome.review.verdict}, auto={outcome.review.auto_verdict}, "
                    f"quality={quality}, suggestions={suggestions}\n"
                )
            )
        self._recent_output_view.setPlainText("\n".join(lines))

    def _update_review_history_view(self, reviews: Optional[List[dict]] = None) -> None:
        """Render review history in the dedicated view."""
        if reviews is None:
            try:
                reviews = self._memory_manager.list_layer("review")
            except MemoryError as exc:
                LOGGER.error("Failed to load review history: %s", exc)
                self._review_history_view.setPlainText(f"Unable to load review history: {exc}")
                return

        if not reviews:
            self._review_history_view.setPlainText("No review records yet.")
            return

        def _parse_timestamp(payload: dict) -> datetime:
            value = payload.get("created_at")
            if isinstance(value, datetime):
                return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
            if isinstance(value, str):
                try:
                    parsed = datetime.fromisoformat(value)
                except ValueError:
                    return datetime.now(timezone.utc)
                return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
            return datetime.now(timezone.utc)

        sorted_records = sorted(reviews, key=_parse_timestamp, reverse=True)[:5]
        lines = []
        for record in sorted_records:
            notes = record.get("notes") or "n/a"
            quality = record.get("quality_score")
            quality_text = f"{quality:.2f}" if isinstance(quality, (int, float)) else "n/a"
            suggestions = record.get("suggestions") or []
            suggestions_text = "; ".join(suggestions) if suggestions else "None"
            lines.append(
                (
                    f"{record.get('created_at', 'unknown')} | verdict={record.get('verdict')} "
                    f"(auto={record.get('auto_verdict')})\n"
                    f"quality={quality_text} | suggestions={suggestions_text}\n"
                    f"notes: {notes}\n"
                )
            )
        self._review_history_view.setPlainText("\n".join(lines))

    def _set_interaction_enabled(self, enabled: bool) -> None:
        """Enable or disable task interaction widgets."""
        self._workflow_selector.setEnabled(enabled)
        self._task_input.setEnabled(enabled)
        self._run_button.setEnabled(enabled)


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
