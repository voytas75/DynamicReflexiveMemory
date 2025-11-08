"""PySide6 GUI harness for the DRM application.

Updates:
    v0.1 - 2025-11-06 - Implemented minimal GUI window with runtime diagnostics and graceful fallback when PySide6 is unavailable.
    v0.2 - 2025-11-06 - Added memory snapshots and drift advisory display.
    v0.3 - 2025-11-07 - Integrated LiveTaskLoop with interactive task execution and drift advisory history.
    v0.4 - 2025-11-06 - Surfaced controller workflow biases in the telemetry panel.
    v0.5 - 2025-11-07 - Displayed recent memory revision history alongside other telemetry.
    v0.6 - 2025-11-07 - Guarded headless environments to avoid Qt crashes and fall back to CLI.
    v0.7 - 2025-11-07 - Added human review feedback capture and drift mitigation controls.
    v0.8 - 2025-11-07 - Restored user preferences for workflow selection and window geometry.
    v0.9 - 2025-11-07 - Added settings dialog for editing configuration values.
    v0.10 - 2025-11-08 - Wired telemetry-driven metrics, drift advisories, and review panes.
    v0.11 - 2025-11-08 - Visualised drift analytics trends with persistent telemetry data.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, TYPE_CHECKING, cast

from config.settings import AppConfig
from core.controller import SelfAdjustingController
from core.exceptions import DRMError, MemoryError, WorkflowError
from core.live_loop import LiveTaskLoop
from core.memory_manager import MemoryManager
from core.telemetry import TelemetryEvent, drain_telemetry
from core.user_settings import UserSettings, UserSettingsManager
from gui.settings_dialog import SettingsDialog
from models.memory import WorkingMemoryItem
from models.workflows import TaskRunOutcome

if TYPE_CHECKING:  # pragma: no cover - typing-only imports
    from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot
    from PySide6.QtWidgets import (
        QApplication,
        QDialog,
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
else:  # pragma: no cover - runtime optional dependency
    try:
        from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot
        from PySide6.QtWidgets import (
            QApplication,
            QDialog,
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
    except ImportError:
        QApplication = None

        class QObject:
            def __init__(self, *_: Any, **__: Any) -> None:
                pass

        class QThread:
            def __init__(self, *_: Any, **__: Any) -> None:
                pass

            def start(self) -> None:
                raise RuntimeError("PySide6 is required for GUI operation")

            def isRunning(self) -> bool:  # pragma: no cover - stub
                return False

            def quit(self) -> None:
                pass

            def deleteLater(self) -> None:
                pass

            @property
            def finished(self) -> "Signal":  # pragma: no cover - stub accessor
                return Signal()

        class Signal:
            def __init__(self, *_: Any, **__: Any) -> None:
                pass

            def connect(self, *_: Any, **__: Any) -> None:
                pass

            def emit(self, *_: Any, **__: Any) -> None:
                pass

        class QTimer:
            def __init__(self, *_: Any, **__: Any) -> None:
                self.timeout = Signal()

            def setInterval(self, *_: Any, **__: Any) -> None:
                pass

            def start(self) -> None:
                pass

            def stop(self) -> None:
                pass

            def isActive(self) -> bool:
                return False

        def Slot(*_args: Any, **_kwargs: Any):
            def decorator(func: Any) -> Any:
                return func

            return decorator

        class QWidget:
            def __init__(self, *_: Any, **__: Any) -> None:
                pass

            def show(self) -> None:
                pass

        class QLabel(QWidget):
            def setWordWrap(self, *_: Any, **__: Any) -> None:
                pass

            def setText(self, *_: Any, **__: Any) -> None:
                pass

        class QHBoxLayout:
            def __init__(self, *_: Any, **__: Any) -> None:
                pass

            def addWidget(self, *_: Any, **__: Any) -> None:
                pass

            def addLayout(self, *_: Any, **__: Any) -> None:
                pass

        class QVBoxLayout(QHBoxLayout):
            pass

        class QComboBox(QWidget):
            def setEditable(self, *_: Any, **__: Any) -> None:
                pass

            def addItem(self, *_: Any, **__: Any) -> None:
                pass

            def currentData(self) -> Any:
                return None

            def setEnabled(self, *_: Any, **__: Any) -> None:
                pass

        class QPlainTextEdit(QWidget):
            def setPlaceholderText(self, *_: Any, **__: Any) -> None:
                pass

            def setFixedHeight(self, *_: Any, **__: Any) -> None:
                pass

            def toPlainText(self) -> str:
                return ""

            def clear(self) -> None:
                pass

            def setEnabled(self, *_: Any, **__: Any) -> None:
                pass

        class QTextEdit(QPlainTextEdit):
            def setReadOnly(self, *_: Any, **__: Any) -> None:
                pass

        class QPushButton(QWidget):
            def __init__(self, *_: Any, **__: Any) -> None:
                self.clicked = Signal()

        class QTabWidget(QWidget):
            def addTab(self, *_: Any, **__: Any) -> None:
                pass

        class QDialog(QWidget):
            Accepted = 1

            def exec(self) -> int:
                raise RuntimeError("PySide6 is required for GUI operation")

        class QMessageBox:
            @staticmethod
            def information(*_: Any, **__: Any) -> None:
                pass

            @staticmethod
            def warning(*_: Any, **__: Any) -> None:
                pass

            @staticmethod
            def critical(*_: Any, **__: Any) -> None:
                pass

        class QApplication:
            def __init__(self, *_: Any, **__: Any) -> None:
                raise RuntimeError("PySide6 is required for GUI operation")

            def exec(self) -> int:  # pragma: no cover - stub
                return 0

LOGGER = logging.getLogger("drm.gui")


def _is_gui_environment_configured() -> bool:
    """Return True when the environment advertises GUI support."""
    if sys.platform.startswith("win"):
        return True
    if os.environ.get("QT_QPA_PLATFORM"):
        return True
    return any(os.environ.get(var) for var in ("DISPLAY", "WAYLAND_DISPLAY"))


def _probe_qt_initialisation() -> tuple[bool, Optional[str]]:
    """Check whether Qt can be initialised without crashing the main process."""
    if QApplication is None:
        return False, "PySide6 is unavailable"

    try:
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                "from PySide6.QtWidgets import QApplication\napp = QApplication([])",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            env=os.environ.copy(),
        )
    except Exception as exc:  # pragma: no cover - defensive
        return False, str(exc)

    if probe.returncode != 0:
        stderr_output = probe.stderr.decode().strip() or None
        return False, stderr_output
    return True, None


class TaskExecutionWorker(QObject):  # pragma: no cover - requires GUI runtime
    """Executes a task in a background thread."""

    finished = Signal(object)
    failed = Signal(object)

    def __init__(
        self,
        task_loop: LiveTaskLoop,
        task_text: str,
        workflow_override: Optional[str],
        human_feedback: Optional[str],
    ) -> None:
        super().__init__()
        self._task_loop = task_loop
        self._task_text = task_text
        self._workflow_override = workflow_override
        self._human_feedback = human_feedback

    @Slot()
    def run(self) -> None:
        """Execute the task and emit completion or failure."""
        try:
            outcome = self._task_loop.run_task(
                task=self._task_text,
                workflow_override=self._workflow_override,
                human_feedback=self._human_feedback,
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
        user_settings: Optional[UserSettingsManager] = None,
        config_path: Optional[Path] = None,
    ) -> None:
        super().__init__()
        self._config = config
        self._memory_manager = memory_manager
        self._controller = controller
        self._task_loop = task_loop
        self._worker_thread: Optional[QThread] = None
        self._current_worker: Optional[TaskExecutionWorker] = None
        self._recent_outcomes: List[TaskRunOutcome] = []
        self._latest_metrics: Dict[str, object] = {}
        self._drift_events: Deque[TelemetryEvent] = deque(maxlen=15)
        self._drift_analytics: Deque[Dict[str, object]] = deque(maxlen=30)
        self._review_history_buffer: List[Dict[str, object]] = []
        self._user_settings = user_settings
        self._config_path = config_path or Path("config/config.json")
        self.setWindowTitle("Dynamic Reflexive Memory")
        layout = QVBoxLayout(self)

        self._header_label = QLabel(
            "<b>Dynamic Reflexive Memory</b><br>"
            f"Default Workflow: {config.llm.default_workflow}<br>"
            f"Review Enabled: {config.review.enabled}"
        )
        self._header_label.setWordWrap(True)
        layout.addWidget(self._header_label)

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

        settings_button = QPushButton("Settings")
        cast(Any, settings_button.clicked).connect(self._open_settings_dialog)
        controls_row.addWidget(settings_button)

        if user_settings is not None:
            self._apply_saved_preferences(user_settings.settings)

        self._task_input = QPlainTextEdit()
        self._task_input.setPlaceholderText("Enter task prompt...")
        self._task_input.setFixedHeight(80)

        run_button = QPushButton("Run Task")
        cast(Any, run_button.clicked).connect(self._handle_run_task)
        self._run_button = run_button
        controls_row.addWidget(run_button)

        layout.addLayout(controls_row)
        layout.addWidget(self._task_input)

        layout.addWidget(QLabel("Optional human review feedback:"))
        self._feedback_input = QPlainTextEdit()
        self._feedback_input.setPlaceholderText(
            "Provide human notes or corrections to blend with automated review."
        )
        self._feedback_input.setFixedHeight(60)
        layout.addWidget(self._feedback_input)

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

        self._telemetry_tab = QTabWidget()
        self._metrics_view = QTextEdit()
        self._metrics_view.setReadOnly(True)
        self._metrics_view.setPlaceholderText("Awaiting telemetry metrics…")
        self._telemetry_tab.addTab(self._metrics_view, "Memory Metrics")

        self._drift_history_view = QTextEdit()
        self._drift_history_view.setReadOnly(True)
        self._drift_history_view.setPlaceholderText("No drift advisories yet.")
        self._telemetry_tab.addTab(self._drift_history_view, "Drift Advisories")

        self._drift_trend_view = QTextEdit()
        self._drift_trend_view.setReadOnly(True)
        self._drift_trend_view.setPlaceholderText("No drift analytics yet.")
        self._telemetry_tab.addTab(self._drift_trend_view, "Drift Trends")

        layout.addWidget(self._telemetry_tab)

        self._memory_view = QTextEdit()
        self._memory_view.setReadOnly(True)
        layout.addWidget(self._memory_view)

        self._drift_label = QLabel("No drift advisories yet.")
        self._drift_label.setWordWrap(True)
        layout.addWidget(self._drift_label)

        self._mitigation_label = QLabel("No mitigation actions recorded.")
        self._mitigation_label.setWordWrap(True)
        layout.addWidget(self._mitigation_label)

        self._bias_label = QLabel("Controller biases: none recorded.")
        self._bias_label.setWordWrap(True)
        layout.addWidget(self._bias_label)

        refresh_button = QPushButton("Refresh Memory Snapshot")
        cast(Any, refresh_button.clicked).connect(self._refresh_memory_snapshot)
        layout.addWidget(refresh_button)

        mitigation_button = QPushButton("Apply Drift Mitigation Now")
        cast(Any, mitigation_button.clicked).connect(self._handle_manual_mitigation)
        layout.addWidget(mitigation_button)

        self._refresh_memory_snapshot()

        self._telemetry_timer = QTimer(self)
        self._telemetry_timer.setInterval(1500)
        cast(Any, self._telemetry_timer.timeout).connect(self._poll_telemetry_feed)
        self._telemetry_timer.start()

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

        feedback_text = self._feedback_input.toPlainText().strip()
        human_feedback = feedback_text or None

        worker = TaskExecutionWorker(self._task_loop, task_text, workflow, human_feedback)
        thread = QThread(self)
        worker.moveToThread(thread)
        cast(Any, thread.started).connect(worker.run)
        cast(Any, worker.finished).connect(self._on_task_finished)
        cast(Any, worker.failed).connect(self._on_task_failed)
        cast(Any, worker.finished).connect(thread.quit)
        cast(Any, worker.failed).connect(thread.quit)
        cast(Any, worker.finished).connect(worker.deleteLater)
        cast(Any, worker.failed).connect(worker.deleteLater)
        cast(Any, thread.finished).connect(thread.deleteLater)

        self._current_worker = worker
        self._worker_thread = thread
        thread.start()

    def _handle_manual_mitigation(self) -> None:
        """Apply drift mitigation immediately and surface the outcome."""
        summary = self._memory_manager.apply_drift_mitigation()
        if summary:
            self._mitigation_label.setText(
                self._build_mitigation_label("Manual", summary)
            )
            actions = self._summarise_actions(summary)
            QMessageBox.information(
                self,
                "Mitigation Applied",
                f"Mitigation actions executed: {actions}",
            )
            self._refresh_memory_snapshot()
        else:
            self._mitigation_label.setText("No mitigation actions recorded.")
            QMessageBox.information(
                self,
                "Mitigation Applied",
                "No mitigation changes were required.",
            )

    def _on_task_finished(self, outcome: object) -> None:
        """Handle successful task completion."""
        self._set_interaction_enabled(True)
        self._status_label.setText("Task completed.")
        self._task_input.clear()
        self._feedback_input.clear()
        self._worker_thread = None
        self._current_worker = None

        if not isinstance(outcome, TaskRunOutcome):
            LOGGER.warning("Received unexpected outcome type: %s", type(outcome))
            return

        self._recent_outcomes.append(outcome)
        self._recent_outcomes = self._recent_outcomes[-5:]
        self._render_recent_outputs()
        self._refresh_memory_snapshot()

        if outcome.mitigation_summary:
            self._mitigation_label.setText(
                self._build_mitigation_label("Automatic", outcome.mitigation_summary)
            )
        else:
            self._mitigation_label.setText("No mitigation actions recorded.")

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
            analytics_records = self._memory_manager.list_drift_analytics(limit=15)
            revisions = self._memory_manager.get_revision_history(limit=5)
        except MemoryError as exc:
            LOGGER.error("Failed to load memory snapshot: %s", exc)
            self._memory_view.setPlainText(f"Unable to load memory snapshot: {exc}")
            self._review_history_view.setPlainText(f"Unable to load review history: {exc}")
            return

        drift_items: List[WorkingMemoryItem] = [
            item for item in working_items if item.key.endswith(":drift")
        ]
        drift_summary = self._format_drift_summary(drift_items)
        self._replace_review_records(reviews)
        self._update_review_history_view()

        self._drift_analytics.clear()
        for record in analytics_records:
            payload = {
                "id": record.id,
                "task_reference": record.task_reference,
                "workflow": record.workflow,
                "latency_seconds": record.latency_seconds,
                "verdict": record.verdict,
                "slo_breaches": list(record.slo_breaches),
                "drift_advisory": record.drift_advisory,
                "workflow_biases": record.workflow_biases,
                "mitigation_plan": record.mitigation_plan,
                "created_at": record.created_at,
            }
            self._drift_analytics.append(self._normalise_analytics_payload(payload))

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
            "=== Drift Analytics ===",
            *(
                self._format_analytics_line(record)
                for record in list(self._drift_analytics)[-5:]
            ),
            "=== Revision Log ===",
            *(self._format_revision_entry(entry) for entry in revisions),
        ]
        self._memory_view.setPlainText("\n".join(lines))

        self._drift_label.setText(drift_summary)
        self._bias_label.setText(self._format_bias_summary(self._controller.workflow_biases))
        try:
            metrics_snapshot = self._memory_manager.snapshot_metrics()
        except Exception as exc:  # pragma: no cover - defensive path
            LOGGER.debug("Unable to refresh metrics snapshot: %s", exc)
        else:
            self._latest_metrics = {
                "captured_at": datetime.now(timezone.utc).isoformat(),
                **metrics_snapshot,
            }
            self._render_memory_metrics()

        self._render_drift_history()
        self._render_drift_trends()

    @staticmethod
    def _format_working_item(item: WorkingMemoryItem) -> str:
        return f"{item.key}: {item.payload} (ttl={item.ttl_seconds}s)"

    @staticmethod
    def _format_generic_item(item: Dict[str, object]) -> str:
        identifier = item.get("id", "unknown")
        content = item.get("content") or item.get("definition") or item
        return f"{identifier}: {content}"

    @staticmethod
    def _format_revision_entry(entry: Dict[str, object]) -> str:
        revision = entry.get("revision", "?")
        layer = entry.get("layer", "unknown")
        identifier = entry.get("id", "unknown")
        timestamp = entry.get("timestamp", "n/a")
        return f"rev {revision} | {layer}:{identifier} @ {timestamp}"

    def _format_drift_summary(self, items: Sequence[WorkingMemoryItem]) -> str:
        if not items:
            return "<b>Drift Advisories:</b> None recorded."
        latest = max(items, key=lambda entry: entry.created_at)
        return (
            f"<b>Drift Advisories:</b> {len(items)} recorded. "
            f"Latest ({latest.created_at.isoformat()}): {latest.payload.get('advisory')}"
        )

    @staticmethod
    def _format_bias_summary(biases: Mapping[str, float]) -> str:
        if not biases:
            return "<b>Controller Biases:</b> None."
        segments = [f"{name}: {value:+.2f}" for name, value in sorted(biases.items())]
        return "<b>Controller Biases:</b> " + ", ".join(segments)

    @staticmethod
    def _summarise_actions(summary: Optional[Mapping[str, object]]) -> str:
        if not summary:
            return "None"
        parts = [f"{key}={summary[key]}" for key in sorted(summary.keys())]
        return ", ".join(parts)

    @staticmethod
    def _build_mitigation_label(kind: str, summary: Mapping[str, object]) -> str:
        actions = DRMWindow._summarise_actions(summary)
        return f"<b>{kind} Mitigation:</b> {actions}"

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
                    f"Mitigation: {self._summarise_actions(outcome.mitigation_summary) if outcome.mitigation_summary else 'None'}\n"
                    f"Output:\n{outcome.result.content}\n"
                    f"Review: verdict={outcome.review.verdict}, auto={outcome.review.auto_verdict}, "
                    f"quality={quality}, suggestions={suggestions}\n"
                )
            )
        self._recent_output_view.setPlainText("\n".join(lines))

    def _update_review_history_view(
        self,
        reviews: Optional[List[Dict[str, object]]] = None,
    ) -> None:
        """Render review history in the dedicated view."""
        if reviews is not None:
            self._replace_review_records(reviews)

        records = list(self._review_history_buffer)
        if not records and reviews is None:
            try:
                fetched = self._memory_manager.list_layer("review")
            except MemoryError as exc:
                LOGGER.error("Failed to load review history: %s", exc)
                self._review_history_view.setPlainText(f"Unable to load review history: {exc}")
                return
            self._replace_review_records(fetched)
            records = list(self._review_history_buffer)

        if not records:
            self._review_history_view.setPlainText("No review records yet.")
            return

        sorted_records = sorted(
            records,
            key=lambda payload: self._coerce_datetime(payload.get("created_at")),
            reverse=True,
        )[:5]
        lines: List[str] = []
        for record in sorted_records:
            notes_raw = record.get("notes")
            notes = str(notes_raw) if notes_raw is not None else "n/a"
            quality = record.get("quality_score")
            quality_text = f"{quality:.2f}" if isinstance(quality, (int, float)) else "n/a"
            suggestions_value = record.get("suggestions")
            if isinstance(suggestions_value, list) and suggestions_value:
                suggestions_text = "; ".join(str(item) for item in suggestions_value)
            else:
                suggestions_text = "None"
            created_at_display = record.get("created_at", "unknown")
            verdict_display = record.get("verdict", "n/a")
            auto_display = record.get("auto_verdict", "n/a")
            lines.append(
                (
                    f"{created_at_display} | verdict={verdict_display} "
                    f"(auto={auto_display})\n"
                    f"quality={quality_text} | suggestions={suggestions_text}\n"
                    f"notes: {notes}\n"
                )
            )
        self._review_history_view.setPlainText("\n".join(lines))

    def _poll_telemetry_feed(self) -> None:
        """Drain telemetry queue and refresh UI panels."""
        events = drain_telemetry(48)
        if not events:
            return

        metrics_updated = False
        drift_updated = False
        review_updated = False
        analytics_updated = False

        for event in events:
            if event.name == "memory.metrics":
                metrics_updated = True
                snapshot = dict(event.payload)
                snapshot["captured_at"] = event.timestamp.isoformat()
                self._latest_metrics = snapshot
            elif event.name == "controller.drift":
                drift_updated = True
                self._drift_events.append(event)
            elif event.name == "memory.drift.mitigation":
                drift_updated = True
                summary_payload = event.payload.get("summary")
                if isinstance(summary_payload, Mapping):
                    self._mitigation_label.setText(
                        self._build_mitigation_label(
                            "Mitigation",
                            cast(Mapping[str, object], summary_payload),
                        )
                    )
            elif event.name == "controller.performance":
                biases = event.payload.get("workflow_biases")
                if isinstance(biases, Mapping):
                    self._bias_label.setText(
                        self._format_bias_summary(cast(Mapping[str, float], biases))
                    )
            elif event.name == "controller.analytics.persisted":
                analytics_payload = event.payload.get("analytics")
                if isinstance(analytics_payload, Mapping):
                    self._drift_analytics.append(
                        self._normalise_analytics_payload(
                            cast(Mapping[str, object], analytics_payload)
                        )
                    )
                    analytics_updated = True
            elif event.name == "review.recorded":
                review_payload = event.payload.get("review")
                if isinstance(review_payload, dict):
                    review_updated = True
                    self._store_review_record(review_payload)

        if metrics_updated:
            self._render_memory_metrics()
        if drift_updated:
            self._render_drift_history()
            self._update_drift_label_from_events()
        if review_updated:
            self._update_review_history_view()
        if analytics_updated:
            self._render_drift_trends()

    def _render_memory_metrics(self) -> None:
        """Render the latest memory metrics snapshot."""
        if not self._latest_metrics:
            self._metrics_view.setPlainText("Awaiting telemetry metrics…")
            return

        metrics = self._latest_metrics
        captured = str(metrics.get("captured_at", "n/a"))
        working = metrics.get("working_items", "n/a")
        episodic = metrics.get("episodic_records", "n/a")
        semantic = metrics.get("semantic_records", "n/a")
        review = metrics.get("review_records", "n/a")
        drift = metrics.get("drift_advisories", "n/a")
        analytics = metrics.get("analytics_records", "n/a")
        lines = [
            f"Captured: {captured}",
            f"Working items: {working}",
            f"Episodic records: {episodic}",
            f"Semantic nodes: {semantic}",
            f"Review records: {review}",
            f"Drift advisories in working memory: {drift}",
            f"Drift analytics records: {analytics}",
        ]
        self._metrics_view.setPlainText("\n".join(lines))

    def _render_drift_history(self) -> None:
        """Render controller drift advisories."""
        if not self._drift_events:
            self._drift_history_view.setPlainText("No drift advisories yet.")
            return

        lines: List[str] = []
        for event in reversed(self._drift_events):
            timestamp = event.timestamp.isoformat()
            workflow = str(event.payload.get("workflow") or "unknown")
            advisory = str(event.payload.get("advisory") or "n/a")
            slo_raw = event.payload.get("slo_breaches")
            if isinstance(slo_raw, list) and slo_raw:
                slo_text = ", ".join(str(item) for item in slo_raw)
            else:
                slo_text = "none"
            lines.append(
                f"{timestamp} | workflow={workflow} | slo={slo_text}\n{advisory}"
            )
        self._drift_history_view.setPlainText("\n\n".join(lines))

    def _render_drift_trends(self) -> None:
        """Render controller drift analytics trends."""
        if not self._drift_analytics:
            self._drift_trend_view.setPlainText("No drift analytics yet.")
            return

        records = list(self._drift_analytics)
        latencies: List[float] = []
        for record in records:
            value = record.get("latency_seconds")
            try:
                latencies.append(float(value))
            except (TypeError, ValueError):
                continue

        average_latency: Optional[float] = None
        if latencies:
            average_latency = sum(latencies) / len(latencies)

        drift_hits = sum(1 for record in records if record.get("drift_advisory"))
        slo_total = 0
        for record in records:
            slo_value = record.get("slo_breaches")
            if isinstance(slo_value, (list, tuple, set)):
                slo_total += len(slo_value)

        summary_lines = [
            f"Records tracked: {len(records)}",
            (
                f"Average latency: {average_latency:.2f}s"
                if average_latency is not None
                else "Average latency: n/a"
            ),
            f"Runs with drift advisories: {drift_hits}",
            f"Cumulative SLO breaches: {slo_total}",
        ]

        latest_biases = records[-1].get("workflow_biases")
        if isinstance(latest_biases, Mapping) and latest_biases:
            bias_segments: List[str] = []
            for name, value in sorted(latest_biases.items()):
                try:
                    bias_segments.append(f"{name}:{float(value):+.2f}")
                except (TypeError, ValueError):
                    continue
            if bias_segments:
                summary_lines.append("Latest bias snapshot: " + ", ".join(bias_segments))

        summary_lines.append("")
        summary_lines.append("Recent records:")
        for record in reversed(records[-5:]):
            summary_lines.append(self._format_analytics_line(record))
            advisory = record.get("drift_advisory")
            if advisory:
                summary_lines.append(f"  advisory: {advisory}")
            mitigation = record.get("mitigation_plan")
            if isinstance(mitigation, Mapping) and mitigation:
                summary_lines.append(
                    "  mitigation: "
                    + self._summarise_actions(cast(Mapping[str, object], mitigation))
                )

        self._drift_trend_view.setPlainText("\n".join(summary_lines))

    @staticmethod
    def _format_analytics_line(record: Mapping[str, object]) -> str:
        created_at = record.get("created_at", "unknown")
        workflow = record.get("workflow", "unknown")
        latency_raw = record.get("latency_seconds")
        latency_text: str
        if isinstance(latency_raw, (int, float)):
            latency_text = f"{float(latency_raw):.2f}s"
        elif isinstance(latency_raw, str):
            try:
                latency_text = f"{float(latency_raw):.2f}s"
            except ValueError:
                latency_text = latency_raw
        else:
            latency_text = "n/a"

        verdict = record.get("verdict")
        verdict_text = str(verdict) if verdict is not None else "n/a"
        drift_flag = "yes" if record.get("drift_advisory") else "no"
        slo_value = record.get("slo_breaches")
        if isinstance(slo_value, (list, tuple, set)) and slo_value:
            slo_text = ", ".join(str(item) for item in slo_value)
        else:
            slo_text = "none"

        return (
            f"{created_at} | workflow={workflow} | latency={latency_text} | "
            f"verdict={verdict_text} | drift={drift_flag} | slo={slo_text}"
        )

    def _normalise_analytics_payload(
        self, payload: Mapping[str, object]
    ) -> Dict[str, object]:
        created_raw = payload.get("created_at")
        if isinstance(created_raw, datetime):
            created_at = created_raw.isoformat()
        elif isinstance(created_raw, str):
            created_at = created_raw
        else:
            created_at = datetime.now(timezone.utc).isoformat()

        slo_raw = payload.get("slo_breaches")
        if isinstance(slo_raw, (list, tuple, set)):
            slo_breaches = [str(item) for item in slo_raw]
        elif slo_raw:
            slo_breaches = [str(slo_raw)]
        else:
            slo_breaches = []

        biases_raw = payload.get("workflow_biases")
        workflow_biases: Dict[str, float] = {}
        if isinstance(biases_raw, Mapping):
            for name, value in biases_raw.items():
                try:
                    workflow_biases[str(name)] = float(value)
                except (TypeError, ValueError):
                    continue

        mitigation_raw = payload.get("mitigation_plan")
        if isinstance(mitigation_raw, Mapping):
            mitigation_plan = dict(mitigation_raw)
        elif mitigation_raw:
            mitigation_plan = {"value": mitigation_raw}
        else:
            mitigation_plan = {}

        latency_raw = payload.get("latency_seconds")
        try:
            latency = float(latency_raw)
        except (TypeError, ValueError):
            latency = None

        advisory_raw = payload.get("drift_advisory")
        drift_advisory = None
        if advisory_raw not in (None, "", "None"):
            drift_advisory = str(advisory_raw)

        return {
            "id": str(payload.get("id") or "unknown"),
            "task_reference": str(payload.get("task_reference") or "unknown"),
            "workflow": str(payload.get("workflow") or "unknown"),
            "latency_seconds": latency,
            "verdict": str(payload.get("verdict") or "unknown"),
            "slo_breaches": slo_breaches,
            "drift_advisory": drift_advisory,
            "workflow_biases": workflow_biases,
            "mitigation_plan": mitigation_plan,
            "created_at": created_at,
        }

    def _update_drift_label_from_events(self) -> None:
        """Show latest drift advisory in the headline label."""
        if not self._drift_events:
            return
        latest = self._drift_events[-1]
        advisory = str(latest.payload.get("advisory") or "None")
        workflow = latest.payload.get("workflow")
        workflow_part = f"[{workflow}] " if workflow else ""
        timestamp = latest.timestamp.isoformat()
        self._drift_label.setText(
            f"<b>Drift Advisory:</b> {workflow_part}{advisory} ({timestamp})"
        )

    def _store_review_record(self, record: Dict[str, object]) -> None:
        """Persist a review record in the local telemetry buffer."""
        record_id_raw = record.get("id")
        record_id = str(record_id_raw) if record_id_raw else None
        if not record_id:
            record_id = f"review:{len(self._review_history_buffer) + 1}"

        existing = [
            item
            for item in self._review_history_buffer
            if str(item.get("id") or "") != record_id
        ]
        stored = dict(record)
        stored["id"] = record_id
        created_at_value = stored.get("created_at")
        if isinstance(created_at_value, datetime):
            stored["created_at"] = created_at_value.isoformat()
        existing.append(stored)
        existing.sort(key=lambda payload: self._coerce_datetime(payload.get("created_at")))
        if len(existing) > 15:
            existing = existing[-15:]
        self._review_history_buffer = existing

    def _replace_review_records(self, records: Sequence[Dict[str, object]]) -> None:
        """Replace local review buffer with provided records."""
        self._review_history_buffer = []
        for record in records:
            if isinstance(record, dict):
                self._store_review_record(record)

    @staticmethod
    def _coerce_datetime(value: object) -> datetime:
        """Normalise values into timezone-aware datetimes."""
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        if isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value)
            except ValueError:
                return datetime.now(timezone.utc)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc)

    def _set_interaction_enabled(self, enabled: bool) -> None:
        """Enable or disable task interaction widgets."""
        self._workflow_selector.setEnabled(enabled)
        self._task_input.setEnabled(enabled)
        self._feedback_input.setEnabled(enabled)
        self._run_button.setEnabled(enabled)

    def _apply_saved_preferences(self, settings: UserSettings) -> None:
        """Restore workflow selection and window geometry from saved settings."""
        if settings.window_width and settings.window_height:
            self.resize(settings.window_width, settings.window_height)

        desired = settings.last_workflow
        index = self._workflow_selector.findData(desired)
        if index == -1 and desired is not None:
            index = self._workflow_selector.findData(None)
        if index != -1:
            self._workflow_selector.setCurrentIndex(index)

    def _open_settings_dialog(self) -> None:
        """Launch the settings dialog and apply a new configuration if saved."""
        if self._worker_thread and self._worker_thread.isRunning():
            QMessageBox.warning(
                self,
                "Task Running",
                "A task is currently executing. Wait for it to finish before opening settings.",
            )
            return

        dialog = SettingsDialog(self._config, self._config_path, parent=self)
        if dialog.exec() != QDialog.Accepted:
            return

        new_config = dialog.result_config()
        if new_config is None:
            return

        self._apply_new_config(new_config)

    def _apply_new_config(self, new_config: AppConfig) -> None:
        """Rebuild runtime components after a configuration update."""
        self._config = new_config
        self._controller = SelfAdjustingController(new_config)
        self._memory_manager = MemoryManager(new_config)
        self._task_loop = LiveTaskLoop(
            new_config,
            memory_manager=self._memory_manager,
            controller=self._controller,
            user_settings=self._user_settings,
        )

        self._header_label.setText(
            "<b>Dynamic Reflexive Memory</b><br>"
            f"Default Workflow: {new_config.llm.default_workflow}<br>"
            f"Review Enabled: {new_config.review.enabled}"
        )

        self._workflow_selector.blockSignals(True)
        self._workflow_selector.clear()
        self._workflow_selector.addItem(
            f"Auto ({new_config.llm.default_workflow})", userData=None
        )
        for name in new_config.llm.workflows.keys():
            self._workflow_selector.addItem(name, userData=name)
        self._workflow_selector.blockSignals(False)

        self._refresh_memory_snapshot()

    def closeEvent(self, event: Any) -> None:  # pragma: no cover - GUI runtime
        """Persist user preferences when the window closes."""
        timer = getattr(self, "_telemetry_timer", None)
        if timer is not None:
            try:
                timer.stop()
            except Exception:
                pass
        if self._user_settings:
            workflow_data = self._workflow_selector.currentData()
            workflow = workflow_data if isinstance(workflow_data, str) else None
            size = self.size()
            try:
                self._user_settings.update(
                    last_workflow=workflow,
                    window_width=size.width(),
                    window_height=size.height(),
                )
            except Exception as exc:
                LOGGER.warning("Failed to persist user settings on close: %s", exc)
        super().closeEvent(event)


def launch_gui(
    config: AppConfig,
    *,
    user_settings: Optional[UserSettingsManager] = None,
    config_path: Optional[Path] = None,
) -> Optional[int]:
    """Launch the PySide6 GUI; returns exit code or None if GUI unavailable."""
    if QApplication is None:
        LOGGER.error(
            "PySide6 is not installed; cannot launch GUI. "
            "Install PySide6 or use CLI mode."
        )
        return None

    if not _is_gui_environment_configured():
        LOGGER.warning(
            "No graphical display detected; skipping GUI launch. "
            "Set DISPLAY/WAYLAND_DISPLAY or QT_QPA_PLATFORM to enable GUI mode."
        )
        return None

    probe_ok, probe_error = _probe_qt_initialisation()
    if not probe_ok:
        details = f": {probe_error}" if probe_error else ""
        LOGGER.error(
            "Qt platform backend unavailable%s; falling back to CLI mode.", details
        )
        return None

    try:
        app = QApplication([])
    except Exception as exc:  # pragma: no cover - depends on Qt runtime
        LOGGER.error("Failed to initialise Qt application: %s", exc, exc_info=True)
        return None
    memory_manager = MemoryManager(config)
    controller = SelfAdjustingController(config)
    task_loop = LiveTaskLoop(
        config,
        memory_manager=memory_manager,
        controller=controller,
        user_settings=user_settings,
    )
    window = DRMWindow(
        config,
        memory_manager,
        controller,
        task_loop,
        user_settings=user_settings,
        config_path=config_path,
    )
    window.show()
    return app.exec()
