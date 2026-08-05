"""Regression tests for the GUI runtime boundary and task completion state."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import gui.app as gui_app
from config.settings import AppConfig
from models.workflows import TaskRunOutcome


def test_partial_persistence_status_is_safe_and_actionable() -> None:
    outcome = cast(
        TaskRunOutcome,
        SimpleNamespace(
            persistence_status="partial",
            persistence_failures=("episodic:result", "review"),
        ),
    )

    status = gui_app._format_task_completion_status(outcome)

    assert status == (
        "Task completed with partial persistence. Remediate storage and retry "
        "(failed boundaries: episodic:result, review)."
    )


def test_qt_probe_initialises_a_real_offscreen_application(
    monkeypatch,
) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    configured = gui_app._is_gui_environment_configured()
    probe_ok, probe_error = gui_app._probe_qt_initialisation()

    assert configured is True
    assert probe_ok is True, probe_error


def test_launch_gui_falls_back_when_pyside_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(gui_app, "QApplication", None)

    result = gui_app.launch_gui(cast(AppConfig, object()))

    assert result is None
