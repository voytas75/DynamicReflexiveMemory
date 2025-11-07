"""Dialog for editing DRM application settings within the GUI.

Updates:
    v0.1 - 2025-11-07 - Added tabbed configuration editor covering LLM, memory,
        review, embedding, and telemetry settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from config.settings import AppConfig, save_app_config
from core.exceptions import ConfigError


@dataclass(slots=True)
class _DialogResult:
    """Container for the dialog outcome."""

    config: AppConfig


class SettingsDialog(QDialog):
    """Tabbed dialog allowing edits to the DRM configuration."""

    def __init__(
        self,
        config: AppConfig,
        config_path,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Application Settings")
        self.resize(720, 560)
        self._original_config = config
        self._config_path = config_path
        self._result: Optional[_DialogResult] = None

        layout = QVBoxLayout(self)
        self._tabs = QTabWidget(self)
        layout.addWidget(self._tabs)

        self._build_llm_tab()
        self._build_memory_tab()
        self._build_review_tab()
        self._build_embedding_tab()
        self._build_telemetry_tab()

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._handle_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._populate_from_config()

    # region tab builders

    def _build_llm_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        form = QFormLayout()
        self._default_workflow_combo = QComboBox()
        form.addRow("Default workflow:", self._default_workflow_combo)

        self._workflows_table = QTableWidget(0, 4)
        self._workflows_table.setHorizontalHeaderLabels(
            ["Name", "Provider", "Model", "Temperature"]
        )
        self._workflows_table.horizontalHeader().setStretchLastSection(True)
        layout.addLayout(form)
        layout.addWidget(QLabel("Workflows"))
        layout.addWidget(self._workflows_table)

        workflow_buttons = QHBoxLayout()
        add_btn = QPushButton("Add workflow")
        add_btn.clicked.connect(self._add_workflow_row)
        remove_btn = QPushButton("Remove selected")
        remove_btn.clicked.connect(self._remove_selected_workflows)
        workflow_buttons.addWidget(add_btn)
        workflow_buttons.addWidget(remove_btn)
        workflow_buttons.addStretch()
        layout.addLayout(workflow_buttons)

        timeout_group = QGroupBox("Timeouts")
        timeout_layout = QFormLayout(timeout_group)
        self._timeout_request_spin = QSpinBox()
        self._timeout_request_spin.setRange(1, 3600)
        timeout_layout.addRow("Request seconds:", self._timeout_request_spin)
        self._timeout_attempts_spin = QSpinBox()
        self._timeout_attempts_spin.setRange(0, 10)
        timeout_layout.addRow("Retry attempts:", self._timeout_attempts_spin)
        self._timeout_backoff_spin = QSpinBox()
        self._timeout_backoff_spin.setRange(0, 600)
        timeout_layout.addRow("Retry backoff seconds:", self._timeout_backoff_spin)
        layout.addWidget(timeout_group)

        self._enable_debug_checkbox = QCheckBox("Enable LiteLLM debug logging")
        layout.addWidget(self._enable_debug_checkbox)

        layout.addStretch()
        self._tabs.addTab(tab, "LLM")

    def _build_memory_tab(self) -> None:
        tab = QWidget()
        layout = QGridLayout(tab)

        redis_group = QGroupBox("Redis (Working Memory)")
        redis_form = QFormLayout(redis_group)
        self._redis_host_edit = QLineEdit()
        redis_form.addRow("Host:", self._redis_host_edit)
        self._redis_port_spin = QSpinBox()
        self._redis_port_spin.setRange(0, 65535)
        redis_form.addRow("Port:", self._redis_port_spin)
        self._redis_db_spin = QSpinBox()
        self._redis_db_spin.setRange(0, 15)
        redis_form.addRow("Database:", self._redis_db_spin)
        self._redis_ttl_spin = QSpinBox()
        self._redis_ttl_spin.setRange(1, 86400)
        redis_form.addRow("TTL seconds:", self._redis_ttl_spin)

        chroma_group = QGroupBox("ChromaDB (Long-term Memory)")
        chroma_form = QFormLayout(chroma_group)
        self._chroma_path_edit = QLineEdit()
        chroma_form.addRow("Persist directory:", self._chroma_path_edit)
        self._chroma_collection_edit = QLineEdit()
        chroma_form.addRow("Collection:", self._chroma_collection_edit)

        layout.addWidget(redis_group, 0, 0)
        layout.addWidget(chroma_group, 0, 1)
        layout.setColumnStretch(1, 1)
        self._tabs.addTab(tab, "Memory")

    def _build_review_tab(self) -> None:
        tab = QWidget()
        form = QFormLayout(tab)
        self._review_enabled_checkbox = QCheckBox("Enable automated review")
        form.addRow(self._review_enabled_checkbox)
        self._review_model_edit = QLineEdit()
        form.addRow("Reviewer model:", self._review_model_edit)
        self._review_provider_edit = QLineEdit()
        form.addRow("Reviewer provider:", self._review_provider_edit)
        self._tabs.addTab(tab, "Review")

    def _build_embedding_tab(self) -> None:
        tab = QWidget()
        form = QFormLayout(tab)
        self._embedding_enabled_checkbox = QCheckBox("Use embedding provider")
        form.addRow(self._embedding_enabled_checkbox)
        self._embedding_provider_edit = QLineEdit()
        form.addRow("Embedding provider:", self._embedding_provider_edit)
        self._embedding_model_edit = QLineEdit()
        form.addRow("Embedding model:", self._embedding_model_edit)
        self._embedding_enabled_checkbox.toggled.connect(
            self._handle_embedding_toggle
        )
        self._tabs.addTab(tab, "Embedding")

    def _build_telemetry_tab(self) -> None:
        tab = QWidget()
        form = QFormLayout(tab)
        self._telemetry_level_combo = QComboBox()
        self._telemetry_level_combo.addItems(
            ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        )
        form.addRow("Log level:", self._telemetry_level_combo)
        self._tabs.addTab(tab, "Telemetry")

    # endregion

    def _populate_from_config(self) -> None:
        cfg = self._original_config

        self._default_workflow_combo.clear()
        self._default_workflow_combo.addItem(cfg.llm.default_workflow)
        for name in cfg.llm.workflows.keys():
            if name != cfg.llm.default_workflow:
                self._default_workflow_combo.addItem(name)

        workflows = cfg.llm.workflows
        self._workflows_table.setRowCount(0)
        for name, data in workflows.items():
            row = self._workflows_table.rowCount()
            self._workflows_table.insertRow(row)
            self._workflows_table.setItem(row, 0, QTableWidgetItem(name))
            self._workflows_table.setItem(row, 1, QTableWidgetItem(data.provider))
            self._workflows_table.setItem(row, 2, QTableWidgetItem(data.model))
            temp_item = QTableWidgetItem(f"{data.temperature:.2f}")
            temp_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._workflows_table.setItem(row, 3, temp_item)

        self._rebuild_default_workflow_choices()
        default_index = self._default_workflow_combo.findText(cfg.llm.default_workflow)
        if default_index != -1:
            self._default_workflow_combo.setCurrentIndex(default_index)

        timeouts = cfg.llm.timeouts
        self._timeout_request_spin.setValue(timeouts.request_seconds)
        self._timeout_attempts_spin.setValue(timeouts.retry_attempts)
        self._timeout_backoff_spin.setValue(timeouts.retry_backoff_seconds)
        self._enable_debug_checkbox.setChecked(cfg.llm.enable_debug)

        redis_cfg = cfg.memory.redis
        self._redis_host_edit.setText(redis_cfg.host)
        self._redis_port_spin.setValue(redis_cfg.port)
        self._redis_db_spin.setValue(redis_cfg.db)
        self._redis_ttl_spin.setValue(redis_cfg.ttl_seconds)

        chroma_cfg = cfg.memory.chromadb
        self._chroma_path_edit.setText(chroma_cfg.persist_directory)
        self._chroma_collection_edit.setText(chroma_cfg.collection)

        review_cfg = cfg.review
        self._review_enabled_checkbox.setChecked(review_cfg.enabled)
        self._review_model_edit.setText(review_cfg.auto_reviewer_model or "")
        self._review_provider_edit.setText(review_cfg.auto_reviewer_provider or "")

        embedding_cfg = cfg.embedding
        enabled = embedding_cfg is not None
        self._embedding_enabled_checkbox.setChecked(enabled)
        if embedding_cfg:
            self._embedding_provider_edit.setText(embedding_cfg.provider)
            self._embedding_model_edit.setText(embedding_cfg.model)
        self._handle_embedding_toggle(enabled)

        self._telemetry_level_combo.setCurrentText(cfg.telemetry.log_level.upper())

    # region actions

    def _add_workflow_row(self) -> None:
        row = self._workflows_table.rowCount()
        self._workflows_table.insertRow(row)
        self._workflows_table.setItem(row, 0, QTableWidgetItem("new-workflow"))
        self._workflows_table.setItem(row, 1, QTableWidgetItem("provider"))
        self._workflows_table.setItem(row, 2, QTableWidgetItem("model"))
        temp_item = QTableWidgetItem("0.20")
        temp_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._workflows_table.setItem(row, 3, temp_item)
        self._rebuild_default_workflow_choices()

    def _remove_selected_workflows(self) -> None:
        selected = self._workflows_table.selectionModel().selectedRows()
        for index in sorted(selected, key=lambda idx: idx.row(), reverse=True):
            self._workflows_table.removeRow(index.row())
        self._rebuild_default_workflow_choices()

    def _handle_embedding_toggle(self, enabled: bool) -> None:
        self._embedding_provider_edit.setEnabled(enabled)
        self._embedding_model_edit.setEnabled(enabled)

    # endregion

    def _handle_accept(self) -> None:
        try:
            self._rebuild_default_workflow_choices()
            new_config = self._collect_config()
            save_app_config(new_config, self._config_path)
        except (ValueError, ConfigError) as exc:
            QMessageBox.critical(self, "Unable to save settings", str(exc))
            return
        self._result = _DialogResult(config=new_config)
        self.accept()

    def result_config(self) -> Optional[AppConfig]:
        return self._result.config if self._result else None

    def _collect_config(self) -> AppConfig:
        workflows: Dict[str, Dict[str, object]] = {}
        for row in range(self._workflows_table.rowCount()):
            name_item = self._workflows_table.item(row, 0)
            provider_item = self._workflows_table.item(row, 1)
            model_item = self._workflows_table.item(row, 2)
            temp_item = self._workflows_table.item(row, 3)
            if not name_item or not name_item.text().strip():
                raise ValueError("Workflow name cannot be empty.")
            name = name_item.text().strip()
            if name in workflows:
                raise ValueError(f"Duplicate workflow name '{name}'.")
            provider = provider_item.text().strip() if provider_item else ""
            model = model_item.text().strip() if model_item else ""
            if not provider or not model:
                raise ValueError(f"Workflow '{name}' requires provider and model.")
            try:
                temperature = float(temp_item.text()) if temp_item else 0.0
            except ValueError as exc:
                raise ValueError(
                    f"Workflow '{name}' has invalid temperature value."
                ) from exc
            workflows[name] = {
                "provider": provider,
                "model": model,
                "temperature": temperature,
            }

        if not workflows:
            raise ValueError("At least one workflow must be defined.")

        default_workflow = self._default_workflow_combo.currentText().strip()
        if default_workflow not in workflows:
            raise ValueError(
                f"Default workflow '{default_workflow}' is not present in the workflow list."
            )

        llm_payload = {
            "default_workflow": default_workflow,
            "workflows": workflows,
            "timeouts": {
                "request_seconds": self._timeout_request_spin.value(),
                "retry_attempts": self._timeout_attempts_spin.value(),
                "retry_backoff_seconds": self._timeout_backoff_spin.value(),
            },
            "enable_debug": self._enable_debug_checkbox.isChecked(),
        }

        memory_payload = {
            "redis": {
                "host": self._redis_host_edit.text().strip(),
                "port": self._redis_port_spin.value(),
                "db": self._redis_db_spin.value(),
                "ttl_seconds": self._redis_ttl_spin.value(),
            },
            "chromadb": {
                "persist_directory": self._chroma_path_edit.text().strip(),
                "collection": self._chroma_collection_edit.text().strip(),
            },
        }

        review_payload = {
            "enabled": self._review_enabled_checkbox.isChecked(),
            "auto_reviewer_model": self._review_model_edit.text().strip() or None,
            "auto_reviewer_provider": self._review_provider_edit.text().strip() or None,
        }

        embedding_payload = None
        if self._embedding_enabled_checkbox.isChecked():
            provider = self._embedding_provider_edit.text().strip()
            model = self._embedding_model_edit.text().strip()
            if not provider or not model:
                raise ValueError(
                    "Embedding provider and model must be specified when embeddings are enabled."
                )
            embedding_payload = {
                "provider": provider,
                "model": model,
            }

        telemetry_payload = {
            "log_level": self._telemetry_level_combo.currentText().strip().upper(),
        }

        new_payload = {
            "version": self._original_config.version,
            "llm": llm_payload,
            "memory": memory_payload,
            "review": review_payload,
            "embedding": embedding_payload,
            "telemetry": telemetry_payload,
        }

        return AppConfig.model_validate(new_payload)

    def _rebuild_default_workflow_choices(self) -> None:
        """Refresh the default workflow combo to reflect the table contents."""
        names: list[str] = []
        for row in range(self._workflows_table.rowCount()):
            item = self._workflows_table.item(row, 0)
            if item and item.text().strip():
                names.append(item.text().strip())

        current = self._default_workflow_combo.currentText().strip()
        self._default_workflow_combo.blockSignals(True)
        self._default_workflow_combo.clear()
        for name in names:
            self._default_workflow_combo.addItem(name)
        if current and current in names:
            self._default_workflow_combo.setCurrentText(current)
        self._default_workflow_combo.blockSignals(False)
