"""Coordinated live task loop orchestrating memory, execution, and review.

Updates:
    v0.1 - 2025-11-07 - Added LiveTaskLoop to persist task outputs, reviews, and drift advisories.
    v0.2 - 2025-11-07 - Normalised hydrated timestamps to timezone-aware UTC.
    v0.3 - 2025-11-06 - Persisted semantic summaries and integrated controller-aware
        task selection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

from config.settings import AppConfig
from core.controller import SelfAdjustingController
from core.exceptions import MemoryError
from core.memory_manager import MemoryManager
from core.prompt_engine import AdaptivePromptEngine, PromptContext
from core.review import ReviewEngine
from core.task_executor import TaskExecutor
from models.memory import (
    EpisodicMemoryEntry,
    ReviewRecord,
    SemanticNode,
    WorkingMemoryItem,
)
from models.workflows import TaskRequest, TaskResult, TaskRunOutcome, WorkflowSelection

LOGGER = logging.getLogger("drm.loop")


@dataclass(slots=True)
class _LayerSnapshot:
    episodic: List[dict]
    semantic: List[dict]
    reviews: List[ReviewRecord]


class LiveTaskLoop:
    """Drive the end-to-end execution cycle for DRM."""

    def __init__(
        self,
        config: AppConfig,
        memory_manager: Optional[MemoryManager] = None,
        prompt_engine: Optional[AdaptivePromptEngine] = None,
        executor: Optional[TaskExecutor] = None,
        review_engine: Optional[ReviewEngine] = None,
        controller: Optional[SelfAdjustingController] = None,
    ) -> None:
        self._config = config
        self._memory_manager = memory_manager or MemoryManager(config)
        self._prompt_engine = prompt_engine or AdaptivePromptEngine(config)
        self._controller = controller or SelfAdjustingController(config)
        self._executor = executor or TaskExecutor(config, controller=self._controller)
        self._review_engine = review_engine or ReviewEngine(config)
        self._logger = LOGGER

    def run_task(
        self,
        task: str,
        workflow_override: Optional[str] = None,
        human_feedback: Optional[str] = None,
        context_overrides: Optional[Dict[str, object]] = None,
    ) -> TaskRunOutcome:
        """Execute a task, persist artefacts, and return the run summary."""
        selection = self._executor.select_workflow(workflow_override)
        request_context = dict(context_overrides or {})
        request = TaskRequest(
            workflow=selection.workflow,
            prompt=task,
            context=request_context,
        )
        working_key_prefix = f"task:{request.task_id}"
        working_payload = {
            "task": task,
            "requested_workflow": workflow_override or "auto",
            "selected_workflow": selection.workflow,
            "selection_rationale": selection.rationale,
        }
        self._store_working_item(
            key=f"{working_key_prefix}:context",
            payload=working_payload,
        )

        layers = self._load_memory_snapshots()
        prompt_context = PromptContext(
            task=task,
            workflow=selection.workflow,
            working_memory=working_payload,
            episodic_memory=layers.episodic,
            semantic_memory=layers.semantic,
            recent_reviews=layers.reviews,
            drift_indicator=self._controller.last_advisory,
        )
        prompt = self._prompt_engine.build_prompt(prompt_context)
        request.prompt = prompt
        request.context.setdefault(
            "system",
            "You are the Dynamic Reflexive Memory task agent. "
            "Use supplied memory context to produce actionable, concise outputs.",
        )

        result = self._executor.execute(request)
        self._persist_result(
            request=request,
            selection=selection,
            result=result,
            compiled_prompt=prompt,
            user_task=task,
        )
        self._persist_semantic_summary(
            request=request,
            selection=selection,
            result=result,
            user_task=task,
        )

        review = self._review_engine.perform_review(
            request=request,
            result=result,
            human_feedback=human_feedback,
        )
        self._persist_review(review)

        drift_advisory = self._controller.register_result(selection, result, review)
        if drift_advisory:
            self._persist_drift_advisory(request.task_id, selection.workflow, drift_advisory)

        result_payload = {
            "workflow": result.workflow,
            "latency_seconds": result.latency_seconds,
            "metadata": result.metadata,
        }
        self._store_working_item(
            key=f"{working_key_prefix}:result",
            payload=result_payload,
        )

        return TaskRunOutcome(
            selection=selection,
            request=request,
            result=result,
            review=review,
            drift_advisory=drift_advisory,
        )

    def _persist_result(
        self,
        request: TaskRequest,
        selection: WorkflowSelection,
        result: TaskResult,
        *,
        compiled_prompt: str,
        user_task: str,
    ) -> None:
        metadata = {
            "compiled_prompt": compiled_prompt,
            "user_task": user_task,
            "workflow": selection.workflow,
            "latency_seconds": result.latency_seconds,
            "metadata": result.metadata,
        }
        entry = EpisodicMemoryEntry(
            id=request.task_id,
            content=result.content,
            metadata=metadata,
        )
        self._safe_record_episodic(entry)

    def _persist_review(self, review: ReviewRecord) -> None:
        try:
            self._memory_manager.record_review(review)
        except MemoryError as exc:
            self._logger.error("Failed to persist review %s: %s", review.id, exc)

    def _persist_semantic_summary(
        self,
        *,
        request: TaskRequest,
        selection: WorkflowSelection,
        result: TaskResult,
        user_task: str,
    ) -> None:
        """Create a lightweight semantic summary node derived from the task outcome."""
        if not result.content.strip():
            return

        label = self._build_semantic_label(user_task)
        definition = self._build_semantic_definition(result)
        if not definition:
            return

        node = SemanticNode(
            id=f"concept:{request.task_id}",
            label=label,
            definition=definition,
            sources=[selection.workflow],
            relations={f"workflow:{selection.workflow}": selection.score},
        )
        try:
            self._memory_manager.record_semantic(node)
        except MemoryError as exc:
            self._logger.error(
                "Failed to persist semantic concept %s: %s",
                node.id,
                exc,
            )

    @staticmethod
    def _build_semantic_label(user_task: str) -> str:
        summary = user_task.strip()
        if len(summary) <= 80:
            return summary
        return f"{summary[:77]}…"

    @staticmethod
    def _build_semantic_definition(result: TaskResult) -> str:
        snippet = result.content.strip().splitlines()[0] if result.content.strip() else ""
        snippet = snippet.strip()
        if not snippet:
            return ""
        if len(snippet) <= 240:
            return snippet
        return f"{snippet[:237]}…"

    def _persist_drift_advisory(
        self, task_id: str, workflow: str, advisory: str
    ) -> None:
        metadata = {
            "task_reference": task_id,
            "workflow": workflow,
            "type": "drift_advisory",
        }
        entry = EpisodicMemoryEntry(
            id=f"drift:{task_id}",
            content=advisory,
            metadata=metadata,
        )
        self._safe_record_episodic(entry)
        self._store_working_item(
            key=f"task:{task_id}:drift",
            payload=metadata | {"advisory": advisory},
        )

    def _store_working_item(self, key: str, payload: Dict[str, object]) -> None:
        item = WorkingMemoryItem(
            key=key,
            payload=payload,
            ttl_seconds=self._config.memory.redis.ttl_seconds,
        )
        try:
            self._memory_manager.put_working_item(item)
        except MemoryError as exc:
            self._logger.error("Failed to persist working memory item %s: %s", key, exc)

    def _load_memory_snapshots(self, limit: int = 5) -> _LayerSnapshot:
        episodic = self._safe_layer_slice("episodic", limit)
        semantic = self._safe_layer_slice("semantic", limit)
        reviews = self._load_recent_reviews(limit)
        return _LayerSnapshot(episodic=episodic, semantic=semantic, reviews=reviews)

    def _safe_layer_slice(self, layer: str, limit: int) -> List[dict]:
        try:
            items = self._memory_manager.list_layer(layer)
        except MemoryError as exc:
            self._logger.warning("Unable to access %s memory: %s", layer, exc)
            return []
        if not items:
            return []
        sorted_items = sorted(
            items,
            key=lambda item: item.get("timestamp") or item.get("created_at") or "",
        )
        return sorted_items[-limit:]

    def _load_recent_reviews(self, limit: int) -> List[ReviewRecord]:
        try:
            items = self._memory_manager.list_layer("review")
        except MemoryError as exc:
            self._logger.warning("Unable to access review memory: %s", exc)
            return []
        hydrated: List[ReviewRecord] = []
        for payload in items:
            record = self._hydrate_review(payload)
            if record:
                hydrated.append(record)
        hydrated.sort(key=lambda record: record.created_at)
        return hydrated[-limit:]

    def _hydrate_review(self, payload: Dict[str, object]) -> Optional[ReviewRecord]:
        try:
            data = dict(payload)
            if "created_at" in data:
                data["created_at"] = self._coerce_timestamp(data["created_at"])
            data.setdefault("suggestions", [])
            data.setdefault("quality_score", None)
            data.setdefault("auto_verdict", None)
            data.setdefault("notes", None)
            return ReviewRecord(**data)
        except Exception as exc:  # pragma: no cover - defensive path
            self._logger.debug("Failed to hydrate review payload %s: %s", payload, exc)
            return None

    def _safe_record_episodic(self, entry: EpisodicMemoryEntry) -> None:
        try:
            self._memory_manager.record_episodic(entry)
        except MemoryError as exc:
            self._logger.error("Failed to persist episodic memory %s: %s", entry.id, exc)

    @staticmethod
    def _coerce_timestamp(value: object) -> datetime:
        """Convert timestamps to timezone-aware UTC values."""
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        if isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value)
            except ValueError:
                return datetime.now(timezone.utc)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc)
