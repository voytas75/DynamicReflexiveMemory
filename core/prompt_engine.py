"""Adaptive prompt construction utilities for DRM workflows.

Updates:
    v0.1 - 2025-11-06 - Added AdaptivePromptEngine with memory-aware prompt
        composition and drift annotations.
    v0.2 - 2025-11-07 - Included semantic relation summaries in generated prompts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from textwrap import dedent
from typing import Dict, List, Optional, Sequence

from config.settings import AppConfig
from models.memory import ReviewRecord

LOGGER = logging.getLogger("drm.prompt")


@dataclass(slots=True)
class PromptContext:
    """Container for the dynamic context used to build prompts."""

    task: str
    workflow: str
    working_memory: Dict[str, object]
    episodic_memory: Sequence[Dict[str, object]]
    semantic_memory: Sequence[Dict[str, object]]
    recent_reviews: List[ReviewRecord]
    semantic_relations: Dict[str, List[Dict[str, object]]] = field(default_factory=dict)
    drift_indicator: Optional[str] = None


class AdaptivePromptEngine:
    """Composes prompts using memory artifacts and task metadata."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._logger = LOGGER

    def build_prompt(self, context: PromptContext) -> str:
        """Create a structured prompt tailored to the desired workflow."""
        self._logger.debug(
            "Building prompt for workflow %s with task '%s'",
            context.workflow,
            context.task,
        )
        prompt_sections = [
            self._format_header(context),
            self._format_memory_section("Working Memory", context.working_memory),
            self._format_list_section("Recent Episodes", context.episodic_memory),
            self._format_list_section("Semantic Concepts", context.semantic_memory),
            self._format_semantic_relations(context.semantic_relations),
            self._format_reviews(context.recent_reviews),
            "### Task Instruction",
            context.task.strip(),
        ]
        if context.drift_indicator:
            prompt_sections.append(
                dedent(
                    f"""
                    ### Drift Advisory
                    {context.drift_indicator.strip()}
                    """
                ).strip()
            )
        return "\n\n".join(section for section in prompt_sections if section)

    def _format_header(self, context: PromptContext) -> str:
        default_workflow = self._config.llm.default_workflow
        return dedent(
            f"""
            ### DRM Adaptive Prompt
            - Workflow: {context.workflow}
            - Default Workflow: {default_workflow}
            - Memory Units: working={len(context.working_memory)}, episodes={len(context.episodic_memory)}, semantics={len(context.semantic_memory)}
            """
        ).strip()

    @staticmethod
    def _format_memory_section(title: str, payload: Dict[str, object]) -> str:
        if not payload:
            return ""
        formatted_items = "\n".join(f"- {key}: {value}" for key, value in payload.items())
        return dedent(
            f"""
            ### {title}
            {formatted_items}
            """
        ).strip()

    @staticmethod
    def _format_list_section(
        title: str, items: Sequence[Dict[str, object]]
    ) -> str:
        if not items:
            return ""
        formatted_items = "\n".join(
            f"- {item.get('id', 'unknown')}: {item.get('content', item)}" for item in items
        )
        return dedent(
            f"""
            ### {title}
            {formatted_items}
            """
        ).strip()

    @staticmethod
    def _format_semantic_relations(
        relations: Dict[str, List[Dict[str, object]]]
    ) -> str:
        if not relations:
            return ""

        lines: List[str] = []
        for node_id, neighbours in relations.items():
            if not neighbours:
                continue
            neighbour_segments: List[str] = []
            for neighbour in neighbours:
                label = neighbour.get("label") or neighbour.get("id", "unknown")
                weight = neighbour.get("weight")
                if isinstance(weight, (int, float)):
                    neighbour_segments.append(f"{label} ({weight:.2f})")
                else:
                    neighbour_segments.append(str(label))
            if neighbour_segments:
                lines.append(f"- {node_id}: " + ", ".join(neighbour_segments))

        if not lines:
            return ""

        formatted = "\n".join(lines)
        return dedent(
            f"""
            ### Semantic Relations
            {formatted}
            """
        ).strip()

    @staticmethod
    def _format_reviews(reviews: List[ReviewRecord]) -> str:
        if not reviews:
            return ""
        formatted_lines = []
        for record in reviews:
            details = record.notes or "n/a"
            if record.quality_score is not None:
                details = f"{details} | quality={record.quality_score:.2f}"
            if record.suggestions:
                suggestions = "; ".join(record.suggestions)
                details = f"{details} | suggestions: {suggestions}"
            formatted_lines.append(
                f"- {record.created_at.isoformat()} | {record.verdict}: {details}"
            )
        formatted = "\n".join(formatted_lines)
        return dedent(
            f"""
            ### Review Feedback
            {formatted}
            """
        ).strip()
