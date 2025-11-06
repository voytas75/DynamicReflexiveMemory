"""Adaptive prompt construction utilities for DRM workflows.

Updates: v0.1 - 2025-11-06 - Added AdaptivePromptEngine with memory-aware prompt
composition and drift annotations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from textwrap import dedent
from typing import Dict, List, Optional

from config.settings import AppConfig
from models.memory import ReviewRecord

LOGGER = logging.getLogger("drm.prompt")


@dataclass(slots=True)
class PromptContext:
    """Container for the dynamic context used to build prompts."""

    task: str
    workflow: str
    working_memory: Dict[str, object]
    episodic_memory: List[dict]
    semantic_memory: List[dict]
    recent_reviews: List[ReviewRecord]
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
    def _format_list_section(title: str, items: List[dict]) -> str:
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
    def _format_reviews(reviews: List[ReviewRecord]) -> str:
        if not reviews:
            return ""
        formatted = "\n".join(
            f"- {record.created_at.isoformat()} | {record.verdict}: {record.notes or 'n/a'}"
            for record in reviews
        )
        return dedent(
            f"""
            ### Review Feedback
            {formatted}
            """
        ).strip()

