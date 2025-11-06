"""Hybrid review engine integrating automated checks.

Updates:
    v0.1 - 2025-11-06 - Added ReviewEngine with optional LiteLLM-based automated
        audits and structured review records.
    v0.2 - 2025-11-06 - Expanded automated review rubric and context framing.
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

import json

from config.settings import AppConfig
from core.exceptions import ReviewError
from models.memory import ReviewRecord
from models.workflows import TaskRequest, TaskResult

try:  # pragma: no cover - optional dependency
    import litellm  # type: ignore
except ImportError:  # pragma: no cover
    litellm = None

LOGGER = logging.getLogger("drm.review")

REVIEW_SYSTEM_PROMPT = (
    "You are the DRM audit agent. Examine task requests and outputs for logic, "
    "factuality, redundancy, and policy alignment. Respond using the format:\n"
    "VERDICT: PASS|FAIL\n"
    "REASONING: <2 sentence justification>\n"
    "QUALITY_SCORE: <0-1 float>\n"
    "SUGGESTIONS: bullet list of corrections or improvements."
)


class ReviewEngine:
    """Performs automated and human-in-the-loop reviews."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._logger = LOGGER

    def perform_review(
        self,
        request: TaskRequest,
        result: TaskResult,
        human_feedback: Optional[str] = None,
    ) -> ReviewRecord:
        """Run automated audit and merge with optional human feedback."""
        if not self._config.review.enabled:
            return self._build_record("skipped", "Review disabled via configuration.")

        auto_notes = self._run_automated_review(request, result)
        notes = auto_notes
        verdict = "pass"

        if human_feedback:
            notes = f"{auto_notes or ''}\nHuman feedback: {human_feedback}".strip()
            if human_feedback.lower().startswith(("fail", "reject")):
                verdict = "fail-human"

        return self._build_record(
            task_reference=request.task_id,
            verdict=verdict,
            notes=notes or "No issues detected.",
        )

    def _run_automated_review(
        self, request: TaskRequest, result: TaskResult
    ) -> Optional[str]:
        model = self._config.review.auto_reviewer_model
        if not model:
            self._logger.debug("Automated review skipped; no model configured.")
            return None
        if litellm is None:
            raise ReviewError(
                "liteLLM is required for automated review but is not installed."
            )

        try:
            self._logger.debug("Running automated review with model %s", model)
            review_payload = json.dumps(
                {
                    "task_prompt": request.prompt,
                    "workflow": request.workflow,
                    "context": request.context,
                    "result": result.content,
                    "metadata": result.metadata,
                },
                ensure_ascii=False,
                indent=2,
            )
            response = litellm.completion(  # type: ignore[attr-defined]
                model=model,
                messages=[
                    {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            "Evaluate the provided task exchange. Use the rubric and "
                            "return a structured audit.\n\n"
                            f"```json\n{review_payload}\n```"
                        ),
                    },
                ],
                temperature=0.0,
                request_timeout=self._config.llm.timeouts.request_seconds,
            )
            auto_notes = response["choices"][0]["message"]["content"]
            return auto_notes
        except litellm.Timeout as exc:  # type: ignore[attr-defined]
            self._logger.warning("Automated review timeout: %s", exc)
            return "Automated review timed out."
        except Exception as exc:  # pragma: no cover - runtime failure
            raise ReviewError(f"Automated review failed: {exc}") from exc

    @staticmethod
    def _build_record(task_reference: str, verdict: str, notes: str) -> ReviewRecord:
        return ReviewRecord(
            id=str(uuid.uuid4()),
            task_reference=task_reference,
            verdict=verdict,
            notes=notes,
        )
