"""Hybrid review engine integrating automated checks.

Updates:
    v0.1 - 2025-11-06 - Added ReviewEngine with optional LiteLLM-based automated audits and structured review records.
    v0.2 - 2025-11-06 - Expanded automated review rubric and context framing.
    v0.3 - 2025-11-07 - Parsed automated review verdict, score, and suggestions into structured fields.
    v0.4 - 2025-11-07 - Honoured LiteLLM debug toggle for automated reviews.
    v0.5 - 2025-11-07 - Normalised metadata serialisation for automated review payloads.
    v0.6 - 2025-11-07 - Applied provider-aware routing for automated review models.
    v0.7 - 2025-11-07 - Logged automated review failures before surfacing to callers.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass
from os import getenv
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, cast

from config.settings import AppConfig
from core.exceptions import ReviewError
from models.memory import ReviewRecord
from models.workflows import TaskRequest, TaskResult

try:  # pragma: no cover - optional dependency
    import litellm as _litellm
except ImportError:  # pragma: no cover
    _litellm = None

litellm = cast(Any, _litellm)

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
        self._activate_litellm_debug()

    def perform_review(
        self,
        request: TaskRequest,
        result: TaskResult,
        human_feedback: Optional[str] = None,
    ) -> ReviewRecord:
        """Run automated audit and merge with optional human feedback."""
        if not self._config.review.enabled:
            return self._build_record(
                task_reference=request.task_id,
                verdict="skipped",
                notes="Review disabled via configuration.",
                quality_score=None,
                suggestions=[],
                auto_verdict=None,
            )

        automated = self._run_automated_review(request, result)
        notes_parts: List[str] = []
        quality_score = automated.quality_score if automated else None
        suggestions = automated.suggestions if automated else []
        auto_verdict = automated.verdict if automated else None

        if automated and automated.reasoning:
            notes_parts.append(automated.reasoning)
        elif automated and automated.raw_text:
            notes_parts.append(automated.raw_text)

        verdict = self._normalise_verdict(auto_verdict)

        if human_feedback:
            notes_parts.append(f"Human feedback: {human_feedback}")
            if human_feedback.lower().startswith(("fail", "reject")):
                verdict = "fail-human"

        notes = "\n".join(part for part in notes_parts if part) or "No issues detected."

        return self._build_record(
            task_reference=request.task_id,
            verdict=verdict,
            notes=notes,
            quality_score=quality_score,
            suggestions=suggestions,
            auto_verdict=auto_verdict,
        )

    def _run_automated_review(
        self, request: TaskRequest, result: TaskResult
    ) -> Optional["AutomatedReview"]:
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
            model_name, provider_kwargs = self._resolve_model_configuration()
            payload = {
                "task_prompt": request.prompt,
                "workflow": request.workflow,
                "context": self._to_json_safe(request.context),
                "result": result.content,
                "metadata": self._to_json_safe(result.metadata),
            }
            review_payload = json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            )
            response = litellm.completion(
                model=model_name,
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
                **provider_kwargs,
            )
            auto_notes = response["choices"][0]["message"]["content"]
            parsed = self._parse_automated_review(auto_notes)
            return parsed
        except litellm.Timeout as exc:
            self._logger.warning("Automated review timeout: %s", exc)
            return AutomatedReview(
                verdict="timeout",
                quality_score=None,
                suggestions=["Automated review timed out."],
                reasoning="Automated review timed out.",
                raw_text="Automated review timed out.",
            )
        except Exception as exc:  # pragma: no cover - runtime failure
            self._logger.error(
                "Automated review failed for task %s: %s",
                request.task_id,
                exc,
                exc_info=True,
            )
            raise ReviewError(f"Automated review failed: {exc}") from exc

    @staticmethod
    def _build_record(
        task_reference: str,
        verdict: str,
        notes: str,
        quality_score: Optional[float],
        suggestions: Sequence[str],
        auto_verdict: Optional[str],
    ) -> ReviewRecord:
        return ReviewRecord(
            id=str(uuid.uuid4()),
            task_reference=task_reference,
            verdict=verdict,
            notes=notes,
            quality_score=quality_score,
            suggestions=list(suggestions),
            auto_verdict=auto_verdict,
        )

    @staticmethod
    def _normalise_verdict(raw_verdict: Optional[str]) -> str:
        if not raw_verdict:
            return "pass"
        verdict = raw_verdict.strip().lower()
        if verdict in {"pass", "passed", "approve"}:
            return "pass"
        if verdict in {"fail", "failed", "reject"}:
            return "fail-auto"
        return verdict

    def _activate_litellm_debug(self) -> None:
        """Enable LiteLLM debug logging for automated review when configured."""
        if not self._config.llm.enable_debug:
            return

        if litellm is None:
            self._logger.warning(
                "LiteLLM debug requested for reviews but the library is not installed."
            )
            return

        debug_hook = getattr(litellm, "_turn_on_debug", None)
        if callable(debug_hook):
            debug_hook()
            self._logger.info("LiteLLM debug logging enabled for review engine.")
        else:
            self._logger.warning(
                "LiteLLM debug requested but '_turn_on_debug' is unavailable on the library."
            )

    @staticmethod
    def _to_json_safe(value: Any) -> Any:
        """Convert the value into JSON-serialisable primitives."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, Mapping):
            return {str(key): ReviewEngine._to_json_safe(item) for key, item in value.items()}

        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [ReviewEngine._to_json_safe(item) for item in value]

        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            return ReviewEngine._to_json_safe(model_dump())

        if hasattr(value, "__dict__"):
            return ReviewEngine._to_json_safe(vars(value))

        return str(value)

    def _resolve_model_configuration(self) -> Tuple[str, Dict[str, object]]:
        """Return the provider-aware model identifier and kwargs for LiteLLM."""
        model_name = self._config.review.auto_reviewer_model
        if not model_name:
            raise ReviewError("Automated review model is not configured.")

        provider = self._config.review.auto_reviewer_provider
        if not provider:
            default = self._config.llm.default_workflow
            default_cfg = self._config.llm.workflows.get(default)
            provider = default_cfg.provider if default_cfg else None

        if not provider:
            return model_name, {}

        provider_lower = provider.lower()
        if provider_lower == "azure":
            api_key = getenv("AZURE_OPENAI_API_KEY")
            endpoint = getenv("AZURE_OPENAI_ENDPOINT")
            api_version = getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
            if not api_key or not endpoint:
                raise ReviewError(
                    "Azure OpenAI credentials missing for automated review."
                )
            base = endpoint.rstrip("/")
            if not model_name.startswith("azure/"):
                model_name = f"azure/{model_name}"
            return model_name, {
                "api_key": api_key,
                "api_base": base,
                "base_url": base,
                "api_version": api_version,
                "custom_llm_provider": "azure",
            }

        if provider_lower == "ollama":
            base_url = getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            return model_name, {"base_url": base_url.rstrip("/")}

        return model_name, {}

    def _parse_automated_review(self, content: str) -> "AutomatedReview":
        lines = [line.rstrip() for line in content.splitlines()]
        verdict: Optional[str] = None
        quality_score: Optional[float] = None
        suggestions: List[str] = []
        reasoning_chunks: List[str] = []
        collecting_suggestions = False

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            upper = line.upper()
            if upper.startswith("VERDICT"):
                verdict = line.split(":", 1)[1].strip() if ":" in line else line[7:].strip()
                collecting_suggestions = False
                continue
            if upper.startswith("QUALITY_SCORE"):
                value = line.split(":", 1)[1].strip() if ":" in line else ""
                quality_score = self._extract_float(value)
                collecting_suggestions = False
                continue
            if upper.startswith("SUGGESTIONS"):
                remainder = line.split(":", 1)[1].strip() if ":" in line else ""
                if remainder:
                    suggestions.append(self._normalise_bullet(remainder))
                collecting_suggestions = True
                continue
            if upper.startswith("REASONING"):
                remainder = line.split(":", 1)[1].strip() if ":" in line else ""
                if remainder:
                    reasoning_chunks.append(remainder)
                collecting_suggestions = False
                continue
            if collecting_suggestions and raw_line.lstrip().startswith(("-", "*")):
                suggestions.append(self._normalise_bullet(raw_line))
                continue
            if collecting_suggestions and re.match(r"^\d+[\.\)]\s+", raw_line.lstrip()):
                suggestions.append(self._normalise_bullet(raw_line))
                continue
            if reasoning_chunks:
                reasoning_chunks.append(line)
            else:
                reasoning_chunks.append(line)

        reasoning = "\n".join(reasoning_chunks).strip() or None
        return AutomatedReview(
            verdict=verdict,
            quality_score=quality_score,
            suggestions=suggestions,
            reasoning=reasoning,
            raw_text=content,
        )

    @staticmethod
    def _extract_float(value: str) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            match = re.search(r"[-+]?\d*\.?\d+", value or "")
            if match:
                try:
                    return float(match.group())
                except ValueError:
                    return None
        return None

    @staticmethod
    def _normalise_bullet(line: str) -> str:
        cleaned = line.strip()
        cleaned = re.sub(r"^[\-\*\d\.\)\s]+", "", cleaned)
        return cleaned.strip()


@dataclass(slots=True)
class AutomatedReview:
    """Structured representation of automated review output."""

    verdict: Optional[str]
    quality_score: Optional[float]
    suggestions: List[str]
    reasoning: Optional[str]
    raw_text: str
