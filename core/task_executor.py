"""LLM workflow executor with LiteLLM routing.

Updates:
    v0.1 - 2025-11-06 - Added workflow selection heuristic and LiteLLM integration
        scaffold with retries and telemetry hooks.
    v0.2 - 2025-11-06 - Wired provider credential handling for Azure and Ollama.
    v0.3 - 2025-11-06 - Integrated controller bias into workflow selection metadata.
"""

from __future__ import annotations

import logging
import time
from os import getenv
from typing import Optional

from config.settings import AppConfig
from core.controller import SelfAdjustingController
from core.exceptions import WorkflowError
from models.workflows import TaskRequest, TaskResult, WorkflowSelection

try:  # pragma: no cover - optional dependency
    import litellm  # type: ignore
except ImportError:  # pragma: no cover
    litellm = None

LOGGER = logging.getLogger("drm.executor")


class TaskExecutor:
    """Executes prompts through configured LLM workflows."""

    def __init__(
        self,
        config: AppConfig,
        controller: Optional[SelfAdjustingController] = None,
    ) -> None:
        self._config = config
        self._logger = LOGGER
        self._controller = controller

    def select_workflow(self, requested: Optional[str] = None) -> WorkflowSelection:
        """Select the best workflow given request metadata."""
        workflows = self._config.llm.workflows
        fallback = self._config.llm.default_workflow
        chosen = requested or fallback
        rationale = (
            "Requested workflow"
            if requested and chosen == requested
            else "Fallback to default configuration"
        )

        if chosen not in workflows:
            self._logger.warning(
                "Requested workflow '%s' unavailable; defaulting to '%s'.",
                chosen,
                fallback,
            )
            chosen = fallback
            rationale = "Fallback to default configuration"

        if self._controller and not requested:
            bias_choice, bias_reason = self._apply_controller_bias(chosen, workflows)
            if bias_choice != chosen:
                chosen = bias_choice
                rationale = bias_reason
            elif bias_reason:
                rationale = f"{rationale} ({bias_reason})"

        score = 0.8 if chosen == fallback else 1.0
        metadata = {}
        if self._controller:
            metadata["biases"] = self._controller.workflow_biases
        return WorkflowSelection(workflow=chosen, rationale=rationale, score=score, metadata=metadata)

    def _apply_controller_bias(self, current: str, workflows: dict) -> tuple[str, Optional[str]]:
        biases = self._controller.workflow_biases
        if not biases:
            return current, None

        best_workflow = current
        best_bias = biases.get(current, 0.0)
        for name in workflows.keys():
            bias = biases.get(name, 0.0)
            if bias > best_bias + 0.05:
                best_workflow = name
                best_bias = bias

        if best_workflow != current:
            reason = f"Controller preference bias={best_bias:.2f}"
            return best_workflow, reason

        if best_bias:
            return current, f"Controller bias retained ({best_bias:.2f})"

        return current, None

    def execute(self, request: TaskRequest) -> TaskResult:
        """Execute a task request via LiteLLM with retry handling."""
        if litellm is None:
            raise WorkflowError(
                "liteLLM is not installed. Install it to execute workflows."
            )

        workflows = self._config.llm.workflows
        if request.workflow not in workflows:
            raise WorkflowError(f"Workflow '{request.workflow}' is not configured.")

        workflow_cfg = workflows[request.workflow]
        timeout_cfg = self._config.llm.timeouts
        provider_kwargs = self._build_provider_kwargs(workflow_cfg.provider)

        attempt = 0
        delay = timeout_cfg.retry_backoff_seconds
        start_time = time.perf_counter()

        while attempt <= timeout_cfg.retry_attempts:
            attempt += 1
            try:
                self._logger.debug(
                    "Executing workflow '%s' attempt %s", request.workflow, attempt
                )
                response = litellm.completion(  # type: ignore[attr-defined]
                    model=workflow_cfg.model,
                    messages=[
                        {"role": "system", "content": request.context.get("system", "")},
                        {"role": "user", "content": request.prompt},
                    ],
                    temperature=workflow_cfg.temperature,
                    request_timeout=timeout_cfg.request_seconds,
                    **provider_kwargs,
                )
                content = response["choices"][0]["message"]["content"]
                latency = time.perf_counter() - start_time
                metadata = {
                    "usage": response.get("usage", {}),
                    "provider": workflow_cfg.provider,
                    "attempts": attempt,
                    "provider_kwargs": self._redact_sensitive(provider_kwargs),
                }
                return TaskResult(
                    workflow=request.workflow,
                    content=content,
                    latency_seconds=latency,
                    metadata=metadata,
                )
            except litellm.Timeout as exc:  # type: ignore[attr-defined]
                self._logger.warning(
                    "Workflow '%s' timed out (attempt %s): %s",
                    request.workflow,
                    attempt,
                    exc,
                )
            except Exception as exc:  # pragma: no cover - runtime failure
                self._logger.error(
                    "Workflow '%s' failed (attempt %s): %s",
                    request.workflow,
                    attempt,
                    exc,
                )
                last_exc = exc
                break

            time.sleep(delay)

        latency = time.perf_counter() - start_time
        message = f"Workflow '{request.workflow}' failed after {attempt} attempts."
        if "last_exc" in locals():
            raise WorkflowError(message) from last_exc  # type: ignore[misc]
        raise WorkflowError(message)

    def _build_provider_kwargs(self, provider: str) -> dict:
        """Prepare provider-specific keyword arguments for LiteLLM."""
        if provider.lower() == "azure":
            api_key = getenv("AZURE_OPENAI_API_KEY")
            endpoint = getenv("AZURE_OPENAI_ENDPOINT")
            api_version = getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
            if not api_key or not endpoint:
                raise WorkflowError(
                    "Azure OpenAI credentials missing. Set AZURE_OPENAI_API_KEY and "
                    "AZURE_OPENAI_ENDPOINT environment variables."
                )
            return {
                "api_key": api_key,
                "base_url": endpoint.rstrip("/"),
                "api_version": api_version,
            }

        if provider.lower() == "ollama":
            base_url = getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            return {"base_url": base_url.rstrip("/")}

        return {}

    @staticmethod
    def _redact_sensitive(payload: dict) -> dict:
        """Redact known sensitive values before logging metadata."""
        redacted = {}
        for key, value in payload.items():
            if "key" in key.lower():
                redacted[key] = "***redacted***"
            else:
                redacted[key] = value
        return redacted
