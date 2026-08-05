"""LLM workflow executor with LiteLLM routing.

Updates:
    v0.1 - 2025-11-06 - Added workflow selection heuristic and LiteLLM integration
        scaffold with retries and telemetry hooks.
    v0.2 - 2025-11-06 - Wired provider credential handling for Azure and Ollama.
    v0.3 - 2025-11-06 - Integrated controller bias into workflow selection metadata.
    v0.4 - 2025-11-07 - Enabled optional LiteLLM debug toggling from configuration.
    v0.5 - 2025-11-07 - Normalised Azure provider routing for LiteLLM compatibility.
    v0.6 - 2025-11-07 - Prefixed Ollama models for LiteLLM provider resolution.
    v0.7 - 2025-11-07 - Added explicit Ollama provider hints for LiteLLM routing.
    v0.8 - 2025-11-07 - Emitted detailed error context when workflows fail.
    v0.9 - 2025-11-07 - Auto-detected Ollama base URL when running under WSL.
    v1.0 - 2026-08-05 - Made workflow failure exception state explicit for strict typing.
    v1.1 - 2026-08-05 - Redacted provider failure details from default logs and
        workflow exceptions.
    v1.2 - 2026-08-05 - Delegated Azure/Ollama routing to the shared provider
        routing seam.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Mapping, Optional, Tuple, cast

from config.settings import AppConfig, WorkflowModelConfig
from core.controller import SelfAdjustingController
from core.exceptions import WorkflowError
from core.provider_routing import (
    DEFAULT_OLLAMA_BASE as _DEFAULT_OLLAMA_BASE,
    ProviderRoutingError,
    resolve_ollama_base_url,
    resolve_provider_configuration,
    resolve_provider_model,
)
from models.workflows import TaskRequest, TaskResult, WorkflowSelection

try:  # pragma: no cover - optional dependency
    import litellm as _litellm
except ImportError:  # pragma: no cover
    _litellm = cast(Any, None)

litellm = cast(Any, _litellm)

LOGGER = logging.getLogger("drm.executor")


class TaskExecutor:
    """Executes prompts through configured LLM workflows."""

    DEFAULT_OLLAMA_BASE = _DEFAULT_OLLAMA_BASE

    def __init__(
        self,
        config: AppConfig,
        controller: Optional[SelfAdjustingController] = None,
    ) -> None:
        self._config = config
        self._logger = LOGGER
        self._controller = controller
        self._activate_litellm_debug()

    def select_workflow(self, requested: Optional[str] = None) -> WorkflowSelection:
        """Select the best workflow given request metadata."""
        workflows = self._config.llm.workflows
        fallback = self._config.llm.default_workflow
        if requested is not None:
            chosen = requested
        else:
            chosen = fallback
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
            biases = self._controller.workflow_biases
            bias_choice, bias_reason = self._apply_controller_bias(
                chosen, workflows, biases
            )
            if bias_choice != chosen:
                chosen = bias_choice
                rationale = bias_reason or "Controller preference override"
            elif bias_reason:
                rationale = f"{rationale} ({bias_reason})"

        score = 0.8 if chosen == fallback else 1.0
        metadata: Dict[str, object] = {}
        if self._controller and not requested:
            metadata["biases"] = dict(self._controller.workflow_biases)
        return WorkflowSelection(
            workflow=chosen,
            rationale=rationale,
            score=score,
            metadata=metadata,
        )

    def _apply_controller_bias(
        self,
        current: str,
        workflows: Mapping[str, object],
        biases: Mapping[str, float],
    ) -> Tuple[str, Optional[str]]:
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
        try:
            model_identifier, provider_kwargs = resolve_provider_configuration(
                workflow_cfg.provider, workflow_cfg.model
            )
        except ProviderRoutingError as exc:
            raise WorkflowError(str(exc)) from None

        attempt = 0
        delay = timeout_cfg.retry_backoff_seconds
        start_time = time.perf_counter()

        while attempt <= timeout_cfg.retry_attempts:
            attempt += 1
            try:
                self._logger.debug(
                    "Executing workflow '%s' attempt %s", request.workflow, attempt
                )
                response = litellm.completion(
                    model=model_identifier,
                    messages=[
                        {
                            "role": "system",
                            "content": request.context.get("system", ""),
                        },
                        {"role": "user", "content": request.prompt},
                    ],
                    temperature=workflow_cfg.temperature,
                    request_timeout=timeout_cfg.request_seconds,
                    **provider_kwargs,
                )
                content = response["choices"][0]["message"]["content"]
                latency = time.perf_counter() - start_time
                metadata: Dict[str, object] = {
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
            except litellm.Timeout as exc:
                self._logger.warning(
                    "Workflow '%s' timed out (attempt %s; error_type=%s).",
                    request.workflow,
                    attempt,
                    type(exc).__name__,
                )
            except Exception as exc:  # pragma: no cover - runtime failure
                self._logger.error(
                    "Workflow '%s' failed (attempt %s; error_type=%s).",
                    request.workflow,
                    attempt,
                    type(exc).__name__,
                )
                break

            time.sleep(delay)

        latency = time.perf_counter() - start_time
        message = f"Workflow '{request.workflow}' failed after {attempt} attempts."
        raise WorkflowError(message) from None

    def _build_provider_kwargs(self, provider: str) -> Dict[str, object]:
        """Prepare provider-specific keyword arguments for LiteLLM."""
        try:
            _, provider_kwargs = resolve_provider_configuration(provider, "")
        except ProviderRoutingError as exc:
            raise WorkflowError(str(exc)) from None
        return provider_kwargs

    @staticmethod
    def _redact_sensitive(payload: Dict[str, object]) -> Dict[str, object]:
        """Redact known sensitive values before logging metadata."""
        redacted: Dict[str, object] = {}
        for key, value in payload.items():
            if "key" in key.lower():
                redacted[key] = "***redacted***"
            else:
                redacted[key] = value
        return redacted

    def _activate_litellm_debug(self) -> None:
        """Enable LiteLLM debug logging when requested via configuration."""
        if not self._config.llm.enable_debug:
            return

        if litellm is None:
            self._logger.warning(
                "LiteLLM debug requested but the library is not installed."
            )
            return

        debug_hook = getattr(litellm, "_turn_on_debug", None)
        if callable(debug_hook):
            debug_hook()
            self._logger.info("LiteLLM debug logging enabled.")
        else:
            self._logger.warning(
                "LiteLLM debug requested but '_turn_on_debug' is unavailable on the library."
            )

    def _resolve_model_name(self, workflow_cfg: WorkflowModelConfig) -> str:
        """Normalise provider-specific model identifiers for LiteLLM."""
        return resolve_provider_model(workflow_cfg.provider, workflow_cfg.model)

    def _resolve_ollama_base_url(self) -> str:
        """Resolve the Ollama base URL with WSL host detection fallback."""
        return resolve_ollama_base_url()
