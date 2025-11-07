"""Entry point for the Dynamic Reflexive Memory application.

Updates:
    v0.1 - 2025-11-06 - Added CLI/GUI bootstrap with configuration loading, logging setup, and sample task execution pipeline.
    v0.2 - 2025-11-07 - Routed CLI execution through the LiveTaskLoop orchestrator.
    v0.3 - 2025-11-06 - Added startup health checks with warning surface.
    v0.4 - 2025-11-07 - Loaded environment variables from .env during startup.
    v0.5 - 2025-11-07 - Captured optional human review feedback in CLI workflows.
"""

from __future__ import annotations

import argparse
import logging
import logging.config
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from config.settings import AppConfig, get_app_config, resolve_config_path
from core.exceptions import DRMError, HealthCheckError, WorkflowError
from core.live_loop import LiveTaskLoop
from core.health import run_startup_checks
from gui.app import launch_gui


def setup_logging(logging_config: Optional[Path] = None) -> None:
    """Configure Python logging using the provided configuration file."""
    config_path = logging_config or Path(__file__).parent / "config" / "logging.conf"
    if not config_path.exists():
        logging.basicConfig(level=logging.INFO)
        logging.getLogger("drm").warning(
            "Logging configuration %s not found; using basicConfig.", config_path
        )
        return
    logging.config.fileConfig(config_path, disable_existing_loggers=False)


def run_cli(
    config: AppConfig,
    task: Optional[str] = None,
    workflow: Optional[str] = None,
    human_feedback: Optional[str] = None,
) -> None:
    """Run a simple CLI workflow as a fallback when GUI is unavailable."""
    logger = logging.getLogger("drm.cli")
    task_loop = LiveTaskLoop(config)
    prompt_text = task or "Summarise today's objectives based on existing memory."
    try:
        outcome = task_loop.run_task(
            task=prompt_text,
            workflow_override=workflow,
            human_feedback=human_feedback,
        )
    except DRMError as exc:
        if isinstance(exc, WorkflowError):
            logger.warning(
                "Workflow execution unavailable: %s. The prompt can be sent manually.",
                exc,
            )
        else:
            logger.error("Task execution failed: %s", exc)
        return

    logger.info(
        "Executed workflow '%s' (reason: %s, score=%.2f)",
        outcome.selection.workflow,
        outcome.selection.rationale,
        outcome.selection.score,
    )
    logger.info("Compiled prompt:\n%s", outcome.request.prompt)
    logger.info("Task result (%s): %s", outcome.result.workflow, outcome.result.content)
    logger.info(
        "Review verdict: %s (auto=%s, quality=%s)",
        outcome.review.verdict,
        outcome.review.auto_verdict,
        f"{outcome.review.quality_score:.2f}" if outcome.review.quality_score is not None else "n/a",
    )
    if outcome.review.suggestions:
        logger.info("Review suggestions: %s", "; ".join(outcome.review.suggestions))
    logger.info("Review notes: %s", outcome.review.notes)
    if human_feedback:
        logger.info("Human feedback applied: %s", human_feedback)
    if outcome.drift_advisory:
        logger.warning("Controller advisory: %s", outcome.drift_advisory)
    if outcome.mitigation_summary:
        logger.info("Mitigation actions: %s", outcome.mitigation_summary)


def main(argv: Optional[list[str]] = None) -> int:
    """Parse CLI arguments and dispatch the selected mode."""
    load_dotenv()

    parser = argparse.ArgumentParser(description="Dynamic Reflexive Memory runner.")
    parser.add_argument("--mode", choices=["gui", "cli"], default="gui")
    parser.add_argument("--config", type=Path, help="Path to configuration file.")
    parser.add_argument("--task", type=str, help="Task prompt for CLI mode.")
    parser.add_argument("--workflow", type=str, help="Workflow override for execution.")
    parser.add_argument("--logging-config", type=Path, help="Path to logging configuration.")
    parser.add_argument("--feedback", type=str, help="Optional human review feedback for the task.")
    args = parser.parse_args(argv)

    try:
        config_path = resolve_config_path(args.config) if args.config else None
        config = get_app_config(config_path)
        setup_logging(args.logging_config)
        warnings = run_startup_checks(config)
        for warning in warnings:
            logging.getLogger("drm").warning("Startup check: %s", warning)
    except HealthCheckError as exc:
        logging.basicConfig(level=logging.ERROR)
        logging.getLogger("drm").error("Startup health check failed: %s", exc)
        return 1
    except DRMError as exc:
        logging.basicConfig(level=logging.ERROR)
        logging.getLogger("drm").error("Startup failed: %s", exc)
        return 1

    if args.mode == "gui":
        gui_result = launch_gui(config)
        if gui_result is not None:
            return gui_result
        logging.getLogger("drm").info("Falling back to CLI mode.")
        run_cli(config, args.task, args.workflow, args.feedback)
        return 0

    run_cli(config, args.task, args.workflow, args.feedback)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
