"""Entry point for the Dynamic Reflexive Memory application.

Updates: v0.1 - 2025-11-06 - Added CLI/GUI bootstrap with configuration loading,
logging setup, and sample task execution pipeline.
"""

from __future__ import annotations

import argparse
import logging
import logging.config
from pathlib import Path
from typing import Optional

from config.settings import AppConfig, get_app_config, resolve_config_path
from core.controller import SelfAdjustingController
from core.memory_manager import MemoryManager
from core.prompt_engine import AdaptivePromptEngine, PromptContext
from core.review import ReviewEngine
from core.task_executor import TaskExecutor
from core.exceptions import DRMError, WorkflowError
from gui.app import launch_gui
from models.memory import WorkingMemoryItem
from models.workflows import TaskRequest


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


def run_cli(config: AppConfig, task: Optional[str] = None, workflow: Optional[str] = None) -> None:
    """Run a simple CLI workflow as a fallback when GUI is unavailable."""
    logger = logging.getLogger("drm.cli")
    memory_manager = MemoryManager(config)
    prompt_engine = AdaptivePromptEngine(config)
    executor = TaskExecutor(config)
    review_engine = ReviewEngine(config)
    controller = SelfAdjustingController(config)

    prompt_text = task or "Summarise today's objectives based on existing memory."
    selection = executor.select_workflow(workflow)
    request = TaskRequest(workflow=selection.workflow, prompt=prompt_text)

    # Seed working memory for the prompt context
    memory_manager.put_working_item(
        WorkingMemoryItem(
            key=request.task_id,
            payload={"task_overview": prompt_text},
            ttl_seconds=config.memory.redis.ttl_seconds,
        )
    )

    prompt_context = PromptContext(
        task=prompt_text,
        workflow=selection.workflow,
        working_memory={"task_overview": prompt_text},
        episodic_memory=memory_manager.list_layer("episodic") if config.review.enabled else [],
        semantic_memory=memory_manager.list_layer("semantic") if config.review.enabled else [],
        recent_reviews=[],
    )
    prompt = prompt_engine.build_prompt(prompt_context)

    logger.info("Prepared prompt for workflow %s:\n%s", selection.workflow, prompt)

    try:
        result = executor.execute(request)
    except WorkflowError as exc:
        logger.warning(
            "Workflow execution unavailable: %s. The prompt above can be sent manually.",
            exc,
        )
        return

    review = review_engine.perform_review(request, result)
    memory_manager.record_review(review)
    drift_notice = controller.register_result(selection, result, review)

    logger.info("Task result (%s): %s", result.workflow, result.content)
    logger.info("Review verdict: %s - %s", review.verdict, review.notes)
    if drift_notice:
        logger.warning("Controller advisory: %s", drift_notice)


def main(argv: Optional[list[str]] = None) -> int:
    """Parse CLI arguments and dispatch the selected mode."""
    parser = argparse.ArgumentParser(description="Dynamic Reflexive Memory runner.")
    parser.add_argument("--mode", choices=["gui", "cli"], default="gui")
    parser.add_argument("--config", type=Path, help="Path to configuration file.")
    parser.add_argument("--task", type=str, help="Task prompt for CLI mode.")
    parser.add_argument("--workflow", type=str, help="Workflow override for execution.")
    parser.add_argument("--logging-config", type=Path, help="Path to logging configuration.")
    args = parser.parse_args(argv)

    try:
        config_path = resolve_config_path(args.config) if args.config else None
        config = get_app_config(config_path)
        setup_logging(args.logging_config)
    except DRMError as exc:
        logging.basicConfig(level=logging.ERROR)
        logging.getLogger("drm").error("Startup failed: %s", exc)
        return 1

    if args.mode == "gui":
        gui_result = launch_gui(config)
        if gui_result is not None:
            return gui_result
        logging.getLogger("drm").info("Falling back to CLI mode.")
        run_cli(config, args.task, args.workflow)
        return 0

    run_cli(config, args.task, args.workflow)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
