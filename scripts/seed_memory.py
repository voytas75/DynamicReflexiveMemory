"""Populate the memory stores with sample DRM entries for demos.

Updates:
    v0.1 - 2025-11-06 - Added seeding script for working, episodic, semantic, and review memories.
    v0.2 - 2025-11-07 - Normalised seeded timestamps to timezone-aware UTC.
    v0.3 - 2025-11-07 - Clarified dependency on in-process ChromaDB package (no Docker service).
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from uuid import uuid4

from config.settings import get_app_config, resolve_config_path
from core.exceptions import DRMError
from core.memory_manager import MemoryManager
from models.memory import EpisodicMemoryEntry, ReviewRecord, SemanticNode, WorkingMemoryItem


LOGGER = logging.getLogger("drm.seed")


def seed_memory(config_path: Path | None = None) -> None:
    """Insert sample memory objects into Redis and embedded ChromaDB layers."""
    config = get_app_config(config_path)
    manager = MemoryManager(config)

    seed_id = uuid4().hex[:8]
    working_item = WorkingMemoryItem(
        key=f"demo-task-{seed_id}",
        payload={
            "objective": "Demonstrate DRM memory seeding.",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        ttl_seconds=config.memory.redis.ttl_seconds,
    )

    episodic_entries: Iterable[EpisodicMemoryEntry] = [
        EpisodicMemoryEntry(
            id=f"episode-{seed_id}-1",
            content="Completed GUI onboarding walkthrough.",
            metadata={"tags": ["onboarding", "ui"]},
        ),
        EpisodicMemoryEntry(
            id=f"episode-{seed_id}-2",
            content="Captured user feedback about prompt clarity.",
            metadata={"tags": ["feedback", "prompting"]},
        ),
    ]

    semantic_nodes: Iterable[SemanticNode] = [
        SemanticNode(
            id=f"concept-{seed_id}-memory",
            label="Memory Fusion",
            definition="Technique for blending working and episodic context.",
            sources=["seed_script"],
            relations={"TaskReview": 0.7},
        ),
        SemanticNode(
            id=f"concept-{seed_id}-review",
            label="Hybrid Review Loop",
            definition="Combined automated and human evaluation pipeline.",
            sources=["seed_script"],
            relations={"Memory Fusion": 0.6},
        ),
    ]

    review_records: Iterable[ReviewRecord] = [
        ReviewRecord(
            id=f"review-{seed_id}-1",
            task_reference="demo-task",
            verdict="pass",
            notes="Automated review confirms seeding scenario is valid.",
        )
    ]

    LOGGER.info("Storing working memory item %s", working_item.key)
    manager.put_working_item(working_item)

    for entry in episodic_entries:
        LOGGER.info("Recording episodic memory %s", entry.id)
        manager.record_episodic(entry)

    for node in semantic_nodes:
        LOGGER.info("Recording semantic node %s", node.id)
        manager.record_semantic(node)

    for record in review_records:
        LOGGER.info("Recording review %s", record.id)
        manager.record_review(record)

    LOGGER.info("Seed operation completed. Use GUI/CLI to inspect the entries.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed DRM memory layers with demo data.")
    parser.add_argument("--config", type=Path, help="Optional path to config.json.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    try:
        config_path = resolve_config_path(args.config) if args.config else None
        seed_memory(config_path)
    except DRMError as exc:
        LOGGER.error("Seeding failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
