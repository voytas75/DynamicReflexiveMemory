"""Lightweight telemetry helpers for metrics and span logging.

Updates:
    v0.1 - 2025-11-07 - Provided logging wrappers for metrics and spans.
    v0.2 - 2025-11-08 - Added in-process telemetry feed for GUI consumption.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from queue import Empty, Queue
from threading import Lock
from typing import Deque, Dict, Iterator, List, Optional

_METRICS_LOGGER = logging.getLogger("drm.metrics")
_SPAN_LOGGER = logging.getLogger("drm.span")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class TelemetryEvent:
    """Structured telemetry payload captured for GUI consumption."""

    name: str
    timestamp: datetime
    payload: Dict[str, object]


class TelemetryFeed:
    """In-process fan-out queue for telemetry events."""

    def __init__(self, max_history: int = 256) -> None:
        self._queue: "Queue[TelemetryEvent]" = Queue()
        self._history: Deque[TelemetryEvent] = deque(maxlen=max_history)
        self._lock = Lock()

    def publish(self, event: TelemetryEvent) -> None:
        with self._lock:
            self._history.append(event)
        self._queue.put(event)

    def drain(self, limit: int = 64) -> List[TelemetryEvent]:
        drained: List[TelemetryEvent] = []
        for _ in range(max(0, limit)):
            try:
                drained.append(self._queue.get_nowait())
            except Empty:
                break
        return drained

    def latest(self, name: Optional[str] = None, limit: int = 10) -> List[TelemetryEvent]:
        with self._lock:
            if name is None:
                events = list(self._history)
            else:
                events = [event for event in self._history if event.name == name]
        if limit <= 0:
            return []
        return events[-limit:]


GLOBAL_TELEMETRY_FEED = TelemetryFeed()


def publish_event(
    name: str,
    payload: Optional[Dict[str, object]] = None,
    **fields: object,
) -> None:
    """Publish a telemetry event into the global feed."""
    event_payload: Dict[str, object] = dict(payload or {})
    if fields:
        event_payload.update(fields)
    event = TelemetryEvent(
        name=name,
        timestamp=_utcnow(),
        payload=event_payload,
    )
    GLOBAL_TELEMETRY_FEED.publish(event)


def drain_telemetry(limit: int = 64) -> List[TelemetryEvent]:
    """Retrieve queued telemetry events for consumers such as the GUI."""
    return GLOBAL_TELEMETRY_FEED.drain(limit)


def latest_telemetry(name: str, limit: int = 5) -> List[TelemetryEvent]:
    """Return the newest telemetry events for *name*."""
    return GLOBAL_TELEMETRY_FEED.latest(name, limit)


def emit_metric(name: str, value: float = 1.0, **tags: object) -> None:
    """Emit a metric via logging for later aggregation."""
    payload: Dict[str, object] = {
        "metric_name": name,
        "metric_value": value,
        "metric_tags": tags or {},
    }
    _METRICS_LOGGER.info("metric", extra=payload)
    publish_event(
        "metric",
        payload=payload,
    )


@contextmanager
def log_span(name: str, **fields: object) -> Iterator[None]:
    """Log a start/end span around a block of work."""
    start = time.perf_counter()
    _SPAN_LOGGER.debug(
        "span.start",
        extra={"span_name": name, "span_fields": fields or {}},
    )
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        span_fields = dict(fields or {})
        span_fields["duration_seconds"] = round(duration, 4)
        _SPAN_LOGGER.debug(
            "span.end",
            extra={"span_name": name, "span_fields": span_fields},
        )
