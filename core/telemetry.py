"""Lightweight telemetry helpers for metrics and span logging."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Dict, Iterator

_METRICS_LOGGER = logging.getLogger("drm.metrics")
_SPAN_LOGGER = logging.getLogger("drm.span")


def emit_metric(name: str, value: float = 1.0, **tags: object) -> None:
    """Emit a metric via logging for later aggregation."""
    payload: Dict[str, object] = {
        "metric_name": name,
        "metric_value": value,
        "metric_tags": tags or {},
    }
    _METRICS_LOGGER.info("metric", extra=payload)


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
