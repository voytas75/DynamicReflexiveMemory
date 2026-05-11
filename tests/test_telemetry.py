"""Tests for telemetry queue, history, metrics, and spans.

Updates:
    v0.1 - 2026-05-11 - Added telemetry helper coverage for quality gates.
"""

from __future__ import annotations

from core import telemetry
from core.telemetry import TelemetryEvent, TelemetryFeed


def test_telemetry_feed_drain_and_latest() -> None:
    feed = TelemetryFeed(max_history=3)
    feed.publish(
        TelemetryEvent(name="alpha", timestamp=telemetry._utcnow(), payload={})
    )
    feed.publish(TelemetryEvent(name="beta", timestamp=telemetry._utcnow(), payload={}))
    feed.publish(
        TelemetryEvent(name="alpha", timestamp=telemetry._utcnow(), payload={"n": 1})
    )

    assert [event.name for event in feed.latest(limit=2)] == ["beta", "alpha"]
    assert [event.name for event in feed.latest("alpha", limit=5)] == [
        "alpha",
        "alpha",
    ]
    assert feed.latest(limit=0) == []
    assert [event.name for event in feed.drain(limit=2)] == ["alpha", "beta"]
    assert [event.name for event in feed.drain(limit=5)] == ["alpha"]
    assert feed.drain(limit=1) == []


def test_global_telemetry_helpers() -> None:
    telemetry.drain_telemetry(limit=1024)
    telemetry.publish_event("custom", payload={"a": 1}, b=2)
    telemetry.emit_metric("requests", value=2.0, route="/")

    drained = telemetry.drain_telemetry(limit=10)
    names = [event.name for event in drained]
    assert "custom" in names
    assert "metric" in names

    latest_metric = telemetry.latest_telemetry("metric", limit=1)
    assert latest_metric
    assert latest_metric[-1].payload["metric_name"] == "requests"


def test_log_span_records_without_error() -> None:
    with telemetry.log_span("unit", value=1):
        pass
