"""Stable, JSON-compatible observability events."""

from __future__ import annotations

from typing import Any, Mapping

from .types import EventHandler, OptimizationPlan, TokenSavingsReport

EVENT_VERSION = "1"


def emit(handler: EventHandler | None, event: str, **payload: Any) -> None:
    if handler is None:
        return
    try:
        handler({"version": EVENT_VERSION, "event": event, **payload})
    except Exception:
        # Observability must not become a new failure path for a model call.
        return


def emit_compaction_decision(
    handler: EventHandler | None, plan: OptimizationPlan, *, dry_run: bool
) -> None:
    payload = plan.as_dict()
    emit(handler, "compaction.decided", **payload)
    if dry_run:
        emit(handler, "compaction.dry_run_previewed", **payload)


def emit_report(handler: EventHandler | None, report: TokenSavingsReport) -> None:
    emit(handler, "token_savings.reported", **report.as_dict())


def emit_fallback(
    handler: EventHandler | None, reason: str, detail: Mapping[str, Any] | None = None
) -> None:
    emit(handler, "compaction.fallback", reason=reason, detail=dict(detail or {}))
