"""Public value types for token-saver.

The package deliberately keeps measurements separate from decisions.  A caller
can therefore log a plan without applying it, and can distinguish a provider's
authoritative usage data from deterministic local estimates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Sequence


class TokenUsageSource(str, Enum):
    """Where a usage number came from."""

    PROVIDER = "provider"
    ESTIMATED = "estimated"
    MISSING = "missing"


@dataclass(frozen=True)
class TokenUsage:
    """Comparable token measurements for one model interaction.

    ``None`` means that a value was not supplied, rather than zero tokens.
    ``source`` applies to every populated field in this value.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    source: TokenUsageSource = TokenUsageSource.MISSING

    def __post_init__(self) -> None:
        for name in ("input_tokens", "output_tokens", "total_tokens"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} cannot be negative")
        if (
            self.total_tokens is None
            and self.input_tokens is not None
            and self.output_tokens is not None
        ):
            object.__setattr__(self, "total_tokens", self.input_tokens + self.output_tokens)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source"] = self.source.value
        return data


@dataclass(frozen=True)
class CompactionConfig:
    """Explicit guardrails for reducing older context."""

    threshold_tokens: int = 4_000
    preserve_recent_messages: int = 4
    min_net_savings_tokens: int = 128
    dry_run: bool = False
    compact_tool_outputs: bool = False

    def __post_init__(self) -> None:
        if self.threshold_tokens < 1:
            raise ValueError("threshold_tokens must be at least 1")
        if self.preserve_recent_messages < 0:
            raise ValueError("preserve_recent_messages cannot be negative")
        if self.min_net_savings_tokens < 0:
            raise ValueError("min_net_savings_tokens cannot be negative")


@dataclass(frozen=True)
class OptimizationConfig:
    """All optimizations are disabled by default except measurement."""

    compaction: CompactionConfig = field(default_factory=CompactionConfig)
    terse: bool = False
    allow_brevity_with_tools: bool = False


@dataclass(frozen=True)
class OptimizationPlan:
    """A transparent preflight result, usable for dry-runs and apply mode."""

    messages: Sequence[Any]
    before_tokens: int
    after_tokens: int
    compaction_cost_tokens: int = 0
    compacted: bool = False
    reason: str = "not_requested"
    candidate_sections: tuple[str, ...] = ()
    brevity_applied: bool = False

    @property
    def predicted_net_savings_tokens(self) -> int:
        return self.before_tokens - self.after_tokens - self.compaction_cost_tokens

    def as_dict(self) -> dict[str, Any]:
        return {
            "before_tokens": self.before_tokens,
            "after_tokens": self.after_tokens,
            "compaction_cost_tokens": self.compaction_cost_tokens,
            "predicted_net_savings_tokens": self.predicted_net_savings_tokens,
            "compacted": self.compacted,
            "reason": self.reason,
            "candidate_sections": list(self.candidate_sections),
            "brevity_applied": self.brevity_applied,
        }


@dataclass(frozen=True)
class TokenSavingsReport:
    """One run's auditable optimization accounting.

    The baseline is normally a local estimate of the original prompt because
    sending the baseline merely to measure it would itself consume tokens.
    Consequently ``net_input_savings_tokens`` is the only net-saving claim made
    by the wrapper unless an application supplies its own baseline run.
    """

    baseline_usage: TokenUsage
    optimized_usage: TokenUsage
    compaction_usage: TokenUsage
    predicted_net_savings_tokens: int
    net_input_savings_tokens: int | None
    plan_reason: str
    terse_requested: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "baseline_usage": self.baseline_usage.as_dict(),
            "optimized_usage": self.optimized_usage.as_dict(),
            "compaction_usage": self.compaction_usage.as_dict(),
            "predicted_net_savings_tokens": self.predicted_net_savings_tokens,
            "net_input_savings_tokens": self.net_input_savings_tokens,
            "plan_reason": self.plan_reason,
            "terse_requested": self.terse_requested,
        }


EventHandler = Callable[[Mapping[str, Any]], None]
