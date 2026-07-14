"""Repeatable baseline-versus-optimized benchmark support."""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .capabilities import extract_usage
from .types import TokenSavingsReport, TokenUsage, TokenUsageSource
from .wrapper import TokenSavingChatWrapper


QualityGate = Callable[[Any], bool]
_REQUIRED_SAFETY_TAGS = frozenset(
    {
        "code_block",
        "compaction_failure",
        "critical_fact",
        "prompt_injection",
        "structured_output",
        "tool_protocol",
        "url_or_id",
    }
)


@dataclass(frozen=True)
class BenchmarkTrace:
    name: str
    input: Any
    quality_gate: QualityGate | None = None
    safety_tags: frozenset[str] = frozenset()


def load_benchmark_traces(path: str | Path) -> tuple[BenchmarkTrace, ...]:
    """Load a portable JSON trace fixture for a repeatable benchmark run.

    Each item needs a unique ``name`` and an ``input`` accepted by the selected
    chat model. Quality gates remain code because they are application-specific.
    """

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("benchmark trace fixture must be a JSON array")
    traces: list[BenchmarkTrace] = []
    names: set[str] = set()
    for item in raw:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("name"), str)
            or "input" not in item
        ):
            raise ValueError("each benchmark trace needs string name and input")
        tags = item.get("safety_tags", [])
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise ValueError("benchmark trace safety_tags must be a string list")
        if item["name"] in names:
            raise ValueError(f"duplicate benchmark trace name: {item['name']}")
        names.add(item["name"])
        traces.append(
            BenchmarkTrace(name=item["name"], input=item["input"], safety_tags=frozenset(tags))
        )
    return tuple(traces)


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    baseline_usage: dict[str, Any]
    optimized_report: dict[str, Any]
    baseline_latency_ms: float
    optimized_latency_ms: float
    baseline_success: bool
    optimized_success: bool
    quality_gate_passed: bool
    quality_gate_configured: bool
    safety_tags: frozenset[str]
    paired_token_savings: dict[str, Any]
    model_configuration_matched: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkSummary:
    results: tuple[BenchmarkResult, ...]

    @property
    def median_net_input_savings_tokens(self) -> float:
        values = [
            result.paired_token_savings["net_input_savings_tokens"]
            for result in self.results
            if result.paired_token_savings.get("source") == TokenUsageSource.PROVIDER.value
            and result.paired_token_savings["net_input_savings_tokens"] is not None
        ]
        return statistics.median(values) if values else 0.0

    @property
    def median_net_total_savings_tokens(self) -> float:
        values = [
            result.paired_token_savings["net_total_savings_tokens"]
            for result in self.results
            if result.paired_token_savings.get("source") == TokenUsageSource.PROVIDER.value
            and result.paired_token_savings["net_total_savings_tokens"] is not None
        ]
        return statistics.median(values) if values else 0.0

    @property
    def quality_gate_failures(self) -> int:
        return sum(not result.quality_gate_passed for result in self.results)

    @property
    def release_ready(self) -> bool:
        return (
            20 <= len(self.results) <= 50
            and self.median_net_total_savings_tokens > 0
            and self.quality_gate_failures == 0
            and all(result.quality_gate_configured for result in self.results)
            and all(result.model_configuration_matched for result in self.results)
            and _REQUIRED_SAFETY_TAGS.issubset(
                set().union(*(result.safety_tags for result in self.results))
            )
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "results": [result.as_dict() for result in self.results],
            "median_net_input_savings_tokens": self.median_net_input_savings_tokens,
            "median_net_total_savings_tokens": self.median_net_total_savings_tokens,
            "quality_gate_failures": self.quality_gate_failures,
            "release_ready": self.release_ready,
        }


def _measure(model: Any, input: Any) -> tuple[Any | None, bool, float]:
    start = time.perf_counter()
    try:
        return model.invoke(input), True, (time.perf_counter() - start) * 1_000
    except Exception:
        return None, False, (time.perf_counter() - start) * 1_000


def _model_configuration(model: Any) -> tuple[str, Any, Any]:
    candidate = getattr(model, "base_model", model)
    model_name = getattr(candidate, "model_name", getattr(candidate, "model", None))
    params = getattr(candidate, "_default_params", getattr(candidate, "model_kwargs", None))
    return type(candidate).__qualname__, model_name, repr(params)


def _paired_token_savings(
    baseline: TokenUsage, report: TokenSavingsReport | None
) -> dict[str, Any]:
    if report is None:
        return {
            "net_input_savings_tokens": None,
            "net_output_savings_tokens": None,
            "net_total_savings_tokens": None,
            "source": TokenUsageSource.MISSING.value,
        }
    optimized = report.optimized_usage
    if (
        baseline.source is not TokenUsageSource.PROVIDER
        or optimized.source is not TokenUsageSource.PROVIDER
    ):
        return {
            "net_input_savings_tokens": None,
            "net_output_savings_tokens": None,
            "net_total_savings_tokens": None,
            "source": TokenUsageSource.MISSING.value,
        }
    overhead = report.compaction_usage.total_tokens or 0
    return {
        "net_input_savings_tokens": (
            baseline.input_tokens - optimized.input_tokens
            if baseline.input_tokens is not None and optimized.input_tokens is not None
            else None
        ),
        "net_output_savings_tokens": (
            baseline.output_tokens - optimized.output_tokens
            if baseline.output_tokens is not None and optimized.output_tokens is not None
            else None
        ),
        "net_total_savings_tokens": (
            baseline.total_tokens - optimized.total_tokens - overhead
            if baseline.total_tokens is not None and optimized.total_tokens is not None
            else None
        ),
        "compaction_overhead_tokens": overhead,
        "source": TokenUsageSource.PROVIDER.value
        if overhead == 0
        else TokenUsageSource.ESTIMATED.value,
    }


def run_benchmark(
    traces: Iterable[BenchmarkTrace],
    *,
    baseline_model_factory: Callable[[], Any],
    optimized_model_factory: Callable[[], TokenSavingChatWrapper],
) -> BenchmarkSummary:
    """Run isolated baseline and optimized model instances for every trace.

    Factories avoid state leakage between the paired calls, which is essential
    when a provider model or an agent maintains conversational state.
    """

    results: list[BenchmarkResult] = []
    for trace in traces:
        baseline_model = baseline_model_factory()
        optimized_model = optimized_model_factory()
        baseline_response, baseline_success, baseline_ms = _measure(baseline_model, trace.input)
        optimized_response, optimized_success, optimized_ms = _measure(optimized_model, trace.input)

        report: TokenSavingsReport | None = optimized_model.last_report
        if report is None:
            # The failed call did not consume a complete response and must not
            # masquerade as a token-saving result.
            report_data = {
                "net_input_savings_tokens": None,
                "plan_reason": "model_error",
            }
        else:
            report_data = report.as_dict()
        baseline_usage = (
            extract_usage(baseline_response) if baseline_response is not None else TokenUsage()
        )
        quality = bool(
            optimized_success
            and trace.quality_gate is not None
            and trace.quality_gate(optimized_response)
        )
        results.append(
            BenchmarkResult(
                name=trace.name,
                baseline_usage=baseline_usage.as_dict(),
                optimized_report=report_data,
                baseline_latency_ms=baseline_ms,
                optimized_latency_ms=optimized_ms,
                baseline_success=baseline_success,
                optimized_success=optimized_success,
                quality_gate_passed=quality,
                quality_gate_configured=trace.quality_gate is not None,
                safety_tags=trace.safety_tags,
                paired_token_savings=_paired_token_savings(baseline_usage, report),
                model_configuration_matched=(
                    _model_configuration(baseline_model) == _model_configuration(optimized_model)
                ),
            )
        )
    return BenchmarkSummary(tuple(results))
