"""Repeatable baseline-versus-optimized benchmark support."""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable

from .capabilities import extract_usage
from .types import TokenSavingsReport
from .wrapper import TokenSavingChatWrapper


QualityGate = Callable[[Any], bool]


@dataclass(frozen=True)
class BenchmarkTrace:
    name: str
    input: Any
    quality_gate: QualityGate = lambda _response: True


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

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkSummary:
    results: tuple[BenchmarkResult, ...]

    @property
    def median_net_input_savings_tokens(self) -> float:
        values = [
            result.optimized_report["net_input_savings_tokens"]
            for result in self.results
            if result.optimized_report["net_input_savings_tokens"] is not None
        ]
        return statistics.median(values) if values else 0.0

    @property
    def quality_gate_failures(self) -> int:
        return sum(not result.quality_gate_passed for result in self.results)

    def as_dict(self) -> dict[str, Any]:
        return {
            "results": [result.as_dict() for result in self.results],
            "median_net_input_savings_tokens": self.median_net_input_savings_tokens,
            "quality_gate_failures": self.quality_gate_failures,
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
        baseline_start = time.perf_counter()
        try:
            baseline_response = baseline_model.invoke(trace.input)
            baseline_success = True
        except Exception:
            baseline_response = None
            baseline_success = False
        baseline_ms = (time.perf_counter() - baseline_start) * 1_000

        optimized_start = time.perf_counter()
        try:
            optimized_response = optimized_model.invoke(trace.input)
            optimized_success = True
        except Exception:
            optimized_response = None
            optimized_success = False
        optimized_ms = (time.perf_counter() - optimized_start) * 1_000

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
        quality = bool(optimized_success and trace.quality_gate(optimized_response))
        results.append(
            BenchmarkResult(
                name=trace.name,
                baseline_usage=extract_usage(baseline_response).as_dict()
                if baseline_response is not None
                else {},
                optimized_report=report_data,
                baseline_latency_ms=baseline_ms,
                optimized_latency_ms=optimized_ms,
                baseline_success=baseline_success,
                optimized_success=optimized_success,
                quality_gate_passed=quality,
            )
        )
    return BenchmarkSummary(tuple(results))
