"""Safe, measurable token optimization for LangChain chat models."""

from .benchmark import BenchmarkResult, BenchmarkSummary, BenchmarkTrace, run_benchmark
from .capabilities import (
    CAPABILITY_MATRIX,
    Provider,
    ProviderCapabilities,
    capabilities_for,
    extract_usage,
)
from .compaction import (
    CompactionResult,
    ContextCompactionStrategy,
    ExtractiveCompactor,
    ToolOutputCompactionStrategy,
    build_optimization_plan,
)
from .types import (
    CompactionConfig,
    OptimizationConfig,
    OptimizationPlan,
    TokenSavingsReport,
    TokenUsage,
    TokenUsageSource,
)
from .wrapper import TokenSavingChatWrapper

__all__ = [
    "CAPABILITY_MATRIX",
    "BenchmarkResult",
    "BenchmarkSummary",
    "BenchmarkTrace",
    "CompactionConfig",
    "CompactionResult",
    "ContextCompactionStrategy",
    "ExtractiveCompactor",
    "OptimizationConfig",
    "OptimizationPlan",
    "Provider",
    "ProviderCapabilities",
    "TokenSavingChatWrapper",
    "TokenSavingsReport",
    "TokenUsage",
    "TokenUsageSource",
    "ToolOutputCompactionStrategy",
    "build_optimization_plan",
    "capabilities_for",
    "extract_usage",
    "run_benchmark",
]
