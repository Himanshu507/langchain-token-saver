"""Provider capabilities and response usage extraction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from .messages import get_value
from .tokenization import estimate_text_tokens
from .types import TokenUsage, TokenUsageSource


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ProviderCapabilities:
    provider: Provider
    reports_usage: bool
    supports_tool_calls: bool
    supports_stream_usage: bool


CAPABILITY_MATRIX: dict[Provider, ProviderCapabilities] = {
    Provider.OPENAI: ProviderCapabilities(Provider.OPENAI, True, True, True),
    Provider.ANTHROPIC: ProviderCapabilities(Provider.ANTHROPIC, True, True, True),
    Provider.UNKNOWN: ProviderCapabilities(Provider.UNKNOWN, False, False, False),
}


def infer_provider(model: Any) -> Provider:
    text = f"{type(model).__module__}.{type(model).__name__}".lower()
    if "openai" in text:
        return Provider.OPENAI
    if "anthropic" in text:
        return Provider.ANTHROPIC
    return Provider.UNKNOWN


def capabilities_for(model: Any) -> ProviderCapabilities:
    return CAPABILITY_MATRIX[infer_provider(model)]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_number(data: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = data.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and value >= 0:
            return int(value)
    return None


def _find_usage(response: Any) -> Mapping[str, Any]:
    direct = _mapping(get_value(response, "usage_metadata", {}))
    if direct:
        return direct
    response_metadata = _mapping(get_value(response, "response_metadata", {}))
    for key in ("usage", "usage_metadata", "token_usage"):
        nested = _mapping(response_metadata.get(key))
        if nested:
            return nested
    additional = _mapping(get_value(response, "additional_kwargs", {}))
    nested = _mapping(additional.get("usage_metadata") or additional.get("usage"))
    return nested


def extract_usage(response: Any) -> TokenUsage:
    """Prefer provider metadata and otherwise estimate only response text."""

    usage = _find_usage(response)
    input_tokens = _first_number(usage, "input_tokens", "prompt_tokens", "input_token_count")
    output_tokens = _first_number(usage, "output_tokens", "completion_tokens", "output_token_count")
    total_tokens = _first_number(usage, "total_tokens", "total_token_count")
    if input_tokens is not None or output_tokens is not None or total_tokens is not None:
        return TokenUsage(input_tokens, output_tokens, total_tokens, TokenUsageSource.PROVIDER)

    content = get_value(response, "content")
    if isinstance(content, str):
        return TokenUsage(
            input_tokens=None,
            output_tokens=estimate_text_tokens(content),
            total_tokens=None,
            source=TokenUsageSource.ESTIMATED,
        )
    return TokenUsage(source=TokenUsageSource.MISSING)
