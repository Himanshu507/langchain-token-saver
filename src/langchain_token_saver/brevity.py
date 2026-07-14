"""Safe, opt-in response brevity selection."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .messages import make_system_message, message_content, message_tool_calls
from .types import OptimizationConfig

_BREVITY_INSTRUCTION = (
    "Response formatting preference: when the user has not requested detail, "
    "answer concisely. Preserve meaning, policy, code blocks, JSON, citations, "
    "and all structured-output requirements exactly."
)


def _has_structured_request(
    messages: Sequence[Any], kwargs: Mapping[str, Any], guarded_output: bool
) -> bool:
    if guarded_output or any(key in kwargs for key in ("tools", "response_format", "tool_choice")):
        return True
    return any(
        message_tool_calls(message)
        or (message_content(message) or "").lstrip().startswith(("{", "["))
        for message in messages
    )


def apply_brevity_preference(
    messages: Sequence[Any],
    config: OptimizationConfig,
    kwargs: Mapping[str, Any],
    *,
    guarded_output: bool = False,
) -> tuple[list[Any], bool, str]:
    """Return a new list only for an unstructured, opt-in interaction."""

    original = list(messages)
    if not config.terse:
        return original, False, "not_requested"
    if (
        _has_structured_request(original, kwargs, guarded_output)
        and not config.allow_brevity_with_tools
    ):
        return original, False, "structured_or_tool_request"
    if any("```" in (message_content(message) or "") for message in original):
        return original, False, "code_block_present"

    insertion = 0
    while insertion < len(original) and getattr(original[insertion], "type", None) == "system":
        insertion += 1
    return (
        [
            *original[:insertion],
            make_system_message(_BREVITY_INSTRUCTION, original[0] if original else None),
            *original[insertion:],
        ],
        True,
        "applied",
    )
