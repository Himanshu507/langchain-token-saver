"""Duck-typed message helpers.

Keeping this adapter at the package edge lets the core work with LangChain
``BaseMessage`` objects without making its safety checks dependent on a
particular LangChain release.  The helpers also support the dict messages used
by the CLI preview API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class SyntheticSystemMessage:
    content: str
    type: str = "system"
    additional_kwargs: dict[str, Any] = field(default_factory=dict)


def get_value(message: Any, name: str, default: Any = None) -> Any:
    if isinstance(message, Mapping):
        return message.get(name, default)
    return getattr(message, name, default)


def message_role(message: Any) -> str:
    role = get_value(message, "type") or get_value(message, "role")
    if role is None:
        name = type(message).__name__.lower()
        if "system" in name:
            return "system"
        if "human" in name or "user" in name:
            return "human"
        if "tool" in name:
            return "tool"
        if "ai" in name or "assistant" in name:
            return "ai"
        return "unknown"
    aliases = {"user": "human", "assistant": "ai"}
    return aliases.get(str(role).lower(), str(role).lower())


def message_content(message: Any) -> str | None:
    content = get_value(message, "content")
    return content if isinstance(content, str) else None


def message_tool_calls(message: Any) -> list[Any]:
    calls = get_value(message, "tool_calls", None)
    if calls:
        return list(calls)
    additional = get_value(message, "additional_kwargs", {}) or {}
    if isinstance(additional, Mapping):
        calls = additional.get("tool_calls") or additional.get("function_call")
        if calls:
            return list(calls) if isinstance(calls, list) else [calls]
    return []


def message_additional_kwargs(message: Any) -> Mapping[str, Any]:
    value = get_value(message, "additional_kwargs", {}) or {}
    return value if isinstance(value, Mapping) else {}


def is_message_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def make_system_message(content: str, template: Any | None = None) -> Any:
    """Create a real LangChain ``SystemMessage`` when it is available."""

    if isinstance(template, Mapping):
        return {"type": "system", "content": content}
    try:
        from langchain_core.messages import SystemMessage

        return SystemMessage(content=content)
    except ImportError:
        return SyntheticSystemMessage(content=content)


def replace_message_content(message: Any, content: str) -> Any:
    """Copy a message while preserving its protocol metadata and concrete type."""

    if isinstance(message, Mapping):
        return {**message, "content": content}
    copier = getattr(message, "model_copy", None)
    if callable(copier):
        return copier(update={"content": content})
    copy_method = getattr(message, "copy", None)
    if callable(copy_method):
        try:
            return copy_method(update={"content": content})
        except TypeError:
            pass
    try:
        clone = object.__new__(type(message))
        clone.__dict__.update(message.__dict__)
        clone.content = content
        return clone
    except (AttributeError, TypeError):
        return message
