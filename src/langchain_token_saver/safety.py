"""Protocol and content invariants for safe context transformation."""

from __future__ import annotations

import re
from typing import Any, Iterable

from .messages import message_additional_kwargs, message_content, message_role, message_tool_calls

_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_IDENTIFIER_RE = re.compile(
    r"\b(?:id|call|tool|run|trace|thread|message|msg|request|response|file)"
    r"(?:[_:-][A-Za-z0-9]+|[0-9][A-Za-z0-9]*)\b",
    re.IGNORECASE,
)
_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b", re.I
)


def protected_fragments(text: str) -> tuple[str, ...]:
    """Find opaque fragments that must survive exactly if a strategy transforms text."""

    fragments = [
        *(_URL_RE.findall(text)),
        *(_IDENTIFIER_RE.findall(text)),
        *(_UUID_RE.findall(text)),
    ]
    return tuple(dict.fromkeys(fragments))


def is_protocol_critical(message: Any) -> bool:
    role = message_role(message)
    if role in {"system", "tool"}:
        return True
    if message_tool_calls(message):
        return True
    additional = message_additional_kwargs(message)
    return any(key in additional for key in ("tool_call_id", "function_call", "tool_calls"))


def is_rewrite_safe_text(message: Any) -> bool:
    """Permit only plain, non-protocol text in the default compactor.

    We intentionally skip a candidate rather than attempting to parse opaque
    material.  Users can add a domain-specific strategy if they can provide a
    stronger invariant than this package can prove generically.
    """

    if is_protocol_critical(message) or message_role(message) not in {"human", "ai"}:
        return False
    content = message_content(message)
    if not content:
        return False
    stripped = content.strip()
    if (
        "```" in content
        or _URL_RE.search(content)
        or _IDENTIFIER_RE.search(content)
        or _UUID_RE.search(content)
    ):
        return False
    if stripped.startswith(("{", "[")) or (stripped.endswith(("}", "]")) and ":" in stripped):
        return False
    return True


def is_rewrite_safe_tool_output(message: Any) -> bool:
    """Return whether an opt-in strategy may compact a tool result's text.

    The surrounding tool message, its ``tool_call_id``, and its position are
    never changed. Opaque data is excluded before a user-supplied semantic
    strategy sees it.
    """

    if message_role(message) != "tool":
        return False
    content = message_content(message)
    if not content:
        return False
    stripped = content.strip()
    if "```" in content or _URL_RE.search(content) or _IDENTIFIER_RE.search(content):
        return False
    if _UUID_RE.search(content) or stripped.startswith(("{", "[")):
        return False
    return not (stripped.endswith(("}", "]")) and ":" in stripped)


def protocol_signature(messages: Iterable[Any]) -> tuple[tuple[str, str], ...]:
    """The exact ordering key for messages that participate in tool protocol."""

    signature: list[tuple[str, str]] = []
    for message in messages:
        role = message_role(message)
        if role == "tool":
            additional = message_additional_kwargs(message)
            signature.append(("tool", str(additional.get("tool_call_id", ""))))
        for call in message_tool_calls(message):
            if isinstance(call, dict):
                call_id = call.get("id", "")
            else:
                call_id = getattr(call, "id", "")
            signature.append(("call", str(call_id)))
    return tuple(signature)


def preserves_protected_fragments(original: Iterable[Any], transformed_text: str) -> bool:
    fragments = [
        fragment
        for message in original
        if (content := message_content(message))
        for fragment in protected_fragments(content)
    ]
    return all(fragment in transformed_text for fragment in fragments)
