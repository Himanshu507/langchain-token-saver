"""Deterministic local token estimates.

This is deliberately an estimate, not a provider tokenizer impersonator.  The
same input always receives the same count, and every caller is told that the
value is estimated.  Provider usage wins whenever the response contains it.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from .messages import message_content, message_role
from .types import TokenUsage, TokenUsageSource

_TOKEN_RE = re.compile(r"\w+(?:['’-]\w+)?|[^\s\w]", re.UNICODE)
_MESSAGE_OVERHEAD = 3


def estimate_text_tokens(text: str) -> int:
    """Return a conservative deterministic lexical-token estimate."""

    return len(_TOKEN_RE.findall(text))


def estimate_messages_tokens(messages: Iterable[Any]) -> int:
    total = 0
    for message in messages:
        content = message_content(message)
        if content is None:
            total += _MESSAGE_OVERHEAD
            continue
        total += _MESSAGE_OVERHEAD + estimate_text_tokens(content)
        if message_role(message) == "tool":
            total += 2
    return total


def estimated_prompt_usage(messages: Iterable[Any]) -> TokenUsage:
    return TokenUsage(
        input_tokens=estimate_messages_tokens(messages),
        output_tokens=None,
        total_tokens=estimate_messages_tokens(messages),
        source=TokenUsageSource.ESTIMATED,
    )
