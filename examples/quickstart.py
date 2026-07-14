"""Run with ``OPENAI_API_KEY=... python examples/quickstart.py``."""

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from langchain_token_saver import CompactionConfig, OptimizationConfig, TokenSavingChatWrapper


class DemonstrateFallback:
    """A stand-in for a failed semantic summarizer; never use this in production."""

    def compact(self, _messages):
        raise RuntimeError("compaction provider unavailable")


model = ChatOpenAI(model="gpt-4.1-mini")
demo_compaction = CompactionConfig(
    threshold_tokens=1,
    preserve_recent_messages=1,
    min_net_savings_tokens=1,
)
chat = TokenSavingChatWrapper(
    model,
    config=OptimizationConfig(
        terse=True,
        compaction=demo_compaction,
    ),
    event_handler=lambda event: print(event),
)

repeated_decision = (
    "Use PostgreSQL for the order store because transactional writes and the existing "
    "operational tooling satisfy the product's reliability requirements."
)
history = [
    HumanMessage(content=repeated_decision),
    HumanMessage(content=repeated_decision),
    HumanMessage(content=repeated_decision),
    HumanMessage(content="Now explain the dependency-injection trade-off briefly."),
]

# No provider request is made here. This tells you exactly whether a compaction
# will apply, the candidate text, and why it would fall back unchanged.
preview = chat.preview(history)
print("preview:", preview.as_dict())

fallback_preview = TokenSavingChatWrapper(
    model,
    config=OptimizationConfig(compaction=demo_compaction),
    compactor=DemonstrateFallback(),
).preview(history)
print("safe fallback preview:", fallback_preview.as_dict())

answer = chat.invoke(history)
print(answer.content)
print(chat.last_report.as_dict())
