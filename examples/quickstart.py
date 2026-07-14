"""Run with ``OPENAI_API_KEY=... python examples/quickstart.py``."""

from langchain_openai import ChatOpenAI

from langchain_token_saver import CompactionConfig, OptimizationConfig, TokenSavingChatWrapper

model = ChatOpenAI(model="gpt-4.1-mini")
chat = TokenSavingChatWrapper(
    model,
    config=OptimizationConfig(
        terse=True,
        compaction=CompactionConfig(threshold_tokens=1_500, preserve_recent_messages=4),
    ),
    event_handler=lambda event: print(event),
)

answer = chat.invoke("Give me a short explanation of dependency injection.")
print(answer.content)
print(chat.last_report.as_dict())
