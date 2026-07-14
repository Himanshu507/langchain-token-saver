"""A provider-agnostic LangChain chat model proxy.

The wrapper deliberately uses the public ``invoke`` family rather than private
provider hooks.  It is therefore compatible with ``ChatOpenAI``,
``ChatAnthropic``, and custom LangChain chat models, while still behaving
usefully in a lightweight environment with a duck-typed model.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any, AsyncIterator, Iterator, Mapping, Sequence

from langchain_core.runnables import Runnable

from .brevity import apply_brevity_preference
from .capabilities import capabilities_for, extract_usage
from .compaction import ContextCompactionStrategy, build_optimization_plan
from .events import emit, emit_compaction_decision, emit_fallback, emit_report
from .messages import is_message_sequence
from .tokenization import estimate_messages_tokens
from .types import (
    EventHandler,
    OptimizationConfig,
    OptimizationPlan,
    TokenSavingsReport,
    TokenUsage,
    TokenUsageSource,
)


class TokenSavingChatWrapper(Runnable[Any, Any]):
    """An opt-in safe optimization proxy around a LangChain chat model.

    ``base_model`` is intentionally accepted structurally: a model only needs
    the applicable method from LangChain's normal ``invoke``/``ainvoke``/
    ``stream``/``astream`` surface. Because it is itself a ``Runnable``, it
    composes directly in runnable sequences and LangGraph nodes.
    """

    def __init__(
        self,
        base_model: Any,
        *,
        config: OptimizationConfig | None = None,
        compactor: ContextCompactionStrategy | None = None,
        event_handler: EventHandler | None = None,
        _guarded_output: bool = False,
    ) -> None:
        self.base_model = base_model
        self.config = config or OptimizationConfig()
        self.compactor = compactor
        self.event_handler = event_handler
        self._guarded_output = _guarded_output
        self._last_report: TokenSavingsReport | None = None

    @property
    def last_report(self) -> TokenSavingsReport | None:
        """The report from the latest completed call on this wrapper instance."""

        return self._last_report

    @property
    def provider_capabilities(self):
        return capabilities_for(self.base_model)

    def preview(self, input: Any, **kwargs: Any) -> OptimizationPlan:
        """Build the exact compaction decision used by a subsequent call."""

        plan = self._plan(input, kwargs)
        emit_compaction_decision(self.event_handler, plan, dry_run=True)
        return plan

    def _plan(self, input: Any, kwargs: Mapping[str, Any]) -> OptimizationPlan:
        if not is_message_sequence(input):
            # A string can be a valid LangChain input.  Preserve the original
            # model surface instead of coercing it to a message list.
            estimated = 0
            if isinstance(input, str):
                estimated = estimate_messages_tokens([{"type": "human", "content": input}])
            return OptimizationPlan([input], estimated, estimated, reason="non_message_input")
        try:
            plan = build_optimization_plan(input, self.config, compactor=self.compactor)
        except Exception as exc:  # never block an application request on optimization
            before = estimate_messages_tokens(input)
            emit_fallback(
                self.event_handler, "planner_exception", {"exception": type(exc).__name__}
            )
            return OptimizationPlan(
                list(input), before, before, reason=f"fallback:{type(exc).__name__}"
            )
        return plan

    def _prepare(self, input: Any, kwargs: Mapping[str, Any]) -> tuple[Any, OptimizationPlan]:
        plan = self._plan(input, kwargs)
        emit_compaction_decision(self.event_handler, plan, dry_run=self.config.compaction.dry_run)
        if not is_message_sequence(input):
            return input, plan
        messages: Sequence[Any] = plan.messages
        messages, applied, reason = apply_brevity_preference(
            messages,
            self.config,
            kwargs,
            guarded_output=self._guarded_output,
        )
        if applied:
            # Account for the additional formatting preference exactly through
            # the same local estimator used for the preflight plan.
            plan = replace(
                plan,
                messages=messages,
                after_tokens=estimate_messages_tokens(messages),
                brevity_applied=True,
            )
        elif self.config.terse:
            emit(self.event_handler, "brevity.skipped", reason=reason)
        if self.config.compaction.dry_run:
            return input, plan
        if not plan.compacted and not applied:
            # A skipped optimization must be observationally inert, including
            # preserving the caller's original list instance.
            return input, plan
        return messages, plan

    @staticmethod
    def _call(method: Any, input: Any, config: Any, kwargs: Mapping[str, Any]) -> Any:
        if config is None:
            return method(input, **kwargs)
        return method(input, config=config, **kwargs)

    def _record(self, plan: OptimizationPlan, response: Any) -> TokenSavingsReport:
        optimized = extract_usage(response)
        baseline = TokenUsage(
            input_tokens=plan.before_tokens,
            output_tokens=None,
            total_tokens=plan.before_tokens,
            source=TokenUsageSource.ESTIMATED,
        )
        compaction_usage = TokenUsage(
            input_tokens=plan.compaction_cost_tokens if plan.compaction_cost_tokens else 0,
            output_tokens=None,
            total_tokens=plan.compaction_cost_tokens if plan.compaction_cost_tokens else 0,
            source=TokenUsageSource.ESTIMATED,
        )
        comparable_input = optimized.input_tokens
        if comparable_input is None:
            comparable_input = plan.after_tokens
        net_input = plan.before_tokens - comparable_input - plan.compaction_cost_tokens
        report = TokenSavingsReport(
            baseline_usage=baseline,
            optimized_usage=optimized,
            compaction_usage=compaction_usage,
            predicted_net_savings_tokens=plan.predicted_net_savings_tokens,
            net_input_savings_tokens=net_input,
            plan_reason=plan.reason,
            terse_requested=self.config.terse,
        )
        self._last_report = report
        emit_report(self.event_handler, report)
        return report

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        prepared, plan = self._prepare(input, kwargs)
        response = self._call(self.base_model.invoke, prepared, config, kwargs)
        self._record(plan, response)
        return response

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        prepared, plan = self._prepare(input, kwargs)
        method = getattr(self.base_model, "ainvoke", None)
        if method is None:
            response = await asyncio.to_thread(
                self._call, self.base_model.invoke, prepared, config, kwargs
            )
        else:
            if config is None:
                response = await method(prepared, **kwargs)
            else:
                response = await method(prepared, config=config, **kwargs)
        self._record(plan, response)
        return response

    def stream(self, input: Any, config: Any = None, **kwargs: Any) -> Iterator[Any]:
        prepared, plan = self._prepare(input, kwargs)
        stream = self._call(self.base_model.stream, prepared, config, kwargs)
        last_chunk: Any = None
        for chunk in stream:
            last_chunk = chunk
            yield chunk
        if last_chunk is not None:
            self._record(plan, last_chunk)

    async def astream(self, input: Any, config: Any = None, **kwargs: Any) -> AsyncIterator[Any]:
        prepared, plan = self._prepare(input, kwargs)
        method = getattr(self.base_model, "astream", None)
        last_chunk: Any = None
        if method is None:
            for chunk in self._call(self.base_model.stream, prepared, config, kwargs):
                last_chunk = chunk
                yield chunk
        else:
            stream = (
                method(prepared, **kwargs)
                if config is None
                else method(prepared, config=config, **kwargs)
            )
            async for chunk in stream:
                last_chunk = chunk
                yield chunk
        if last_chunk is not None:
            self._record(plan, last_chunk)

    def batch(
        self,
        inputs: list[Any],
        config: Any = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> list[Any]:
        configs = config if isinstance(config, list) else [config] * len(inputs)
        if len(configs) != len(inputs):
            raise ValueError("config must be one config or a list matching inputs")
        results: list[Any] = []
        for item, item_config in zip(inputs, configs, strict=True):
            try:
                results.append(self.invoke(item, config=item_config, **kwargs))
            except Exception as exc:
                if not return_exceptions:
                    raise
                results.append(exc)
        return results

    async def abatch(
        self,
        inputs: list[Any],
        config: Any = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> list[Any]:
        configs = config if isinstance(config, list) else [config] * len(inputs)
        if len(configs) != len(inputs):
            raise ValueError("config must be one config or a list matching inputs")
        coroutines = [
            self.ainvoke(item, config=item_config, **kwargs)
            for item, item_config in zip(inputs, configs, strict=True)
        ]
        return list(await asyncio.gather(*coroutines, return_exceptions=return_exceptions))

    def _clone(self, model: Any, *, guarded_output: bool | None = None) -> "TokenSavingChatWrapper":
        return type(self)(
            model,
            config=self.config,
            compactor=self.compactor,
            event_handler=self.event_handler,
            _guarded_output=self._guarded_output if guarded_output is None else guarded_output,
        )

    def bind_tools(self, *args: Any, **kwargs: Any) -> "TokenSavingChatWrapper":
        return self._clone(self.base_model.bind_tools(*args, **kwargs), guarded_output=True)

    def with_structured_output(self, *args: Any, **kwargs: Any) -> "TokenSavingChatWrapper":
        return self._clone(
            self.base_model.with_structured_output(*args, **kwargs), guarded_output=True
        )

    def bind(self, *args: Any, **kwargs: Any) -> "TokenSavingChatWrapper":
        return self._clone(self.base_model.bind(*args, **kwargs))

    def with_config(self, *args: Any, **kwargs: Any) -> "TokenSavingChatWrapper":
        return self._clone(self.base_model.with_config(*args, **kwargs))

    def __getattr__(self, name: str) -> Any:
        # Preserve the underlying public model surface, including provider
        # metadata and LangChain inspection helpers.
        return getattr(self.base_model, name)
