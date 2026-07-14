# Project Requirement: `langchain-token-saver`

## 1. Purpose

`langchain-token-saver` is a pip-installable LangChain ecosystem package focused on one problem only: reduce token usage in long-running LLM workflows without breaking task quality, tool protocol correctness, or structured outputs.

This project is not for audio generation, image generation, or general multimodal abstraction work. The entire scope is token optimization for text workflows.

## 2. Problem Statement

LangChain applications often spend too many tokens on:

- repeated conversation history
- verbose assistant responses
- uncompressed tool output
- unnecessary context carried forward between steps

This creates higher cost, higher latency, and in some cases degraded reliability because the model sees too much irrelevant text.

The project should solve this by making token reduction measurable, opt-in, and safe.

## 3. Product Goal

Build a reusable drop-in chat-model wrapper that:

- reduces prompt and completion token usage
- preserves correctness of tool calls and structured outputs
- provides transparent measurements of savings
- works as a standalone LangChain and LangGraph compatible package
- intercepts model calls before they leave the application

## 4. Non-Goals

Do not build:

- audio generation support
- image model support
- a generic multimodal abstraction layer
- a LangChain core fork
- opaque prompt rewriting with no measurement
- any feature that silently changes meaning, policy, code, JSON, or tool arguments
- a generic vision/audio abstraction layer

## 5. Target Users

Primary users:

- developers building LangChain agents
- developers building LangGraph agents
- developers running long conversations
- teams paying attention to token cost and latency

Secondary users:

- maintainers who want a reusable LangChain wrapper package
- developers who want benchmarkable prompt optimization strategies

## 6. Core Principles

The project must follow these principles:

- Measure before claiming savings.
- Optimize only when savings are proven.
- Preserve protocol correctness over compression.
- Keep features opt-in.
- Keep the package provider-aware, not provider-locked.
- Prefer transparent reporting over hidden heuristics.
- Preserve the original model surface as closely as possible.

## 7. Scope

### In scope

- direct chat-model wrappers
- response brevity optimization
- context compaction
- tool output compaction
- token usage reporting
- benchmark harness
- safety/invariant checks
- provider capability handling
- examples and docs
- CLI-based dry-run and comparison mode
- example app for quick validation

### Out of scope

- vision support
- audio support
- image support
- core LangChain changes before evidence exists
- prompt generation for unrelated use cases
- provider support beyond OpenAI and Anthropic in v0.1

## 8. Features

### 8.1 Token Usage Reporting

The package must report token usage in a way that is visible to users.

Requirements:

- capture provider-reported input tokens
- capture provider-reported output tokens
- capture total tokens when available
- report tokens spent by compaction itself
- report net token savings
- label estimated counts as estimates when provider data is unavailable

Acceptance criteria:

- users can inspect token savings for each run
- reports separate baseline usage from optimization overhead
- reports are deterministic for a fixed trace and model version
- reports distinguish exact provider usage from local estimates

### 8.2 Response Brevity Optimization

Add a wrapper behavior that makes model responses shorter when the task allows it.

Requirements:

- opt-in terse/telegraphic style
- do not alter system policy
- do not rewrite tool schemas
- do not rewrite tool call arguments
- do not change code blocks, JSON, citations, or structured output requests
- preserve meaning while reducing verbosity

Acceptance criteria:

- response length decreases on benchmarked tasks where brevity is acceptable
- no increase in critical correctness failures
- output remains parseable where structured output is required

### 8.3 Context Compaction

Add a wrapper behavior that reduces future prompt size by compressing older history and tool output.

Requirements:

- trigger only after a configurable threshold
- compact only eligible older text context
- preserve the most recent turns
- preserve protocol-critical material exactly
- preserve tool-call/result ordering
- preserve IDs, URLs, code blocks, and structured payloads
- treat compacted text as untrusted quoted data

Compacted memory should be structured, for example:

- facts
- decisions
- constraints
- open work
- references

Acceptance criteria:

- long traces use fewer tokens after compaction
- the model still completes tasks successfully
- tool protocol stays valid
- critical facts are not lost

### 8.4 Dry-Run and Comparison Mode

The package and CLI must support a mode that previews optimization before mutation.

Requirements:

- show candidate content for compaction
- show estimated before/after tokens
- show compaction cost
- show predicted net savings
- allow skipping compaction if savings are not sufficient
- expose the exact sections that would be compacted

Acceptance criteria:

- users can see why compaction did or did not happen
- dry-run output matches actual behavior closely enough for debugging

### 8.5 Benchmark Harness

Add a benchmark harness to compare baseline LangChain behavior against optimized behavior.

Requirements:

- support a set of 20 to 50 representative traces
- compare baseline and optimized runs with the same model/settings
- record provider-reported tokens
- record latency
- record success/failure status
- record quality gate failures
- compare exact provider usage against estimated usage where applicable

Acceptance criteria:

- benchmarks run repeatably
- benchmark output is easy to inspect
- benchmark results can be used to justify ship/no-ship decisions

### 8.6 Provider Capability Matrix

The package must know which providers support which measurements and behaviors.

Requirements:

- handle provider-reported usage when available
- use local estimates only as fallback
- clearly mark missing data
- avoid claiming universal percentages across all providers

Acceptance criteria:

- unsupported providers fail gracefully
- the package does not misreport estimated counts as exact counts

### 8.7 Structured Observability Events

The package must emit structured events for important runtime decisions.

Requirements:

- emit token savings events
- emit compaction decision events
- emit fallback events when compression fails or is skipped
- emit dry-run preview events
- keep event payloads machine-readable and stable

Acceptance criteria:

- event consumers can distinguish exact usage from estimated usage
- event consumers can identify when the wrapper fell back to uncompressed context
- event payloads are suitable for logging and metrics systems

## 9. Success Criteria

The project is successful only if the following are true:

1. Median net token savings are positive on the benchmark set.
2. Quality gate passes on benchmark traces.
3. Tool protocol correctness is preserved.
4. Critical facts are retained after compaction.
5. Users can see a clear token savings report.
6. The package works as a standalone reusable LangChain ecosystem project.
7. The package works as a drop-in wrapper for OpenAI and Anthropic chat models in LangChain and LangGraph workflows.

### Quantitative targets

Use these as initial ship targets:

- median net token savings > 0
- p95 protocol invariant failures = 0
- p95 critical fact loss rate = 0
- benchmark coverage = at least 20 representative traces
- compaction skipped whenever predicted savings do not exceed compaction cost plus margin
- fallback to original uncompressed context occurs when compaction fails

These thresholds can be tightened after the first benchmark run.

## 10. Quality Gates

Do not merge or ship a feature unless all of these pass:

- task still succeeds end to end
- no invalid tool calls are introduced
- JSON and structured outputs remain valid
- code fences and code snippets are preserved
- URLs, IDs, and references remain intact
- token savings are net positive or the feature is explicitly marked experimental

## 11. Test Plan

### 11.1 Unit Tests

Write unit tests for:

- token accounting report generation
- compaction threshold logic
- break-even logic
- dry-run output
- provider capability handling
- response brevity mode selection
- text blocks that must not be rewritten
- wrapper construction and config validation
- sync and async call paths
- streaming pass-through behavior
- fallback-on-compaction-failure behavior

### 11.2 Integration Tests

Write integration tests for:

- one or more LangChain agent traces
- OpenAI wrapper traces
- Anthropic wrapper traces
- wrapper call ordering
- tool-call preservation
- context compaction with real message objects
- fallback behavior when provider usage metadata is missing
- CLI dry-run and apply modes
- example app execution

### 11.3 Regression Tests

Add regression tests for:

- broken tool/result ordering
- malformed JSON after compaction
- lost URLs or IDs
- accidental rewriting of code blocks
- compaction happening too early
- false token savings claims
- wrapper surface regressions against ChatOpenAI and ChatAnthropic constructor/call patterns
- fallback-to-original-context on summarization errors
- structured observability event payloads and sequence

### 11.4 Benchmark Tests

Benchmark tests should verify:

- baseline vs optimized token totals
- output token reduction where applicable
- input token reduction after compaction
- net savings after compaction cost
- latency impact
- quality gate pass rate
- dry-run accuracy against actual apply mode

### 11.5 Safety Tests

Add safety tests for:

- prompt injection inside old history
- prompt injection inside tool output
- quoted untrusted text in compacted memory
- preservation of system instructions
- preservation of tool schema constraints
- preservation of streaming behavior
- preservation of sync and async parity

## 12. Recommended Architecture

The first release should be a standalone package with the following shape:

- `TokenSavingChatWrapper` or equivalent drop-in proxy around `ChatOpenAI` and `ChatAnthropic`
- `ContextCompactionStrategy`
- `TokenSavingsReport`
- benchmark runner
- provider capability helpers
- CLI for dry-run/apply comparison
- example app for quick validation
- example traces and docs

Implementation expectations:

- use LangChain chat-model interfaces and wrapper/proxy hooks
- keep optimization logic isolated from app code
- keep config explicit
- keep reporting separate from transformation logic
- support sync, async, and streaming entry points
- preserve the original constructor/call shape as closely as possible
- fall back to original uncompressed context if compaction fails
- support OpenAI and Anthropic first, with other providers deferred

## 13. Release Strategy

Release in this order:

1. baseline benchmark and invariants
2. direct wrapper for OpenAI and Anthropic
3. response brevity behavior
4. compaction behavior
5. structured observability events
6. dry-run and reporting
7. CLI and example app
8. benchmark suite
9. documentation

Do not add unrelated features before the token optimization path is proven.

## 14. Documentation Requirements

The repository must include:

- project overview
- install instructions
- configuration examples
- benchmark instructions
- explanation of token savings measurement
- explanation of safety boundaries
- examples for terse mode and compaction mode
- CLI usage
- example app walkthrough
- wrapper adoption notes for LangChain and LangGraph users

## 15. Development Workflow

The project should be developed with the Matt Pocock-style workflow and supporting skills:

- use TDD for feature work
- keep pre-commit checks enabled
- break work into small, reviewable units
- write explicit tests before broad refactors
- run code review on the final branch before merge

Recommended supporting skills/processes:

- `tdd`
- `setup-pre-commit`
- `to-issues` or `to-spec` for work breakdown
- `code-review` before merging

## 16. Final Definition of Done

The project is done for v0.1 when:

- the package can be installed and imported
- the main token optimization wrapper works
- benchmark output shows net savings
- tests protect protocol correctness
- docs explain when to use the package
- the scope remains limited to token optimization only
- OpenAI and Anthropic wrappers support sync, async, and streaming flows
- CLI dry-run and apply modes work
- example app demonstrates token savings and fallback behavior
