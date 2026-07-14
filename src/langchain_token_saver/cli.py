"""CLI support for transparent dry-runs and comparison output."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable, Sequence

from .compaction import build_optimization_plan
from .messages import get_value, message_content, message_role
from .types import CompactionConfig, OptimizationConfig


def _messages(raw: str) -> list[dict[str, Any]]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"messages must be valid JSON: {exc.msg}") from exc
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise argparse.ArgumentTypeError("messages must be a JSON array of message objects")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="langchain-token-saver")
    subparsers = parser.add_subparsers(dest="command", required=True)
    dry_run = subparsers.add_parser("dry-run", help="preview an exact compaction decision")
    dry_run.add_argument("--messages-json", required=True, type=_messages)
    dry_run.add_argument("--threshold-tokens", type=int, default=4_000)
    dry_run.add_argument("--preserve-recent", type=int, default=4)
    dry_run.add_argument("--min-net-savings", type=int, default=128)

    apply = subparsers.add_parser("apply", help="apply a safe compaction plan to JSON messages")
    apply.add_argument("--messages-json", required=True, type=_messages)
    apply.add_argument("--threshold-tokens", type=int, default=4_000)
    apply.add_argument("--preserve-recent", type=int, default=4)
    apply.add_argument("--min-net-savings", type=int, default=128)

    compare = subparsers.add_parser(
        "compare", help="compare two saved benchmark/report JSON objects"
    )
    compare.add_argument("--baseline-json", required=True)
    compare.add_argument("--optimized-json", required=True)
    return parser


def _compaction_config(args: argparse.Namespace, *, dry_run: bool) -> OptimizationConfig:
    return OptimizationConfig(
        compaction=CompactionConfig(
            threshold_tokens=args.threshold_tokens,
            preserve_recent_messages=args.preserve_recent,
            min_net_savings_tokens=args.min_net_savings,
            dry_run=dry_run,
        )
    )


def _message_as_json(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return dict(message)
    dumped = getattr(message, "model_dump", None)
    if callable(dumped):
        return dumped(mode="json")
    return {
        "type": message_role(message),
        "content": message_content(message),
        "additional_kwargs": dict(get_value(message, "additional_kwargs", {}) or {}),
    }


def main(argv: Sequence[str] | None = None, *, output: Callable[[str], None] = print) -> int:
    args = _parser().parse_args(argv)
    if args.command in {"dry-run", "apply"}:
        config = _compaction_config(args, dry_run=args.command == "dry-run")
        plan = build_optimization_plan(args.messages_json, config)
        payload = plan.as_dict()
        if args.command == "apply":
            payload["messages"] = [_message_as_json(message) for message in plan.messages]
        output(json.dumps(payload, sort_keys=True))
        return 0

    try:
        baseline = json.loads(args.baseline_json)
        optimized = json.loads(args.optimized_json)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"comparison inputs must be JSON: {exc.msg}") from exc
    baseline_tokens = baseline.get("total_tokens", baseline.get("input_tokens"))
    optimized_tokens = optimized.get("total_tokens", optimized.get("input_tokens"))
    if not isinstance(baseline_tokens, (int, float)) or not isinstance(
        optimized_tokens, (int, float)
    ):
        raise SystemExit("comparison inputs need total_tokens or input_tokens")
    output(
        json.dumps(
            {
                "baseline_tokens": baseline_tokens,
                "optimized_tokens": optimized_tokens,
                "delta_tokens": baseline_tokens - optimized_tokens,
            },
            sort_keys=True,
        )
    )
    return 0


def run() -> None:
    raise SystemExit(main(sys.argv[1:]))
