"""Evaluate a CSV/JSON batch against one approved role and hiring pack.

This is a bounded prototype lane, not a production queue or load tester.
Execution is strictly sequential by default (``--max-concurrency`` defaults
to 1); a caller can opt into a bounded ``ThreadPoolExecutor`` via that flag,
still with results reassembled by original candidate order. Candidate text
always enters the existing ``evaluate_candidate`` boundary, which applies the
same untrusted-data delimiters and validators as the single-candidate UI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.batch import (  # noqa: E402
    BatchInputError,
    load_batch_input,
    metrics_payload,
    run_batch,
    write_metrics_json,
    write_review_queue_csv,
)
from src.config import ConfigurationError, Settings  # noqa: E402
from src.llm_client import OpenRouterClient  # noqa: E402
from src.models import HiringPack, RoleSpecification  # noqa: E402


DEFAULT_ROLE = (
    ROOT
    / "data"
    / "fixtures"
    / "demo_fallback"
    / "session_demo_fallback"
    / "role_specification.json"
)
DEFAULT_PACK = (
    ROOT
    / "data"
    / "fixtures"
    / "demo_fallback"
    / "session_demo_fallback"
    / "hiring_pack.json"
)


def _model_from_file(path: Path, model_type: type):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return model_type.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise BatchInputError(f"Could not load {path}: {exc}") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate multiple candidates sequentially and write a review-"
            "ordered CSV. This does not make hiring decisions."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role", type=Path, default=DEFAULT_ROLE)
    parser.add_argument("--hiring-pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--actor", default="Batch Review Prototype")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--request-delay-seconds", type=float, default=0.25)
    parser.add_argument("--base-backoff-seconds", type=float, default=1.0)
    parser.add_argument("--maximum-backoff-seconds", type=float, default=30.0)
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help=(
            "Candidates evaluated at once (default 1, strictly sequential). "
            "Opt-in only: live calibration should start at 2-3 and step up, "
            "never jump straight to a high value."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        settings = Settings.from_env()
        settings.require_api_configuration()
        role = _model_from_file(args.role, RoleSpecification)
        hiring_pack = _model_from_file(args.hiring_pack, HiringPack)
        response_sets = load_batch_input(
            args.input,
            role=role,
            hiring_pack=hiring_pack,
        )
        batch_run = run_batch(
            response_sets,
            role=role,
            hiring_pack=hiring_pack,
            llm_client=OpenRouterClient(settings),
            actor=args.actor,
            max_attempts=args.max_attempts,
            request_delay_seconds=args.request_delay_seconds,
            base_backoff_seconds=args.base_backoff_seconds,
            maximum_backoff_seconds=args.maximum_backoff_seconds,
            max_concurrency=args.max_concurrency,
        )
    except (BatchInputError, ConfigurationError, ValidationError, ValueError) as exc:
        print(f"Batch evaluation could not start: {exc}", file=sys.stderr)
        return 2

    metrics_path = args.metrics_output or args.output.with_suffix(
        ".metrics.json"
    )
    write_review_queue_csv(args.output, batch_run.rows)
    assert settings.openrouter_model is not None
    write_metrics_json(
        metrics_path,
        batch_run,
        model=settings.openrouter_model,
    )
    metrics = metrics_payload(batch_run, model=settings.openrouter_model)
    print(f"Review-order CSV: {args.output}")
    print(f"Metrics: {metrics_path}")
    print(
        "Candidates: "
        f"{metrics['sample_candidates']}; accepted: "
        f"{metrics['accepted_candidates']}; failed: "
        f"{metrics['failed_candidates']}"
    )
    print(
        f"Elapsed wall clock: {metrics['elapsed_wall_clock_seconds']} seconds"
    )
    print(
        "Linear 500-candidate extrapolation (not a load test): "
        f"{metrics['linear_extrapolation_500_minutes']} minutes"
    )
    return 1 if metrics["failed_candidates"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
