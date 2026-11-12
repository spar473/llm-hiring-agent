"""Run a reproducible live candidate-evaluation consistency experiment.

Five synthetic candidate response sets are evaluated three times each against
the same approved role and hiring pack. Every trial calls ``evaluate_candidate``
independently. Bounded outer retries handle strict semantic-output rejection;
all rejected attempts remain visible in the result fixture.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
from statistics import pvariance
import sys
from time import perf_counter, sleep

from pydantic import ValidationError


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.batch import BatchInputError, load_batch_input  # noqa: E402
from src.config import ConfigurationError, Settings  # noqa: E402
from src.evaluation import (  # noqa: E402
    CandidateEvaluationValidationError,
    EvaluationBlockedError,
    evaluate_candidate,
)
from src.llm_client import (  # noqa: E402
    InvalidStructuredOutputError,
    LLMClientError,
    OpenRouterClient,
)
from src.models import HiringPack, RoleSpecification  # noqa: E402


EXPERIMENT_CANDIDATE_IDS = (
    "batch_candidate_001",
    "batch_candidate_004",
    "batch_candidate_005",
    "batch_candidate_007",
    "batch_candidate_008",
)
REPEATS_PER_CANDIDATE = 3
DEFAULT_INPUT = ROOT / "data" / "fixtures" / "batch_candidate_responses_10.json"
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
DEFAULT_OUTPUT = (
    ROOT / "data" / "fixtures" / "consistency_experiment_results.json"
)


def _load_model(path: Path, model_type: type):
    try:
        return model_type.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError) as exc:
        raise BatchInputError(f"Could not load {path}: {exc}") from exc


def _aggregate_score(evaluation) -> float:
    return round(
        sum(item.score for item in evaluation.assessments)
        / len(evaluation.assessments),
        3,
    )


def summarise_candidate(trials: list[dict[str, object]]) -> dict[str, object]:
    """Compute exact descriptive statistics over accepted trial results."""
    accepted = [
        trial
        for trial in trials
        if trial.get("status") == "accepted"
    ]
    scores = [float(trial["score"]) for trial in accepted]
    confidences = [float(trial["confidence"]) for trial in accepted]
    return {
        "accepted_trials": len(accepted),
        "requested_trials": len(trials),
        "scores": scores,
        "score_min": min(scores) if scores else None,
        "score_max": max(scores) if scores else None,
        "score_range": round(max(scores) - min(scores), 6) if scores else None,
        "score_population_variance": (
            round(pvariance(scores), 6) if len(scores) > 1 else None
        ),
        "confidences": confidences,
        "confidence_min": min(confidences) if confidences else None,
        "confidence_max": max(confidences) if confidences else None,
        "confidence_range": (
            round(max(confidences) - min(confidences), 6)
            if confidences
            else None
        ),
        "confidence_population_variance": (
            round(pvariance(confidences), 6)
            if len(confidences) > 1
            else None
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run five synthetic candidates through three independent live "
            "evaluations each and save score/confidence consistency evidence."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--role", type=Path, default=DEFAULT_ROLE)
    parser.add_argument("--hiring-pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-attempts-per-trial", type=int, default=5)
    parser.add_argument("--retry-delay-seconds", type=float, default=0.5)
    parser.add_argument("--request-delay-seconds", type=float, default=0.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_attempts_per_trial < 1:
        print("max-attempts-per-trial must be at least one.", file=sys.stderr)
        return 2
    if min(args.retry_delay_seconds, args.request_delay_seconds) < 0:
        print("Experiment delays cannot be negative.", file=sys.stderr)
        return 2

    try:
        settings = Settings.from_env()
        settings.require_api_configuration()
        role = _load_model(args.role, RoleSpecification)
        hiring_pack = _load_model(args.hiring_pack, HiringPack)
        all_response_sets = load_batch_input(
            args.input,
            role=role,
            hiring_pack=hiring_pack,
        )
    except (
        BatchInputError,
        ConfigurationError,
        ValidationError,
        ValueError,
    ) as exc:
        print(f"Consistency experiment could not start: {exc}", file=sys.stderr)
        return 2

    by_candidate = {item.candidate_id: item for item in all_response_sets}
    missing = [
        candidate_id
        for candidate_id in EXPERIMENT_CANDIDATE_IDS
        if candidate_id not in by_candidate
    ]
    if missing:
        print(
            "Consistency input is missing candidates: " + ", ".join(missing),
            file=sys.stderr,
        )
        return 2

    client = OpenRouterClient(settings)
    experiment_started = datetime.now(UTC)
    stopwatch = perf_counter()
    candidate_results: list[dict[str, object]] = []
    trial_number = 0
    for candidate_id in EXPERIMENT_CANDIDATE_IDS:
        response_set = by_candidate[candidate_id]
        trials: list[dict[str, object]] = []
        for repeat in range(1, REPEATS_PER_CANDIDATE + 1):
            trial_number += 1
            trial_started = datetime.now(UTC)
            trial_stopwatch = perf_counter()
            rejected_attempts: list[dict[str, str]] = []
            evaluation = None
            attempts = 0
            for attempt in range(1, args.max_attempts_per_trial + 1):
                attempts = attempt
                try:
                    evaluation = evaluate_candidate(
                        role=role,
                        hiring_pack=hiring_pack,
                        response_set=response_set,
                        llm_client=client,
                        actor="Consistency Experiment Reviewer",
                    )
                    break
                except (
                    CandidateEvaluationValidationError,
                    InvalidStructuredOutputError,
                ) as exc:
                    rejected_attempts.append(
                        {
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        }
                    )
                except LLMClientError as exc:
                    rejected_attempts.append(
                        {
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        }
                    )
                    if exc.diagnostics is None or not exc.diagnostics.retryable:
                        break
                except (EvaluationBlockedError, ValueError) as exc:
                    rejected_attempts.append(
                        {
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        }
                    )
                    break
                if attempt < args.max_attempts_per_trial:
                    sleep(args.retry_delay_seconds)

            elapsed = round(perf_counter() - trial_stopwatch, 3)
            if evaluation is None:
                trial = {
                    "trial": repeat,
                    "status": "failed",
                    "started_at_utc": trial_started.isoformat().replace(
                        "+00:00", "Z"
                    ),
                    "elapsed_wall_clock_seconds": elapsed,
                    "outer_attempts": attempts,
                    "rejected_attempts": rejected_attempts,
                }
                print(
                    f"{candidate_id} trial {repeat}: failed after "
                    f"{attempts} attempt(s)"
                )
            else:
                trial = {
                    "trial": repeat,
                    "status": "accepted",
                    "started_at_utc": trial_started.isoformat().replace(
                        "+00:00", "Z"
                    ),
                    "elapsed_wall_clock_seconds": elapsed,
                    "outer_attempts": attempts,
                    "rejected_attempts": rejected_attempts,
                    "evaluation_id": evaluation.evaluation_id,
                    "model": evaluation.model,
                    "provider": evaluation.provider,
                    "score": _aggregate_score(evaluation),
                    "confidence": evaluation.overall_confidence,
                    "routing": (
                        evaluation.routing.value
                        if evaluation.routing is not None
                        else None
                    ),
                    "prompt_injection_detected": (
                        evaluation.prompt_injection_detected
                    ),
                    "missing_evidence_count": len(
                        evaluation.missing_evidence
                    ),
                    "contradiction_count": len(evaluation.contradictions),
                    "input_tokens": evaluation.input_tokens,
                    "output_tokens": evaluation.output_tokens,
                    "total_tokens": evaluation.total_tokens,
                }
                print(
                    f"{candidate_id} trial {repeat}: "
                    f"score={trial['score']}, "
                    f"confidence={trial['confidence']}, "
                    f"attempts={attempts}"
                )
            trials.append(trial)
            if trial_number < (
                len(EXPERIMENT_CANDIDATE_IDS) * REPEATS_PER_CANDIDATE
            ):
                sleep(args.request_delay_seconds)

        candidate_results.append(
            {
                "candidate_id": candidate_id,
                "trials": trials,
                "summary": summarise_candidate(trials),
            }
        )

    elapsed = round(perf_counter() - stopwatch, 3)
    accepted_trials = sum(
        int(item["summary"]["accepted_trials"])
        for item in candidate_results
    )
    total_tokens = sum(
        int(trial["total_tokens"])
        for item in candidate_results
        for trial in item["trials"]
        if trial.get("status") == "accepted"
        and trial.get("total_tokens") is not None
    )
    output = {
        "experiment_label": (
            "Live synthetic consistency experiment; descriptive only"
        ),
        "model": settings.openrouter_model,
        "source_role_id": role.role_id,
        "source_role_version": role.version,
        "source_hiring_pack_id": hiring_pack.hiring_pack_id,
        "source_hiring_pack_version": hiring_pack.version,
        "source_input_fixture": str(args.input.relative_to(ROOT)),
        "candidate_ids": list(EXPERIMENT_CANDIDATE_IDS),
        "repeats_per_candidate": REPEATS_PER_CANDIDATE,
        "requested_trials": (
            len(EXPERIMENT_CANDIDATE_IDS) * REPEATS_PER_CANDIDATE
        ),
        "accepted_trials": accepted_trials,
        "started_at_utc": experiment_started.isoformat().replace(
            "+00:00", "Z"
        ),
        "completed_at_utc": datetime.now(UTC).isoformat().replace(
            "+00:00", "Z"
        ),
        "elapsed_wall_clock_seconds": elapsed,
        "accepted_output_total_tokens": total_tokens or None,
        "token_scope": (
            "Accepted evaluations only; rejected provider outputs do not "
            "expose usage through evaluate_candidate."
        ),
        "variance_definition": (
            "Population variance across the three accepted repeats for each "
            "candidate; descriptive, not inferential."
        ),
        "candidates": candidate_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Results: {args.output}")
    print(
        f"Accepted trials: {accepted_trials}/{output['requested_trials']}; "
        f"elapsed: {elapsed} seconds; accepted-output tokens: "
        f"{output['accepted_output_total_tokens']}"
    )
    return 0 if accepted_trials == output["requested_trials"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
