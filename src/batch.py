"""Batch candidate-evidence evaluation over the existing trusted boundary.

CSV and JSON records are untrusted candidate inputs. They are materialized as
``CandidateResponseSet`` objects and passed to ``evaluate_candidate`` so the
same delimiters, prompt-injection detection, traceability checks, deterministic
scoring, and human-review routing apply as in the single-candidate UI.
"""

from __future__ import annotations

import csv
import json
import random
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, ROUND_HALF_UP
from io import StringIO
from pathlib import Path
from time import perf_counter

from src.evaluation import (
    CandidateEvaluationValidationError,
    EvaluationBlockedError,
    assessment_is_demonstrated,
    evaluate_candidate,
)
from src.llm_client import (
    InvalidStructuredOutputError,
    LLMClient,
    LLMClientError,
)
from src.models import (
    CandidateEvaluation,
    CandidateQuestionResponse,
    CandidateResponseSet,
    HiringPack,
    RequirementPriority,
    RoleSpecification,
)


BATCH_RESULT_FIELDS = (
    "review_order",
    "candidate_id",
    "unresolved_must_haves",
    "score",
    "confidence",
    "human_review_flags",
)
MAX_BATCH_INPUT_BYTES = 5 * 1024 * 1024
MAX_BATCH_CANDIDATES = 1_000
_BATCH_INPUT_FIELDS = ("candidate_id", "question_id", "answer_text")
_FORMULA_PREFIXES = ("=", "+", "-", "@")


class BatchInputError(ValueError):
    """Raised when a batch file cannot be mapped to candidate responses."""


def _round_metric(value: float, places: int = 3) -> float:
    """Round benchmark metrics deterministically using conventional half-up."""
    quantum = Decimal(1).scaleb(-places)
    return float(
        Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP)
    )


@dataclass(frozen=True, slots=True)
class BatchEvaluationRow:
    """One candidate's evidence summary and operational metadata."""

    candidate_id: str
    score: float | None
    confidence: float | None
    human_review_flags: tuple[str, ...]
    status: str
    attempts: int
    elapsed_wall_clock_seconds: float
    unresolved_must_haves: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    error_type: str | None = None


@dataclass(frozen=True, slots=True)
class BatchRun:
    """Complete batch outcome, including timing used for extrapolation."""

    rows: tuple[BatchEvaluationRow, ...]
    started_at_utc: str
    completed_at_utc: str
    elapsed_wall_clock_seconds: float


def _candidate_records_from_json(payload: object) -> list[dict[str, object]]:
    if isinstance(payload, dict):
        candidates = payload.get("candidates")
    else:
        candidates = payload
    if not isinstance(candidates, list) or not candidates:
        raise BatchInputError(
            "JSON input must contain a non-empty 'candidates' list."
        )
    if not all(isinstance(item, dict) for item in candidates):
        raise BatchInputError("Every JSON candidate must be an object.")
    return candidates


def _candidate_records_from_csv(text: str) -> list[dict[str, object]]:
    reader = csv.DictReader(StringIO(text))
    if reader.fieldnames is None or not set(_BATCH_INPUT_FIELDS).issubset(
        reader.fieldnames
    ):
        raise BatchInputError(
            "CSV input requires candidate_id, question_id, and answer_text."
        )
    grouped: dict[str, list[dict[str, str]]] = {}
    for row_number, row in enumerate(reader, start=2):
        candidate_id = (row.get("candidate_id") or "").strip()
        question_id = (row.get("question_id") or "").strip()
        if not candidate_id or not question_id:
            raise BatchInputError(
                f"CSV row {row_number} needs candidate_id and question_id."
            )
        grouped.setdefault(candidate_id, []).append(
            {
                "question_id": question_id,
                "answer_text": row.get("answer_text") or "",
            }
        )
    if not grouped:
        raise BatchInputError("CSV input contains no candidate responses.")
    return [
        {"candidate_id": candidate_id, "responses": responses}
        for candidate_id, responses in grouped.items()
    ]


def _normalise_responses(value: object) -> list[dict[str, str]]:
    if isinstance(value, dict):
        return [
            {"question_id": str(question_id), "answer_text": str(answer)}
            for question_id, answer in value.items()
        ]
    if not isinstance(value, list):
        raise BatchInputError(
            "Each candidate's responses must be an object or a list."
        )
    responses: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            raise BatchInputError("Every candidate response must be an object.")
        question_id = item.get("question_id")
        answer_text = item.get("answer_text", "")
        if not isinstance(question_id, str) or not isinstance(answer_text, str):
            raise BatchInputError(
                "Each response needs string question_id and answer_text values."
            )
        responses.append(
            {"question_id": question_id, "answer_text": answer_text}
        )
    return responses


def load_batch_input(
    path: Path,
    *,
    role: RoleSpecification,
    hiring_pack: HiringPack,
) -> list[CandidateResponseSet]:
    """Load CSV/JSON candidates and bind them to one role/pack snapshot."""
    try:
        if path.stat().st_size > MAX_BATCH_INPUT_BYTES:
            raise BatchInputError("Batch input is larger than the 5 MiB limit.")
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise BatchInputError(f"Batch input could not be read: {exc}") from exc
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise BatchInputError(
                f"Batch JSON is invalid at line {exc.lineno}, column {exc.colno}."
            ) from exc
        records = _candidate_records_from_json(payload)
        ingest_mode = "batch_json"
    elif path.suffix.lower() == ".csv":
        records = _candidate_records_from_csv(text)
        ingest_mode = "batch_csv"
    else:
        raise BatchInputError("Batch input must use a .csv or .json suffix.")
    if len(records) > MAX_BATCH_CANDIDATES:
        raise BatchInputError(
            f"Batch input exceeds the {MAX_BATCH_CANDIDATES}-candidate limit."
        )

    valid_question_ids = {
        question.question_id for question in hiring_pack.screening_questions
    }
    seen_candidate_ids: set[str] = set()
    response_sets: list[CandidateResponseSet] = []
    for candidate_index, record in enumerate(records, start=1):
        candidate_id = record.get("candidate_id")
        if not isinstance(candidate_id, str):
            raise BatchInputError("Every candidate needs a string candidate_id.")
        if candidate_id in seen_candidate_ids:
            raise BatchInputError(f"Candidate ID is duplicated: {candidate_id}")
        seen_candidate_ids.add(candidate_id)
        responses = _normalise_responses(record.get("responses"))
        unknown_ids = sorted(
            {
                item["question_id"]
                for item in responses
                if item["question_id"] not in valid_question_ids
            }
        )
        if unknown_ids:
            raise BatchInputError(
                f"Candidate {candidate_id} references unknown questions: "
                + ", ".join(unknown_ids)
            )
        try:
            response_sets.append(
                CandidateResponseSet(
                    response_set_id=f"batch_response_set_{candidate_index:05d}",
                    candidate_id=candidate_id,
                    source_role_id=role.role_id,
                    source_role_version=role.version,
                    source_hiring_pack_id=hiring_pack.hiring_pack_id or "",
                    source_hiring_pack_version=hiring_pack.version,
                    responses=[
                        CandidateQuestionResponse(
                            response_id=(
                                f"batch_response_{candidate_index:05d}_"
                                f"{response_index:03d}"
                            ),
                            question_id=item["question_id"],
                            answer_text=item["answer_text"],
                        )
                        for response_index, item in enumerate(
                            responses, start=1
                        )
                    ],
                    source_metadata={"ingest_mode": ingest_mode},
                )
            )
        except ValueError as exc:
            raise BatchInputError(
                f"Candidate {candidate_id} is invalid: {exc}"
            ) from exc
    return response_sets


def _aggregate_score(evaluation: CandidateEvaluation) -> float:
    if not evaluation.assessments:
        return 0.0
    return round(
        sum(item.score for item in evaluation.assessments)
        / len(evaluation.assessments),
        3,
    )


def _human_review_flags(
    evaluation: CandidateEvaluation,
) -> tuple[str, ...]:
    flags: list[str] = []
    if evaluation.routing is not None:
        flags.append(f"routing:{evaluation.routing.value}")
    if evaluation.prompt_injection_detected:
        flags.append("prompt_injection_detected")
    if evaluation.missing_evidence:
        flags.append("missing_evidence")
    if evaluation.contradictions:
        flags.append("contradictory_evidence")
    if any(item.review_required for item in evaluation.assessments):
        flags.append("human_review_required")
    return tuple(dict.fromkeys(flags))


def _retry_delay(
    *,
    attempt: int,
    exc: Exception,
    base_backoff_seconds: float,
    maximum_backoff_seconds: float,
    jitter_seconds: float | None = None,
) -> float:
    """Return the next bounded, jittered retry delay. Pure and stateless per
    call: jitter comes from ``random.uniform`` (or an injected override for
    deterministic tests), not from any counter shared across candidates or
    threads. Jitter applies to a Retry-After-driven delay too, since a
    synchronised Retry-After value across concurrent workers is exactly the
    retry storm this is meant to avoid."""
    if isinstance(exc, LLMClientError) and exc.diagnostics is not None:
        retry_after = exc.diagnostics.retry_after_seconds
        base_delay = (
            float(retry_after)
            if retry_after is not None
            else base_backoff_seconds * (2 ** max(0, attempt - 1))
        )
    else:
        base_delay = base_backoff_seconds * (2 ** max(0, attempt - 1))
    jitter = (
        random.uniform(0, base_backoff_seconds)
        if jitter_seconds is None
        else jitter_seconds
    )
    return min(base_delay + jitter, maximum_backoff_seconds)


def _evaluate_one_candidate(
    response_set: CandidateResponseSet,
    *,
    role: RoleSpecification,
    hiring_pack: HiringPack,
    llm_client: LLMClient,
    actor: str,
    max_attempts: int,
    base_backoff_seconds: float,
    maximum_backoff_seconds: float,
    sleeper: Callable[[float], None],
) -> BatchEvaluationRow:
    """Evaluate one candidate with bounded, safe retries.

    Reads only its own arguments and calls the existing ``evaluate_candidate``
    boundary, so this is safe to run from multiple threads at once: nothing
    here is shared or mutated across candidates.
    """
    candidate_stopwatch = perf_counter()
    evaluation: CandidateEvaluation | None = None
    last_error: Exception | None = None
    attempts = 0
    for attempt in range(1, max_attempts + 1):
        attempts = attempt
        try:
            evaluation = evaluate_candidate(
                role=role,
                hiring_pack=hiring_pack,
                response_set=response_set,
                llm_client=llm_client,
                actor=actor,
            )
            break
        except CandidateEvaluationValidationError as exc:
            last_error = exc
        except InvalidStructuredOutputError as exc:
            last_error = exc
        except LLMClientError as exc:
            last_error = exc
            if exc.diagnostics is None or not exc.diagnostics.retryable:
                break
        except (EvaluationBlockedError, ValueError) as exc:
            last_error = exc
            break
        if attempt < max_attempts:
            sleeper(
                _retry_delay(
                    attempt=attempt,
                    exc=last_error,
                    base_backoff_seconds=base_backoff_seconds,
                    maximum_backoff_seconds=maximum_backoff_seconds,
                )
            )

    elapsed = round(perf_counter() - candidate_stopwatch, 3)
    if evaluation is None:
        return BatchEvaluationRow(
            candidate_id=response_set.candidate_id,
            score=None,
            confidence=None,
            human_review_flags=("evaluation_failed",),
            status="failed",
            attempts=attempts,
            elapsed_wall_clock_seconds=elapsed,
            error_type=(
                type(last_error).__name__
                if last_error is not None
                else "UnknownBatchError"
            ),
        )
    return BatchEvaluationRow(
        candidate_id=evaluation.candidate_id,
        score=_aggregate_score(evaluation),
        confidence=evaluation.overall_confidence,
        unresolved_must_haves=sum(
            assessment.requirement_priority is RequirementPriority.MUST_HAVE
            and not assessment_is_demonstrated(assessment)
            for assessment in evaluation.assessments
        ),
        human_review_flags=_human_review_flags(evaluation),
        status="accepted",
        attempts=attempts,
        elapsed_wall_clock_seconds=elapsed,
        input_tokens=evaluation.input_tokens,
        output_tokens=evaluation.output_tokens,
        total_tokens=evaluation.total_tokens,
    )


def run_batch(
    response_sets: Sequence[CandidateResponseSet],
    *,
    role: RoleSpecification,
    hiring_pack: HiringPack,
    llm_client: LLMClient,
    actor: str,
    max_attempts: int = 3,
    request_delay_seconds: float = 0.25,
    base_backoff_seconds: float = 1.0,
    maximum_backoff_seconds: float = 30.0,
    max_concurrency: int = 1,
    sleeper: Callable[[float], None] = time.sleep,
) -> BatchRun:
    """Evaluate candidates with bounded, safe retries.

    ``max_concurrency`` defaults to 1: candidates are evaluated one at a time,
    exactly as before, including the ``request_delay_seconds`` pacing between
    them. This keeps the documented sequential baseline true unless a caller
    explicitly opts into concurrency. With ``max_concurrency > 1``, candidates
    run across a bounded ``ThreadPoolExecutor`` -- safe because
    ``OpenRouterClient.generate_structured`` opens a fresh synchronous
    ``httpx.Client`` per call and touches no shared mutable state -- and
    ``request_delay_seconds`` does not apply, since the concurrency cap and
    each retry's jitter already bound the request rate. Rows are always
    reassembled by each candidate's original position, never by whichever
    future completes first.
    """
    if not response_sets:
        raise ValueError("At least one candidate response set is required.")
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least one.")
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be at least one.")
    if min(
        request_delay_seconds,
        base_backoff_seconds,
        maximum_backoff_seconds,
    ) < 0:
        raise ValueError("Batch timing controls cannot be negative.")

    started_at = datetime.now(UTC)
    stopwatch = perf_counter()

    def evaluate_one(response_set: CandidateResponseSet) -> BatchEvaluationRow:
        return _evaluate_one_candidate(
            response_set,
            role=role,
            hiring_pack=hiring_pack,
            llm_client=llm_client,
            actor=actor,
            max_attempts=max_attempts,
            base_backoff_seconds=base_backoff_seconds,
            maximum_backoff_seconds=maximum_backoff_seconds,
            sleeper=sleeper,
        )

    rows: list[BatchEvaluationRow]
    if max_concurrency == 1:
        rows = []
        for candidate_index, response_set in enumerate(response_sets):
            rows.append(evaluate_one(response_set))
            if (
                candidate_index < len(response_sets) - 1
                and request_delay_seconds
            ):
                sleeper(request_delay_seconds)
    else:
        rows_by_index: list[BatchEvaluationRow | None] = [None] * len(
            response_sets
        )
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            future_to_index = {
                executor.submit(evaluate_one, response_set): index
                for index, response_set in enumerate(response_sets)
            }
            for future in as_completed(future_to_index):
                rows_by_index[future_to_index[future]] = future.result()
        rows = [row for row in rows_by_index if row is not None]

    completed_at = datetime.now(UTC)
    return BatchRun(
        rows=tuple(rows),
        started_at_utc=started_at.isoformat().replace("+00:00", "Z"),
        completed_at_utc=completed_at.isoformat().replace("+00:00", "Z"),
        elapsed_wall_clock_seconds=round(perf_counter() - stopwatch, 6),
    )


def review_ordered_rows(
    rows: Sequence[BatchEvaluationRow],
) -> list[BatchEvaluationRow]:
    """Order evidence summaries for review without making a hiring decision."""
    return sorted(
        rows,
        key=lambda item: (
            item.score is None,
            item.unresolved_must_haves,
            -(item.confidence or 0.0),
            -(item.score or 0.0),
            item.candidate_id,
        ),
    )


def _csv_safe(value: object) -> str:
    text = str(value)
    return (
        f"'{text}"
        if text.lstrip().startswith(_FORMULA_PREFIXES)
        else text
    )


def write_review_queue_csv(
    path: Path,
    rows: Sequence[BatchEvaluationRow],
) -> None:
    """Write exactly the reviewer-facing batch columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=BATCH_RESULT_FIELDS)
        writer.writeheader()
        for review_order, row in enumerate(review_ordered_rows(rows), start=1):
            writer.writerow(
                {
                    "review_order": review_order,
                    "candidate_id": _csv_safe(row.candidate_id),
                    "unresolved_must_haves": row.unresolved_must_haves,
                    "score": "" if row.score is None else f"{row.score:.3f}",
                    "confidence": (
                        ""
                        if row.confidence is None
                        else f"{row.confidence:.3f}"
                    ),
                    "human_review_flags": _csv_safe(
                        ";".join(row.human_review_flags)
                    ),
                }
            )


def metrics_payload(
    batch_run: BatchRun,
    *,
    model: str,
) -> dict[str, object]:
    """Return reproducible benchmark metadata and a linear extrapolation."""
    total = len(batch_run.rows)
    accepted = [row for row in batch_run.rows if row.status == "accepted"]
    failed = total - len(accepted)
    # Publish and extrapolate from the same millisecond-resolution value so a
    # reviewer can reproduce the arithmetic shown in the metrics JSON.
    elapsed = max(
        _round_metric(batch_run.elapsed_wall_clock_seconds),
        0.001,
    )
    extrapolated_seconds = elapsed / total * 500
    accepted_token_counts = [
        row.total_tokens for row in accepted if row.total_tokens is not None
    ]
    return {
        "benchmark_kind": "small_sample_linear_extrapolation_not_load_test",
        "model": model,
        "started_at_utc": batch_run.started_at_utc,
        "completed_at_utc": batch_run.completed_at_utc,
        "sample_candidates": total,
        "accepted_candidates": len(accepted),
        "failed_candidates": failed,
        "elapsed_wall_clock_seconds": elapsed,
        "seconds_per_candidate": _round_metric(elapsed / total),
        "candidates_per_minute": _round_metric(total / elapsed * 60),
        "linear_extrapolation_500_seconds": _round_metric(
            extrapolated_seconds, 3
        ),
        "linear_extrapolation_500_minutes": _round_metric(
            extrapolated_seconds / 60, 3
        ),
        "linear_extrapolation_500_hours": _round_metric(
            extrapolated_seconds / 3600, 3
        ),
        "accepted_output_total_tokens": (
            sum(accepted_token_counts) if accepted_token_counts else None
        ),
        "token_measurement_scope": (
            "Accepted evaluations only; rejected provider outputs do not "
            "expose usage through evaluate_candidate."
        ),
        "monetary_token_cost": None,
        "monetary_cost_note": (
            "OpenRouter price metadata was not returned by this evaluation "
            "boundary, so no currency estimate is claimed."
        ),
    }


def write_metrics_json(
    path: Path,
    batch_run: BatchRun,
    *,
    model: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metrics_payload(batch_run, model=model), indent=2) + "\n",
        encoding="utf-8",
    )


def parse_review_queue_csv(text: str) -> list[dict[str, object]]:
    """Validate a batch result before rendering it in Streamlit."""
    reader = csv.DictReader(StringIO(text))
    if reader.fieldnames != list(BATCH_RESULT_FIELDS):
        raise BatchInputError(
            "Batch result columns must be: " + ", ".join(BATCH_RESULT_FIELDS)
        )
    rows: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for row_number, row in enumerate(reader, start=2):
        candidate_id = (row.get("candidate_id") or "").strip()
        human_review_flags = row.get("human_review_flags") or ""
        if candidate_id.lstrip().startswith(_FORMULA_PREFIXES) or (
            human_review_flags.lstrip().startswith(_FORMULA_PREFIXES)
        ):
            raise BatchInputError(
                f"Batch result row {row_number} has an unsafe spreadsheet "
                "formula prefix."
            )
        if not candidate_id or candidate_id in seen_ids:
            raise BatchInputError(
                f"Batch result row {row_number} has a missing or duplicate "
                "candidate_id."
            )
        seen_ids.add(candidate_id)
        try:
            review_order = int(row["review_order"])
            unresolved_must_haves = int(row["unresolved_must_haves"])
            score = (
                None
                if not (row.get("score") or "").strip()
                else float(row["score"])
            )
            confidence = (
                None
                if not (row.get("confidence") or "").strip()
                else float(row["confidence"])
            )
        except (TypeError, ValueError) as exc:
            raise BatchInputError(
                f"Batch result row {row_number} has invalid numeric values."
            ) from exc
        expected_review_order = row_number - 1
        if review_order != expected_review_order:
            raise BatchInputError(
                f"Batch result row {row_number} review_order must be "
                f"{expected_review_order}."
            )
        if unresolved_must_haves < 0:
            raise BatchInputError(
                f"Batch result row {row_number} unresolved_must_haves "
                "cannot be negative."
            )
        if score is not None and not 0 <= score <= 5:
            raise BatchInputError(
                f"Batch result row {row_number} score must be from 0 to 5."
            )
        if confidence is not None and not 0 <= confidence <= 1:
            raise BatchInputError(
                f"Batch result row {row_number} confidence must be from 0 to 1."
            )
        rows.append(
            {
                "review_order": review_order,
                "candidate_id": candidate_id,
                "unresolved_must_haves": unresolved_must_haves,
                "score": score,
                "confidence": confidence,
                "human_review_flags": human_review_flags,
            }
        )
    if not rows:
        raise BatchInputError("Batch result contains no candidate rows.")
    return rows
