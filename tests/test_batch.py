"""Batch-lane tests: parsing, trust-boundary reuse, review order, and metrics."""

from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor as RealThreadPoolExecutor
import json
import threading

import pytest

from src.batch import (
    BatchEvaluationRow,
    BatchInputError,
    MAX_BATCH_CANDIDATES,
    MAX_BATCH_INPUT_BYTES,
    load_batch_input,
    metrics_payload,
    parse_review_queue_csv,
    review_ordered_rows,
    run_batch,
    write_review_queue_csv,
)
from src.evaluation import CandidateEvaluationValidationError
from src.llm_client import InvalidStructuredOutputError, StructuredLLMResponse
from tests.phase6_helpers import approved_marketing_role, pack_from_draft
from tests.phase7_helpers import (
    FakeEvaluationClient,
    strong_response_set,
    valid_evaluation_draft,
)


def test_json_batch_is_bound_to_one_role_and_pack(tmp_path) -> None:
    role = approved_marketing_role()
    pack = pack_from_draft(session_id="batch")
    path = tmp_path / "batch.json"
    path.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "candidate_id": "batch_candidate_001",
                        "responses": {
                            question.question_id: "Synthetic answer"
                            for question in pack.screening_questions
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    response_sets = load_batch_input(path, role=role, hiring_pack=pack)

    assert len(response_sets) == 1
    response_set = response_sets[0]
    assert response_set.source_role_id == role.role_id
    assert response_set.source_hiring_pack_id == pack.hiring_pack_id
    assert response_set.source_metadata == {"ingest_mode": "batch_json"}


def test_csv_batch_groups_rows_by_candidate(tmp_path) -> None:
    role = approved_marketing_role()
    pack = pack_from_draft(session_id="batch")
    path = tmp_path / "batch.csv"
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=["candidate_id", "question_id", "answer_text"],
        )
        writer.writeheader()
        for candidate_id in ("candidate_001", "candidate_002"):
            for question in pack.screening_questions:
                writer.writerow(
                    {
                        "candidate_id": candidate_id,
                        "question_id": question.question_id,
                        "answer_text": "Synthetic answer",
                    }
                )

    response_sets = load_batch_input(path, role=role, hiring_pack=pack)

    assert [item.candidate_id for item in response_sets] == [
        "candidate_001",
        "candidate_002",
    ]
    assert all(
        item.source_metadata == {"ingest_mode": "batch_csv"}
        for item in response_sets
    )


def test_batch_input_rejects_excessive_file_size(tmp_path) -> None:
    role = approved_marketing_role()
    pack = pack_from_draft(session_id="batch")
    path = tmp_path / "oversized.json"
    path.write_text(" " * (MAX_BATCH_INPUT_BYTES + 1), encoding="utf-8")

    with pytest.raises(BatchInputError, match="5 MiB"):
        load_batch_input(path, role=role, hiring_pack=pack)


def test_batch_input_rejects_excessive_candidate_count(tmp_path) -> None:
    role = approved_marketing_role()
    pack = pack_from_draft(session_id="batch")
    path = tmp_path / "too_many.json"
    path.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "candidate_id": f"candidate_{index:04d}",
                        "responses": {},
                    }
                    for index in range(MAX_BATCH_CANDIDATES + 1)
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(BatchInputError, match="candidate limit"):
        load_batch_input(path, role=role, hiring_pack=pack)


def test_batch_reuses_delimited_evaluation_boundary_and_records_tokens() -> None:
    role = approved_marketing_role()
    pack = pack_from_draft(session_id="batch")
    client = FakeEvaluationClient()

    batch_run = run_batch(
        [strong_response_set()],
        role=role,
        hiring_pack=pack,
        llm_client=client,
        actor="Batch Reviewer",
        request_delay_seconds=0,
        sleeper=lambda _: None,
    )

    assert batch_run.rows[0].status == "accepted"
    assert batch_run.rows[0].score == 3.0
    assert batch_run.rows[0].total_tokens == 400
    assert client.messages is not None
    assert (
        "-----BEGIN UNTRUSTED CANDIDATE RESPONSE DATA-----"
        in client.messages[1]["content"]
    )
    assert (
        "-----END UNTRUSTED CANDIDATE RESPONSE DATA-----"
        in client.messages[1]["content"]
    )


def test_batch_retries_strict_validation_rejection() -> None:
    class FlakyEvaluationClient(FakeEvaluationClient):
        def generate_structured(self, **kwargs):
            if self.calls == 0:
                self.calls += 1
                raise CandidateEvaluationValidationError(
                    ["Synthetic score mismatch."]
                )
            return super().generate_structured(**kwargs)

    role = approved_marketing_role()
    pack = pack_from_draft(session_id="batch")
    batch_run = run_batch(
        [strong_response_set()],
        role=role,
        hiring_pack=pack,
        llm_client=FlakyEvaluationClient(),
        actor="Batch Reviewer",
        max_attempts=2,
        request_delay_seconds=0,
        base_backoff_seconds=0,
        sleeper=lambda _: None,
    )

    assert batch_run.rows[0].status == "accepted"
    assert batch_run.rows[0].attempts == 2


def test_sequential_default_ignores_max_concurrency_opt_in() -> None:
    """max_concurrency defaults to 1: the documented sequential baseline
    stays true unless a caller explicitly opts into concurrency."""
    import inspect

    assert inspect.signature(run_batch).parameters["max_concurrency"].default == 1


def test_mocked_n500_batch_scales_under_a_bounded_concurrency_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Structural proof only, not a load test: with 500 mocked candidates and
    max_concurrency=10, (1) a synchronized first wave fills but never exceeds
    the configured cap, (2) every candidate ID appears in the output exactly
    once, and (3) forced failures become status "failed" rows instead of
    crashing the batch. The synchronization is deterministic scaling
    evidence rather than a runner-speed assertion. Real sustained-rate
    provider behaviour remains unmeasured; see the 2026-07-26 batch
    concurrency decision log entry."""
    total_candidates = 500
    max_concurrency = 10
    forced_failure_marker = "FORCE_TEST_FAILURE_MARKER"
    forced_failure_stride = 50
    configured_worker_counts: list[int | None] = []

    def recording_executor(*args, **kwargs):
        configured_worker_counts.append(
            kwargs.get("max_workers", args[0] if args else None)
        )
        return RealThreadPoolExecutor(*args, **kwargs)

    monkeypatch.setattr("src.batch.ThreadPoolExecutor", recording_executor)

    role = approved_marketing_role()
    pack = pack_from_draft(session_id="batch_scale")
    base = strong_response_set()

    response_sets = []
    for index in range(total_candidates):
        responses = list(base.responses)
        if index % forced_failure_stride == 0:
            responses[0] = responses[0].model_copy(
                update={"answer_text": forced_failure_marker}
            )
        response_sets.append(
            base.model_copy(
                update={
                    "response_set_id": f"batch_scale_response_set_{index:04d}",
                    "candidate_id": f"batch_scale_candidate_{index:04d}",
                    "responses": responses,
                }
            )
        )

    class ConcurrencyTrackingClient:
        """Records the observed concurrent in-flight call count; every call
        is otherwise independent and stateless, matching the real
        ``OpenRouterClient`` (fresh ``httpx.Client`` per call)."""

        def __init__(self) -> None:
            self._lock = threading.Lock()
            self._in_flight = 0
            self._calls_started = 0
            self.max_in_flight = 0
            self._first_wave = threading.Barrier(
                max_concurrency,
                timeout=30,
            )

        def generate_structured(
            self, *, messages, response_model, temperature=None
        ):
            with self._lock:
                call_index = self._calls_started
                self._calls_started += 1
                self._in_flight += 1
                self.max_in_flight = max(self.max_in_flight, self._in_flight)
            try:
                if call_index < max_concurrency:
                    self._first_wave.wait()
                if any(
                    forced_failure_marker in message["content"]
                    for message in messages
                ):
                    raise InvalidStructuredOutputError(
                        "Forced synthetic test failure."
                    )
                return StructuredLLMResponse(
                    data=valid_evaluation_draft(),
                    model="synthetic/evaluator",
                    provider="Synthetic",
                    schema_mode="strict_json_schema",
                    input_tokens=150,
                    output_tokens=250,
                    total_tokens=400,
                    latency_ms=2,
                    attempts=1,
                )
            finally:
                with self._lock:
                    self._in_flight -= 1

    client = ConcurrencyTrackingClient()

    batch_run = run_batch(
        response_sets,
        role=role,
        hiring_pack=pack,
        llm_client=client,
        actor="Scale Test",
        max_attempts=1,
        max_concurrency=max_concurrency,
        sleeper=lambda _: None,
    )

    # (1) Never exceeded the cap, and concurrency actually happened (not an
    # accidental fallback to serial execution). The executor configuration
    # pins the upper bound, and the synchronized first wave makes reaching
    # that bound deterministic without a runner-speed threshold.
    assert configured_worker_counts == [max_concurrency]
    assert client.max_in_flight == max_concurrency

    # (2) Every candidate ID present exactly once, reassembled by original
    # index rather than completion order.
    result_ids = [row.candidate_id for row in batch_run.rows]
    expected_ids = [item.candidate_id for item in response_sets]
    assert result_ids == expected_ids
    assert len(set(result_ids)) == total_candidates

    # (3) Forced failures are flagged rows, not a crashed batch.
    failed_rows = [row for row in batch_run.rows if row.status == "failed"]
    assert len(failed_rows) == total_candidates // forced_failure_stride
    assert all(
        row.error_type == "InvalidStructuredOutputError" for row in failed_rows
    )
    accepted_rows = [row for row in batch_run.rows if row.status == "accepted"]
    assert len(accepted_rows) == total_candidates - len(failed_rows)


def test_review_queue_csv_is_safe_and_parseable(tmp_path) -> None:
    path = tmp_path / "results.csv"
    rows = [
        BatchEvaluationRow(
            candidate_id="-formula_candidate",
            score=4.0,
            confidence=0.8,
            human_review_flags=("human_review_required",),
            status="accepted",
            attempts=1,
            elapsed_wall_clock_seconds=1.0,
        ),
        BatchEvaluationRow(
            candidate_id="candidate_lower",
            score=2.0,
            confidence=0.5,
            human_review_flags=("missing_evidence",),
            status="accepted",
            attempts=1,
            elapsed_wall_clock_seconds=1.0,
        ),
    ]

    write_review_queue_csv(path, rows)
    text = path.read_text(encoding="utf-8")
    parsed = parse_review_queue_csv(text)

    assert parsed[0]["candidate_id"] == "'-formula_candidate"
    assert parsed[0]["score"] == 4.0
    assert parsed[0]["review_order"] == 1
    assert text.splitlines()[0] == (
        "review_order,candidate_id,unresolved_must_haves,score,confidence,"
        "human_review_flags"
    )


def test_uploaded_review_queue_csv_rejects_formula_prefixes() -> None:
    text = (
        "review_order,candidate_id,unresolved_must_haves,score,confidence,"
        "human_review_flags\n"
        "1,candidate_001,0,4,0.8,"
        "\"   =HYPERLINK(\"\"https://example.invalid\"\")\"\n"
    )

    with pytest.raises(BatchInputError, match="formula prefix"):
        parse_review_queue_csv(text)


def test_review_order_puts_demonstrated_must_haves_above_probe() -> None:
    all_must_haves_demonstrated = BatchEvaluationRow(
        candidate_id="candidate_complete",
        score=3.0,
        confidence=0.4,
        unresolved_must_haves=0,
        human_review_flags=(),
        status="accepted",
        attempts=1,
        elapsed_wall_clock_seconds=1.0,
    )
    high_preferred_score_with_mandatory_gap = BatchEvaluationRow(
        candidate_id="candidate_probe",
        score=5.0,
        confidence=1.0,
        unresolved_must_haves=1,
        human_review_flags=("routing:mandatory_gap_review",),
        status="accepted",
        attempts=1,
        elapsed_wall_clock_seconds=1.0,
    )

    ordered = review_ordered_rows(
        [high_preferred_score_with_mandatory_gap, all_must_haves_demonstrated]
    )

    assert [row.candidate_id for row in ordered] == [
        "candidate_complete",
        "candidate_probe",
    ]


def test_metrics_are_explicitly_a_small_sample_extrapolation() -> None:
    role = approved_marketing_role()
    pack = pack_from_draft(session_id="batch")
    batch_run = run_batch(
        [strong_response_set()],
        role=role,
        hiring_pack=pack,
        llm_client=FakeEvaluationClient(),
        actor="Batch Reviewer",
        request_delay_seconds=0,
        sleeper=lambda _: None,
    )

    metrics = metrics_payload(batch_run, model="synthetic/evaluator")

    assert metrics["benchmark_kind"] == (
        "small_sample_linear_extrapolation_not_load_test"
    )
    assert metrics["accepted_output_total_tokens"] == 400
    assert metrics["monetary_token_cost"] is None
