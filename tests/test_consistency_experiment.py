"""Descriptive-statistics tests for the live consistency runner."""

import json

from scripts.run_consistency_experiment import (
    DEFAULT_OUTPUT,
    EXPERIMENT_CANDIDATE_IDS,
    REPEATS_PER_CANDIDATE,
    summarise_candidate,
)


def test_consistency_summary_uses_population_variance() -> None:
    trials = [
        {"status": "accepted", "score": 4.0, "confidence": 0.80},
        {"status": "accepted", "score": 3.0, "confidence": 0.70},
        {"status": "accepted", "score": 4.0, "confidence": 0.90},
    ]

    summary = summarise_candidate(trials)

    assert summary["score_range"] == 1.0
    assert summary["score_population_variance"] == 0.222222
    assert summary["confidence_range"] == 0.2
    assert summary["confidence_population_variance"] == 0.006667


def test_consistency_summary_excludes_failed_trials() -> None:
    trials = [
        {"status": "accepted", "score": 2.0, "confidence": 0.40},
        {"status": "failed"},
        {"status": "accepted", "score": 2.0, "confidence": 0.50},
    ]

    summary = summarise_candidate(trials)

    assert summary["accepted_trials"] == 2
    assert summary["requested_trials"] == 3
    assert summary["scores"] == [2.0, 2.0]
    assert summary["confidence_range"] == 0.1


def test_committed_live_fixture_recomputes_exactly() -> None:
    result = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))

    assert result["candidate_ids"] == list(EXPERIMENT_CANDIDATE_IDS)
    assert result["requested_trials"] == (
        len(EXPERIMENT_CANDIDATE_IDS) * REPEATS_PER_CANDIDATE
    )
    assert result["accepted_trials"] == result["requested_trials"] == 15
    assert result["model"] == "openai/gpt-5.6-terra"
    assert result["accepted_output_total_tokens"] == sum(
        trial["total_tokens"]
        for candidate in result["candidates"]
        for trial in candidate["trials"]
        if trial["status"] == "accepted"
    )

    for candidate in result["candidates"]:
        assert len(candidate["trials"]) == REPEATS_PER_CANDIDATE
        assert candidate["summary"] == summarise_candidate(
            candidate["trials"]
        )
