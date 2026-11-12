"""Tests for the observation-only synthetic outcome-calibration harness."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_outcome_calibration_experiment import (
    main,
    summarise_hypothetical_labels,
)


def test_hypothetical_label_summary_reports_group_means_only() -> None:
    summaries, unlabeled, unused = summarise_hypothetical_labels(
        [
            {"candidate_id": "candidate_001", "score": 4.0, "confidence": 0.8},
            {"candidate_id": "candidate_002", "score": 2.0, "confidence": 0.4},
            {"candidate_id": "candidate_003", "score": 3.0, "confidence": 0.6},
        ],
        {
            "candidate_001": "hypothetical_strong_hire",
            "candidate_002": "hypothetical_strong_hire",
            "candidate_004": "hypothetical_weak_hire",
        },
    )

    assert summaries["hypothetical_strong_hire"] == {
        "candidate_count": 2,
        "mean_score": 3.0,
        "mean_confidence": 0.6,
    }
    assert summaries["hypothetical_weak_hire"]["candidate_count"] == 0
    assert unlabeled == ["candidate_003"]
    assert unused == ["candidate_004"]


def test_main_writes_an_explicit_synthetic_disclaimer(tmp_path: Path) -> None:
    evaluation_csv = tmp_path / "evaluation.csv"
    evaluation_csv.write_text(
        "candidate_id,score,confidence\n"
        "candidate_001,4.0,0.8\n"
        "candidate_002,1.0,0.2\n",
        encoding="utf-8",
    )
    evaluation_csv.with_suffix(".metrics.json").write_text(
        json.dumps({"sample_candidates": 2}), encoding="utf-8"
    )
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "candidate_001": "hypothetical_strong_hire",
                "candidate_002": "hypothetical_weak_hire",
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "result.json"

    assert main(
        [
            "--input",
            str(evaluation_csv),
            "--labels",
            str(labels_path),
            "--output",
            str(output_path),
        ]
    ) == 0

    output = json.loads(output_path.read_text(encoding="utf-8"))
    assert "hypothetical labels, not real hiring outcomes" in output[
        "synthetic_disclaimer"
    ]
    assert output["group_summaries"]["hypothetical_strong_hire"]["mean_score"] == 4.0
