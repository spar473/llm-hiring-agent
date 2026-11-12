"""Summarise hypothetical outcome labels against synthetic evaluation output.

This is an auditable observation-only hook for a future, governed outcome
learning process. It does not train, tune, or otherwise alter application
scoring, and its built-in labels are illustrative rather than real outcomes.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_INPUT = ROOT / "data" / "fixtures" / "batch_evaluation_results_10.csv"
DEFAULT_OUTPUT = (
    ROOT / "data" / "fixtures" / "outcome_calibration_experiment_results.json"
)
HYPOTHETICAL_OUTCOME_LABELS = frozenset(
    {"hypothetical_strong_hire", "hypothetical_weak_hire"}
)
DEFAULT_HYPOTHETICAL_LABELS = {
    "batch_candidate_001": "hypothetical_strong_hire",
    "batch_candidate_002": "hypothetical_weak_hire",
    "batch_candidate_003": "hypothetical_strong_hire",
    "batch_candidate_004": "hypothetical_weak_hire",
    "batch_candidate_005": "hypothetical_strong_hire",
    "batch_candidate_006": "hypothetical_weak_hire",
    "batch_candidate_007": "hypothetical_strong_hire",
    "batch_candidate_008": "hypothetical_weak_hire",
    "batch_candidate_009": "hypothetical_strong_hire",
    "batch_candidate_010": "hypothetical_weak_hire",
}


class OutcomeCalibrationInputError(ValueError):
    """Raised when an observation input cannot support the summary."""


def _display_path(path: Path) -> str:
    """Render a repository-relative path when possible for auditable output."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _load_labels(path: Path | None) -> dict[str, str]:
    """Load validated hypothetical labels or use the built-in example mapping."""
    if path is None:
        return dict(DEFAULT_HYPOTHETICAL_LABELS)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OutcomeCalibrationInputError(
            f"Could not load hypothetical labels from {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise OutcomeCalibrationInputError("Hypothetical labels must be a JSON object.")
    labels = {str(candidate_id): label for candidate_id, label in payload.items()}
    invalid = sorted(
        candidate_id
        for candidate_id, label in labels.items()
        if not isinstance(label, str) or label not in HYPOTHETICAL_OUTCOME_LABELS
    )
    if invalid:
        raise OutcomeCalibrationInputError(
            "Labels must be hypothetical_strong_hire or hypothetical_weak_hire; "
            f"invalid candidate ids: {', '.join(invalid)}"
        )
    return labels


def _load_evaluation_rows(path: Path) -> list[dict[str, float | str]]:
    """Read score and confidence from a previously produced batch CSV."""
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required_columns = {"candidate_id", "score", "confidence"}
            if reader.fieldnames is None or not required_columns.issubset(
                reader.fieldnames
            ):
                raise OutcomeCalibrationInputError(
                    "Evaluation CSV must include candidate_id, score, and confidence."
                )
            rows = [
                {
                    "candidate_id": row["candidate_id"],
                    "score": float(row["score"]),
                    "confidence": float(row["confidence"]),
                }
                for row in reader
            ]
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise OutcomeCalibrationInputError(
            f"Could not load evaluation output from {path}: {exc}"
        ) from exc
    if not rows:
        raise OutcomeCalibrationInputError("Evaluation CSV contains no candidates.")
    candidate_ids = [str(row["candidate_id"]) for row in rows]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise OutcomeCalibrationInputError("Evaluation CSV contains duplicate candidate ids.")
    return rows


def _load_metrics(path: Path) -> dict[str, object]:
    """Read the metrics JSON produced alongside the batch evaluation output."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OutcomeCalibrationInputError(
            f"Could not load evaluation metrics from {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise OutcomeCalibrationInputError("Evaluation metrics must be a JSON object.")
    return payload


def summarise_hypothetical_labels(
    rows: list[dict[str, float | str]], labels: dict[str, str]
) -> tuple[dict[str, dict[str, float | int | None]], list[str], list[str]]:
    """Compute group means only; this summary makes no inferential claim."""
    grouped = {label: [] for label in sorted(HYPOTHETICAL_OUTCOME_LABELS)}
    candidate_ids = set()
    unlabeled_candidate_ids = []
    for row in rows:
        candidate_id = str(row["candidate_id"])
        candidate_ids.add(candidate_id)
        label = labels.get(candidate_id)
        if label is None:
            unlabeled_candidate_ids.append(candidate_id)
            continue
        grouped[label].append(row)

    summaries = {}
    for label, group in grouped.items():
        scores = [float(row["score"]) for row in group]
        confidences = [float(row["confidence"]) for row in group]
        summaries[label] = {
            "candidate_count": len(group),
            "mean_score": round(fmean(scores), 6) if scores else None,
            "mean_confidence": (
                round(fmean(confidences), 6) if confidences else None
            ),
        }
    unused_label_candidate_ids = sorted(set(labels).difference(candidate_ids))
    return summaries, sorted(unlabeled_candidate_ids), unused_label_candidate_ids


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Write descriptive score and confidence means by hypothetical "
            "outcome label. This does not claim real calibration."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--labels",
        type=Path,
        help=(
            "JSON object mapping candidate IDs to hypothetical_strong_hire or "
            "hypothetical_weak_hire. Uses an inline illustrative mapping when omitted."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    metrics_path = args.input.with_suffix(".metrics.json")
    try:
        labels = _load_labels(args.labels)
        rows = _load_evaluation_rows(args.input)
        metrics = _load_metrics(metrics_path)
        summaries, unlabeled_candidate_ids, unused_label_candidate_ids = (
            summarise_hypothetical_labels(rows, labels)
        )
    except OutcomeCalibrationInputError as exc:
        print(f"Outcome-calibration experiment could not start: {exc}", file=sys.stderr)
        return 2

    labelled_candidate_count = sum(
        summary["candidate_count"] for summary in summaries.values()
    )
    if labelled_candidate_count == 0:
        print(
            "Outcome-calibration experiment could not start: no evaluation "
            "candidates have hypothetical labels.",
            file=sys.stderr,
        )
        return 2

    output = {
        "experiment_label": "Synthetic outcome-calibration harness; descriptive only",
        "synthetic_disclaimer": (
            "These are hypothetical labels, not real hiring outcomes. This "
            "script exists to prove the calibration hook is auditable, not to "
            "claim calibration has happened."
        ),
        "input_evaluation_output": _display_path(args.input),
        "input_evaluation_metrics": _display_path(metrics_path),
        "source_evaluation_metrics": metrics,
        "label_source": (
            _display_path(args.labels)
            if args.labels is not None
            else "inline illustrative hypothetical labels"
        ),
        "descriptive_statistics_definition": (
            "Group mean evaluation score and confidence by hypothetical label; "
            "descriptive only, with no inferential or calibration claim."
        ),
        "labelled_candidate_count": labelled_candidate_count,
        "unlabeled_candidate_ids": unlabeled_candidate_ids,
        "unused_label_candidate_ids": unused_label_candidate_ids,
        "group_summaries": summaries,
    }
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(output, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        print(
            f"Outcome-calibration experiment could not write {args.output}: {exc}",
            file=sys.stderr,
        )
        return 2
    print(f"Results: {args.output}")
    print(
        "Hypothetically labelled candidates: "
        f"{labelled_candidate_count}/{len(rows)}; groups: "
        f"{', '.join(summaries)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
