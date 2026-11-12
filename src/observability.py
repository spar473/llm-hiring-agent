"""Deterministic, persisted-session guardrail activity summaries.

The aggregation deliberately observes stored quality and evaluation results. It
does not recompute role quality, readiness, confidence, or candidate signals.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.storage import SessionStore, StorageError


@dataclass(frozen=True, slots=True)
class GuardrailActivitySummary:
    """Counts of persisted safety and quality guardrail activity."""

    total_sessions: int = 0
    vague_phrase_hits: int = 0
    excessive_requirement_hits: int = 0
    contradictions_flagged: int = 0
    contradictions_resolved: int = 0
    prompt_injection_detections: int = 0


def _stored_warning_counts(warnings: list[str]) -> tuple[int, int]:
    """Categorise flattened, already-stored quality warning messages.

    ``RoleQuality`` persists warning messages rather than the transient
    ``QualityWarning`` objects. These checks use only their documented display
    prefixes, excluding stored contradiction warnings because contradictions
    have their own persisted collection and counters.
    """
    vague_phrase_hits = sum(
        warning.startswith("Vague phrase ") for warning in warnings
    )
    excessive_requirement_hits = sum(
        not warning.startswith("Vague phrase ")
        and " contradiction:" not in warning.casefold()
        for warning in warnings
    )
    return vague_phrase_hits, excessive_requirement_hits


def aggregate_guardrail_activity(
    session_store: SessionStore,
) -> GuardrailActivitySummary:
    """Summarise stored guardrail activity across all available sessions."""
    try:
        session_ids = session_store.list_sessions()
    except OSError:
        return GuardrailActivitySummary()

    vague_phrase_hits = 0
    excessive_requirement_hits = 0
    contradictions_flagged = 0
    contradictions_resolved = 0
    prompt_injection_detections = 0
    for session_id in session_ids:
        try:
            snapshot = session_store.load_session(session_id)
        except (OSError, StorageError):
            continue

        quality = snapshot.workflow_state.role_specification.quality
        vague_hits, excessive_hits = _stored_warning_counts(quality.warnings)
        vague_phrase_hits += vague_hits
        excessive_requirement_hits += excessive_hits
        contradictions_flagged += len(quality.contradictions)
        contradictions_resolved += sum(
            contradiction.resolved for contradiction in quality.contradictions
        )
        if (
            snapshot.candidate_evaluation is not None
            and snapshot.candidate_evaluation.prompt_injection_detected
        ):
            prompt_injection_detections += 1

    return GuardrailActivitySummary(
        total_sessions=len(session_ids),
        vague_phrase_hits=vague_phrase_hits,
        excessive_requirement_hits=excessive_requirement_hits,
        contradictions_flagged=contradictions_flagged,
        contradictions_resolved=contradictions_resolved,
        prompt_injection_detections=prompt_injection_detections,
    )
