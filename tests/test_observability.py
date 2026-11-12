"""Tests for deterministic, persisted-session guardrail observability."""

from __future__ import annotations

from src.models import (
    CandidateEvaluation,
    Contradiction,
    RoleQuality,
    RoleSpecification,
)
from src.observability import aggregate_guardrail_activity
from src.storage import AuditEventType, AuditLog, SessionStore, append_audit_event
from src.workflow import WorkflowState
from tests.phase6_helpers import pack_from_draft


def _save_session(
    store: SessionStore,
    *,
    session_id: str,
    quality: RoleQuality,
    prompt_injection_detected: bool = False,
) -> None:
    role = RoleSpecification(role_id=f"role_{session_id}", quality=quality)
    audit_log = append_audit_event(
        AuditLog(session_id=session_id),
        AuditEventType.SESSION_CREATED,
        role=role,
        actor="system",
    )
    hiring_pack = pack_from_draft(role=role, session_id=session_id)
    evaluation = (
        CandidateEvaluation(
            evaluation_id=f"evaluation_{session_id}",
            candidate_id=f"candidate_{session_id}",
            role_id=role.role_id,
            role_version=role.version,
            hiring_pack_id=hiring_pack.hiring_pack_id,
            hiring_pack_version=hiring_pack.version,
            prompt_injection_detected=True,
        )
        if prompt_injection_detected
        else None
    )
    store.save_session(
        session_id=session_id,
        workflow_state=WorkflowState(role_specification=role),
        messages=[],
        audit_log=audit_log,
        hiring_pack=hiring_pack,
        candidate_evaluation=evaluation,
    )


def test_aggregate_guardrail_activity_counts_stored_session_data(tmp_path) -> None:
    store = SessionStore(tmp_path)
    _save_session(
        store,
        session_id="first",
        quality=RoleQuality(
            warnings=[
                "Vague phrase 'strategic' in requirement.",
                "An intern or entry-level role includes advanced ownership.",
                "Medium contradiction: role scope is unclear.",
            ],
            contradictions=[
                Contradiction(
                    contradiction_id="contradiction_open",
                    description="Managers disagree about role scope.",
                    severity="high",
                    source_statements=["One", "Two"],
                ),
                Contradiction(
                    contradiction_id="contradiction_resolved",
                    description="Managers aligned on location.",
                    severity="low",
                    source_statements=["One", "Two"],
                    resolved=True,
                    resolution="The role will be Auckland-based.",
                ),
            ],
        ),
        prompt_injection_detected=True,
    )
    _save_session(
        store,
        session_id="second",
        quality=RoleQuality(warnings=["Vague phrase 'culture fit' in role."]),
    )

    summary = aggregate_guardrail_activity(store)

    assert summary.total_sessions == 2
    assert summary.vague_phrase_hits == 2
    assert summary.excessive_requirement_hits == 1
    assert summary.contradictions_flagged == 2
    assert summary.contradictions_resolved == 1
    assert summary.prompt_injection_detections == 1


def test_aggregate_guardrail_activity_handles_an_empty_store(tmp_path) -> None:
    assert aggregate_guardrail_activity(SessionStore(tmp_path)).total_sessions == 0
