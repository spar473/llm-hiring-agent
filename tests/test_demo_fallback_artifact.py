"""The committed offline fallback must satisfy every live-system invariant.

``scripts/prepare_fallback_session.py`` used to hand-construct ``Requirement``
objects and never call ``run_discovery_turn``, so source-quote grounding was
never exercised against the one artifact it exists to protect. The committed
fixture consequently carried a ``source_statement`` that ended in a full stop
where the manager said a comma -- a quote the live validator rejects.

These tests run the real validators over the committed bytes, so the demo
fallback cannot drift away from what the application enforces at runtime.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.discovery import _normalise_source_text
from src.evaluation import EVALUATION_PROMPT_VERSION, validate_candidate_evaluation
from src.generation import validate_hiring_pack
from src.guardrails import scan_pack_protected_content, scan_pack_payload
from src.models import RequirementPriority, ReviewStatus, WorkflowStage
from src.readiness import approval_blockers, calculate_readiness
from src.storage import SessionStore


FALLBACK_ROOT = (
    Path(__file__).resolve().parent.parent / "data" / "fixtures" / "demo_fallback"
)
SESSION_ID = "demo_fallback"


@pytest.fixture(scope="module")
def snapshot():
    return SessionStore(FALLBACK_ROOT).load_session(SESSION_ID)


@pytest.fixture(scope="module")
def role(snapshot):
    return snapshot.workflow_state.role_specification


def test_the_committed_session_loads_with_every_demo_artefact(snapshot) -> None:
    assert snapshot.hiring_pack is not None
    assert snapshot.candidate_evaluation is not None
    assert snapshot.workflow_state.current_stage is WorkflowStage.COMPLETE


# --- source-quote grounding ----------------------------------------------


def _manager_turns(snapshot) -> list[str]:
    return [
        _normalise_source_text(message.content)
        for message in snapshot.messages
        if message.role == "manager"
    ]


def test_every_requirement_quote_is_verbatim_in_a_manager_turn(
    snapshot, role
) -> None:
    """This is the check the hand-built fixture bypassed."""
    turns = _manager_turns(snapshot)
    assert turns, "the fallback must persist the manager turn it quotes from"
    ungrounded = [
        (item.requirement_id, item.source_statement)
        for item in role.requirements
        if not any(
            _normalise_source_text(item.source_statement) in turn for turn in turns
        )
    ]
    assert not ungrounded, ungrounded


def test_every_other_source_quote_is_verbatim_in_a_manager_turn(
    snapshot, role
) -> None:
    turns = _manager_turns(snapshot)
    quoted = [
        (item.outcome_id, item.source_statement) for item in role.success_outcomes
    ]
    quoted += [
        (item.responsibility_id, item.source_statement)
        for item in role.responsibilities
    ]
    quoted += [(item.value, item.source_statement) for item in role.company_behaviours]
    ungrounded = [
        (identifier, quote)
        for identifier, quote in quoted
        if quote and not any(_normalise_source_text(quote) in turn for turn in turns)
    ]
    assert not ungrounded, ungrounded


def test_requirement_source_turn_ids_reference_a_persisted_turn(
    snapshot, role
) -> None:
    expected = {
        f"manager_turn_{index:03d}"
        for index, _ in enumerate(_manager_turns(snapshot), start=1)
    }
    assert {item.source_turn_id for item in role.requirements} <= expected


# --- identifier uniqueness -----------------------------------------------


@pytest.mark.parametrize(
    ("collection", "id_attribute"),
    [
        ("requirements", "requirement_id"),
        ("success_outcomes", "outcome_id"),
        ("responsibilities", "responsibility_id"),
        ("open_ambiguities", "ambiguity_id"),
        ("open_assumptions", "assumption_id"),
    ],
)
def test_role_collection_ids_are_unique(role, collection, id_attribute) -> None:
    identifiers = [getattr(item, id_attribute) for item in getattr(role, collection)]
    assert len(identifiers) == len(set(identifiers))


def test_hiring_pack_and_evaluation_ids_are_unique(snapshot) -> None:
    question_ids = [item.question_id for item in snapshot.hiring_pack.screening_questions]
    assert len(question_ids) == len(set(question_ids))
    evidence_ids = [item.evidence_id for item in snapshot.candidate_evaluation.evidence_items]
    assert len(evidence_ids) == len(set(evidence_ids))


# --- approval and human confirmation -------------------------------------


def test_the_committed_role_is_approved_with_no_outstanding_blockers(role) -> None:
    assert role.human_approved
    assert role.review_status is ReviewStatus.APPROVED
    assert approval_blockers(role) == []


def test_every_requirement_is_individually_confirmed_and_approved(role) -> None:
    """The fixture must not rely on the discovery defaults it started from."""
    unconfirmed = [
        item.requirement_id for item in role.requirements if item.requires_confirmation
    ]
    unapproved = [
        item.requirement_id for item in role.requirements if not item.approved_by_human
    ]
    assert not unconfirmed, unconfirmed
    assert not unapproved, unapproved


def test_the_role_covers_both_priorities_and_the_internship_dates(role) -> None:
    priorities = {item.priority for item in role.requirements}
    assert RequirementPriority.MUST_HAVE in priorities
    assert RequirementPriority.PREFERRED in priorities
    assert role.basic_info.start_date
    assert role.basic_info.end_date


# --- readiness is independently recomputed -------------------------------


def test_the_persisted_readiness_score_is_reproducible(role) -> None:
    """The committed number must be what the scorer actually derives."""
    assert role.quality.readiness_score == calculate_readiness(role).score


# --- pack, guardrails, and evaluation ------------------------------------


def test_the_committed_pack_passes_full_hiring_pack_validation(snapshot, role) -> None:
    validate_hiring_pack(snapshot.hiring_pack, role)


def test_the_committed_pack_is_clean_under_the_free_text_guardrails(
    snapshot,
) -> None:
    findings = scan_pack_payload(snapshot.hiring_pack.model_dump(mode="json"))
    assert not findings, [item.message() for item in findings]
    protected_findings = scan_pack_protected_content(
        snapshot.hiring_pack.model_dump(mode="json")
    )
    assert not protected_findings, protected_findings


def test_the_committed_evaluation_passes_full_validation(snapshot, role) -> None:
    validate_candidate_evaluation(
        snapshot.candidate_evaluation,
        role=role,
        hiring_pack=snapshot.hiring_pack,
    )


def test_the_committed_evaluation_carries_current_evaluator_provenance(
    snapshot,
) -> None:
    assert snapshot.candidate_evaluation.prompt_version == EVALUATION_PROMPT_VERSION


def test_pack_and_evaluation_bind_to_the_committed_role_version(snapshot, role) -> None:
    assert snapshot.hiring_pack.provenance.source_role_id == role.role_id
    assert snapshot.hiring_pack.provenance.source_role_version == role.version
    assert snapshot.candidate_evaluation.role_version == role.version
