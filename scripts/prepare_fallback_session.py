"""Prepare a complete local fallback demo session.

The default live path remains available for a fresh provider-generated demo.
The ``--offline`` path deterministically rebuilds the committed fallback from
local models and its reviewed hiring-pack template. It calls the production
role-quality, hiring-pack, evaluation, audit, storage, and reload validators,
but never constructs an ``OpenRouterClient`` or reads API configuration.

The ordinary demo uses the committed one-click ``Load Demo Fallback`` fixture;
it does not require this script or network access. Use ``--offline`` with an
explicit temporary ``--output-root`` to validate a deterministic rebuilt copy.
The no-flag path is an intentional live provider refresh that consumes credits
and is not required for presentation readiness.

This is fixture preparation and validation, not a second application mode.
The live path uses the application's generation/persistence wrappers; the
offline path uses the same production validators and ``SessionStore``.

The live candidate-evaluation step retries on validation failure (see
``docs/LIMITATIONS.md``, 2026-07-24, "score/recency consistency
check is strict"): this is documented, expected live-model behaviour, not
a bug this script works around silently -- each rejection is printed.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
import sys
import uuid
from pathlib import Path

from pydantic import ValidationError


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ConfigurationError, Settings  # noqa: E402
from src.discovery import (  # noqa: E402
    add_responsibility,
    add_success_outcome,
    add_company_behaviour,
    apply_discovery_turn,
    edit_assessment_plan,
    edit_business_need,
    edit_constraints,
    edit_requirement,
    run_discovery_turn,
)
from src.evaluation import (  # noqa: E402
    EVALUATION_PROMPT_VERSION,
    CandidateEvaluationValidationError,
    evaluate_candidate,
    evaluate_and_persist_candidate,
    validate_candidate_evaluation,
)
from src.generation import (  # noqa: E402
    generate_and_persist_hiring_pack,
    load_reference_files,
    validate_hiring_pack,
)
from src.llm_client import (  # noqa: E402
    LLMClientError,
    OpenRouterClient,
    StructuredLLMResponse,
)
from src.models import (  # noqa: E402
    ApprovalSection,
    AuditMetadata,
    BasicRoleInfo,
    CandidateEvaluationDraft,
    CandidateEvidenceExtraction,
    CandidateQuestionResponse,
    CandidateRequirementAssessmentDraft,
    CandidateResponseSet,
    DiscoveryExtractionResponse,
    DiscoveryProgressRecommendation,
    DiscoveryRequirementExtraction,
    EmploymentType,
    HiringPack,
    JobDescriptionCriterion,
    Learnability,
    ProficiencyLevel,
    RequirementCategory,
    RequirementPriority,
    RoleFamily,
    RoleLevel,
    RoleSpecification,
    RubricAnchor,
    ScreeningQuestion,
    WorkflowStage,
)
from src.readiness import approve_role, evaluate_role_quality  # noqa: E402
from src.storage import (  # noqa: E402
    AuditEventType,
    AuditLog,
    DiscoveryMessage,
    SessionStore,
    StorageError,
    append_audit_event,
    record_candidate_evaluation_generated,
    record_hiring_pack_generated,
)
from src.workflow import WorkflowState  # noqa: E402


DEMO_SESSION_ID = "demo_fallback"
FIXTURE_TIME = datetime(2026, 7, 25, 1, 27, 12, tzinfo=UTC)
COMMITTED_FALLBACK_ROOT = ROOT / "data" / "fixtures" / "demo_fallback"
COMMITTED_PACK_TEMPLATE = (
    COMMITTED_FALLBACK_ROOT
    / f"session_{DEMO_SESSION_ID}"
    / "hiring_pack.json"
)


MANAGER_STATEMENT = (
    "We need a Marketing Intern from 30 November 2026 to 26 February "
    "2027 because the summer campaign calendar has increased. They must "
    "script and film short TikTok videos with the team, targeting three "
    "approved posts per week. They will support one full campaign cycle "
    "and write a results retrospective. Basic visual content editing "
    "experience is preferred and may come from study, volunteering, or "
    "personal projects."
)

# The reviewed extraction this fixture replays, in the provider's own
# response schema. Every quote below is a verbatim span of
# ``MANAGER_STATEMENT``: the real ``run_discovery_turn`` binds each one to a
# manager turn and rejects the turn outright if it cannot, so this fixture
# cannot drift out of the grounding rule the way a hand-built
# ``Requirement`` list silently did.
FALLBACK_EXTRACTION = DiscoveryExtractionResponse(
    incremental_requirements=[
        DiscoveryRequirementExtraction(
            category=RequirementCategory.DOMAIN.value,
            name="Weekly TikTok content scripting",
            description="Script and film short TikTok videos with the team.",
            priority=RequirementPriority.MUST_HAVE.value,
            rationale="The manager explicitly assigned this task.",
            source_statement=(
                "They must script and film short TikTok videos with the team,"
            ),
        ),
        DiscoveryRequirementExtraction(
            category=RequirementCategory.DOMAIN.value,
            name="Basic visual content editing",
            description="Use a basic visual editing tool to prepare social assets.",
            priority=RequirementPriority.PREFERRED.value,
            rationale=(
                "Editing familiarity shortens feedback cycles but can be learned."
            ),
            source_statement=(
                "Basic visual content editing experience is preferred and "
                "may come from study, volunteering, or personal projects."
            ),
        ),
    ],
    assumptions=[],
    ambiguities=[],
    possible_contradictions=[],
    resolved_ambiguity_ids=[],
    next_question=(
        "Which tools should the intern use for the visual editing work?"
    ),
    stage_recommendation=DiscoveryProgressRecommendation.ADVANCE.value,
)


class _OfflineFixtureDiscoveryClient:
    """Replay the reviewed extraction through the real discovery boundary."""

    def __init__(self, extraction: DiscoveryExtractionResponse) -> None:
        self.extraction = extraction

    def generate_structured(self, **kwargs: object) -> StructuredLLMResponse:
        if kwargs.get("response_model") is not DiscoveryExtractionResponse:
            raise TypeError(
                "Offline fixture only supports DiscoveryExtractionResponse."
            )
        return StructuredLLMResponse(
            data=self.extraction,
            model="offline-reviewed-fixture",
            provider="Offline Fixture",
            schema_mode="strict_json_schema",
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            latency_ms=0,
            attempts=1,
        )


def fallback_role() -> RoleSpecification:
    """Build the approved Marketing Intern fixture through the real path.

    Requirements are produced by ``run_discovery_turn`` and confirmed by
    ``edit_requirement`` rather than constructed by hand, so the committed
    fallback is subject to every validator the live system runs -- source
    quote grounding above all. Hand-constructing them is what let a
    non-verbatim ``source_statement`` sit in the committed artifact.
    """
    manager_statement = MANAGER_STATEMENT
    role = RoleSpecification(
        role_id="role_fallback_marketing_intern",
        basic_info=BasicRoleInfo(
            title="Marketing Intern",
            role_family=RoleFamily.MARKETING,
            role_level=RoleLevel.INTERN,
            employment_type=EmploymentType.INTERNSHIP,
            division="Consumer Products",
            team="Brand Marketing",
            location="Auckland",
            work_arrangement="On-site",
            reporting_to="Brand Marketing Manager",
            start_date="2026-11-30",
            end_date="2027-02-26",
            initial_manager_statement=manager_statement,
        ),
        audit=AuditMetadata(
            created_at=FIXTURE_TIME,
            updated_at=FIXTURE_TIME,
            source_ids=["manager_turn_001"],
        ),
    )
    turn = run_discovery_turn(
        role=role,
        stage=WorkflowStage.BASIC_INFO,
        manager_message=manager_statement,
        llm_client=_OfflineFixtureDiscoveryClient(FALLBACK_EXTRACTION),
    )
    role = apply_discovery_turn(role, turn, manager_message=manager_statement)

    # The manager confirms each extracted requirement and supplies the
    # proficiency, learnability, and evidence detail the extractor does not
    # propose. This is the production confirmation path, and it is what sets
    # requires_confirmation=False / approved_by_human=True.
    role = edit_requirement(
        role,
        "requirement_001",
        name="Weekly TikTok content scripting",
        description="Script and film short TikTok videos with the team.",
        priority=RequirementPriority.MUST_HAVE,
        business_rationale="The manager explicitly assigned this task.",
        proficiency=ProficiencyLevel.WORKING,
        learnability=Learnability.DAY_ONE,
        accepted_equivalents=[
            "Comparable short-form video scripting and filming experience"
        ],
        evidence_methods=[
            "Structured behavioural question",
            "Portfolio or personal-project example",
        ],
        edited_at=FIXTURE_TIME,
    )
    role = edit_requirement(
        role,
        "requirement_002",
        name="Basic visual content editing",
        description="Use a basic visual editing tool to prepare social assets.",
        priority=RequirementPriority.PREFERRED,
        business_rationale=(
            "Editing familiarity shortens feedback cycles but can be learned."
        ),
        proficiency=ProficiencyLevel.AWARENESS,
        learnability=Learnability.WITHIN_30_DAYS,
        accepted_equivalents=[
            "Editing completed through study, volunteering, or personal projects"
        ],
        evidence_methods=[
            "Structured behavioural question",
            "Portfolio or personal-project example",
        ],
        edited_at=FIXTURE_TIME,
    )
    role = edit_business_need(
        role,
        problem="The team needs summer campaign delivery capacity.",
        why_now="The summer campaign calendar has increased.",
        cost_of_vacancy=None,
        is_replacement=False,
        edited_at=FIXTURE_TIME,
    )
    role = add_success_outcome(
        role,
        description="Deliver three approved TikTok posts each week.",
        time_horizon="First 30 days",
        measure="Three approved posts per week",
        priority=RequirementPriority.MUST_HAVE,
        source_statement=manager_statement,
        edited_at=FIXTURE_TIME,
    )
    role = add_success_outcome(
        role,
        description="Support one full campaign cycle end to end.",
        time_horizon="First 90 days",
        measure="One completed campaign cycle",
        priority=RequirementPriority.PREFERRED,
        edited_at=FIXTURE_TIME,
    )
    role = add_success_outcome(
        role,
        description="Publish a reusable summary of campaign results and learnings.",
        time_horizon="By internship end",
        measure="One manager-approved campaign retrospective",
        priority=RequirementPriority.PREFERRED,
        edited_at=FIXTURE_TIME,
    )
    role = add_responsibility(
        role,
        description="Draft and schedule weekly campaign content with the team.",
        frequency="Weekly",
        ownership_level="Shared",
        priority=RequirementPriority.MUST_HAVE,
        edited_at=FIXTURE_TIME,
    )
    role = edit_constraints(
        role,
        country="New Zealand",
        location="Auckland",
        work_arrangement="On-site",
        work_rights="Must hold valid New Zealand work rights.",
        weekly_hours="40 hours",
        travel=None,
        edited_at=FIXTURE_TIME,
    )
    role = edit_assessment_plan(
        role,
        assessment_methods=[
            "Structured screening questions",
            "Portfolio or personal-project discussion",
        ],
        decision_owner="Talent Acquisition Partner",
        edited_at=FIXTURE_TIME,
    )
    role = add_company_behaviour(
        role,
        value="Collaboration",
        role_behaviour="Shares draft work early and incorporates feedback.",
        scenario="A campaign draft receives feedback.",
        evidence_method="Behavioural screening response",
        source_statement=manager_statement,
        edited_at=FIXTURE_TIME,
    )
    warnings = evaluate_role_quality(role).warnings
    return approve_role(
        role,
        approver="Fixture Hiring Manager",
        confirmed_sections=[
            ApprovalSection.BUSINESS_PURPOSE,
            ApprovalSection.OUTCOMES,
            ApprovalSection.MUST_HAVES,
            ApprovalSection.BEHAVIOURAL_CRITERIA,
            ApprovalSection.KEY_CONSTRAINTS,
        ],
        acknowledged_warning_ids=[item.warning_id for item in warnings],
        approved_at=FIXTURE_TIME,
    )


def _fallback_answer(question: ScreeningQuestion) -> str:
    """One plausible, evidence-rich answer keyed off what the question asks.

    Questions are distinguished by their wording because several map to the
    same requirement. Each answer is a past-tense, first-person account so
    the fixture demonstrates traceable behavioural evidence.
    """
    topic = question.question.casefold()
    if "visual" in topic and ("edit" in topic or "asset" in topic):
        return (
            "For a university club launch last month, I owned the visual edit "
            "for three social assets in Canva. I resized the supplied product "
            "images, applied the club template, and incorporated two rounds of "
            "feedback before the assets were published on schedule."
        )
    if "script" in topic and "film" in topic and (
        "brief" in topic or "personally" in topic or "result" in topic
    ):
        return (
            "When a TikTok concept I pitched got cut two days before a launch, "
            "I asked the brand lead which KPI mattered most, cut the shoot to "
            "the one scene that hit it, and we still posted on time. I checked "
            "the numbers three days later and it beat our average engagement."
        )
    if "idea" in topic and ("turn" in topic or "draft of the script" in topic):
        return (
            "Two months ago our brand lead gave me a rough campaign idea for a "
            "product launch. I wrote a first script draft the same day, "
            "checked it against our target audience with the content lead, "
            "then revised the hook before we filmed two days later. During "
            "filming I coordinated shot order with our videographer so we "
            "could reuse the footage in a follow-up cut."
        )
    if "feedback" in topic and ("draft" in topic or "share" in topic):
        return (
            "Last month I shared a half-finished campaign draft with the team "
            "before I was fully happy with it, specifically to get feedback "
            "early. Two teammates flagged that the call-to-action was unclear; "
            "I rewrote it that afternoon and used their wording in the final "
            "version we shipped that week."
        )
    if "working with others" in topic or "divide" in topic or "own" in topic:
        return (
            "On a recent product-launch video, I owned the script and the "
            "on-camera segments while a teammate handled the b-roll and "
            "captions. We split the two-day shoot by scene, I sent my cut to "
            "them the same evening, and we combined and published the final "
            "version together the next morning."
        )
    if "week" in topic and ("three" in topic or "plan" in topic or "approved" in topic):
        return (
            "In my current role I already run on this cadence: I draft all "
            "three scripts on Monday, get them approved by the brand lead by "
            "Tuesday, film Tuesday afternoon and Wednesday, and leave Thursday "
            "for revisions so every post is reviewed and signed off before it "
            "goes out Friday."
        )
    if "collaborat" in topic or "feedback" in topic:
        return (
            "Last month I shared a half-finished campaign draft with the team "
            "before I was fully happy with it, specifically to get feedback "
            "early. Two teammates flagged that the call-to-action was unclear; "
            "I rewrote it that afternoon and used their wording in the final "
            "version we shipped that week."
        )
    if "campaign cycle" in topic or "end to end" in topic:
        return (
            "In my last role I owned one campaign from brief through to "
            "posting: I drafted the content calendar, scheduled every post "
            "over the four-week run, and reported the final engagement "
            "numbers to the team lead at the end of the cycle."
        )
    if "post" in topic or "weekly" in topic or "week" in topic:
        return (
            "Over the last six weeks in my current role I have scripted and "
            "scheduled three short-form videos a week without missing a week, "
            "and tracked each one's engagement against the previous week's "
            "average."
        )
    if (
        "work right" in topic
        or "on-site" in topic
        or "hours" in topic
        or "auckland" in topic
    ):
        return (
            "I hold a valid New Zealand work visa with no restriction on "
            "hours, and I am based in Auckland, so I can work on-site five "
            "days a week."
        )
    return (
        "In my current role I took ownership of one specific, measurable "
        "piece of marketing work end to end -- from the initial brief through "
        "to reporting the actual result to my manager within the last two "
        "months."
    )


def fallback_candidate_response_set(
    role: RoleSpecification,
    hiring_pack: HiringPack,
    *,
    submitted_at: datetime | None = None,
    response_set_id: str | None = None,
) -> CandidateResponseSet:
    """One synthetic, evidence-rich answer per generated screening question."""
    responses = [
        CandidateQuestionResponse(
            response_id=f"response_{question.question_id}",
            question_id=question.question_id,
            answer_text=_fallback_answer(question),
        )
        for question in hiring_pack.screening_questions
    ]
    return CandidateResponseSet(
        response_set_id=response_set_id or f"responses_{uuid.uuid4().hex[:8]}",
        candidate_id="candidate_fallback_001",
        source_role_id=role.role_id,
        source_role_version=role.version,
        source_hiring_pack_id=hiring_pack.hiring_pack_id,
        source_hiring_pack_version=hiring_pack.version,
        submitted_at=submitted_at or datetime.now(UTC),
        responses=responses,
    )


def _offline_hiring_pack(role: RoleSpecification) -> HiringPack:
    """Bind the reviewed committed pack template to the refreshed role."""
    template = HiringPack.model_validate_json(
        COMMITTED_PACK_TEMPLATE.read_text(encoding="utf-8")
    )
    preferred = next(
        item
        for item in role.requirements
        if item.priority is RequirementPriority.PREFERRED
    )
    updated_questions: list[ScreeningQuestion] = []
    for question in template.screening_questions:
        if question.question_id != "sq_004":
            updated_questions.append(question)
            continue
        updated_questions.append(
            question.model_copy(
                update={
                    "question": (
                        "Tell us about a visual social asset you edited using "
                        "Canva or a comparable tool. What did you personally "
                        "change, what feedback did you apply, and what was delivered?"
                    ),
                    "requirement_ids": [preferred.requirement_id],
                    "purpose": (
                        "Assess owned evidence of basic visual content editing "
                        "without requiring prior paid work."
                    ),
                    "expected_evidence": [
                        "A specific visual asset from work, study, volunteering, "
                        "or a personal project.",
                        "The editing tool and changes personally completed.",
                        "Feedback incorporated and a concrete delivered result.",
                    ],
                    "rubric": [
                        RubricAnchor(
                            score=0,
                            description=(
                                "States they have not edited a visual asset and "
                                "provides no comparable evidence."
                            ),
                        ),
                        RubricAnchor(
                            score=1,
                            description=(
                                "Provides no relevant visual editing example or "
                                "personal contribution."
                            ),
                        ),
                        RubricAnchor(
                            score=2,
                            description=(
                                "Names an editing tool or asset but gives only a "
                                "generic account with unclear ownership."
                            ),
                        ),
                        RubricAnchor(
                            score=3,
                            description=(
                                "Describes a relevant asset, a personally completed "
                                "edit, and a delivered output."
                            ),
                        ),
                        RubricAnchor(
                            score=4,
                            description=(
                                "Provides a specific example with clear editing "
                                "ownership, feedback applied, and a concrete result."
                            ),
                        ),
                        RubricAnchor(
                            score=5,
                            description=(
                                "Provides externally verified editing evidence with "
                                "clear ownership, trade-offs, results, and reflection."
                            ),
                        ),
                    ],
                    "green_flags": [
                        "Clearly distinguishes personal edits from supplied material.",
                        "Explains how feedback changed the delivered asset.",
                        "Uses an equivalent academic, volunteer, or personal project.",
                    ],
                    "red_flags": [
                        "Names a tool without describing any actual edit.",
                        "Attributes all editing decisions and execution to others.",
                        "Provides no concrete delivered visual output.",
                    ],
                    "follow_up": (
                        "Which edit had the greatest effect on the final asset, "
                        "and how did you decide?"
                    ),
                }
            )
        )

    job_description = template.job_description.model_copy(
        update={
            "outcomes": [
                (
                    f"Within {item.time_horizon.casefold()}, {item.description}"
                    if item.time_horizon
                    else item.description
                )
                for item in role.success_outcomes
            ],
            "preferred_criteria": [
                JobDescriptionCriterion(
                    requirement_id=preferred.requirement_id,
                    text=preferred.description or preferred.name,
                )
            ],
            "logistics": [
                "Internship role in Brand Marketing within Consumer Products.",
                "Location: Auckland, New Zealand.",
                "Work arrangement: On-site.",
                "Hours: 40 hours per week.",
                "Dates: 30 November 2026 to 26 February 2027.",
                "Must hold valid New Zealand work rights.",
            ],
            "assessment_expectations": [
                "Structured screening questions assess approved role requirements.",
                (
                    "Candidates may use academic, volunteer, extracurricular, "
                    "personal-project, or work evidence."
                ),
                "Portfolio examples are discussed but do not replace human review.",
            ],
        }
    )
    provenance = template.provenance.model_copy(
        update={
            "reference_files": load_reference_files(ROOT / "references").provenance,
            "source_role_id": role.role_id,
            "source_role_version": role.version,
            "generated_at": FIXTURE_TIME + timedelta(minutes=1),
            "generated_by": "Fixture Hiring Manager",
            "model": "offline-reviewed-template",
            "provider": "Offline Fixture",
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
    )
    pack = template.model_copy(
        update={
            "source_session_id": DEMO_SESSION_ID,
            "provenance": provenance,
            "job_description": job_description,
            "screening_questions": updated_questions,
        }
    )
    validate_hiring_pack(pack, role)
    return pack


def _offline_evaluation_draft(
    hiring_pack: HiringPack,
    response_set: CandidateResponseSet,
) -> CandidateEvaluationDraft:
    """Return reviewed extraction labels; production code computes all scores."""
    response_by_question = {
        item.question_id: item for item in response_set.responses
    }
    mapped_questions: dict[str, list[str]] = {}
    for question in hiring_pack.screening_questions:
        for requirement_id in question.requirement_ids:
            mapped_questions.setdefault(requirement_id, []).append(
                question.question_id
            )

    evidence_inputs = (
        (
            "evidence_fallback_001",
            "requirement_001",
            "sq_001",
            "Specific scripting, filming, adaptation, and delivery evidence.",
        ),
        (
            "evidence_fallback_002",
            "requirement_002",
            "sq_004",
            "Specific visual editing, feedback, and delivery evidence.",
        ),
    )
    evidence_items = [
        CandidateEvidenceExtraction(
            evidence_id=evidence_id,
            requirement_id=requirement_id,
            source_type="screening_response",
            source_id=response_by_question[question_id].response_id,
            question_id=question_id,
            quote=response_by_question[question_id].answer_text,
            evidence_type="direct",
            relevance=0.95,
            specificity=0.92,
            recency_relevant=False,
            recency=None,
            ownership="owned",
            action_described=True,
            outcome_evidence=True,
            reflection_evidence=False,
            hypothetical=False,
            verification_status="unverified_candidate_claim",
            contradiction_status="none",
            evaluator_explanation=explanation,
        )
        for evidence_id, requirement_id, question_id, explanation in evidence_inputs
    ]
    assessments = [
        CandidateRequirementAssessmentDraft(
            requirement_id=requirement_id,
            relevant_question_ids=mapped_questions[requirement_id],
            evidence_ids=[evidence_id],
            proposed_score=4,
            rubric_question_id=question_id,
            strengths=[strength],
            concerns=[],
            missing_evidence=[],
            contradictory_evidence=[],
            contradiction_evidence_ids=[],
            recommended_follow_up=(
                "Which artefact or referee could a human reviewer use to "
                "verify this example?"
            ),
            reviewer_explanation=(
                "The cited response describes owned actions and a concrete "
                "delivered result. It remains an unverified candidate claim."
            ),
        )
        for evidence_id, requirement_id, question_id, strength in evidence_inputs
    ]
    return CandidateEvaluationDraft(
        evidence_items=evidence_items,
        assessments=assessments,
    )


class _OfflineFixtureEvaluationClient:
    """Local structured-response stub used only to exercise production evaluation."""

    def __init__(self, draft: CandidateEvaluationDraft) -> None:
        self.draft = draft

    def generate_structured(self, **kwargs: object) -> StructuredLLMResponse:
        if kwargs.get("response_model") is not CandidateEvaluationDraft:
            raise TypeError("Offline fixture only supports CandidateEvaluationDraft.")
        return StructuredLLMResponse(
            data=self.draft,
            model="offline-reviewed-fixture",
            provider="Offline Fixture",
            schema_mode="strict_json_schema",
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            latency_ms=0,
            attempts=1,
        )


def prepare_offline_fallback_session(output_root: Path) -> str:
    """Build, persist, reload, and validate the full fallback without a provider."""
    role = fallback_role()
    pack = _offline_hiring_pack(role)
    response_set = fallback_candidate_response_set(
        role,
        pack,
        submitted_at=FIXTURE_TIME + timedelta(seconds=90),
        response_set_id="responses_demo_fallback_001",
    )
    evaluation = evaluate_candidate(
        role=role,
        hiring_pack=pack,
        response_set=response_set,
        llm_client=_OfflineFixtureEvaluationClient(
            _offline_evaluation_draft(pack, response_set)
        ),
        actor="Fixture Hiring Manager",
        evaluated_at=FIXTURE_TIME + timedelta(minutes=2),
        id_factory=lambda: "evaluation_4a809b33619247ab93d047198a5cf0b3",
    )
    evaluation = evaluation.model_copy(
        update={
            "audit": AuditMetadata(
                created_at=FIXTURE_TIME + timedelta(minutes=2),
                updated_at=FIXTURE_TIME + timedelta(minutes=2),
                model=evaluation.model,
                prompt_version=evaluation.prompt_version,
                source_ids=[
                    role.role_id,
                    pack.hiring_pack_id,
                    response_set.response_set_id,
                ],
            )
        }
    )
    validate_candidate_evaluation(
        evaluation,
        role=role,
        hiring_pack=pack,
    )

    audit_log = append_audit_event(
        AuditLog(session_id=DEMO_SESSION_ID),
        AuditEventType.SESSION_CREATED,
        role=role,
        actor="system",
        occurred_at=FIXTURE_TIME,
    )
    audit_log = append_audit_event(
        audit_log,
        AuditEventType.REQUIREMENT_APPROVED,
        role=role,
        actor=role.approved_by or "Fixture Hiring Manager",
        occurred_at=FIXTURE_TIME,
        metadata={
            "requirement_ids": [
                item.requirement_id for item in role.requirements
            ]
        },
    )
    audit_log = append_audit_event(
        audit_log,
        AuditEventType.HUMAN_REVIEW_RECORDED,
        role=role,
        actor=role.approved_by or "Fixture Hiring Manager",
        occurred_at=FIXTURE_TIME,
        metadata={
            "approved_sections": [
                section.value for section in role.approved_sections
            ]
        },
    )
    audit_log = record_hiring_pack_generated(
        audit_log,
        role=role,
        hiring_pack=pack,
    )
    audit_log = record_candidate_evaluation_generated(
        audit_log,
        role=role,
        evaluation=evaluation,
    )

    store = SessionStore(output_root)
    store.save_session(
        session_id=DEMO_SESSION_ID,
        workflow_state=WorkflowState(
            role_specification=role,
            current_stage=WorkflowStage.COMPLETE,
        ),
        messages=[
            DiscoveryMessage(
                role="manager",
                content=role.basic_info.initial_manager_statement or "",
                timestamp=FIXTURE_TIME,
            )
        ],
        audit_log=audit_log,
        hiring_pack=pack,
        candidate_evaluation=evaluation,
    )

    loaded = store.load_session(DEMO_SESSION_ID)
    loaded_role = loaded.workflow_state.role_specification
    if loaded.hiring_pack is None or loaded.candidate_evaluation is None:
        raise StorageError("Offline fallback did not persist all demo artefacts.")
    validate_hiring_pack(loaded.hiring_pack, loaded_role)
    validate_candidate_evaluation(
        loaded.candidate_evaluation,
        role=loaded_role,
        hiring_pack=loaded.hiring_pack,
    )
    if loaded_role.quality.readiness_score != 100:
        raise StorageError("Offline fallback role did not retain 100 readiness.")
    if loaded.candidate_evaluation.prompt_version != EVALUATION_PROMPT_VERSION:
        raise StorageError("Offline fallback evaluator provenance is stale.")
    return DEMO_SESSION_ID


def _prepare_live_fallback_session() -> int:
    try:
        settings = Settings.from_env()
        settings.require_api_configuration()
    except (ValidationError, ConfigurationError) as exc:
        print(f"OpenRouter is not configured, cannot prepare a fallback session: {exc}")
        return 2

    role = fallback_role()
    session_id = f"fallback_{uuid.uuid4().hex[:8]}"
    audit_log = append_audit_event(
        AuditLog(session_id=session_id),
        AuditEventType.SESSION_CREATED,
        role=role,
        actor="system",
    )
    audit_log = append_audit_event(
        audit_log,
        AuditEventType.REQUIREMENT_APPROVED,
        role=role,
        actor=role.approved_by or "system",
    )

    store = SessionStore(ROOT / "data" / "sessions")
    state = WorkflowState(role_specification=role)
    messages = [
        DiscoveryMessage(
            role="manager",
            content=role.basic_info.initial_manager_statement or "",
        )
    ]
    try:
        store.save_session(
            session_id=session_id,
            workflow_state=state,
            messages=messages,
            audit_log=audit_log,
        )
    except StorageError as exc:
        print(f"Could not save the base session: {exc}")
        return 1

    client = OpenRouterClient(settings)
    try:
        result = generate_and_persist_hiring_pack(
            role=role,
            llm_client=client,
            reference_directory=ROOT / "references",
            actor=role.approved_by or "Fixture Hiring Manager",
            session_id=session_id,
            session_store=store,
            audit_log=audit_log,
        )
    except (LLMClientError, StorageError) as exc:
        print(f"Hiring pack generation failed: {exc}")
        print(
            "The approved role alone was not saved either -- rerun once network "
            "access is available."
        )
        return 1

    hiring_pack = result.hiring_pack
    audit_log = result.audit_log
    print(f"Hiring pack ready: version {hiring_pack.version}")

    response_set = fallback_candidate_response_set(role, hiring_pack)
    max_attempts = 5
    evaluation_result = None
    for attempt in range(1, max_attempts + 1):
        try:
            evaluation_result = evaluate_and_persist_candidate(
                role=role,
                hiring_pack=hiring_pack,
                response_set=response_set,
                llm_client=client,
                actor=role.approved_by or "Fixture Hiring Manager",
                session_id=session_id,
                session_store=store,
                audit_log=audit_log,
            )
            break
        except CandidateEvaluationValidationError as exc:
            print(f"Candidate evaluation attempt {attempt}/{max_attempts} rejected: {exc}")
        except (LLMClientError, StorageError) as exc:
            print(f"Candidate evaluation attempt {attempt}/{max_attempts} failed: {exc}")

    print(f"Fallback session ready: {session_id}")
    print(
        "In the app: Define Role tab -> 'Save or reload session' -> select "
        f"'{session_id}' -> Load session."
    )
    print(f"Hiring pack version: {hiring_pack.version}")
    if evaluation_result is None:
        print(
            f"Candidate evaluation was NOT persisted after {max_attempts} attempts -- "
            "this is the documented live-model score/recency validation strictness "
            "(docs/LIMITATIONS.md, 2026-07-24), not a bug in this script. "
            "The saved session still covers the approved role and hiring pack; rerun "
            "this script to try again for a fallback that also covers candidate "
            "evaluation without a live call."
        )
        return 1
    evaluation = evaluation_result.evaluation
    print(
        f"Candidate evaluation persisted: version {evaluation.version}, "
        f"{len(evaluation.assessments)} requirement(s) assessed, "
        f"routing={evaluation.routing.value if evaluation.routing else 'none'}."
    )
    print(
        "This fallback session now covers the demo through Candidate Evidence "
        "with zero further live calls required."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Rebuild a complete fallback without credentials or network calls.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "data" / "fixtures" / "demo_fallback_regenerated",
        help="SessionStore root used by --offline.",
    )
    args = parser.parse_args()
    if not args.offline:
        return _prepare_live_fallback_session()

    try:
        session_id = prepare_offline_fallback_session(args.output_root)
    except (OSError, StorageError, ValueError) as exc:
        print(f"Offline fallback generation failed: {exc}")
        return 1
    print(f"Offline fallback session ready: {session_id}")
    print(f"Output root: {args.output_root.resolve()}")
    print("Validated role, hiring pack, evaluator v2, persistence, and reload.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
