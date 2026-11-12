"""Streamlit AppTest coverage for the Phase 7 Candidate Evidence interface."""

from __future__ import annotations

from pathlib import Path

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"

from streamlit.testing.v1 import AppTest

from scripts.prepare_fallback_session import prepare_offline_fallback_session
from src.evaluation import evaluate_candidate
from src.models import RequirementPriority
from src.storage import AuditLog, SessionStore
from src.workflow import WorkflowState
from tests.phase7_helpers import (
    EVALUATION_TIME,
    FakeEvaluationClient,
    approved_marketing_role,
    pack_from_draft,
    strong_response_set,
)


def _seeded_app(*, include_evaluation: bool) -> AppTest:
    role = approved_marketing_role()
    pack = pack_from_draft(session_id="ui")
    evaluation = (
        evaluate_candidate(
            role=role,
            hiring_pack=pack,
            response_set=strong_response_set(),
            llm_client=FakeEvaluationClient(),
            actor="TA Partner",
            evaluated_at=EVALUATION_TIME,
            id_factory=lambda: "evaluation_ui",
        )
        if include_evaluation
        else None
    )
    app = AppTest.from_file(APP_PATH, default_timeout=20)
    app.session_state["workflow_state"] = WorkflowState(
        role_specification=role
    )
    app.session_state["hiring_pack"] = pack
    app.session_state["candidate_evaluation"] = evaluation
    app.session_state["session_id"] = "ui"
    app.session_state["audit_log"] = AuditLog(session_id="ui")
    app.session_state["discovery_messages"] = []
    return app


def test_empty_application_state_has_no_exceptions() -> None:
    app = AppTest.from_file(APP_PATH, default_timeout=20).run()

    assert not app.exception
    assert "Candidate Evidence" in [item.value for item in app.subheader]
    assert "Batch Review" in [item.value for item in app.subheader]


def test_committed_batch_fixture_renders_review_summary_and_table() -> None:
    app = AppTest.from_file(APP_PATH, default_timeout=20).run()
    metrics = {item.label: item.value for item in app.metric}

    assert not app.exception
    assert metrics["Candidates"] == "10"
    assert metrics["Accepted evaluations"] == "10"
    assert app.dataframe


def test_demo_fallback_loads_without_provider_configuration(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv("OPENROUTER_MODEL", "")
    app = AppTest.from_file(APP_PATH, default_timeout=20).run()

    demo_button = next(
        button for button in app.button if button.label == "Load Offline Demo"
    )
    demo_button.click().run()

    assert not app.exception
    assert app.session_state["session_id"] == "demo_fallback"
    role = app.session_state["workflow_state"].role_specification
    evaluation = app.session_state["candidate_evaluation"]
    assert role.human_approved
    assert role.quality.readiness_score == 100
    assert role.basic_info.start_date == "2026-11-30"
    assert role.basic_info.end_date == "2027-02-26"
    assert {item.priority for item in role.requirements} >= {
        RequirementPriority.MUST_HAVE,
        RequirementPriority.PREFERRED,
    }
    assert all(
        not item.requires_confirmation and item.approved_by_human
        for item in role.requirements
    )
    assert app.session_state["hiring_pack"] is not None
    assert evaluation is not None
    assert evaluation.prompt_version == "candidate_evaluator_v2"
    assert len(evaluation.evidence_items) == len(role.requirements)
    assert len(evaluation.assessments) == len(role.requirements)


def test_offline_fallback_builder_round_trips_without_configuration(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv("OPENROUTER_MODEL", "")
    output_root = tmp_path / "offline_fallback"

    session_id = prepare_offline_fallback_session(output_root)
    snapshot = SessionStore(output_root).load_session(session_id)

    role = snapshot.workflow_state.role_specification
    assert session_id == "demo_fallback"
    assert role.human_approved
    assert role.quality.readiness_score == 100
    assert snapshot.workflow_state.current_stage.value == "complete"
    assert snapshot.hiring_pack is not None
    assert snapshot.candidate_evaluation is not None
    assert snapshot.candidate_evaluation.prompt_version == "candidate_evaluator_v2"
    assert snapshot.candidate_evaluation.requirement_coverage == 1.0
    assert snapshot.candidate_evaluation.must_have_coverage == 1.0


def test_conflicting_stakeholders_demo_loads_an_unresolved_contradiction() -> None:
    app = AppTest.from_file(APP_PATH, default_timeout=20).run()

    demo_button = next(
        button
        for button in app.button
        if button.label == "Load Conflicting-Stakeholders Demo"
    )
    demo_button.click().run()

    role = app.session_state["workflow_state"].role_specification
    assert not app.exception
    assert role.quality.contradictions
    assert any(not item.resolved for item in role.quality.contradictions)
    assert app.session_state["hiring_pack"] is None
    assert app.session_state["candidate_evaluation"] is None


def test_approved_role_and_valid_pack_render_candidate_response_inputs(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "ui-test-placeholder")
    monkeypatch.setenv("OPENROUTER_MODEL", "test/model")
    app = _seeded_app(include_evaluation=False).run()
    response_key = "candidate_answer_hiring_pack_fixed_1_sq_001"

    assert not app.exception
    assert "Run evidence evaluation" in [button.label for button in app.button]
    assert len(
        [
            area
            for area in app.text_area
            if area.key and area.key.startswith("candidate_answer_")
        ]
    ) == 5

    app.text_area(key=response_key).set_value(
        "I changed a campaign hook after reviewing retention."
    )
    next(button for button in app.button if button.label == "Run evidence evaluation").click().run()

    assert (
        app.text_area(key=response_key).value
        == "I changed a campaign hook after reviewing retention."
    )


def test_hiring_pack_reference_comparison_renders_a_guardrail_checklist() -> None:
    app = _seeded_app(include_evaluation=False).run()
    rendered_markdown = [item.value for item in app.markdown]

    assert not app.exception
    assert any("Comparison checklist" in item for item in rendered_markdown)
    assert any(
        "references are fictional examples" in item
        for item in rendered_markdown
    )


def test_persisted_evidence_assessments_and_review_controls_render() -> None:
    app = _seeded_app(include_evaluation=True).run()
    markdown = [item.value for item in app.markdown]
    captions = [item.value for item in app.caption]
    button_labels = [button.label for button in app.button]

    assert not app.exception
    assert "## Evaluation for candidate_strong_001 (version 1)" in markdown
    assert "### Extracted evidence" in markdown
    assert "### Requirement assessments" in markdown
    assert button_labels.count("Save human review") == 3
    assert any(
        metric.label == "Must-have coverage" for metric in app.metric
    )
    assert any(
        "uncalibrated evidence-quality routing heuristic" in value
        and "not a probability of candidate success" in value
        for value in captions
    )


def test_rendering_candidate_evaluation_emits_no_audit_event_or_secret() -> None:
    app = _seeded_app(include_evaluation=True).run()
    rendered_values = [
        str(element.value)
        for collection in (
            app.markdown,
            app.caption,
            app.info,
            app.warning,
            app.error,
        )
        for element in collection
    ]

    assert not app.exception
    assert app.session_state["audit_log"].events == []
    assert all("OPENROUTER_API_KEY" not in value for value in rendered_values)
    assert all("Traceback" not in value for value in rendered_values)


def test_prompt_injection_warning_lists_safe_indicator_codes() -> None:
    app = _seeded_app(include_evaluation=True)
    evaluation = app.session_state["candidate_evaluation"]
    app.session_state["candidate_evaluation"] = evaluation.model_copy(
        update={
            "prompt_injection_detected": True,
            "prompt_injection_indicators": [
                "fake_system_message",
                "score_manipulation",
            ],
        }
    )

    app.run()

    warnings = [item.value for item in app.warning]
    assert any("fake_system_message, score_manipulation" in item for item in warnings)
