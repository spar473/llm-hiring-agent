"""Manual, credit-consuming live discovery runner for the required test personas.

Runs the exact same call path as ``app.py`` (``take_discovery_turn`` against a
live ``OpenRouterClient``) so recorded fixtures reflect the live app, not a
special-cased script. Each invocation performs at most one live discovery
turn; working state persists between calls so a human can read the model's
``next_question`` and choose the next in-character scripted answer, the same
way a hiring manager would type a reply in the Streamlit UI.

Usage:
    python scripts/run_persona_discovery.py <persona_key> --start
    python scripts/run_persona_discovery.py <persona_key> --answer "..."
    python scripts/run_persona_discovery.py <persona_key> --show
    python scripts/run_persona_discovery.py <persona_key> --finalize

State lives at data/fixtures/_live_runs/<persona_key>.json (gitignored working
data). ``--finalize`` writes the reviewed, redacted transcript fixture to
data/fixtures/<persona_key>_discovery_transcript.json and leaves the working
file in place for reference.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from pydantic import ValidationError


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ConfigurationError, Settings  # noqa: E402
from src.llm_client import LLMClientError, OpenRouterClient  # noqa: E402
from src.models import (  # noqa: E402
    BasicRoleInfo,
    DiscoverySemanticValidationError,
    EmploymentType,
    RoleFamily,
    RoleLevel,
    RoleSpecification,
    WorkflowStage,
)
from src.readiness import evaluate_role_quality  # noqa: E402
from src.workflow import WorkflowState  # noqa: E402
from src.discovery import take_discovery_turn  # noqa: E402


LIVE_RUN_DIR = ROOT / "data" / "fixtures" / "_live_runs"
FIXTURE_DIR = ROOT / "data" / "fixtures"


PERSONAS: dict[str, dict[str, object]] = {
    "vague_executive": {
        "persona_label": "Persona A: Vague executive",
        "brief_source": (
            "docs/SCENARIOS.md "
            ""
        ),
        "documented_expected_behaviour": [
            "Break the statement into outcomes, priorities, scope, and trade-offs.",
            "Ask for concrete examples.",
            "Force prioritisation rather than recording vague adjectives.",
        ],
        "role_id": "role_live_vague_executive",
        "basic_info": BasicRoleInfo(
            title="Head of Growth",
            role_family=RoleFamily.EXECUTIVE,
            role_level=RoleLevel.EXECUTIVE,
            employment_type=EmploymentType.PERMANENT,
        ),
        "initial_manager_statement": (
            "I need a superstar who can do a bit of everything."
        ),
    },
    "over_technical_manager": {
        "persona_label": "Persona B: Over-technical manager",
        "brief_source": (
            "docs/SCENARIOS.md "
            ""
        ),
        "documented_expected_behaviour": [
            "Cluster the skills.",
            "Test whether each is genuinely necessary.",
            "Separate day-one requirements from learnable skills.",
            "Calibrate requirements against seniority.",
            "Flag possible role overloading or role splitting.",
        ],
        "role_id": "role_live_over_technical_manager",
        "basic_info": BasicRoleInfo(
            title="AI Integration Intern",
            role_family=RoleFamily.TECHNICAL,
            role_level=RoleLevel.INTERN,
            employment_type=EmploymentType.INTERNSHIP,
        ),
        "initial_manager_statement": (
            "For this entry-level intern I need someone who is already fully "
            "proficient on day one in: Python, SQL, R, Java, C++, Docker, "
            "Kubernetes, AWS, Azure, GCP, Terraform, CI/CD pipelines, REST API "
            "design, GraphQL, microservices architecture, PostgreSQL, MongoDB, "
            "Redis, Apache Kafka, Airflow, Spark, Hadoop, machine learning "
            "model training, deep learning with PyTorch and TensorFlow, "
            "retrieval-augmented generation, LangChain, vector databases, "
            "prompt engineering, fine-tuning LLMs, MLOps, data engineering "
            "pipelines, Tableau, Power BI, Excel macros, statistics, A/B "
            "testing, product management, stakeholder presentations, "
            "technical writing, Figma for internal tool mockups, Photoshop "
            "for the odd deck graphic, project management in Jira, Scrum "
            "mastery, budget forecasting, and public speaking. They need every "
            "one of these on day one, no exceptions."
        ),
    },
    "culture_focused_manager": {
        "persona_label": "Persona C: Culture-focused manager",
        "brief_source": (
            "docs/SCENARIOS.md "
            ""
        ),
        "documented_expected_behaviour": [
            "Convert subjective culture language into observable workplace behaviours.",
            "Avoid assessing similarity, likeability, or background resemblance.",
            "Link behaviours to role situations and Company Values.",
        ],
        "role_id": "role_live_culture_focused_manager",
        "basic_info": BasicRoleInfo(
            title="Brand Coordinator",
            role_family=RoleFamily.MARKETING,
            role_level=RoleLevel.ENTRY,
            employment_type=EmploymentType.PERMANENT,
        ),
        "initial_manager_statement": (
            "Skills can be taught. I need someone who gets our vibe."
        ),
    },
    "marketing_intern": {
        "persona_label": "Real-world scenario: Marketing Intern",
        "brief_source": (
            "docs/SCENARIOS.md"
        ),
        "documented_expected_behaviour": [
            "Gather all required information efficiently and in an easy-to-use manner.",
            "Transform the input into comprehensive requirements.",
            "Resolve team/brand, dates/location, content responsibilities, "
            "platform/tools, design requirement level, campaign "
            "responsibilities, success metrics, collaboration behaviours, "
            "and screening evidence.",
        ],
        "role_id": "role_live_marketing_intern",
        "basic_info": BasicRoleInfo(
            title="Marketing Intern",
            role_family=RoleFamily.MARKETING,
            role_level=RoleLevel.INTERN,
            employment_type=EmploymentType.INTERNSHIP,
        ),
        "initial_manager_statement": (
            "We need a Marketing Intern for summer. They should be creative "
            "and good with social media. Maybe some design skills? They'll "
            "work with the team on TikTok stuff and help with campaigns. "
            "Should be fun to work with."
        ),
    },
    "technical_role": {
        "persona_label": "Component 1 example: technical role (AI Integration Intern)",
        "brief_source": "docs/SCENARIOS.md",
        "documented_expected_behaviour": [
            "Probe systems and architecture, technical depth, data and "
            "integrations, production versus prototype expectations, "
            "security and privacy, reliability and monitoring, debugging, "
            "testing, deployment, documentation, stakeholder translation, "
            "and acceptable equivalent technologies.",
        ],
        "role_id": "role_live_technical_role",
        "basic_info": BasicRoleInfo(
            title="AI Integration Intern",
            role_family=RoleFamily.TECHNICAL,
            role_level=RoleLevel.INTERN,
            employment_type=EmploymentType.INTERNSHIP,
        ),
        "initial_manager_statement": (
            "We need an AI Integration Intern to help embed AI workflows "
            "into the Marketing team. They'll build small internal tools "
            "and prototypes for us."
        ),
    },
    "creative_role": {
        "persona_label": "Component 1 example: creative role (Brand Designer)",
        "brief_source": "docs/SCENARIOS.md",
        "documented_expected_behaviour": [
            "Probe target audience, brand and channel, ideation versus "
            "execution, portfolio evidence, visual/copy/video "
            "responsibilities, production tools, creative approval "
            "process, speed and volume, response to feedback, and "
            "commercial performance measures.",
        ],
        "role_id": "role_live_creative_role",
        "basic_info": BasicRoleInfo(
            title="Brand Designer",
            role_family=RoleFamily.CREATIVE,
            role_level=RoleLevel.INTERMEDIATE,
            employment_type=EmploymentType.PERMANENT,
        ),
        "initial_manager_statement": (
            "We're hiring a Brand Designer to work on packaging and "
            "campaign visuals across our Toys and Edge brands."
        ),
    },
}


def _persona(persona_key: str) -> dict[str, object]:
    try:
        return PERSONAS[persona_key]
    except KeyError:
        raise SystemExit(
            f"Unknown persona '{persona_key}'. Choices: {', '.join(PERSONAS)}"
        ) from None


def _state_path(persona_key: str) -> Path:
    return LIVE_RUN_DIR / f"{persona_key}.json"


def _load_state(persona_key: str) -> dict[str, object]:
    path = _state_path(persona_key)
    if not path.exists():
        raise SystemExit(
            f"No in-progress run for '{persona_key}'. Start it with --start first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _save_state(persona_key: str, payload: dict[str, object]) -> None:
    LIVE_RUN_DIR.mkdir(parents=True, exist_ok=True)
    _state_path(persona_key).write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )


def _utc_now() -> str:
    """Return a stable UTC timestamp for live-run evidence."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _client() -> tuple[OpenRouterClient, str]:
    try:
        settings = Settings.from_env()
        settings.require_api_configuration()
    except (ValidationError, ConfigurationError) as exc:
        raise SystemExit(f"OpenRouter is not configured: {exc}") from None
    assert settings.openrouter_model is not None
    return OpenRouterClient(settings), settings.openrouter_model


def _run_turn(
    persona_key: str,
    *,
    state: WorkflowState,
    manager_message: str,
    turns: list[dict[str, object]],
) -> tuple[WorkflowState, list[dict[str, object]]]:
    stage_before = state.current_stage
    client, model = _client()
    started_at = _utc_now()
    stopwatch = perf_counter()
    try:
        new_state = take_discovery_turn(
            state=state,
            manager_message=manager_message,
            llm_client=client,
            prior_manager_messages=[
                str(turn["manager_message"]) for turn in turns
            ],
        )
    except (LLMClientError, DiscoverySemanticValidationError) as exc:
        elapsed_seconds = round(perf_counter() - stopwatch, 3)
        rejection: dict[str, object] = {
            "turn_index": len(turns) + 1,
            "stage_before": stage_before.value,
            "manager_message": manager_message,
            "model": model,
            "started_at_utc": started_at,
            "elapsed_wall_clock_seconds": elapsed_seconds,
            "error_type": type(exc).__name__,
        }
        if isinstance(exc, DiscoverySemanticValidationError):
            rejection["validation_field_locations"] = [
                location for location, _ in exc.issues
            ]
            rejection["validation_error_types"] = [
                error_type for _, error_type in exc.issues
            ]
            if exc.offending_quotes:
                rejection["offending_quotes"] = list(exc.offending_quotes)
        elif exc.diagnostics is not None:
            rejection["provider_diagnostics"] = exc.diagnostics.as_dict()

        path = _state_path(persona_key)
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            rejected_turns = list(payload.get("rejected_turns", []))
            rejected_turns.append(rejection)
            payload["rejected_turns"] = rejected_turns
            _save_state(persona_key, payload)

        print(f"TURN FAILED: {exc}")
        print(json.dumps(rejection, indent=2))
        raise SystemExit(1) from None

    elapsed_seconds = round(perf_counter() - stopwatch, 3)
    turn_record = {
        "turn_index": len(turns) + 1,
        "stage_before": stage_before.value,
        "manager_message": manager_message,
        "stage_after": new_state.current_stage.value,
        "model": model,
        "started_at_utc": started_at,
        "elapsed_wall_clock_seconds": elapsed_seconds,
        "next_question": (
            new_state.current_question.model_dump(mode="json")
            if new_state.current_question is not None
            else None
        ),
        "new_requirement_count": len(new_state.role_specification.requirements)
        - len(state.role_specification.requirements),
    }
    turns = [*turns, turn_record]
    print(json.dumps(turn_record, indent=2))
    print("\n--- Full role snapshot after this turn ---")
    print(new_state.role_specification.model_dump_json(indent=2))
    return new_state, turns


def cmd_start(persona_key: str) -> None:
    persona = _persona(persona_key)
    recent_start_rejections: list[dict[str, object]] = []
    state_path = _state_path(persona_key)
    if state_path.exists():
        previous_payload = json.loads(state_path.read_text(encoding="utf-8"))
        now = datetime.now(timezone.utc)
        for item in previous_payload.get("rejected_turns", []):
            item_started_at = item.get("started_at_utc")
            if item.get("turn_index") != 1 or not isinstance(item_started_at, str):
                continue
            item_started = datetime.fromisoformat(
                item_started_at.replace("Z", "+00:00")
            )
            age_seconds = (now - item_started).total_seconds()
            if 0 <= age_seconds <= 600:
                recent_start_rejections.append(item)
    basic_info: BasicRoleInfo = persona["basic_info"]  # type: ignore[assignment]
    role = RoleSpecification(
        role_id=str(persona["role_id"]),
        basic_info=basic_info.model_copy(
            update={
                "initial_manager_statement": persona["initial_manager_statement"]
            }
        ),
    )
    state = WorkflowState(role_specification=role, current_stage=WorkflowStage.BASIC_INFO)
    settings = Settings.from_env()
    started_at_utc = _utc_now()
    _save_state(
        persona_key,
        {
            "run_metadata": {
                "provider": "OpenRouter",
                "model": settings.openrouter_model or "not configured",
                "started_at_utc": started_at_utc,
            },
            "role_specification": state.role_specification.model_dump(mode="json"),
            "current_stage": state.current_stage.value,
            "confirmed_stages": [],
            "current_question": None,
            "turns": [],
            "rejected_turns": recent_start_rejections,
        },
    )
    new_state, turns = _run_turn(
        persona_key,
        state=state,
        manager_message=str(persona["initial_manager_statement"]),
        turns=[],
    )
    first_turn = turns[0]
    _save_state(
        persona_key,
        {
            "run_metadata": {
                "provider": "OpenRouter",
                "model": first_turn["model"],
                "started_at_utc": started_at_utc,
            },
            "role_specification": new_state.role_specification.model_dump(mode="json"),
            "current_stage": new_state.current_stage.value,
            "confirmed_stages": sorted(
                stage.value for stage in new_state.confirmed_stages
            ),
            "current_question": (
                new_state.current_question.model_dump(mode="json")
                if new_state.current_question is not None
                else None
            ),
            "turns": turns,
            "rejected_turns": recent_start_rejections,
        },
    )


def cmd_answer(persona_key: str, answer: str) -> None:
    payload = _load_state(persona_key)
    role = RoleSpecification.model_validate(payload["role_specification"])
    state = WorkflowState(
        role_specification=role,
        current_stage=WorkflowStage(payload["current_stage"]),
        confirmed_stages={
            WorkflowStage(item) for item in payload["confirmed_stages"]
        },
        current_question=(
            None
            if payload["current_question"] is None
            else state_question(payload["current_question"])
        ),
    )
    new_state, turns = _run_turn(
        persona_key, state=state, manager_message=answer, turns=payload["turns"]
    )
    _save_state(
        persona_key,
        {
            "run_metadata": payload.get("run_metadata", {}),
            "role_specification": new_state.role_specification.model_dump(mode="json"),
            "current_stage": new_state.current_stage.value,
            "confirmed_stages": sorted(
                stage.value for stage in new_state.confirmed_stages
            ),
            "current_question": (
                new_state.current_question.model_dump(mode="json")
                if new_state.current_question is not None
                else None
            ),
            "turns": turns,
            "rejected_turns": payload.get("rejected_turns", []),
        },
    )


def state_question(payload: dict[str, object]):
    from src.models import ClarificationQuestion

    return ClarificationQuestion.model_validate(payload)


def cmd_show(persona_key: str) -> None:
    payload = _load_state(persona_key)
    print(json.dumps(payload, indent=2))


def cmd_finalize(persona_key: str) -> None:
    persona = _persona(persona_key)
    payload = _load_state(persona_key)
    role = RoleSpecification.model_validate(payload["role_specification"])
    report = evaluate_role_quality(role)
    completed_at_utc = _utc_now()
    run_metadata = payload.get("run_metadata", {})
    started_at_utc = run_metadata.get("started_at_utc")
    run_elapsed_seconds = None
    if isinstance(started_at_utc, str):
        started = datetime.fromisoformat(started_at_utc.replace("Z", "+00:00"))
        completed = datetime.fromisoformat(completed_at_utc.replace("Z", "+00:00"))
        run_elapsed_seconds = round((completed - started).total_seconds(), 3)
    turns = payload["turns"]
    rejected_turns = payload.get("rejected_turns", [])
    provider_elapsed_seconds = round(
        sum(
            float(item.get("elapsed_wall_clock_seconds", 0))
            for item in [*turns, *rejected_turns]
        ),
        3,
    )
    live_run = {
        **run_metadata,
        "completed_at_utc": completed_at_utc,
        "successful_turn_count": len(turns),
        "rejected_turn_count": len(rejected_turns),
        "provider_elapsed_wall_clock_seconds": provider_elapsed_seconds,
        "run_elapsed_wall_clock_seconds": run_elapsed_seconds,
    }
    new_evidence = {
        "persona_key": persona_key,
        "persona_label": persona["persona_label"],
        "brief_source": persona["brief_source"],
        "documented_expected_behaviour": persona["documented_expected_behaviour"],
        "initial_manager_statement": persona["initial_manager_statement"],
        "live_run": live_run,
        "turns": turns,
        "rejected_turns": rejected_turns,
        "final_role_specification": payload["role_specification"],
        "final_quality_report": {
            "readiness_score": report.readiness.score,
            "readiness_interpretation": report.readiness.interpretation,
            "vague_phrases": [
                {
                    "phrase": item.phrase,
                    "category": item.category,
                    "source": item.source,
                }
                for item in report.vague_phrases
            ],
            "excessive_requirement_issues": [
                {
                    "issue_id": item.issue_id,
                    "rule": item.rule,
                    "message": item.message,
                    "requirement_ids": list(item.requirement_ids),
                }
                for item in report.excessive_requirements
            ],
            "blockers": list(report.blockers),
        },
    }
    destination = FIXTURE_DIR / f"{persona_key}_discovery_transcript.json"
    existing = (
        json.loads(destination.read_text(encoding="utf-8"))
        if destination.exists()
        else {}
    )
    fixture = {**existing, **new_evidence}
    destination.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {destination}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("persona_key", choices=list(PERSONAS))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--start", action="store_true")
    group.add_argument("--answer", type=str)
    group.add_argument("--show", action="store_true")
    group.add_argument("--finalize", action="store_true")
    args = parser.parse_args()

    if args.start:
        cmd_start(args.persona_key)
    elif args.answer is not None:
        cmd_answer(args.persona_key, args.answer)
    elif args.show:
        cmd_show(args.persona_key)
    elif args.finalize:
        cmd_finalize(args.persona_key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
