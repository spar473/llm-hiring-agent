"""Validated, offline role fixtures used by the demonstration UI."""

from __future__ import annotations

from pathlib import Path

from src.models import RoleSpecification


CONFLICTING_STAKEHOLDERS_FIXTURE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "fixtures"
    / "conflicting_manager_inputs_role.json"
)


def load_conflicting_stakeholders_demo() -> RoleSpecification:
    """Load the deliberate unresolved-contradiction role fixture."""
    return RoleSpecification.model_validate_json(
        CONFLICTING_STAKEHOLDERS_FIXTURE.read_text(encoding="utf-8")
    )
