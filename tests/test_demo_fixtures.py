"""Tests for validated, offline demonstration fixtures."""

from src.demo_fixtures import load_conflicting_stakeholders_demo
from src.models import RoleSpecification


def test_conflicting_stakeholders_loader_returns_an_unresolved_role() -> None:
    role = load_conflicting_stakeholders_demo()

    assert isinstance(role, RoleSpecification)
    assert role.quality.contradictions
    assert any(not item.resolved for item in role.quality.contradictions)
