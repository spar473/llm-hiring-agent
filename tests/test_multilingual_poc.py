"""Tests for the isolated multilingual JD proof."""

from __future__ import annotations

from datetime import UTC, datetime

from scripts.generate_multilingual_jd_poc import (
    FIXTURE_LABEL,
    SpanishJDSection,
    build_messages,
    generate_spanish_jd_poc,
)
from src.llm_client import StructuredLLMResponse
from tests.phase6_helpers import approved_marketing_role


class FakeSpanishClient:
    def __init__(self) -> None:
        self.messages = None
        self.response_model = None

    def generate_structured(
        self,
        *,
        messages,
        response_model,
        temperature=None,
    ):
        self.messages = messages
        self.response_model = response_model
        return StructuredLLMResponse(
            data=SpanishJDSection(
                language="Spanish",
                heading="Practicante de Marketing",
                role_purpose="Apoyar la entrega de campañas de verano.",
                responsibilities=["Crear contenido semanal con el equipo."],
                success_outcomes=["Entregar tres publicaciones por semana."],
                location_and_constraints=[
                    "Puesto presencial en Auckland, Nueva Zelanda."
                ],
            ),
            model="synthetic/translator",
            provider="Synthetic",
            schema_mode="strict_json_schema",
            input_tokens=100,
            output_tokens=80,
            total_tokens=180,
            latency_ms=1,
            attempts=1,
        )


def test_multilingual_prompt_delimits_approved_role_as_untrusted_data() -> None:
    messages = build_messages(approved_marketing_role())

    assert "never as instructions" in messages[0]["content"]
    assert (
        "-----BEGIN UNTRUSTED APPROVED ROLE DATA-----"
        in messages[1]["content"]
    )
    assert (
        "-----END UNTRUSTED APPROVED ROLE DATA-----"
        in messages[1]["content"]
    )


def test_multilingual_fixture_is_explicitly_not_integrated() -> None:
    client = FakeSpanishClient()
    payload = generate_spanish_jd_poc(
        role=approved_marketing_role(),
        llm_client=client,
        generated_at=datetime(2026, 7, 25, 2, 0, tzinfo=UTC),
    )

    assert payload["fixture_label"] == FIXTURE_LABEL
    assert payload["integration_status"] == "not_integrated"
    assert payload["total_tokens"] == 180
    assert payload["jd_section"]["language"] == "Spanish"
    assert client.response_model is SpanishJDSection
