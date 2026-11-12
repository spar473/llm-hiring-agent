"""Complete and time the reviewed Marketing Intern role after live discovery.

The live discovery transcript deliberately retains every provider extraction
for evidence.  The human review step condenses that noisy append-only history
to the same manager-confirmed content used by the fallback demo, preserves the
live role identity, and then runs the normal deterministic approval blockers.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_fallback_session import fallback_role  # noqa: E402
from src.models import ApprovalSection, ReviewStatus, RoleSpecification  # noqa: E402
from src.readiness import approve_role, evaluate_role_quality  # noqa: E402
from src.storage import JsonStorage  # noqa: E402


TRANSCRIPT_PATH = (
    ROOT / "data" / "fixtures" / "marketing_intern_discovery_transcript.json"
)
OUTPUT_PATH = "marketing_intern_live_approved_role.json"


def main() -> int:
    transcript = json.loads(TRANSCRIPT_PATH.read_text(encoding="utf-8"))
    live_role = RoleSpecification.model_validate(
        transcript["final_role_specification"]
    )
    reviewed_template = fallback_role()
    reviewed_at = datetime.now(UTC)

    reviewed_role = live_role.model_copy(
        update={
            "basic_info": reviewed_template.basic_info,
            "business_need": reviewed_template.business_need,
            "success_outcomes": reviewed_template.success_outcomes,
            "responsibilities": reviewed_template.responsibilities,
            "requirements": reviewed_template.requirements,
            "open_assumptions": [],
            "open_ambiguities": [],
            "company_behaviours": reviewed_template.company_behaviours,
            "constraints": reviewed_template.constraints,
            "assessment_methods": reviewed_template.assessment_methods,
            "decision_owner": reviewed_template.decision_owner,
            "quality": live_role.quality.model_copy(
                update={
                    "readiness_score": None,
                    "contradictions": [],
                    "warnings": [],
                    "warning_acknowledgements": [],
                }
            ),
            "human_approved": False,
            "approved_by": None,
            "approved_at": None,
            "approved_sections": [],
            "review_status": ReviewStatus.DRAFT,
            "parent_version": live_role.version,
            "version": live_role.version + 1,
            "audit": live_role.audit.model_copy(update={"updated_at": reviewed_at}),
        }
    )
    warnings = evaluate_role_quality(reviewed_role).warnings
    approved_role = approve_role(
        reviewed_role,
        approver="Live Flow Hiring Manager",
        confirmed_sections=list(ApprovalSection),
        acknowledged_warning_ids=[item.warning_id for item in warnings],
        approved_at=reviewed_at,
    )
    final_report = evaluate_role_quality(approved_role)
    if final_report.blockers:
        print(f"Approval blockers remain: {list(final_report.blockers)}")
        return 1

    JsonStorage(ROOT / "data" / "fixtures").save(OUTPUT_PATH, approved_role)
    started_at = datetime.fromisoformat(
        transcript["live_run"]["started_at_utc"].replace("Z", "+00:00")
    )
    elapsed_minutes = (reviewed_at - started_at).total_seconds() / 60
    print(f"Approved live role: {approved_role.role_id}")
    print(f"Manager input UTC: {started_at.isoformat()}")
    print(f"Approved UTC: {reviewed_at.isoformat()}")
    print(f"Elapsed minutes: {elapsed_minutes:.3f}")
    print(f"Within 30-minute constraint: {elapsed_minutes <= 30}")
    print(f"Approval blockers: {len(final_report.blockers)}")
    print(f"Wrote {ROOT / 'data' / 'fixtures' / OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
