"""Record one raw browser-viewport capture of the live Streamlit happy path.

This is intentionally a real UI run: discovery, hiring-pack generation, and
candidate evaluation all invoke the configured provider through ``app.py``.
Playwright writes the WebM stream directly; the script only renames the file
after the browser context closes and does not edit or transcode it.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
from playwright.sync_api import Locator, Page, TimeoutError, sync_playwright


ROOT = Path(__file__).resolve().parents[1]
RECORDING_ROOT = ROOT / "recordings"
PORT = 8517
APP_URL = f"http://127.0.0.1:{PORT}"
MINIMUM_RECORDING_SECONDS = 185
MAXIMUM_RECORDING_SECONDS = 295


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _wait_for_streamlit(
    page: Page,
    timeout: int = 120_000,
    *,
    provider_call: bool = False,
) -> None:
    """Wait for any active Streamlit spinner and the subsequent rerun."""
    spinner = page.locator('[data-testid="stSpinner"]')
    try:
        spinner.first.wait_for(
            state="visible",
            timeout=3_000 if provider_call else 700,
        )
    except TimeoutError:
        pass
    try:
        spinner.first.wait_for(state="hidden", timeout=timeout)
    except TimeoutError:
        pass
    page.wait_for_timeout(500)


def _open_tab(page: Page, name: str) -> None:
    page.get_by_role("tab", name=name, exact=True).click()
    page.wait_for_timeout(300)


def _expander(page: Page, text: str | re.Pattern[str]) -> Locator:
    return page.locator('[data-testid="stExpander"]').filter(has_text=text).first


def _ensure_expanded(expander: Locator) -> None:
    if expander.evaluate("(element) => element.tagName.toLowerCase()") == "details":
        details = expander
    else:
        details = expander.locator("details")
    if details.count() and details.get_attribute("open") is None:
        details.locator("summary").click()
    expander.scroll_into_view_if_needed()


def _choose_combobox(
    page: Page,
    scope: Locator,
    label: str,
    option: str,
) -> None:
    field = scope.get_by_label(label, exact=True).last
    if field.input_value() == option:
        return
    field.click()
    field.fill(option)
    field.press("Enter")
    page.wait_for_timeout(300)


def _record_step(steps: list[dict[str, str]], name: str) -> None:
    steps.append({"step": name, "timestamp_utc": _utc_now()})


def _start_server(log_path: Path) -> tuple[subprocess.Popen[bytes], object]:
    log_file = log_path.open("wb")
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app.py",
            "--server.headless",
            "true",
            "--server.port",
            str(PORT),
            "--browser.gatherUsageStats",
            "false",
        ],
        cwd=ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        creationflags=creation_flags,
    )
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        if process.poll() is not None:
            log_file.close()
            raise RuntimeError("Streamlit exited before becoming ready.")
        try:
            response = httpx.get(APP_URL, timeout=1)
        except httpx.HTTPError:
            time.sleep(0.25)
            continue
        if response.status_code == 200:
            return process, log_file
    process.terminate()
    log_file.close()
    raise RuntimeError("Streamlit did not become ready within 45 seconds.")


def _save_role_sections(page: Page) -> None:
    _open_tab(page, "Review Role")
    business = _expander(page, "Business need")
    _ensure_expanded(business)
    business.get_by_label("Business problem").fill(
        "The Brand Marketing team needs additional delivery capacity for its "
        "eight-week summer short-form campaign."
    )
    business.get_by_label("Why now").fill(
        "The approved summer campaign calendar starts in December."
    )
    business.get_by_label("Cost of leaving this vacant").fill(
        "The team would miss its weekly approved-script cadence."
    )
    business.get_by_role("button", name="Save business need").click()
    _wait_for_streamlit(page)

    outcomes = [
        (
            "Deliver one approved TikTok or Instagram Reels script each week.",
            "Eight-week internship",
            "Eight approved scripts by the final day",
        ),
        (
            "Submit every draft to the Brand Marketing Lead before publishing.",
            "Each campaign week",
            "100% of drafts reviewed before publication",
        ),
        (
            "Apply documented manager feedback to each requested revision.",
            "Within one business day of feedback",
            "Every requested revision addressed or explicitly discussed",
        ),
    ]
    for description, horizon, measure in outcomes:
        _open_tab(page, "Review Role")
        outcome = _expander(page, re.compile(r"Success outcomes \(\d+\)"))
        _ensure_expanded(outcome)
        outcome.get_by_label("Description", exact=True).last.fill(description)
        outcome.get_by_label("Time horizon", exact=True).last.fill(horizon)
        outcome.get_by_label("Measure", exact=True).last.fill(measure)
        outcome.get_by_role("button", name="Add outcome").click()
        _wait_for_streamlit(page)

    _open_tab(page, "Review Role")
    responsibility = _expander(page, re.compile(r"Responsibilities \(\d+\)"))
    _ensure_expanded(responsibility)
    responsibility.get_by_label("Description", exact=True).last.fill(
        "Draft and film short-form content with the Brand Marketing team."
    )
    responsibility.get_by_label("Frequency", exact=True).last.fill("Weekly")
    responsibility.get_by_label("Ownership level", exact=True).last.fill("Shared")
    priority = responsibility.get_by_label("Priority", exact=True).last
    priority.click()
    page.get_by_role("option", name="Must Have", exact=True).last.click()
    responsibility.get_by_role("button", name="Add responsibility").click()
    _wait_for_streamlit(page)

    _open_tab(page, "Review Role")
    constraints = _expander(page, "Constraints and logistics")
    _ensure_expanded(constraints)
    constraints.get_by_label("Country").fill("New Zealand")
    constraints.get_by_label("Location", exact=True).fill("Auckland")
    constraints.get_by_label("Work arrangement").fill("On-site")
    constraints.get_by_label("Work rights").fill(
        "Must hold valid New Zealand work rights."
    )
    constraints.get_by_label("Weekly hours").fill("40 hours")
    constraints.get_by_role("button", name="Save constraints").click()
    _wait_for_streamlit(page)

    _open_tab(page, "Review Role")
    assessment = _expander(page, "Assessment plan")
    _ensure_expanded(assessment)
    assessment.get_by_label("Assessment methods (one per line)").fill(
        "Structured screening questions"
    )
    assessment.get_by_label("Decision owner").fill("Talent Acquisition Partner")
    assessment.get_by_role("button", name="Save assessment plan").click()
    _wait_for_streamlit(page)

    _open_tab(page, "Review Role")
    values = _expander(page, re.compile(r"Company behaviours \(\d+\)"))
    _ensure_expanded(values)
    values.get_by_label("Company value", exact=True).last.fill("Collaboration")
    values.get_by_label("Role-relevant behaviour", exact=True).last.fill(
        "Shares draft work early and incorporates specific feedback."
    )
    values.get_by_label("Scenario", exact=True).last.fill(
        "A campaign script receives revision feedback."
    )
    values.get_by_label("Evidence method", exact=True).last.fill(
        "Behavioural screening response"
    )
    values.get_by_role("button", name="Add behaviour").click()
    _wait_for_streamlit(page)


def _complete_requirements(page: Page) -> None:
    def editable_requirements() -> list[Locator]:
        marker = page.get_by_text("Edit requirements", exact=True)
        summary_column = marker.locator(
            "xpath=ancestor::*[@data-testid='stColumn']"
        ).first
        candidates = summary_column.locator('[data-testid="stExpander"]')
        editable: list[Locator] = []
        for candidate_index in range(candidates.count()):
            candidate = candidates.nth(candidate_index)
            _ensure_expanded(candidate)
            if candidate.get_by_role(
                "button", name="Save changes"
            ).count():
                editable.append(candidate)
        return editable

    _open_tab(page, "Define Role")
    marker = page.get_by_text("Edit requirements", exact=True)
    marker.scroll_into_view_if_needed()
    requirement_count = len(editable_requirements())
    if not requirement_count:
        raise RuntimeError("No extracted requirement was available to confirm.")

    # The manager deliberately narrows the noisy provider draft to one
    # assessable must-have before generation.
    for requirement_index in range(requirement_count - 1, 0, -1):
        _open_tab(page, "Define Role")
        editable = editable_requirements()[requirement_index]
        confirmation = editable.get_by_label(
            "Confirm deletion from the current role version",
            exact=True,
        )
        confirmation.evaluate("(element) => element.click()")
        page.wait_for_timeout(350)
        _open_tab(page, "Define Role")
        editable = editable_requirements()[requirement_index]
        editable.get_by_role(
            "button", name="Delete requirement", exact=True
        ).click()
        _wait_for_streamlit(page)

    requirement_count = 1
    for requirement_index in range(requirement_count):
        _open_tab(page, "Define Role")
        editable = editable_requirements()[requirement_index]
        _choose_combobox(
            page,
            editable,
            "Priority",
            "Must Have" if requirement_index == 0 else "Preferred",
        )
        rationale = editable.get_by_label("Business rationale", exact=True)
        rationale.fill(
            "This capability directly supports the approved weekly campaign "
            "delivery plan."
        )
        _choose_combobox(page, editable, "Proficiency", "Working")
        _choose_combobox(page, editable, "Learnability", "Day One")
        editable.get_by_label(
            "Accepted equivalents (one per line)", exact=True
        ).fill("A portfolio or project demonstrating equivalent capability")
        editable.get_by_label(
            "Evidence methods (one per line)", exact=True
        ).fill("Structured screening question")
        editable.get_by_role("button", name="Save changes").click()
        _wait_for_streamlit(page)


def _approve_role(page: Page) -> None:
    _open_tab(page, "Review Role")
    approve = page.get_by_role("button", name="Approve role", exact=True)
    approve.scroll_into_view_if_needed()
    form = approve.locator("xpath=ancestor::*[@data-testid='stForm']")
    form.get_by_label("Approver name").fill("Capture Hiring Manager")
    checkbox_labels = [
        form.get_by_role("checkbox").nth(index).get_attribute("aria-label")
        for index in range(form.get_by_role("checkbox").count())
    ]
    for label in checkbox_labels:
        if not label:
            continue
        checkbox = page.get_by_label(label, exact=True)
        if not checkbox.is_checked():
            checkbox.evaluate("(element) => element.click()")
            page.wait_for_timeout(350)
        if not page.get_by_label(label, exact=True).is_checked():
            raise RuntimeError(f"Approval checkbox did not persist: {label}")
    approve = page.get_by_role("button", name="Approve role", exact=True)
    approve.click()
    _wait_for_streamlit(page)
    _open_tab(page, "Review Role")
    approval_confirmation = page.get_by_text(
        re.compile(r"Approved by Capture Hiring Manager"),
    )
    try:
        approval_confirmation.wait_for(state="visible", timeout=20_000)
    except TimeoutError as exc:
        alerts = page.locator('[data-testid="stAlertContentError"]')
        reasons = [
            alerts.nth(index).inner_text() for index in range(alerts.count())
        ]
        raise RuntimeError(
            "Role approval did not complete: " + " | ".join(reasons)
        ) from exc


def _generate_pack(page: Page) -> None:
    _open_tab(page, "Hiring Pack")
    generate = page.get_by_role("button", name="Generate hiring pack", exact=True)
    generate.scroll_into_view_if_needed()
    generate.click()
    _wait_for_streamlit(page, timeout=150_000, provider_call=True)
    _open_tab(page, "Hiring Pack")
    page.get_by_text(
        re.compile(r"Screening questions \(\d+\)")
    ).wait_for(state="visible", timeout=20_000)


def _evaluate_candidate(page: Page) -> None:
    _open_tab(page, "Candidate Evidence")
    page.get_by_role(
        "textbox", name="Anonymous candidate identifier", exact=True
    ).fill("capture_candidate_001")
    response = (
        "I am available on-site in Auckland from 1 December through 23 January "
        "with no conflicting commitments. On 12 June 2026, I drafted a "
        "short-form campaign script from a brand brief, filmed it with a "
        "teammate, and sent the draft to the marketing lead one business day "
        "early. I incorporated two specific feedback changes before approval. "
        "The published post reached 18,400 views and 9.2% engagement; my "
        "contribution was the hook, script, filming plan, and revision. I can "
        "provide the dated brief, draft history, and analytics screenshot."
    )
    answers = page.locator("textarea:visible")
    if answers.count() < 1:
        raise RuntimeError("No candidate response fields were rendered.")
    for index in range(answers.count()):
        answers.nth(index).fill(response)

    evaluation_heading = page.get_by_text(
        re.compile(r"Evaluation for capture_candidate_001")
    )
    rejection_reasons: list[str] = []
    for _ in range(3):
        run = page.get_by_role(
            "button", name="Run evidence evaluation", exact=True
        )
        run.scroll_into_view_if_needed()
        run.click()
        _wait_for_streamlit(page, timeout=150_000, provider_call=True)
        _open_tab(page, "Candidate Evidence")
        if evaluation_heading.count() and evaluation_heading.first.is_visible():
            evaluation_heading.first.scroll_into_view_if_needed()
            return
        errors = page.locator('[data-testid="stAlertContentError"]')
        rejection_reasons.extend(
            errors.nth(index).inner_text() for index in range(errors.count())
        )
    raise RuntimeError(
        "Candidate evaluation did not validate after three attempts: "
        + " | ".join(rejection_reasons)
    )


def main() -> int:
    RECORDING_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = RECORDING_ROOT / f"live_happy_path_{timestamp}.webm"
    evidence_path = destination.with_suffix(".json")
    server_log = RECORDING_ROOT / f"streamlit_{timestamp}.log"
    failure_screenshot = RECORDING_ROOT / f"live_happy_path_{timestamp}_failure.png"
    steps: list[dict[str, str]] = []
    server, log_file = _start_server(server_log)
    started_monotonic = time.monotonic()
    started_at_utc = _utc_now()

    try:
        with tempfile.TemporaryDirectory(
            prefix="capture-", dir=RECORDING_ROOT
        ) as video_directory:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                context = browser.new_context(
                    viewport={"width": 1440, "height": 900},
                    record_video_dir=video_directory,
                    record_video_size={"width": 1440, "height": 900},
                )
                page = context.new_page()
                page.set_default_timeout(30_000)
                video = page.video
                try:
                    page.goto(APP_URL, wait_until="domcontentloaded")
                    page.get_by_role(
                        "heading", name="LLM Hiring Agent"
                    ).wait_for(state="visible", timeout=30_000)
                    _record_step(steps, "Define Role")

                    page.get_by_label("Role title").fill("Marketing Intern")
                    page.get_by_label("Division").fill("Consumer Products")
                    page.get_by_label("Team").fill("Brand Marketing")
                    page.get_by_label("Location", exact=True).first.fill("Auckland")
                    # Marketing / Intern / Internship are the app's defaults.
                    page.get_by_label("Describe the role in your own words").fill(
                        "We need a Marketing Intern for summer. They should be "
                        "creative and good with social media. Maybe some design "
                        "skills? They'll work with the team on TikTok stuff and "
                        "help with campaigns. Should be fun to work with."
                    )
                    page.get_by_role(
                        "button", name="Start discovery", exact=True
                    ).click()
                    _wait_for_streamlit(page, timeout=120_000)
                    page.get_by_text(
                        re.compile(r"Current stage:")
                    ).wait_for(state="visible", timeout=20_000)
                    _record_step(steps, "Discovery")

                    focused_answers = [
                        (
                            "They'll script and film short TikTok videos with the "
                            "team, then the Brand Marketing Lead handles final "
                            "editing and posting. They coordinate assets between "
                            "the design and social teams, not campaigns solo."
                        ),
                        (
                            "They should share a draft script one business day "
                            "before filming, ask specific questions about feedback, "
                            "revise it, and confirm approval before filming."
                        ),
                        (
                            "Entry-level experience from a student project or "
                            "personal channel is enough. Prior professional social "
                            "media employment is not required."
                        ),
                    ]
                    accepted_discovery = False
                    for focused_answer in focused_answers:
                        answer_field = page.get_by_label("Your answer")
                        if not answer_field.count() or not answer_field.is_visible():
                            break
                        answer_field.fill(focused_answer)
                        page.get_by_role(
                            "button", name="Send", exact=True
                        ).click()
                        page.wait_for_function(
                            """answer => {
                                const text = document.body.innerText;
                                return text.includes(answer) ||
                                    text.includes(
                                        "The discovery turn could not be completed:"
                                    );
                            }""",
                            arg=focused_answer,
                            timeout=120_000,
                        )
                        page.wait_for_timeout(500)
                        manager_message = page.get_by_text(
                            focused_answer, exact=True
                        )
                        requirement_marker = page.get_by_text(
                            "Edit requirements", exact=True
                        )
                        if (
                            manager_message.count()
                            and manager_message.first.is_visible()
                            and requirement_marker.count()
                            and requirement_marker.first.is_visible()
                        ):
                            accepted_discovery = True
                            break
                    if not accepted_discovery:
                        raise RuntimeError(
                            "Live discovery did not produce an editable "
                            "requirement after three focused answers."
                        )

                    _save_role_sections(page)
                    _complete_requirements(page)
                    _record_step(steps, "Review / Approve")
                    _approve_role(page)

                    _record_step(steps, "Generate Pack")
                    _generate_pack(page)

                    _record_step(steps, "Candidate Evidence")
                    _evaluate_candidate(page)
                    _record_step(steps, "Score")

                    elapsed = time.monotonic() - started_monotonic
                    if elapsed < MINIMUM_RECORDING_SECONDS:
                        page.wait_for_timeout(
                            int((MINIMUM_RECORDING_SECONDS - elapsed) * 1000)
                        )
                    elapsed = time.monotonic() - started_monotonic
                    if elapsed > MAXIMUM_RECORDING_SECONDS:
                        raise RuntimeError(
                            f"Recording exceeded five-minute bound: {elapsed:.1f}s"
                        )
                except Exception:
                    page.screenshot(path=failure_screenshot, full_page=True)
                    raise
                finally:
                    page.close()
                    context.close()
                    browser.close()

                if video is None:
                    raise RuntimeError("Playwright did not create a video handle.")
                recorded_path = Path(video.path())
                recorded_path.replace(destination)
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5)
        log_file.close()

    completed_at_utc = _utc_now()
    elapsed_seconds = round(time.monotonic() - started_monotonic, 3)
    evidence = {
        "file": str(destination),
        "capture_type": "raw Playwright browser-viewport WebM",
        "edited": False,
        "model": "openai/gpt-5.6-terra",
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "elapsed_wall_clock_seconds": elapsed_seconds,
        "within_3_to_5_minutes": 180 <= elapsed_seconds <= 300,
        "steps": steps,
    }
    evidence_path.write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(evidence, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
