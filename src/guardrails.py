"""Shared free-text guardrail patterns and the hiring-pack field registry.

Both the generation path and the evaluation path must apply the *same*
fairness and autonomy rules to author-supplied prose. Keeping two private
copies of these pattern tuples is what let them drift: ``race conditions``
was allowlisted on the evaluation path only, so "I diagnosed a race
condition" was accepted by the evaluator and rejected by the generator,
while ``family status`` / ``health condition`` / ``political beliefs`` were
enforced on the evaluation path only. This module is the single definition
both paths import.

Two deliberate design rules live here:

* **Matching is case-insensitive in the pattern itself.** The generation
  path previously relied on its caller casefolding the text first, so any
  new caller that forgot silently disabled the entire protected scan with
  no error. Every pattern below carries ``re.I``.
* **Fields are enumerated, not concatenated.** ``HIRING_PACK_TEXT_FIELDS``
  is an explicit registry rather than one big JSON blob, so an unregistered
  field is structurally visible (``tests/test_guardrails.py`` fails the
  build) and a failure can name the offending field and text.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Callable, Iterable


# A matcher result: the text that was actually searched, plus the match.
_Hit = tuple[str, re.Match[str]] | None


def _hit(searched: str, match: re.Match[str] | None) -> _Hit:
    return None if match is None else (searched, match)


# --- Protected characteristics -------------------------------------------
#
# Matching a protected term is not enough to infer discriminatory intent.
# The shared policy therefore has three tiers:
#
# * a term coupled to a restriction construction is blocked;
# * a curated inclusive or idiomatic construction is accepted; and
# * any other mention is accepted with a field-specific human-review flag.
#
# This is deliberately lexical rather than a claim of semantic understanding.
PROTECTED_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:ages?|aged)\b", re.I),
    re.compile(r"\b(?:dates? of birth|birth dates?)\b", re.I),
    re.compile(r"\b(?:races?|racial)\b", re.I),
    re.compile(r"\b(?:ethnic(?:ity|ities| backgrounds?)?|ethnically)\b", re.I),
    re.compile(r"\b(?:nationalit(?:y|ies)|nationals?)\b", re.I),
    re.compile(r"\b(?:religions?|religious)\b", re.I),
    re.compile(r"\b(?:disabilit(?:y|ies)|disabled)\b", re.I),
    re.compile(r"\bpregnan(?:t|cy|cies)\b", re.I),
    re.compile(r"\bmarital(?: statuses?)?\b", re.I),
    re.compile(r"\b(?:family|familial)[- ]status(?:es)?\b", re.I),
    re.compile(r"\bgender(?:s|ed)?\b", re.I),
    re.compile(r"\b(?:sexual orientations?|sexually oriented)\b", re.I),
    re.compile(r"\bhealth(?:[- ]related)? conditions?\b", re.I),
    re.compile(r"\b(?:political|politically held) beliefs?\b", re.I),
)

# Exact idioms are safe only when the protected match falls inside the matched
# phrase. Sentence-level inclusive constructions are handled separately below.
SAFE_PROTECTED_CONTEXTS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\brace conditions?\b", re.I),
    re.compile(r"\braces? to market\b", re.I),
    re.compile(r"\barms races?\b", re.I),
    re.compile(r"\braces? ahead\b", re.I),
    re.compile(r"\bgender balances?\b", re.I),
    re.compile(r"\bgender[- ]balanced\b", re.I),
    re.compile(r"\bgender[- ]diverse\b", re.I),
)

INCLUSIVE_PROTECTED_CONSTRUCTIONS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:(?:do|does|did|will) not|don['’]t|doesn['’]t|didn['’]t)"
        r"\s+discriminate\b",
        re.I,
    ),
    re.compile(r"\bregardless of\b", re.I),
    re.compile(r"\bwelcom(?:e|es|ed|ing) applicants? of any\b", re.I),
    re.compile(r"\birrespective of\b", re.I),
    re.compile(r"\bequal opportunit(?:y|ies)\b", re.I),
    re.compile(
        r"\baccommodations? (?:are|is|will be|remain|remains) available\b",
        re.I,
    ),
    re.compile(r"\breasonable accommodations?\b", re.I),
    re.compile(r"\bsupport(?:s|ed|ing)? religious observances?\b", re.I),
    re.compile(r"\bdiversity reporting\b", re.I),
)

# Each pattern operates on one sentence with the current protected match
# replaced by the sentinel. Keeping the construction and the term coupled
# avoids treating every bare mention as a restriction.
_PROTECTED_TERM_SENTINEL = "__PROTECTED_TERM__"
PROTECTED_RESTRICTION_CONSTRUCTIONS: tuple[re.Pattern[str], ...] = (
    re.compile(
        rf"\b(?:must|has to|have to|needs? to|required to)\s+"
        rf"(?:be\s+(?:(?:under|over)\s+(?:the\s+)?)?|have\s+no\s+|"
        rf"share\s+(?:(?:our|the|a|an|specified)\s+)?|"
        rf"hold\s+(?:(?:a|an|the|specified)\s+)?)"
        rf"{_PROTECTED_TERM_SENTINEL}\b",
        re.I,
    ),
    re.compile(
        rf"\bno\s+(?:candidates?|applicants?)\b[^.!?;\r\n]{{0,40}}"
        rf"\b(?:has|have)\s+(?:(?:a|an|the)\s+)?"
        rf"{_PROTECTED_TERM_SENTINEL}\b",
        re.I,
    ),
    re.compile(
        rf"\bonly\b[^.!?;\r\n]{{0,100}}{_PROTECTED_TERM_SENTINEL}\b",
        re.I,
    ),
    re.compile(
        rf"\b(?:exclud(?:e|es|ed|ing)|"
        rf"screen(?:s|ed|ing)? out|"
        rf"reject(?:s|ed|ing)?|"
        rf"eliminat(?:e|es|ed|ing)|"
        rf"disqualif(?:y|ies|ied|ying))\b"
        rf"[^.!?;\r\n]{{0,100}}{_PROTECTED_TERM_SENTINEL}\b",
        re.I,
    ),
    re.compile(
        rf"{_PROTECTED_TERM_SENTINEL}\b[^.!?;\r\n]{{0,80}}"
        rf"\b(?:should not apply|need not apply|"
        rf"(?:is|are)\s+(?:unsuitable|ineligible|disqualifying|excluded))\b",
        re.I,
    ),
    re.compile(
        rf"\b(?:what (?:is|are)|stat(?:e|es|ed|ing)|"
        rf"provid(?:e|es|ed|ing)|tell(?:s|ing)? us)\s+"
        rf"(?:(?:your|the)\s+)?"
        rf"{_PROTECTED_TERM_SENTINEL}\b",
        re.I,
    ),
    re.compile(
        rf"\bhow\s+(?:does|do|did)\s+(?:(?:your|a|an|the)\s+)?"
        rf"{_PROTECTED_TERM_SENTINEL}\b[^.!?;\r\n]{{0,50}}"
        rf"\baffect(?:s|ed|ing)?\b[^.!?;\r\n]{{0,30}}"
        rf"\b(?:work|roles?|jobs?|applications?)\b",
        re.I,
    ),
)


class ProtectedContentTier(str, Enum):
    """Disposition for one protected-term match."""

    RESTRICTION = "restriction"
    RECOGNISED_INCLUSIVE = "recognised_inclusive"
    HUMAN_REVIEW = "human_review"


@dataclass(frozen=True)
class ProtectedContentMatch:
    """One protected-term match and its tiered disposition."""

    tier: ProtectedContentTier
    searched: str
    match: re.Match[str]

    @property
    def term(self) -> str:
        return self.match.group(0)

    @property
    def pattern(self) -> str:
        return self.match.re.pattern


# --- Autonomous hiring decisions -----------------------------------------
#
# Previously enforced against candidate-assessment prose only, so
# "automatically reject every candidate scoring below 4 without human
# review" was accepted at generation time. The invariant ("no autonomous
# hiring decisions") is not specific to the evaluation path.
_AUTONOMOUS_ACTION = (
    r"(?:hir(?:e|es|ed|ing)|reject(?:s|ed|ing)?|"
    r"eliminat(?:e|es|ed|ing))"
)
_CANDIDATE_SUBJECT = r"(?:candidates?|applicants?)"
_HUMAN_HIRING_ACTOR = (
    r"(?:hiring managers?|human reviewers?|TA partners?|recruiters?|people teams?)"
)

AUTONOMOUS_DECISION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        rf"\b{_AUTONOMOUS_ACTION}\s+"
        rf"(?:(?:the|an?|any)\s+)?{_CANDIDATE_SUBJECT}\b",
        re.I,
    ),
    re.compile(
        r"\bautomatically\s+(?:progress(?:es|ed|ing)?|"
        r"advanc(?:e|es|ed|ing)|reject(?:s|ed|ing)?|"
        r"eliminat(?:e|es|ed|ing))\b",
        re.I,
    ),
    re.compile(r"\bfinal\s+hiring\s+decisions?\b", re.I),
)

# These spans express the inverse of autonomous decision-making. They are
# removed before the blocking patterns run, while an unqualified decision
# statement still matches.
SAFE_AUTONOMOUS_CONTEXTS: tuple[re.Pattern[str], ...] = (
    re.compile(
        rf"\b(?:the\s+)?{_HUMAN_HIRING_ACTOR}\s+"
        r"(?:makes?|owns?|retains?|approves?)\s+(?:the\s+)?"
        r"final\s+hiring\s+decisions?\b",
        re.I,
    ),
    re.compile(
        r"\b(?:the\s+)?final\s+hiring\s+decisions?\b[^.!?;\r\n]{0,60}"
        r"\b(?:rests?|remains?|belongs?)\s+with\s+(?:the\s+)?"
        rf"{_HUMAN_HIRING_ACTOR}\b",
        re.I,
    ),
    re.compile(
        rf"\b(?:the\s+)?{_HUMAN_HIRING_ACTOR}\s+"
        rf"{_AUTONOMOUS_ACTION}\s+"
        rf"(?:(?:the|an?|any)\s+)?{_CANDIDATE_SUBJECT}\b",
        re.I,
    ),
    re.compile(
        r"\b(?:this|the)\s+(?:tools?|systems?|models?|AI)\s+"
        r"(?:does|do|will|can|should|must)\s+not\s+make\s+(?:the\s+)?"
        r"final\s+hiring\s+decisions?\b",
        re.I,
    ),
    re.compile(
        r"\b(?:does|do|will|can)\s+not\s+make\s+(?:the\s+)?"
        r"final\s+hiring\s+decisions?\b",
        re.I,
    ),
    re.compile(
        rf"\b(?:this|the)\s+(?:tools?|systems?|models?|AI)\s+"
        rf"(?:(?:does|do|will|can|should|must)\s+not|never)\s+"
        rf"(?:automatically\s+)?{_AUTONOMOUS_ACTION}\s+"
        rf"(?:(?:the|an?|any)\s+)?{_CANDIDATE_SUBJECT}\b",
        re.I,
    ),
    re.compile(
        r"\bnever\s+makes?\s+(?:the\s+)?final\s+hiring\s+decisions?\b",
        re.I,
    ),
    re.compile(
        r"\b(?:the\s+)?final\s+hiring\s+decisions?\b[^.!?;\r\n]{0,30}"
        r"\bnot\s+(?:this|the)\s+(?:tools?|systems?|models?|AI)\b",
        re.I,
    ),
)


# --- Unsupported compensation claims -------------------------------------
#
# Scoped to job-description prose, where an unsupported benefit claim is a
# commitment to a candidate. The narrowed equity pattern and its safe
# contexts are carried over intact: bare "equity" is deliberately *not* a
# rejection, because it is ambiguous, and ambiguity routes to human review
# guidance instead (see ``_with_equity_review_guidance``).
UNSUPPORTED_COMPENSATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsalar(?:y|ies|ied)\b", re.I),
    re.compile(r"\bremunerat(?:ion|ions|ed|ing)\b", re.I),
    re.compile(r"\bbonus(?:es|[- ]based)?\b", re.I),
    re.compile(r"\bhealth insurance\b", re.I),
    re.compile(r"\bcompetitive benefits?\b", re.I),
    re.compile(r"\bwell[- ]being benefits?\b", re.I),
    re.compile(r"\bpaid leaves?\b", re.I),
)
SAFE_EQUITY_CONTEXTS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bbrand equity\b", re.I),
    re.compile(r"\bequity research\b", re.I),
)
EQUITY_COMPENSATION_PATTERN = re.compile(
    r"\b(?:employee|stock)\s+equity\b|"
    r"\bequity\s+(?:awards?|compensations?|grants?|packages?|shares?|"
    r"options?|vesting)\b",
    re.I,
)


# --- Instruction-like / injection content ---------------------------------
#
# The role-marker prefix requires a line or sentence boundary so ordinary
# prose ("our CRM system: Salesforce") is not flagged, while a real injected
# turn marker still is. Shared so both paths agree on that boundary.
ROLE_MARKER_PREFIX = (
    r"(?:^|[\r\n]|[.!?]\s+)"
    r"(?:system|developer)\s*(?:messages?)?\s*:"
)
INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "instruction_override",
        re.compile(
            r"\bignor(?:e|es|ed|ing)\s+(?:all\s+|any\s+)?"
            r"(?:previous|prior|above)\s+instructions?\b",
            re.I,
        ),
    ),
    (
        "fake_system_message",
        re.compile(
            ROLE_MARKER_PREFIX
            + r"\s*(?:ignore|disregard|override|follow|use|reveal|show|"
            r"print|repeat|act|pretend|give|award|assign|mark|suppress|"
            r"do\s+not|you\s+are)\b",
            re.I,
        ),
    ),
    ("possible_role_marker", re.compile(ROLE_MARKER_PREFIX, re.I)),
    (
        "score_manipulation",
        re.compile(
            r"\b(?:giv(?:e|es|ing)|gave|award(?:s|ed|ing)?|"
            r"assign(?:s|ed|ing)?|mark(?:s|ed|ing)?)\b.{0,40}\b"
            r"(?:maximum|max|perfect|5\s*/\s*5|score\s+(?:of\s+)?5)\b",
            re.I | re.S,
        ),
    ),
    (
        "missing_evidence_override",
        re.compile(
            r"\bignor(?:e|es|ed|ing)\b.{0,40}\bmissing\s+evidence\b",
            re.I | re.S,
        ),
    ),
    (
        "prompt_disclosure",
        re.compile(
            r"\b(?:reveal(?:s|ed|ing)?|show(?:s|ed|ing|n)?|"
            r"print(?:s|ed|ing)?|repeat(?:s|ed|ing)?|"
            r"disclos(?:e|es|ed|ing))\b.{0,50}\b"
            r"(?:hidden|system|developer|internal)\s+prompts?\b",
            re.I | re.S,
        ),
    ),
    (
        "role_playing",
        re.compile(
            r"\b(?:role[- ]?play(?:s|ed|ing)?|pretend(?:s|ed|ing)?|"
            r"act(?:s|ed|ing)?)\s+as\b",
            re.I,
        ),
    ),
    (
        "evaluation_suppression",
        re.compile(
            r"\b(?:(?:do|does) not|don['’]t|doesn['’]t|never)\s+evaluate\b",
            re.I,
        ),
    ),
    (
        "fake_output",
        re.compile(
            r'["\']?(?:scores?|requirement_ids?)["\']?\s*:\s*',
            re.I,
        ),
    ),
    (
        "boundary_manipulation",
        re.compile(r"(?:BEGIN|END)\s+UNTRUSTED\s+CANDIDATES?", re.I),
    ),
)


# --- Hiring-pack free-text registry ---------------------------------------
#
# Path grammar: ``.`` walks a mapping key, a ``[]`` suffix iterates a list.
# Every author-supplied narrative field in ``HiringPack`` belongs here.
# ``human_review_guidance`` is the field this registry exists for: it sat in
# neither of the two former text blobs and so was scanned by nothing, while
# ``_with_equity_review_guidance`` actively writes into it.
HIRING_PACK_TEXT_FIELDS: tuple[str, ...] = (
    "job_description.title",
    "job_description.location",
    "job_description.purpose",
    "job_description.business_impact",
    "job_description.responsibilities[]",
    "job_description.outcomes[]",
    "job_description.must_have_criteria[].text",
    "job_description.preferred_criteria[].text",
    "job_description.company_behaviours[].value",
    "job_description.company_behaviours[].role_behaviour",
    "job_description.logistics[]",
    "job_description.assessment_expectations[]",
    "screening_questions[].question",
    "screening_questions[].purpose",
    "screening_questions[].expected_evidence[]",
    "screening_questions[].rubric[].description",
    "screening_questions[].green_flags[]",
    "screening_questions[].red_flags[]",
    "screening_questions[].follow_up",
    "human_review_guidance[]",
)

# The compensation rule is a promise-to-candidate rule, so it stays scoped
# to job-description prose rather than running over every field.
JOB_DESCRIPTION_TEXT_FIELDS: tuple[str, ...] = tuple(
    path for path in HIRING_PACK_TEXT_FIELDS if path.startswith("job_description.")
)

# String-typed fields that carry identifiers, provenance, or actor names
# rather than author-written narrative. Listed explicitly with a reason so
# the completeness test still accounts for them instead of ignoring them.
HIRING_PACK_EXEMPT_FIELDS: dict[str, str] = {
    "schema_version": "Fixed literal, not author-supplied narrative.",
    "hiring_pack_id": "Generated identifier.",
    "source_session_id": "Generated identifier.",
    "last_edited_by": "Actor name captured for audit, not generated prose.",
    "job_description.must_have_criteria[].requirement_id": "Cross-reference identifier.",
    "job_description.preferred_criteria[].requirement_id": "Cross-reference identifier.",
    "screening_questions[].question_id": "Generated identifier.",
    "screening_questions[].requirement_ids[]": "Cross-reference identifiers.",
    "provenance": "Model, provider, prompt version, and reference-file metadata.",
}


@dataclass(frozen=True)
class GuardrailFinding:
    """One guardrail hit, named precisely enough for a TA reviewer to act."""

    field: str
    check: str
    pattern: str
    excerpt: str
    matched_text: str
    confirmation: str

    def message(self) -> str:
        return (
            f"{self.field} requires review for a possible {self.check}: matched "
            f"{self.matched_text!r} using {self.pattern!r} in "
            f'"{self.excerpt}". {self.confirmation}'
        )


def _first_match(
    value: str, patterns: Iterable[re.Pattern[str]]
) -> re.Match[str] | None:
    return next((found for found in (p.search(value) for p in patterns) if found), None)


def _sentence_bounds(value: str, start: int, end: int) -> tuple[int, int]:
    boundaries = ".!?;\r\n"
    left = max((value.rfind(marker, 0, start) for marker in boundaries), default=-1)
    right_candidates = [
        position
        for marker in boundaries
        if (position := value.find(marker, end)) != -1
    ]
    right = min(right_candidates) if right_candidates else len(value)
    return left + 1, right


def _match_is_inside(
    match: re.Match[str], contexts: Iterable[re.Pattern[str]], value: str
) -> bool:
    return any(
        context_match.start() <= match.start()
        and match.end() <= context_match.end()
        for pattern in contexts
        for context_match in pattern.finditer(value)
    )


def classify_protected_content(value: str) -> tuple[ProtectedContentMatch, ...]:
    """Classify each protected-term match as restriction, inclusive, or review."""
    term_matches = sorted(
        (
            match
            for pattern in PROTECTED_PATTERNS
            for match in pattern.finditer(value)
        ),
        key=lambda match: (match.start(), match.end()),
    )
    classified: list[ProtectedContentMatch] = []
    for match in term_matches:
        sentence_start, sentence_end = _sentence_bounds(
            value, match.start(), match.end()
        )
        sentence = value[sentence_start:sentence_end]
        relative_start = match.start() - sentence_start
        relative_end = match.end() - sentence_start

        if _match_is_inside(match, SAFE_PROTECTED_CONTEXTS, value) or any(
            pattern.search(sentence)
            for pattern in INCLUSIVE_PROTECTED_CONSTRUCTIONS
        ):
            tier = ProtectedContentTier.RECOGNISED_INCLUSIVE
        else:
            restriction_text = (
                sentence[:relative_start]
                + _PROTECTED_TERM_SENTINEL
                + sentence[relative_end:]
            )
            tier = (
                ProtectedContentTier.RESTRICTION
                if any(
                    pattern.search(restriction_text)
                    for pattern in PROTECTED_RESTRICTION_CONSTRUCTIONS
                )
                else ProtectedContentTier.HUMAN_REVIEW
            )
        classified.append(ProtectedContentMatch(tier, value, match))
    return tuple(classified)


def equity_policy_text(value: str) -> str:
    """Blank out safe equity phrases before applying compensation patterns."""
    for pattern in SAFE_EQUITY_CONTEXTS:
        value = pattern.sub(" ", value)
    return value


# Each matcher returns the text it actually searched plus the match, so a
# finding can quote the offending excerpt from the same string that matched.
def _protected_match(value: str) -> _Hit:
    finding = next(
        (
            finding
            for finding in classify_protected_content(value)
            if finding.tier is ProtectedContentTier.RESTRICTION
        ),
        None,
    )
    return None if finding is None else (finding.searched, finding.match)


def _autonomous_match(value: str) -> _Hit:
    policy_text = value
    for pattern in SAFE_AUTONOMOUS_CONTEXTS:
        policy_text = pattern.sub(" ", policy_text)
    return _hit(
        policy_text,
        _first_match(policy_text, AUTONOMOUS_DECISION_PATTERNS),
    )


def _instruction_match(value: str) -> _Hit:
    return _hit(
        value,
        _first_match(
            value,
            tuple(
                pattern
                for code, pattern in INJECTION_PATTERNS
                if code != "possible_role_marker"
            ),
        ),
    )


def _compensation_match(value: str) -> _Hit:
    policy_text = equity_policy_text(value)
    return _hit(
        policy_text,
        _first_match(policy_text, UNSUPPORTED_COMPENSATION_PATTERNS)
        or EQUITY_COMPENSATION_PATTERN.search(policy_text),
    )


def contains_protected_content(value: str) -> bool:
    """Return whether text contains a protected-term restriction."""
    return _protected_match(value) is not None


def first_protected_restriction(value: str) -> ProtectedContentMatch | None:
    """Return the first blocking protected-term match, if one exists."""
    return next(
        (
            finding
            for finding in classify_protected_content(value)
            if finding.tier is ProtectedContentTier.RESTRICTION
        ),
        None,
    )


def first_unresolved_protected_mention(
    value: str,
) -> ProtectedContentMatch | None:
    """Return a non-inclusive mention that must not become scored evidence."""
    return next(
        (
            finding
            for finding in classify_protected_content(value)
            if finding.tier is not ProtectedContentTier.RECOGNISED_INCLUSIVE
        ),
        None,
    )


def contains_autonomous_decision(value: str) -> bool:
    return _autonomous_match(value) is not None


def contains_instruction_like_content(value: str) -> bool:
    """Detect injected instructions, ignoring the bare role-marker signal."""
    return _instruction_match(value) is not None


def contains_unsupported_compensation(value: str) -> bool:
    return _compensation_match(value) is not None


def _excerpt(value: str, match: re.Match[str], window: int = 40) -> str:
    start = max(0, match.start() - window)
    end = min(len(value), match.end() + window)
    text = " ".join(value[start:end].split())
    return f"{'...' if start else ''}{text}{'...' if end < len(value) else ''}"


def iter_text_fields(payload: object, paths: Iterable[str]) -> list[tuple[str, str]]:
    """Resolve registry paths against a dumped pack into (path, text) pairs."""
    resolved: list[tuple[str, str]] = []
    for path in paths:
        nodes: list[tuple[str, object]] = [("", payload)]
        for segment in path.split("."):
            iterate = segment.endswith("[]")
            key = segment[:-2] if iterate else segment
            following: list[tuple[str, object]] = []
            for prefix, node in nodes:
                if not isinstance(node, dict) or key not in node:
                    continue
                child = node[key]
                here = f"{prefix}.{key}" if prefix else key
                if iterate:
                    following.extend(
                        (f"{here}[{index}]", item)
                        for index, item in enumerate(child or [])
                    )
                else:
                    following.append((here, child))
            nodes = following
        resolved.extend(
            (field, value)
            for field, value in nodes
            if isinstance(value, str) and value.strip()
        )
    return resolved


# (label, matcher, fields, requested confirmation). Protected-characteristic,
# autonomy, and instruction-like checks run over every registered field; the
# compensation rule is a promise-to-candidate rule and stays on the JD.
_PACK_CHECKS: tuple[
    tuple[str, Callable[[str], _Hit], tuple[str, ...], str], ...
] = (
    (
        "protected-characteristic restriction",
        _protected_match,
        HIRING_PACK_TEXT_FIELDS,
        "Confirm the wording does not limit applicant eligibility based on "
        "this characteristic.",
    ),
    (
        "autonomous hiring decision",
        _autonomous_match,
        HIRING_PACK_TEXT_FIELDS,
        "Confirm that progression and hiring authority remains with a named "
        "human reviewer.",
    ),
    (
        "instruction-like content",
        _instruction_match,
        HIRING_PACK_TEXT_FIELDS,
        "Confirm that this field contains role content, not instructions to "
        "the tool.",
    ),
    (
        "unsupported compensation or benefit claim",
        _compensation_match,
        JOB_DESCRIPTION_TEXT_FIELDS,
        "Confirm that this candidate-facing claim is grounded in approved "
        "compensation or benefit information.",
    ),
)


def scan_pack_payload(payload: object) -> list[GuardrailFinding]:
    """Scan every registered free-text field and name each specific failure."""
    findings: list[GuardrailFinding] = []
    for label, matcher, paths, confirmation in _PACK_CHECKS:
        for field, value in iter_text_fields(payload, paths):
            hit = matcher(value)
            if hit is None:
                continue
            searched, match = hit
            findings.append(
                GuardrailFinding(
                    field,
                    label,
                    match.re.pattern,
                    _excerpt(searched, match),
                    match.group(0),
                    confirmation,
                )
            )
    return findings


@dataclass(frozen=True)
class PackProtectedContentMatch:
    """A tiered protected-term match tied to a concrete pack field."""

    field: str
    tier: ProtectedContentTier
    term: str
    pattern: str
    excerpt: str


def scan_pack_protected_content(
    payload: object,
) -> list[PackProtectedContentMatch]:
    """Return every protected-term match, including accepted Tier 2 matches."""
    findings: list[PackProtectedContentMatch] = []
    for field, value in iter_text_fields(payload, HIRING_PACK_TEXT_FIELDS):
        findings.extend(
            PackProtectedContentMatch(
                field=field,
                tier=match.tier,
                term=match.term,
                pattern=match.pattern,
                excerpt=_excerpt(match.searched, match.match),
            )
            for match in classify_protected_content(value)
        )
    return findings


_PROTECTED_REVIEW_GUIDANCE_PREFIX = "Human review required: field "


def protected_review_guidance(payload: object) -> list[str]:
    """Build deterministic guidance for ambiguous Tier 3 pack mentions."""
    guidance: list[str] = []
    seen: set[tuple[str, str]] = set()
    for field, value in iter_text_fields(payload, HIRING_PACK_TEXT_FIELDS):
        # Do not recursively create guidance about guidance generated here.
        if (
            field.startswith("human_review_guidance[")
            and value.startswith(_PROTECTED_REVIEW_GUIDANCE_PREFIX)
        ):
            continue
        for match in classify_protected_content(value):
            if match.tier is not ProtectedContentTier.HUMAN_REVIEW:
                continue
            key = (field, match.term.casefold())
            if key in seen:
                continue
            seen.add(key)
            guidance.append(
                f'{_PROTECTED_REVIEW_GUIDANCE_PREFIX}{field} mentions the term '
                f'"{match.term}". Confirm the wording is relevant and does not '
                "limit applicant eligibility based on that characteristic."
            )
    return guidance
