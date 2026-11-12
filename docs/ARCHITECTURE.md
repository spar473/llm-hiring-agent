# Architecture and design

The application is organised around typed role, hiring-pack, and candidate-evaluation records. Streamlit calls Python services; those services can call an injected `LLMClient` interface. Provider transport is separate from state transitions, validation, and persistence.

## Responsibilities and trust boundaries

| Component | Owns | Does not own |
| --- | --- | --- |
| Streamlit (`app.py`) | Forms, tabs, session selection, orchestration, error display | Scoring policy |
| Domain models (`src/models.py`) | Field types, enums, shape constraints, cross-field contracts | Provider transport |
| Discovery (`src/discovery.py`) | Prompt assembly, source grounding, extraction application, manual section edits | Final role approval |
| Workflow and readiness | Fixed stage order, completeness, quality warnings, approval conditions | Language interpretation |
| Generation | Local reference extraction, draft generation, requirement links, edits | Publishing job advertisements |
| Evaluation | Evidence materialisation, validation, arithmetic, review routing, human edits | External fact verification or hiring decisions |
| Batch | CSV/JSON input, bounded workers, retries, result files | Durable distributed execution |
| Storage | Local sessions, versioned artifacts, audit events, reload checks | Authentication or tamper-proof storage |

Manager notes, candidate answers, and reference documents are all untrusted input. Prompts tell the model to treat them as data; Python additionally checks output shape, IDs, quotes, content patterns, and consistency. These checks enforce application contracts, not the truth of every interpretation.

## Role discovery and approval

The role workflow has twelve ordered stages: basic information, business need, outcomes, responsibilities, day-one requirements, learnable requirements, behaviours, constraints, assessment plan, quality review, manager approval, and complete.

The model proposes `stay` or `advance`. Python maps that suggestion to the next stage and checks whether the current stage is complete. The model cannot directly move the application past an incomplete stage.

Discovery extracts requirements, assumptions, ambiguities, and possible contradictions. Requirement source statements must be grounded in manager text. Ambiguities have stable IDs so later turns can explicitly resolve them. Business need, outcomes, responsibilities, constraints, the assessment plan, and company behaviours are completed through human forms; they are not all inferred from the chat.

Explicit approval requires a named approver, no blocking gaps, confirmation of five sections (business purpose, outcomes, must-haves, behavioural criteria, key constraints), and acknowledgement of quality warnings. High or critical unresolved contradictions block approval. A readiness score by itself never approves a role.

## Readiness is completeness, not hiring quality

`src/readiness.py` computes a 0–100 weighted completeness score. The weights are project design choices, not validated predictors.

| Dimension | Ongoing role | Bounded-duration role |
| --- | ---: | ---: |
| Business purpose | 10 | 10 |
| Measurable outcomes | 20 | 15 |
| Prioritised responsibilities | 10 | 10 |
| Must-have versus preferred | 15 | 15 |
| Proficiency and equivalents | 10 | 10 |
| Evidence and assessment | 15 | 15 |
| Observable behaviours | 10 | 10 |
| Logistics and constraints | 5 | 10 |
| Contradictions resolved | 5 | 5 |

Bounded-duration roles receive more weight on timing and constraints. Independent approval blockers still apply. The role-quality engine also flags vague language, excessive day-one requirements, and contradictions.

## Hiring-pack generation

Generation requires an explicitly approved role satisfying the approval contract. The generator supplies that role, local reference text, and the output schema to the model. The result must include a structured job description and five to seven screening questions. Every question maps to approved requirement IDs and has six ordered rubric anchors, from 0 to 5. Must-have requirements must be covered.

`load_reference_files()` accepts local Markdown, text, and text-extractable PDFs, up to 5 MiB per file. It ignores unrelated names and sorts accepted files deterministically. A filename containing `company` and `values` supplies the `company_values` category; one containing `example` and the word `JD` supplies `example_jd`. Both categories are mandatory. The bundled references are original fictional examples. PDF support here loads reference text; it is not CV parsing or OCR.

Provenance records reference filenames, SHA-256 hashes, byte sizes, extraction methods, role ID/version, model, provider, prompt version, timestamps, and token metadata where available. An edited role can leave an existing pack stale; candidate evaluation must use a matching approved role/pack pair.

Generated materials remain drafts for review. There is no separate job-publication workflow or approval event.

## Candidate evidence and review signals

The main UI evaluates written responses to the hiring pack's questions. Provider output proposes source excerpts, requirement mappings, evidence classifications, ownership, relevance, specificity, action/result/reflection flags, and assessment prose. Python checks the structure and source grounding and derives evidence quality, score, confidence, coverage, and routing.

Scores are in the range 0–5. Negative evidence can force 0; no evidence produces 1; other scores follow the derived evidence-quality category. They are deterministic given the accepted model classifications. The classifications themselves can be wrong or change between model runs.

Confidence uses these weighted inputs before penalties:

| Input | Weight |
| --- | ---: |
| Specificity | 0.25 |
| Relevance | 0.20 |
| Ownership | 0.15 |
| Verifiability | 0.15 |
| Consistency | 0.15 |
| Action/result/reflection completeness | 0.10 |

Penalties apply for inference, hypothetical or stale evidence, contradictions, missing information, and detected instruction-like text. No-evidence confidence starts at 0.20. The final value is bounded to 0–1. This is an uncalibrated review heuristic, not a probability.

Unresolved must-haves take priority in routing, followed by contradictions and explicit human-review flags. Other routing labels reflect evidence strength. They do not trigger autonomous candidate progression or rejection. Human edits create new evaluation versions and recalculate signals under the same rules.

## Transport and failure handling

`src/llm_client.py` uses `httpx` to send strict JSON-schema chat-completion requests to the configured OpenRouter API base. The application boundary returns typed data and safe call metadata. It classifies provider errors and limits retries instead of treating every failure as a valid answer. `src/llm_diagnostics.py` contains separate redacted probe utilities; it uses the same HTTP transport library behind a separate diagnostic boundary.

Tests inject controlled clients rather than contacting the provider. Script-level experiments and batch processing add their own bounded retry behavior. There is no guarantee that a model available through OpenRouter will satisfy these schemas; live verification remains a separate step.

## Persistence and audit limits

`SessionStore` writes local JSON containing the role, discovery history, pack, evaluation, and audit events. Artifact versions and source IDs support reload checks and stale-output detection. Candidate text can remain in these files.

Audit events are append-only through the application API and meaningful edits record content hashes. A person with filesystem access can still change or remove files. This is not a database transaction system, signed ledger, secure multi-user service, or complete regulatory audit solution.

## Why these choices?

- **Streamlit:** exposes a complete workflow quickly while keeping the project Python-based. The large UI module would benefit from splitting into tab components.
- **Pydantic contracts:** make malformed outputs fail visibly and make provider substitution testable. Rigid schemas can reject otherwise useful model responses.
- **Deterministic rules:** make approval and arithmetic inspectable. The policy thresholds remain assumptions that need evaluation.
- **JSON storage:** makes sessions easy to inspect and the demo easy to run. It limits concurrent use and operational reliability.
- **Human role completion:** keeps business decisions explicit. It increases manual input and means chat alone cannot finish the entire role.

See [limitations](LIMITATIONS.md) for the remaining failure modes and [scenarios](SCENARIOS.md) for the evidence covered by tests.
