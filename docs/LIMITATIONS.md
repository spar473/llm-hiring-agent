# Limitations and next steps

This project demonstrates application engineering around an LLM. It does not establish that an LLM can reliably select suitable employees.

## What is verified

The automated suite exercises domain validation, workflow gates, readiness rules, approval, source grounding, hiring-pack mappings, candidate-evidence arithmetic, persistence, redacted diagnostics, batch handling, and the Streamlit interface using controlled provider responses. The prepared offline session passes the production role, hiring-pack, candidate-evaluation, and reload validators.

No live provider request was made to verify this public edition. A working OpenRouter key was unavailable. Historical transcripts were sanitized and retained as regression examples; provider/model identifiers and recorded metrics in those files describe old examples, not supported current configuration or reproduced benchmarks.

## Known boundaries

| Area | Boundary and implication |
| --- | --- |
| Candidate quality | There is no validated relationship between review signals and job performance. Scores must not be interpreted as hiring suitability probabilities. |
| Confidence | The formula describes evidence quality using hand-selected weights and model-derived labels. It has not been calibrated against real hiring outcomes. |
| Truth and verification | A quoted claim is grounded in supplied text, not independently proven. The application does not contact referees, open portfolio links, or verify employment. |
| Fairness | Content rules catch some explicit terms and patterns. They do not prove fairness, remove proxy discrimination, or resolve contextual interpretation. |
| Prompt injection | Delimiters, instruction-pattern checks, and validators cover tested cases. They are not a comprehensive defence against every adversarial input. |
| Role discovery | The model extracts requirements and issues. Several business sections require manual entry. Role-family probing is prompt-led, not a complete engineered decision tree. |
| Dense manager answers | A model may quote the same long sentence for both a must-have and an unresolved ambiguity. The strict overlap guard can reject a useful answer; splitting statements can help. |
| Ambiguity resolution | The model must cite the correct ambiguity IDs. An omitted ID can leave an issue open even when the manager supplied enough detail. |
| Candidate input | Written screening responses are supported. CV parsing, OCR, automated portfolio ingestion, and broad candidate research are not implemented. |
| Publication and outreach | The app drafts materials. It does not publish job advertisements, schedule interviews, or message candidates. |
| Local storage | JSON files may contain complete candidate text. There is no login, tenant isolation, encrypted storage layer, retention policy engine, or production database. |
| Audit trail | Application-managed append-only events and hashes help trace edits. They cannot prevent someone with filesystem access from changing records. |
| Scale | Batch processing has bounded threads and retries, with mocked structural coverage up to 500 candidates. This is not a live load test, durable queue, or throughput benchmark. |
| Model behavior | Provider availability, schema support, latency, cost, and response quality can change. Offline tests cannot establish current live performance. |
| Jurisdiction | Local employment requirements and data-handling duties are not configured or validated for production use. |
| Multilingual support | A separate Spanish JD experiment exists. The main application is not a validated multilingual hiring system. |

## Design lessons

Schema validation and source quotation make outputs easier to inspect, but do not prove that the model chose the right requirement or interpreted a candidate fairly. Deterministic arithmetic makes a result reproducible from accepted labels; it does not make the labels objective.

A role approval gate is useful because it prevents generation from silently relying on missing business decisions. It also adds manual work, and the interface should make that work clear. Rejected model outputs need actionable explanations rather than generic failures.

Synthetic examples are useful for testing known conditions and keeping a public demo free of candidate data. They cannot establish hiring accuracy, real-world fairness, or calibration. Repeated consistency runs measure variation on the chosen examples, not general reliability.

## Next steps, in order

1. Verify a live model against the existing schemas and record a reproducible end-to-end run with request metadata and redacted errors.
2. Define a held-out evaluation set, review guidelines, and independent human labels. Measure extraction errors, source-grounding failures, inconsistent evidence classification, and reviewer disagreement.
3. Evaluate bias and adversarial behavior across realistic role and response variations. Review the confidence weights before using them to prioritise real work.
4. Split the Streamlit module into smaller tab components and add clearer stale-artifact recovery.
5. For multi-user use, design authentication, access controls, durable transactional storage, retention/deletion, secret handling, and reliable job execution before deployment.

These are proposed improvements. They are not features already delivered by this repository.
