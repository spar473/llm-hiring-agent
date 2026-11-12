# Fixtures and example scenarios

All public examples are fictional/synthetic demonstration data or sanitized historical model transcripts. They are not real applicant records. Generic team names, company behaviours, and original fictional reference documents replace the original development context.

## Start here

| Example | Location | What it demonstrates |
| --- | --- | --- |
| Complete offline session | `data/fixtures/demo_fallback/session_demo_fallback/` | Approved Marketing Intern role, hiring pack, candidate evidence, artifact versions, and audit events |
| Conflicting stakeholders | `data/fixtures/conflicting_manager_inputs_role.json` | An unresolved conflict that needs human review |
| Ten candidate responses | `data/fixtures/batch_candidate_responses_10.json` | Batch input shape and different evidence patterns |
| Ten prepared review results | `data/fixtures/batch_evaluation_results_10.csv` | Sortable UI review queue, loaded automatically without inference |
| Evidence scenarios | `data/fixtures/candidate_evidence_scenarios.json` | Stronger and weaker evidence patterns for controlled tests |
| Instruction-like answers | `data/fixtures/candidate_prompt_injection.json` | Candidate text that must remain untrusted input |
| Fictional writing references | `references/` | Company values and two example job descriptions used for style/context |

The offline builder uses reviewed extraction and evaluation responses through injected clients, then invokes production validators and persistence. It is a deterministic fixture replay, not an offline language model. Rebuilding the demo does not prove that a live model will produce the same result.

The prepared batch results are separate sample output. They are not regenerated when the offline role is loaded or edited.

## Discovery regression scenarios

The persona scripts and recorded JSON cover several deliberately different manager situations:

- **Marketing intern:** distinguish essential weekly content work from preferred editing familiarity, and account for bounded employment dates.
- **Vague executive:** turn broad aspirations into questions and reviewable requirements.
- **Culture-focused manager:** turn vague culture language into observable work behaviours, with human completion of business sections.
- **Over-technical entry role:** challenge an excessive day-one list and make priority/learnability explicit.
- **Technical role:** ask about concrete integration and system responsibilities.
- **Creative role:** ask about deliverables, channels, feedback, and review processes.

The scripts' `brief_source` fields describe scenario provenance. References to this guide do not imply a model was freshly run for the public release. Some tests intentionally document a known gap in a stored transcript, rather than asserting the model solved that situation.

## Historical experiments

Consistency results, the Spanish JD proof, and batch metrics are retained as inspectable examples. Their scope is limited to the scenarios and settings recorded in the files. They are not current model recommendations, independent benchmarks, hiring accuracy estimates, or production performance measurements.

The outcome-calibration script uses explicitly hypothetical labels. Its output is a format/method experiment, not empirical calibration from real employees.

## Changing the examples

If you change a role or pack, update the linked IDs and versions together. Reference provenance should describe the actual local bytes: filename, category, size, SHA-256, and extraction method. The offline builder refreshes reference provenance from `references/`.

After changes, run:

```bash
python scripts/prepare_fallback_session.py --offline --output-root data/outputs/demo-check
python -m pytest -q
```

The fallback tests check source quotes, approval, readiness, ID uniqueness, pack validation, evaluation validation, and version linkage. Keep runtime sessions and generated results out of Git; their paths are ignored.
