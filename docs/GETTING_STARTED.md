# Getting started

Start with the offline demo. You can inspect the whole prepared role-to-evidence workflow without an account, credential, or model request.

## Install

Use Python 3.12 and run commands from the repository root.

macOS/Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/check_setup.py
python -m streamlit run app.py
```

Windows/PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/check_setup.py
python -m streamlit run app.py
```

Open the address printed in the terminal. Keep the terminal running while you use the app. Stop it with Ctrl+C.

## Five-minute offline walkthrough

1. **Define Role → Load Offline Demo.** This loads validated JSON files from `data/fixtures/demo_fallback/`. No API configuration is read by the fixture builder and no inference request is needed to load the files.
2. **Review Role.** Inspect the business need, success outcomes, responsibilities, must-have and preferred requirements, constraints, assessment plan, and observable company behaviours. The example role is already approved. Its readiness score of 100 means the required role information is complete under this application's rules; it says nothing about candidate ability.
3. **Hiring Pack.** Inspect the prepared job description and its requirement links. Read the screening questions, expected evidence, scoring anchors, and follow-ups. Expand the reference comparison to see the fictional writing examples.
4. **Candidate Evidence.** Inspect the stored written-response evaluation. Read the quoted sources and requirement-level assessments before looking at summary signals. “Confidence” describes evidence completeness and consistency under a heuristic; it is not a probability of candidate success.
5. **Batch Review.** The sample result file is prefilled and loaded automatically. The table shows ten prepared examples. Sorting is for human evidence review, not automatic rejection or selection.
6. **Define Role → Load Conflicting-Stakeholders Demo.** Open Review Role to inspect the unresolved contradiction. This is a separate example of an approval blocker, not an additional candidate.

You can inspect and edit local data offline. A meaningful role edit creates a new role version and can invalidate the prepared pack's source version. A fresh pack or candidate assessment needs live LLM access. Reload the offline demo to restore the prepared example in the UI.

## Run a fresh live workflow

1. Copy `.env.example` to `.env` and set your own `OPENROUTER_API_KEY` and `OPENROUTER_MODEL`. Do not paste credentials into role notes, candidate answers, or source files.
2. Restart Streamlit and open System Status. A configured key is not proof that the provider will accept a request.
3. In Define Role, create a new role with the basic information and an initial manager statement. Continue the discovery conversation.
4. In Review Role, complete the sections that discovery does not fill automatically: business need, outcomes, responsibilities, constraints, assessment plan, and company behaviours. Confirm requirement priority, learnability, proficiency, and evidence methods. Resolve blocking contradictions and confirm the required sections.
5. Approve the role. Python checks approval conditions before accepting the action.
6. In Hiring Pack, enter the reviewer identity and request generation. Inspect the result, references, and rubric before using it.
7. In Candidate Evidence, enter written responses against the pack's questions and request evaluation. Review the returned quotes, gaps, contradictions, and follow-ups. A person decides what to do next.

The provider/model must support the application's structured-output requests. The public edition has not been tested against a live provider. Errors may reflect credentials, model capability, service availability, rate limits, or output validation; System Status and redacted diagnostic tools help distinguish them.

## Commands and network use

| Command | Network or inference? | Purpose |
| --- | --- | --- |
| `python scripts/check_setup.py` | No | Check local imports, files, and configuration shape |
| `python -m pytest -q` | No provider requests | Run mocked-provider application tests |
| `python scripts/prepare_fallback_session.py --offline --output-root data/outputs/demo-check` | No | Rebuild, save, reload, and validate the prepared demo |
| `python scripts/openrouter_probe.py preflight` | Authenticated provider HTTP request; no inference | Check configured provider access |
| `python scripts/openrouter_smoke.py` | Live inference; may spend credits | Exercise a discovery request |
| `python scripts/prepare_fallback_session.py` | Live inference; may spend credits | Recreate demo materials through the provider |
| `python scripts/batch_evaluate.py --help` | No | Inspect batch arguments |
| `python scripts/batch_evaluate.py --input data/fixtures/batch_candidate_responses_10.json --output data/outputs/review.csv` | Live inference; may spend credits | Evaluate a prepared candidate batch |

Other research scripts include persona discovery, repeated consistency runs, an isolated Spanish JD proof of concept, and synthetic outcome calibration. Inspect their `--help` and source before running them. These are experiments, not additional production features.

## Batch input

JSON accepts a candidate list, or an object containing `candidates`. Each candidate contains `candidate_id` and a `responses` list with `question_id` and `answer_text`. The committed JSON fixture is the complete example.

CSV accepts `candidate_id,question_id,answer_text`, with one row per answer. A candidate's repeated ID groups their answers together. Question IDs must exist in the selected hiring pack. The CLI defaults to the prepared demo role and pack; use `--role` and `--hiring-pack` to supply another matching approved pair.

Processing is sequential by default. `--max-concurrency` enables bounded threads. Retries and validation errors are recorded in result metrics. This is not a durable job queue or a measured production throughput claim.

## Where information goes

The app saves runtime sessions under ignored `data/sessions/`. Batch outputs can go under ignored `data/outputs/`. Stored JSON can contain full role notes, candidate responses, evidence, and audit metadata. The repository examples are fictional/synthetic; use the local prototype with synthetic data.

For Docker, use the README commands. To retain container sessions, mount a local directory at `/app/data/sessions`. Restarting or removing an unmounted container can discard runtime data.

## Troubleshooting

- **Missing dependencies:** activate the environment and run `python -m pip install -r requirements.txt` again.
- **No API key:** use Load Offline Demo. New generation remains unavailable until configuration is complete.
- **Stale hiring pack:** the role changed after that pack was created. Reload the prepared demo or generate a fresh pack against the new approved version.
- **Rejected discovery output:** split long manager answers into separate concrete statements and inspect the reported quote overlap or grounding issue.
- **Approval blocked:** review the listed gaps and contradictions; a high readiness score alone is not explicit approval.
