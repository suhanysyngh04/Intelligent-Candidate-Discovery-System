# Sandbox demo

A small Streamlit app satisfying `submission_spec.docx` Section 10.5 ("a hosted
environment where organizers can verify your ranking system runs reproducibly on a
small sample"). It wraps the exact same `src/` code used for the full 100K-candidate
submission — nothing in `src/` is duplicated or reimplemented here.

## Run locally

```bash
pip install -r sandbox/requirements.txt
streamlit run sandbox/app.py
```

Opens at `http://localhost:8501`. Uses the bundled `data/sample_candidates.json`
(50 candidates) by default, or accepts an uploaded `.jsonl`/`.json` sample of up to
100 candidates.

## Deploying to Streamlit Cloud (for the portal's `sandbox_link` field)

1. Push this repository to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), create a new app pointing at
   this repo, branch `main`, entry point `sandbox/app.py`.
3. Set the app's requirements file to `sandbox/requirements.txt` (or merge it with
   the root `requirements.txt` — both are equivalent here).
4. No secrets, API keys, or network access are required at runtime.

Confirmed locally: the app boots and serves a 200 response within ~8 seconds, and
ranks the bundled 50-candidate sample in well under a second once loaded.
