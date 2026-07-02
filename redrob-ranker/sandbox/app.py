"""
sandbox/app.py — a small Streamlit app for the hackathon's required sandbox /
demo link (submission_spec.docx Section 10.5).

This is NOT the production ranking path — it's a thin, hosted-environment wrapper
around the exact same src/ code, built to accept a small candidate sample (<=100
candidates) and demonstrate the ranker runs end-to-end within the compute budget.

To deploy: push this repo to GitHub, then create a Streamlit Cloud app pointing at
sandbox/app.py (entry point), with requirements.txt at the repo root already covering
the dependencies. No secrets or network access are needed at runtime.

Run locally with:
    pip install streamlit
    streamlit run sandbox/app.py
"""
import json
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jd_profile import JD_TEXT
from semantic import SemanticSimilarityScorer
from scoring import score_candidate
from reasoning import build_reasoning
import random

st.set_page_config(page_title="Redrob Ranker — Sandbox", layout="wide")

st.title("Redrob Hackathon Ranker — Sandbox")
st.caption(
    "Small-sample reproduction demo. Upload a candidates.jsonl sample (≤100 rows), "
    "or use the bundled 50-candidate sample, and the exact same src/ ranking code "
    "used for the full 100K submission runs against it below."
)

DEFAULT_SAMPLE = Path(__file__).parent.parent / "data" / "sample_candidates.json"

uploaded = st.file_uploader("Upload a small candidates.jsonl or sample_candidates.json sample", type=["jsonl", "json"])


def load_candidates_from_upload(file) -> list[dict]:
    raw = file.read().decode("utf-8")
    text = raw.strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


if uploaded is not None:
    candidates = load_candidates_from_upload(uploaded)
elif DEFAULT_SAMPLE.exists():
    candidates = json.loads(DEFAULT_SAMPLE.read_text())
    st.info(f"No file uploaded — using the bundled {len(candidates)}-candidate sample.")
else:
    candidates = []
    st.warning("No sample file available. Upload a candidates.jsonl or sample_candidates.json file to proceed.")

if candidates:
    n = min(len(candidates), 100)
    candidates = candidates[:n]
    st.write(f"Running ranker on **{len(candidates)}** candidates...")

    t0 = time.time()
    sim_scorer = SemanticSimilarityScorer(JD_TEXT).fit(candidates)
    semantic_scores = sim_scorer.scores()

    results = []
    for c in candidates:
        sem = semantic_scores.get(c["candidate_id"], 0.0)
        results.append((c, score_candidate(c, sem)))

    results.sort(key=lambda pair: (-pair[1]["score"], pair[1]["candidate_id"]))
    elapsed = time.time() - t0

    st.success(f"Ranked {len(candidates)} candidates in {elapsed:.2f}s (CPU only, no network).")

    rng = random.Random(42)
    rows = []
    for rank, (candidate, result) in enumerate(results, start=1):
        reasoning = build_reasoning(candidate, result, rng)
        rows.append({
            "rank": rank,
            "candidate_id": result["candidate_id"],
            "score": round(result["score"], 4),
            "title": candidate["profile"]["current_title"],
            "company": candidate["profile"]["current_company"],
            "location": f"{candidate['profile']['location']}, {candidate['profile']['country']}",
            "reasoning": reasoning,
        })

    st.dataframe(rows, use_container_width=True, hide_index=True)

    csv_lines = ["candidate_id,rank,score,reasoning"]
    for r in rows:
        reasoning_escaped = r["reasoning"].replace('"', '""')
        csv_lines.append(f'{r["candidate_id"]},{r["rank"]},{r["score"]:.4f},"{reasoning_escaped}"')
    csv_text = "\n".join(csv_lines)

    st.download_button("Download ranked CSV", csv_text, file_name="sandbox_submission.csv", mime="text/csv")
