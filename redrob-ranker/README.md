# Redrob Hackathon — Intelligent Candidate Discovery & Ranking

A hybrid lexical-semantic + structured-feature ranker that scores all 100,000
candidates in `candidates.jsonl` against the Senior AI Engineer JD and outputs the
top 100 as a ranked CSV with per-candidate reasoning.

**TL;DR on approach:** this is not a single embedding model or a single LLM prompt. It
is eight independently-validated scoring components (title relevance, TF-IDF semantic
similarity, trust-adjusted skill matching, a hand-validated career-narrative pattern
scanner, experience-band fit, product-vs-consulting-company history, location, notice
period) combined into a weighted composite, multiplied by penalty factors for known
trap patterns (keyword-stuffing, CV/speech-without-NLP, title-chasing, no-recent-code,
self-reported-skill-vs-platform-assessment mismatch), with a final multiplicative
discount for behavioral unavailability (inactive, low recruiter response rate,
incomplete interview history). Honeypots are detected and hard-suppressed up front.
See `docs/methodology.md` for the full design rationale and `docs/data_analysis.md`
for the actual EDA queries and findings that justify each design choice.

## Quick start

```bash
pip install -r requirements.txt
python src/rank.py --candidates ./data/candidates.jsonl --out ./artifacts/submission.csv
python qa_check.py --candidates ./data/candidates.jsonl --submission ./artifacts/submission.csv
```

That's it — no pretraining step, no model download, no GPU, no network access needed
at ranking time. `candidates.jsonl` (or `candidates.jsonl.gz`, both are supported) is
the only required input, and it isn't included in this repo due to size (~465MB
uncompressed) — drop the hackathon-provided file into `data/` before running.

Single-command reproduction, exactly as required by `submission_spec.docx` Section
10.3:

```bash
python src/rank.py --candidates ./data/candidates.jsonl --out ./submission.csv
```

## Repository layout

```
.
├── src/
│   ├── jd_profile.py    # structured JD interpretation: titles, skills, location
│   │                    # tiers, experience band, etc., each justified against
│   │                    # specific JD passages (see comments inline)
│   ├── features.py      # per-candidate feature extraction (title/skills/career/
│   │                    # location/availability/honeypot/etc.)
│   ├── semantic.py       # TF-IDF JD<->candidate narrative similarity
│   ├── scoring.py        # combines features into the final composite score
│   ├── reasoning.py      # generates the per-row reasoning string for the CSV
│   └── rank.py           # CLI entrypoint; the single command graders will run
├── tests/
│   └── test_features.py  # unit tests for every scoring component
├── qa_check.py            # content-level sanity checks on a produced submission
│                          # (honeypot rate, consulting-only leakage, hallucination
│                          # spot-check, score monotonicity, reasoning duplication)
├── docs/
│   ├── methodology.md     # full architecture writeup and design rationale
│   └── data_analysis.md   # the actual EDA queries run and what they found
├── artifacts/
│   └── submission.csv     # the top-100 ranked output produced by this code
├── data/                  # (gitignored) drop candidates.jsonl here to reproduce
├── requirements.txt
└── submission_metadata.yaml
```

## Why this architecture (short version)

The competition's own compute constraints (5 min wall-clock, 16GB RAM, **CPU only**,
**no network during ranking**, ≤5GB intermediate state) directly rule out calling a
hosted LLM per candidate, and make a from-scratch neural embedding model a real
reproducibility risk for Stage 3 (an unfamiliar sandbox, an extra multi-hundred-MB
dependency, slower CPU inference). We benchmarked TF-IDF + cosine similarity over the
full 100K-candidate pool at **under 30 seconds** on a single CPU core using only
`scikit-learn` — and on this dataset's small, fixed, templated vocabulary (133 skill
tags, 47 job titles), lexical similarity captures real signal without that risk.

Critically, semantic similarity is **one of eight weighted inputs**, not the ranking
itself — and skills are deliberately excluded from the similarity text so that loading
a profile's skills section with buzzwords can't inflate it. The JD explicitly warns
against keyword-matching as "the right answer," so the system is built around several
independent, individually-validated signals (career narrative content, trust-adjusted
skill claims, product-vs-consulting company history, behavioral availability) that
each try to capture a different part of what an actual recruiter would notice.

Every non-obvious rule in the code (the honeypot fingerprint, the consulting-firm
detection, the narrative-pattern regexes, the location tiers) was derived by directly
querying the full 100K-candidate dataset first, not assumed and then rationalized —
see `docs/data_analysis.md` for the exact queries and results.

## Measured performance

Run on the full, unmodified 100,000-candidate file, on a single CPU core with ~4GB RAM
available (i.e. *more* constrained than the competition's stated 16GB/CPU-only
target):

| Stage | Time |
|---|---|
| Load 100K JSON records | ~7.5s |
| TF-IDF fit + similarity | ~28s |
| Feature extraction + scoring (100K candidates) | ~54s |
| Write top-100 CSV | <1s |
| **Total wall-clock** | **~90s** (budget: 300s) |
| **Peak RAM** | **~2.9 GB** (budget: 16 GB) |

No GPU, no network calls, no model weights to download or ship.

## Validating the output

```bash
python /path/to/hackathon/validate_submission.py artifacts/submission.csv   # format
python qa_check.py --candidates ./data/candidates.jsonl --submission ./artifacts/submission.csv  # content
python -m pytest tests/ -v   # or: python tests/test_features.py
```

`qa_check.py` independently re-verifies (separately from the official format
validator): zero honeypots in the submission, zero candidates whose entire career is
at a consulting/IT-services firm with no product-company history, score
monotonicity, no exact-duplicate reasoning strings, and a regex-based spot-check that
no reasoning string claims a skill the candidate's profile doesn't actually have.

## Honest limitations

See "Known limitations and explicit tradeoffs" in `docs/methodology.md` for the full
list. The short version: TF-IDF doesn't capture true paraphrase-level semantics the
way a neural embedding model would; career-history description text isn't always
tightly coupled to the listed job title for "filler" candidates in this dataset (we
found this directly and designed around it rather than assuming clean data); and the
composite scoring weights are hand-specified from reasoning about the JD text plus
validation against known example candidates, not learned from labeled data (no ground
truth is available to participants).

## AI tool usage

Declared in `submission_metadata.yaml`. Used for architecture discussion, EDA query
drafting, and code authoring; no candidate data was sent to any external LLM API as
part of the ranking pipeline itself (which runs fully offline, per the compute
constraints).
