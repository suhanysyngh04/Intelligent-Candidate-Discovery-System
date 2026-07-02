#!/usr/bin/env python3
"""
qa_check.py -- sanity checks on a produced submission CSV against the full
candidate pool. Goes beyond validate_submission.py's format checks to verify
the *content* invariants our methodology is supposed to guarantee:

  1. Zero honeypots in the top 100 (spec disqualifies at >10%; we target 0%).
  2. No candidate whose entire career is at a pure consulting/IT-services firm
     with no product-company experience.
  3. Score is non-increasing with rank (re-verified independently of the
     official validator).
  4. Reasoning field has no exact-duplicate strings across rows.
  5. Reasoning field never mentions a skill name that isn't actually present
     in that candidate's skills list (no-hallucination spot check).

Usage:
    python qa_check.py --candidates ./data/candidates.jsonl --submission ./artifacts/submission.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from jd_profile import CONSULTING_INDUSTRIES, PRODUCT_INDUSTRIES  # noqa: E402


def load_candidates_by_id(path: Path, ids: set[str]) -> dict:
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d["candidate_id"] in ids:
                out[d["candidate_id"]] = d
    return out


def is_honeypot(candidate: dict) -> bool:
    skills = candidate.get("skills", [])
    n = sum(1 for s in skills if s.get("proficiency") == "expert" and (s.get("duration_months", 0) or 0) == 0)
    return n >= 2


def is_consulting_only(candidate: dict) -> bool:
    profile = candidate["profile"]
    history = candidate.get("career_history", [])
    industries = {j.get("industry") for j in history} | {profile.get("current_industry")}
    has_product = bool(industries & PRODUCT_INDUSTRIES)
    all_consulting = industries.issubset(CONSULTING_INDUSTRIES | {None})
    return all_consulting and not has_product


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--submission", required=True, type=Path)
    args = parser.parse_args()

    with open(args.submission, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    ids = {row["candidate_id"] for row in rows}
    candidates = load_candidates_by_id(args.candidates, ids)

    issues = []

    # 1. Honeypot check
    honeypot_hits = [row["candidate_id"] for row in rows if is_honeypot(candidates[row["candidate_id"]])]
    if honeypot_hits:
        issues.append(f"FAIL: {len(honeypot_hits)} honeypot(s) found in submission: {honeypot_hits}")
    else:
        print("PASS: 0 honeypots in submission (0.0% rate, well under the 10% disqualification threshold)")

    # 2. Consulting-only check
    consulting_only_hits = [row["candidate_id"] for row in rows if is_consulting_only(candidates[row["candidate_id"]])]
    if consulting_only_hits:
        issues.append(f"WARN: {len(consulting_only_hits)} candidate(s) with consulting-only career history: {consulting_only_hits}")
    else:
        print("PASS: no candidates with a pure consulting/IT-services-only career history")

    # 3. Score monotonicity (independent re-check)
    scores = [float(row["score"]) for row in rows]
    if all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)):
        print("PASS: scores are non-increasing with rank")
    else:
        issues.append("FAIL: scores are NOT non-increasing with rank")

    # 4. No exact-duplicate reasoning strings
    reasonings = [row["reasoning"] for row in rows]
    dupes = len(reasonings) - len(set(reasonings))
    if dupes == 0:
        print("PASS: no exact-duplicate reasoning strings")
    else:
        issues.append(f"WARN: {dupes} duplicate reasoning string(s) found")

    # 5. No-hallucination spot check: any skill NAME mentioned in reasoning text
    # (via simple substring match against the dataset's known skill vocabulary)
    # must actually be in that candidate's skill list.
    #
    # NOTE: we deliberately check only the text *after* the first semicolon.
    # The reasoning template's first clause is "<title> at <company>", and
    # several job titles in this dataset legitimately contain skill-like
    # substrings (e.g. "Recommendation Systems Engineer", "Machine Learning
    # Engineer", "NLP Engineer") -- matching against the title clause produces
    # false positives on the candidate's own real job title, not a hallucinated
    # skill claim. The "hands-on with ..." clause -- the part that actually
    # asserts skill possession -- always comes after that first semicolon.
    from jd_profile import MUST_HAVE_SKILLS
    hallucinations = []
    for row in rows:
        cand = candidates[row["candidate_id"]]
        cand_skill_names = {s["name"] for s in cand.get("skills", [])}
        reasoning_text = row["reasoning"]
        claim_clause = reasoning_text.split(";", 1)[1] if ";" in reasoning_text else ""
        for skill_name in MUST_HAVE_SKILLS:
            if not re.search(r"\b" + re.escape(skill_name) + r"\b", claim_clause):
                continue
            # Guard against false positives where the matched name is a
            # substring of a DIFFERENT skill name the candidate actually has
            # (e.g. "Information Retrieval" inside "Information Retrieval
            # Systems", or "LLMs" inside "Fine-tuning LLMs").
            if any(skill_name in real_name for real_name in cand_skill_names):
                continue
            if skill_name not in cand_skill_names:
                hallucinations.append((row["candidate_id"], skill_name))
    if hallucinations:
        issues.append(f"FAIL: hallucinated skill mentions found: {hallucinations}")
    else:
        print("PASS: no hallucinated skill mentions detected in reasoning text")

    print()
    if issues:
        print(f"{len(issues)} issue(s) found:")
        for i in issues:
            print(" -", i)
        sys.exit(1)
    else:
        print("All QA checks passed.")


if __name__ == "__main__":
    main()
