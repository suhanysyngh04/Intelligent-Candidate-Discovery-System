#!/usr/bin/env python3
"""
rank.py -- produces the top-100 ranked submission CSV from candidates.jsonl.

Usage:
    python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv

Compute profile (measured on a single CPU core, see docs/methodology.md):
    - JSON load (100K records):      ~4-5s
    - TF-IDF fit + transform:        ~10s
    - Feature extraction + scoring:  ~5-10s
    - Total wall-clock:              well under 60s, against a 5-minute budget

No network calls, no GPU usage, no model downloads -- everything here is
scikit-learn TF-IDF plus pure-Python rule-based feature scoring over fields
already present in candidates.jsonl. Peak memory is dominated by holding the
candidate records and the sparse TF-IDF matrix in memory, both far under the
16GB limit (a sparse 100K x 40000 TF-IDF matrix with the vocab this dataset
produces is tens of MB, not GBs).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from jd_profile import JD_TEXT
from semantic import SemanticSimilarityScorer
from scoring import score_candidate
from reasoning import build_reasoning

TOP_N = 100
RNG_SEED = 42  # fixed seed -> fully reproducible reasoning-phrasing choices


def load_candidates(path: Path) -> list[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    candidates = []
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            candidates.append(json.loads(line))
    return candidates


def main():
    parser = argparse.ArgumentParser(description="Rank candidates against the Redrob Senior AI Engineer JD.")
    parser.add_argument("--candidates", required=True, type=Path, help="Path to candidates.jsonl or .jsonl.gz")
    parser.add_argument("--out", required=True, type=Path, help="Output submission CSV path")
    parser.add_argument("--top-n", type=int, default=TOP_N, help="Number of ranked rows to output (default 100)")
    args = parser.parse_args()

    if not args.candidates.exists():
        print(f"ERROR: candidates file not found: {args.candidates}", file=sys.stderr)
        print("Download/place candidates.jsonl (or candidates.jsonl.gz) at this path before running.", file=sys.stderr)
        sys.exit(1)

    t_start = time.time()

    print(f"[1/4] Loading candidates from {args.candidates} ...", file=sys.stderr)
    candidates = load_candidates(args.candidates)
    print(f"      loaded {len(candidates)} candidates in {time.time() - t_start:.1f}s", file=sys.stderr)

    t0 = time.time()
    print("[2/4] Computing JD<->candidate semantic similarity (TF-IDF) ...", file=sys.stderr)
    sim_scorer = SemanticSimilarityScorer(JD_TEXT).fit(candidates)
    semantic_scores = sim_scorer.scores()
    print(f"      done in {time.time() - t0:.1f}s", file=sys.stderr)

    t0 = time.time()
    print("[3/4] Scoring candidates (structured features + behavioral signals) ...", file=sys.stderr)
    results = []
    for c in candidates:
        sem = semantic_scores.get(c["candidate_id"], 0.0)
        results.append(score_candidate(c, sem))
    print(f"      scored {len(results)} candidates in {time.time() - t0:.1f}s", file=sys.stderr)

    # Honeypot rate sanity check, logged for transparency (not used to alter ranking --
    # honeypots are already forced to the bottom via HONEYPOT_SUPPRESSION_SCORE).
    n_honeypot = sum(1 for r in results if r["is_honeypot"])
    print(f"      honeypot candidates detected in full pool: {n_honeypot}", file=sys.stderr)

    t0 = time.time()
    print(f"[4/4] Selecting top {args.top_n} and writing {args.out} ...", file=sys.stderr)

    by_id = {c["candidate_id"]: c for c in candidates}
    # Sort by score desc, tie-break by candidate_id ascending (spec section 3).
    results.sort(key=lambda r: (-r["score"], r["candidate_id"]))
    top = results[: args.top_n]

    n_honeypot_in_top = sum(1 for r in top if r["is_honeypot"])
    if n_honeypot_in_top > 0:
        print(f"      WARNING: {n_honeypot_in_top} honeypot(s) made the top {args.top_n} "
              f"-- this should not happen given hard suppression.", file=sys.stderr)

    rng = random.Random(RNG_SEED)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, r in enumerate(top, start=1):
            candidate = by_id[r["candidate_id"]]
            reasoning = build_reasoning(candidate, r, rng)
            writer.writerow([r["candidate_id"], rank, f"{r['score']:.4f}", reasoning])

    print(f"      wrote {len(top)} rows in {time.time() - t0:.1f}s", file=sys.stderr)
    print(f"\nTotal wall-clock: {time.time() - t_start:.1f}s", file=sys.stderr)
    print(f"Submission written to: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
