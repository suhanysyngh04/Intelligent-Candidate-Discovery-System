# Methodology

## 1. The actual problem

The JD (`job_description.docx`) is unusually explicit about what it's testing for. Read
literally, it tells participants:

> "The 'right answer' to this JD is not 'find candidates whose skills section contains
> the most AI keywords.' ... A Tier 5 candidate may not use the words 'RAG' or
> 'Pinecone' in their profile, but if their career history shows they built a
> recommendation system at a product company, they're a fit. A candidate who has all
> the AI keywords listed as skills but whose title is 'Marketing Manager' is not a fit."

So the system being graded is not "does it retrieve documents whose text is close to
the query" -- it's "does it reproduce the judgment a thoughtful recruiter would form
after actually reading a profile, including reading between the lines and discounting
unreachable candidates." That reframes the task: keyword/embedding similarity is *one
input*, not the answer.

## 2. What we found by actually inspecting the data

Before writing any scoring code we profiled the full 100K-candidate pool (see
`docs/data_analysis.md` for the full set of queries run). The headline findings that
shaped the architecture:

- **Only 47 distinct job titles exist** in the whole pool, and only ~1,000 candidates
  carry an AI/ML-flavored title at all. The truly senior titles the JD describes
  ("Senior AI Engineer", "Lead AI Engineer", "Staff Machine Learning Engineer",
  "Senior Applied Scientist") total **29 candidates** out of 100,000.
- **~75,000 candidates are in completely unrelated professions** (Accountant, HR
  Manager, Civil Engineer, Sales Executive, etc.) -- pure distractor volume.
- **Honeypots have a clean, deliberate fingerprint**: `proficiency: "expert"` paired
  with `duration_months: 0` on the same skill. Across the full pool this occurs in
  exactly 21 candidates, and the count is sharply bimodal (0, or 3-5 -- never 1 or 2),
  which is a strong signal of deliberate construction rather than natural noise. Every
  one of these 21 also carries an unrelated job title (Accountant, HR Manager, Mobile
  Developer, etc.), so they were never going to surface on title/narrative grounds
  either -- but we hard-suppress them explicitly as a safety margin against the
  Stage 3 honeypot-rate disqualification threshold.
- **Keyword-stuffing is real and common**: candidates with backend/data-engineering
  careers (e.g. at IT-services firms) carrying long lists of flashy AI skill *tags*
  (NLP, GANs, TTS, Speech Recognition) with low proficiency and near-zero
  `duration_months` behind them. `sample_candidates.json`'s very first record is
  exactly this pattern: a "Backend Engineer" at Mindtree with 17 skill tags including
  GANs, TTS, and Speech Recognition, none used for more than a few months.
- **Real elite product companies are seeded as rare anchors.** Meta, Google, Netflix,
  Amazon, Microsoft, Salesforce, LinkedIn, Apple, Adobe, and Uber appear only 1-7 times
  each in `current_company` (43 candidates total) against ~7,500 occurrences each for
  generic large-company placeholders (Wayne Enterprises, Hooli, Stark Industries,
  Globex, etc.) and equally large counts for real-but-explicitly-disqualified
  consulting firms (TCS, Infosys, Wipro: ~7,500 each). This rarity pattern is itself a
  signal: the small set of named global/Indian product companies (Swiggy, Zomato,
  Razorpay, CRED, Flipkart, PhonePe, Sarvam AI, and the FAANG-adjacent names) lines up
  with the JD's "product company, not pure services" framing.
- **"Behavioral twins" are real and locatable.** Grouping by
  `(title, company, years_of_experience, location)` finds thousands of near-identical
  profiles differing only in `redrob_signals` -- e.g. two "AI Specialist" candidates
  at HCL with identical tenure, one active in the last few days with an 84% recruiter
  response rate, the other inactive for 3 months with a 32% response rate. A ranker
  that ignores behavioral signals cannot tell these two apart; the JD explicitly says
  it should.
- **The IT-services/consulting exclusion has a clean field-level signal.** Every
  candidate at TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, Mindtree, HCL,
  Tech Mahindra, or Mphasis carries `current_industry: "IT Services"` (or
  `"Consulting"` for Accenture). Checking whether a candidate's *entire* career
  history industry set is contained in `{IT Services, Consulting}` cleanly
  operationalizes the JD's stated exception ("if you're currently at one of these
  companies but have prior product-company experience, that's fine").
- **Generic English words in career descriptions are a trap for naive NLP.** Patterns
  like "production", "evaluate", and "scale" appear in 20,000-46,000 *unrelated*
  candidate descriptions (sales people "evaluate vendors", ops people "scale the
  team") -- we validated every narrative-matching regex against the full pool before
  using it, keeping only patterns with a confirmed ~0% false-positive rate against a
  set of clearly irrelevant titles (Accountant, Mechanical Engineer, HR Manager, Sales
  Executive, Operations Manager, etc.). See "narrative pattern validation" below.

These findings directly motivated specific code paths; they were not guessed at and
then loosely confirmed -- the EDA queries that produced each finding above are in
`docs/data_analysis.md` so the design choices are auditable end to end.

## 3. Architecture: hybrid scoring, not a single model

```
candidates.jsonl
      |
      v
+-----------------------+        +--------------------------+
| TF-IDF semantic layer |        | Structured feature layer |
| (semantic.py)         |        | (features.py)            |
| JD text <-> candidate |        | title / skills / career  |
| headline+summary+     |        | narrative / location /   |
| career descriptions   |        | consulting-exclusion /   |
+-----------------------+        | experience-band / notice |
      |                          +--------------------------+
      |  semantic_score (0-1)              |  8 component scores (0-1 each)
      |                                    |  + 6 penalty multipliers
      v                                    v
            +---------------------------+
            |   scoring.py composite    |
            |  weighted sum * penalties |
            +---------------------------+
                          |
                          v  fit_score (0-1)
            +---------------------------+
            | availability multiplier   |
            | (behavioral signals)      |
            +---------------------------+
                          |
                          v  final_score
                  sort, take top 100
                          |
                          v
            +---------------------------+
            |  reasoning.py             |
            |  per-row justification    |
            +---------------------------+
                          |
                          v
                 submission.csv
```

### Why hybrid, and why not an embedding model or an LLM call per candidate

The compute constraints (`submission_spec.docx` Section 3) are explicit and binding:
5-minute wall clock, 16GB RAM, CPU only, **no network during ranking**, ≤5GB
intermediate state. An LLM-per-candidate re-ranker is ruled out by the spec itself
("running an LLM call for each of 100,000 candidates will not fit the 5-minute CPU
budget, even if the model runs locally"). A neural sentence-embedding model
(sentence-transformers, BGE, E5) is technically usable within the constraints if its
weights are pre-bundled in the repo, but it adds real Stage-3 reproduction risk
(larger dependency surface, model-loading edge cases, slower CPU inference) for a
dataset whose vocabulary is small, fixed, and templated (133 distinct skill tags, 47
distinct titles). We benchmarked the alternative -- TF-IDF + cosine similarity over the
full 100K pool -- at **under 30 seconds** on a single CPU core using only
`scikit-learn`, which is about as close to zero-dependency-risk as this kind of system
gets. We treat this as the right engineering tradeoff for *this* dataset and *these*
constraints, not as a universal claim that lexical similarity beats neural embeddings
in general.

Critically, **semantic similarity is one weighted component among eight**, not the
ranking itself. Skills are deliberately excluded from the TF-IDF input text (only
headline + summary + career-history descriptions go in) specifically so that loading
up a skills list with AI buzzwords cannot inflate the semantic-similarity score on top
of the separate, trust-adjusted `skills_match_score` -- otherwise keyword stuffing
would just move from one layer of the system into another.

### Composite score formula

```
fit_score = 0.20 * title_score
          + 0.20 * semantic_score        (TF-IDF JD <-> candidate narrative)
          + 0.20 * skills_score          (trust-adjusted must-have skill match)
          + 0.15 * narrative_score       (validated career-history pattern scan)
          + 0.10 * experience_score      (soft band around 5-9 yrs)
          + 0.08 * product_company_score (consulting-only exclusion lives here)
          + 0.05 * location_score        (Pune/Noida highest, India broadly good,
                                           non-India case-by-case per the JD)
          + 0.02 * notice_score

fit_score *= keyword_stuffing_penalty
           * cv_speech_without_nlp_penalty
           * title_chaser_penalty
           * recent_code_penalty
           * skill_assessment_credibility
           * experience_consistency_penalty

final_score = fit_score * availability_multiplier   (behavioral signals)
```

Honeypots are detected first and short-circuited directly to a near-zero score,
bypassing the rest of the pipeline entirely (see `scoring.score_candidate`).

### Why penalties are multiplicative, and why availability is applied last

Two deliberate design choices that map directly to specific JD language:

1. **Penalties (keyword-stuffing, consulting-only, title-chasing, etc.) multiply the
   fit score rather than subtracting from it.** A multiplicative penalty can never
   flip a clearly-bad profile into a good one just by stacking unrelated positive
   features, and it scales proportionally -- a 50% keyword-stuffing penalty removes
   half the candidate's *earned* score, whatever that score was, rather than a fixed
   amount that would over-punish weak profiles and under-punish strong ones.

2. **The availability multiplier (behavioral signals) is applied last, after the fit
   score, and only multiplicatively.** This directly encodes the JD's own framing:
   "a perfect-on-paper candidate who hasn't logged in for 6 months and has a 5%
   recruiter response rate is, for hiring purposes, not actually available. Down-weight
   them appropriately." A multiplier can discount an excellent fit's visible rank
   (because they're hard to actually reach right now) but it can never let a poor
   fit's high activity/responsiveness outscore a strong fit's substance -- a 0.9
   fit-score candidate with a rough 0.5 availability multiplier (0.45 final) still
   beats a 0.3 fit-score candidate with perfect availability (0.30 final).

### Narrative pattern validation (the "plain-language Tier 5" detector)

`features.career_narrative_signal` scans career-history description text for evidence
of having actually built retrieval/ranking/search systems, in plain English, without
requiring the candidate to have used any specific skill-tag noun. Every regex pattern
in that scanner was checked against the **full 100K-candidate pool**, specifically
against a control group of titles with no plausible connection to AI/ML/search
(Accountant, Mechanical Engineer, Civil Engineer, HR Manager, Content Writer, Sales
Executive, Graphic Designer, Customer Support, Operations Manager, Marketing Manager,
Project Manager, Business Analyst -- ~69,000 candidates). Patterns that fired on any
meaningful fraction of that control group were removed; the strongest false-positive
offenders were generic business English ("production" -- 46,624 false hits,
"evaluate" -- 21,505, "scale" -- 21,246). The patterns that survived (e.g. "retriev",
"ranking model", "click-through / dwell time / conversion", "offline...online",
"vector search/database/index", "discovery feed", "matching layer") have a **confirmed
0% hit rate** on that same control group, while hitting ~12.5% of candidates with
genuinely AI/ML-flavored titles -- a clean, validated signal rather than a guess.

### Honeypot detection

`features.honeypot_flags` looks for `proficiency == "expert"` combined with
`duration_months == 0` on the same skill entry, requiring at least 2 such entries to
fire (a single occurrence is treated as tolerable data noise rather than a confirmed
trap, giving a small safety margin against false positives). This was reverse-engineered
directly from the data, not assumed: scanning the full pool shows the count of such
entries per candidate is exactly bimodal (99,979 candidates have 0; the rest have 3, 4,
or 5 -- never 1 or 2), which is strong evidence of deliberate construction. All 21
candidates matching this fingerprint also carry unrelated job titles, so in practice
they were already going to rank far outside the top 100 on fit grounds alone -- the
explicit hard-suppression in `scoring.py` exists as a safety margin against the
Stage 3 honeypot-rate disqualification threshold, per the spec's own guidance that "we
expect a good ranking system to naturally avoid them; you don't need to special-case
them" (we special-case them anyway, because the cost of doing so is zero and the
downside of a false negative is disqualification).

## 4. Known limitations and explicit tradeoffs

- **TF-IDF cannot capture true paraphrase-level semantics** the way a neural embedding
  model can (e.g. "I made search results better" vs. "improved retrieval relevance"
  would not score as similar under TF-IDF unless they share n-grams). We accept this
  because the dataset's career-history text is templated/formulaic enough that bigram
  overlap captures most of the real signal (validated in Section 3), and because the
  compute-constraint tradeoff favors a simpler, more reproducible approach for this
  specific dataset. A production system at Redrob's actual scale, free of a 5-minute
  CPU-only competition constraint, would likely use a real embedding model.
- **Career-history description text is not always coupled to the listed job title.**
  We found this directly: for many "filler" candidates (Mechanical Engineer, Sales
  Executive, etc.) the description text reads as a fully unrelated job's
  description, while for AI/ML-titled candidates and the genuinely strong matches,
  description content reliably matches title content. We did not find a reliable way
  to detect this decoupling at the individual-candidate level (it appears to be a
  property of which "track" a candidate was generated on, not something visible from
  a single field), so `career_narrative_signal` is one of eight weighted components
  rather than a dominant signal, which limits the damage from any single
  description/title mismatch.
- **The TF-IDF similarity score is normalized per-run via min-max scaling across the
  candidate pool.** This means the *absolute* semantic_score value is only meaningful
  relative to this specific 100K-candidate run, not as a portable similarity metric --
  acceptable for a ranking task (we only need relative ordering) but worth flagging
  explicitly.
- **Weights in the composite formula were set by reasoning from the JD text and
  validated against known-good/known-bad example candidates (see
  `docs/data_analysis.md`), not learned from labeled data** -- there is no ground-truth
  relevance signal available to participants, so this is a deliberately interpretable,
  hand-specified rule system rather than a trained ranker. This is consistent with the
  hackathon's own framing that "the architecture is your call" and the emphasis on
  defensible, explainable design choices at the Stage 5 interview.
