# Data analysis log

This file documents the exploratory queries that were actually run against
`candidates.jsonl` before any scoring code was written, and the findings that shaped
each design decision in `src/`. Every claim in `docs/methodology.md` traces back to one
of the queries below. Re-run any of these against the released dataset to verify.

## Title vocabulary

```python
import json, collections
titles = collections.Counter()
with open("candidates.jsonl") as f:
    for line in f:
        d = json.loads(line)
        titles[d["profile"]["current_title"]] += 1
print(len(titles))            # 47 distinct titles total
```

Result: 100,000 candidates use only 47 distinct title strings. The bulk (~70K) are
non-tech professions (Business Analyst, HR Manager, Mechanical Engineer, Accountant,
Project Manager, Customer Support, Operations Manager, Content Writer, Sales
Executive, Civil Engineer, Graphic Designer, Marketing Manager -- ~5,500-5,800 each).
General tech titles (Software Engineer, Full Stack Developer, Cloud Engineer, etc.)
total ~25K. AI/ML-flavored titles total well under 1,000, and the most senior ones
(Senior AI Engineer, Lead AI Engineer, Staff Machine Learning Engineer, Senior Applied
Scientist, Senior Machine Learning Engineer, Senior NLP Engineer) total **29
candidates** -- this directly informed treating title match as a strong but not
dominant signal (0.20 weight) since the JD explicitly says title isn't the point.

## Country / location distribution

```python
countries = collections.Counter()
... countries[d["profile"]["country"]] += 1
```

Result: 75,113 India, 9,978 USA, then Australia/Canada/UK/Germany/Singapore/UAE at
~2,400-2,600 each. Within India, 18 cities are represented near-evenly (~4,000-4,300
each). This confirmed building a tiered location score (Pune/Noida highest, then
Hyderabad/Mumbai/Delhi/Gurgaon/Bangalore, then other India, then non-India scaled by
relocation willingness) rather than a hard in/out filter.

## Skill vocabulary

```python
skill_counts = collections.Counter()
... for s in d["skills"]: skill_counts[s["name"]] += 1
print(len(skill_counts))  # 133 distinct skills
```

Result: 133 distinct skill tags. ~75 generic tech/business skills appear ~12,000 times
each (HTML, Java, SQL, Excel, Salesforce CRM, etc. -- present across the whole
distractor population). ~48 AI-core skills (Embeddings, FAISS, LangChain, RAG, NLP,
etc.) appear 1,300-5,100 times. A small set of highly specific IR/search skills
(Search Backend, Text Encoders, Ranking Systems, Search & Discovery, Search
Infrastructure, Indexing Algorithms, Vector Representations, Content Matching, Model
Adaptation) appear only 1-7 times each -- these read as deliberately hand-placed on the
very best-fit candidates and are weighted at 0.8-1.0 in `MUST_HAVE_SKILLS`.

## Honeypot fingerprint

```python
n = 0
for s in skills:
    if s["proficiency"] == "expert" and s.get("duration_months", 0) == 0:
        n += 1
```

Distribution of this count across the full pool: `{0: 99979, 3: 8, 5: 8, 4: 5}` --
**exactly bimodal**, never 1 or 2. All 21 candidates with count >= 3 carry occupations
with zero plausible AI/ML connection (Mobile Developer, Full Stack Developer, HR
Manager, Operations Manager, Business Analyst, Accountant, QA Engineer, Civil
Engineer, Content Writer, .NET Developer, Software Engineer, Frontend Engineer,
Backend Engineer, Civil Engineer -- spot-checked several directly). This is the
honeypot fingerprint used in `features.honeypot_flags` (threshold set at >=2 to leave a
small safety margin below the observed minimum of 3).

We also checked and explicitly rejected three other candidate "honeypot" signals as too
noisy to use as hard flags (all are common dataset-generation artifacts, not deliberate
traps):
  - `years_of_experience` vs. sum of `career_history` durations: 47 candidates differ
    by >3 years, but `career_history` is schema-capped at 10 entries, so long careers
    naturally truncate -- this is expected, not a trap.
  - Education `end_year` later than first job's `start_date` year: 19,499 candidates
    (~20% of the pool) -- far too common to be deliberate; treated as generic synthetic-
    data noise.
  - Skill `duration_months` exceeding total years of experience by 24+ months: 2,821
    candidates -- also too common; used only as a very mild input to
    `skill_assessment_credibility`, never as a hard flag.

## Behavioral twins

```python
groups = collections.defaultdict(list)
key = (title, current_company, round(years_of_experience, 1), location)
groups[key].append(candidate_id)
```

Result: 5,028 groups share an identical (title, company, years_of_experience,
location) tuple across the full pool; one such pair exists even within the narrow
AI/ML-titled population (`CAND_0009200` / `CAND_0024878`, both "AI Specialist" at HCL,
5.7 yrs, Vizag) -- their `redrob_signals` differ sharply (recruiter_response_rate 0.32
vs 0.84; interview_completion_rate 0.58 vs 1.0; last_active_date Feb vs May). This
directly confirms the dataset is designed so that behavioral signals are sometimes the
*only* differentiator between two candidates, motivating the availability multiplier
in `features.availability_modifier`.

## Consulting-firm industry tagging

```python
comp_industry = collections.defaultdict(collections.Counter)
... comp_industry[current_company][current_industry] += 1
```

Result: every candidate at TCS, Infosys, Wipro, Capgemini, Cognizant, Mindtree, HCL,
Tech Mahindra, and Mphasis carries `current_industry: "IT Services"`; Accenture
candidates carry `"Consulting"`. This is a clean, exception-free field-level signal
that operationalizes the JD's explicit "people who have only worked at consulting
firms... in their entire career" disqualifier --
`features.product_company_score` checks whether the *union* of current + all historical
`industry` values is fully contained in `{IT Services, Consulting}`, which also
correctly implements the JD's stated exception ("if you're currently at one of these
companies but have prior product-company experience, that's fine") -- confirmed 21,410
candidates currently at a consulting/IT-services firm DO have non-consulting industries
elsewhere in their history, so the exception path is real and exercised, not
theoretical.

## Narrative pattern false-positive validation

```python
nonrelevant_titles = {"Accountant", "Mechanical Engineer", "Civil Engineer",
                       "HR Manager", "Content Writer", "Sales Executive",
                       "Graphic Designer", "Customer Support", "Operations Manager",
                       "Marketing Manager", "Project Manager", "Business Analyst"}
# for each candidate with title in nonrelevant_titles, test each candidate regex
# pattern against the concatenated career_history description text
```

Result (hits out of 68,821 non-relevant-titled candidates):

| Pattern | False-positive hits |
|---|---|
| `production` | 46,624 |
| `evaluat` | 21,505 |
| `scal(e\|ing)` | 21,246 |
| `ranking model`, `retriev`, `embedding`, `relevance`, `search (backend\|infrastructure\|product\|result)`, `A/?B test`, `offline...online`, `click-through/dwell time/conversion`, `vector (search\|database\|index)`, `hybrid (search\|retrieval)`, `feature pipeline/feature engineering`, `learning.to.rank`, `discovery feed`, `matching layer` | **0** (every one) |

The first three patterns were removed from `features.career_narrative_signal`
entirely. The surviving patterns were then checked against the AI/ML-titled
population and hit ~12.5% of those candidates (148 / 1,179) -- a real, validated
signal rather than noise.

## Company rarity (product-company anchor check)

```python
companies = collections.Counter()
... companies[current_company] += 1
```

Result: 63 distinct companies total. Generic placeholder "big companies" (Wayne
Enterprises, Hooli, Stark Industries, Globex Inc, Acme Corp, Dunder Mifflin, Pied
Piper, Initech) and the explicitly-disqualified consulting firms (TCS, Infosys, Wipro)
each appear ~7,300-7,600 times. Real Indian product/startup companies (Swiggy, Zomato,
Razorpay, CRED, Flipkart, PhonePe, Paytm, etc.) appear ~150-1,300 times each, and a
small set of AI-native Indian startups (Sarvam AI, Niramai, Mad Street Den, Yellow.ai,
Krutrim, Observe.AI, etc.) appear 20-42 times each. Real global tech giants (Meta,
Google, Netflix, Amazon, Microsoft, Salesforce, LinkedIn, Apple, Adobe, Uber) appear
only **1-7 times each, 43 candidates total** -- a clear rarity signal that these are
deliberately seeded "ideal anchor" companies rather than a natural distribution.

## Compute budget validation

```python
import time
t0 = time.time(); texts = [load all 100K summaries]; print(time.time()-t0)   # ~4s
t0 = time.time(); TfidfVectorizer().fit_transform(texts); print(time.time()-t0)  # ~10s
```

Full `rank.py` run on the complete 100,000-candidate file, single CPU core, ~4GB-RAM
sandbox (i.e. a *more* constrained environment than the competition's 16GB/CPU-only
target):

```
loaded 100000 candidates in 7.5s
TF-IDF fit+transform in 28.0s
scored 100000 candidates in 54.0s
wrote 100 rows in 0.3s
Total wall-clock: 89.8s
Peak RSS: ~2.9 GB
```

~90 seconds against a 300-second (5-minute) budget, ~2.9GB against a 16GB budget --
comfortable margin on a single core; the target machine is expected to have more
headroom still.
