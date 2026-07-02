"""
Feature extraction from a raw candidate JSON record.

Every function here takes a candidate dict (matching candidate_schema.json) and
returns a small, named, auditable piece of signal. scoring.py combines these into
the final composite score. Keeping extraction separate from weighting means we can
explain every number in a reasoning string without re-deriving it.
"""
from __future__ import annotations

import math
import re
from datetime import date, datetime
from typing import Any

from jd_profile import (
    CORE_AI_TITLES,
    ADJACENT_TITLES,
    MUST_HAVE_SKILLS,
    SURFACE_AI_SKILLS,
    CV_SPEECH_ROBOTICS_SKILLS,
    NLP_IR_SKILLS,
    CONSULTING_INDUSTRIES,
    PRODUCT_INDUSTRIES,
    LOCATION_TIERS,
    OTHER_INDIA_SCORE,
    NON_INDIA_NO_VISA_SCORE,
    EXPERIENCE_BAND,
    EXPERIENCE_SOFT_MIN,
    EXPERIENCE_SOFT_MAX,
    NOTICE_IDEAL_DAYS,
    TITLE_CHASER_AVG_TENURE_MONTHS,
)

# "Today" for recency calculations. Fixed rather than datetime.now() so the
# ranking is fully reproducible regardless of when the script is rerun.
REFERENCE_DATE = date(2026, 6, 27)


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def title_score(current_title: str) -> tuple[float, str]:
    """Returns (score 0-1, tier label) for the candidate's current title."""
    if current_title in CORE_AI_TITLES:
        return CORE_AI_TITLES[current_title], "core_ai_title"
    if current_title in ADJACENT_TITLES:
        return ADJACENT_TITLES[current_title], "adjacent_title"
    return 0.05, "unrelated_title"


def experience_fit_score(years: float) -> float:
    """Soft band score around the JD's 5-9yr range. 1.0 inside the band,
    decaying outside it. JD explicitly frames this as a range, not a cutoff."""
    lo, hi = EXPERIENCE_BAND
    if lo <= years <= hi:
        return 1.0
    if years < lo:
        # below band: linear decay down to 0 at EXPERIENCE_SOFT_MIN
        span = max(lo - EXPERIENCE_SOFT_MIN, 0.01)
        return max(0.0, 1.0 - (lo - years) / span)
    # above band: gentler decay -- JD says "we'll seriously consider candidates
    # outside the band if other signals are strong" but separately flags
    # architecture-only senior folks, handled by recent_code_score instead.
    span = max(EXPERIENCE_SOFT_MAX - hi, 0.01)
    return max(0.0, 1.0 - 0.6 * (years - hi) / span)


def skills_match_score(skills: list[dict]) -> tuple[float, dict]:
    """
    Weighted skill match against MUST_HAVE_SKILLS, trust-adjusted by proficiency
    and duration_months so a claimed "expert" with 0 months counts for nothing
    (this is also our honeypot fingerprint, see anomaly_flags).

    Returns (score 0-1, detail dict for reasoning/debugging).
    """
    PROF_WEIGHT = {"beginner": 0.25, "intermediate": 0.5, "advanced": 0.8, "expert": 1.0}

    weighted_sum = 0.0
    max_possible = sum(MUST_HAVE_SKILLS.values())
    matched = []
    for s in skills:
        name = s.get("name")
        if name not in MUST_HAVE_SKILLS:
            continue
        weight = MUST_HAVE_SKILLS[name]
        prof = PROF_WEIGHT.get(s.get("proficiency"), 0.25)
        dur = s.get("duration_months", 0) or 0
        # Trust factor: a claim needs SOME duration behind it to count fully.
        # 0 months -> 0 trust regardless of claimed proficiency (honeypot pattern).
        # Ramps up to full trust by 6 months.
        trust = min(1.0, dur / 6.0) if dur > 0 else 0.0
        contribution = weight * prof * trust
        weighted_sum += contribution
        if contribution > 0:
            matched.append((name, round(contribution, 3)))

    matched.sort(key=lambda x: -x[1])
    score = min(1.0, weighted_sum / (max_possible * 0.35))  # 35% coverage = full score
    return score, {"matched_skills": matched[:8], "raw_weighted_sum": round(weighted_sum, 2)}


def keyword_stuffing_penalty(skills: list[dict]) -> tuple[float, bool]:
    """
    Detects the JD's explicit trap: many flashy AI skill *tags* with little
    substance behind them (low proficiency / low duration / few must-have
    skills relative to surface-level ones).

    Returns (penalty_multiplier in (0,1], is_flagged).
    """
    surface_hits = [s for s in skills if s.get("name") in SURFACE_AI_SKILLS]
    must_have_hits = [s for s in skills if s.get("name") in MUST_HAVE_SKILLS]

    if len(surface_hits) < 3:
        return 1.0, False  # not enough surface skills to be a stuffing pattern

    # Average trust (duration-based) across the surface hits.
    avg_dur = sum((s.get("duration_months", 0) or 0) for s in surface_hits) / len(surface_hits)
    substantive_ratio = len(must_have_hits) / max(len(skills), 1)

    is_stuffer = avg_dur < 12 and substantive_ratio < 0.25 and len(surface_hits) >= 5
    if is_stuffer:
        return 0.5, True
    return 1.0, False


def cv_speech_without_nlp_penalty(skills: list[dict]) -> float:
    """JD: de-prioritize candidates whose primary expertise is CV/speech/robotics
    without significant NLP/IR exposure."""
    names = {s.get("name") for s in skills}
    cv_count = len(names & CV_SPEECH_ROBOTICS_SKILLS)
    nlp_count = len(names & NLP_IR_SKILLS)
    if cv_count >= 3 and nlp_count == 0:
        return 0.55
    return 1.0


def career_narrative_signal(career_history: list[dict]) -> tuple[float, list[str]]:
    """
    Scans career history descriptions for substantive evidence of having built
    ranking/retrieval/search/recommendation systems -- the "plain language Tier 5"
    case the JD calls out: a candidate may never say "RAG" or "Pinecone" but the
    career history itself shows they built the real thing.

    Returns (score 0-1, list of matched evidence phrases for reasoning text).
    """
    # NOTE: each pattern below was validated against the full 100K-candidate pool
    # during development to confirm a near-zero false-positive rate on titles
    # with no plausible AI/ML/search connection (Accountant, Mechanical Engineer,
    # HR Manager, Sales Executive, etc. -- see docs/methodology.md "narrative
    # pattern validation"). Generic business-English patterns that DID false-fire
    # on those titles -- "production", "evaluate", "scale" -- were deliberately
    # excluded; they matched 20-45K irrelevant profiles (sales/ops language like
    # "production tooling", "evaluate vendors", "scale the team") and carried no
    # real signal.
    pattern_groups = [
        (re.compile(r"\branking model", re.I), "shipped ranking models"),
        (re.compile(r"\bretriev", re.I), "retrieval systems experience"),
        (re.compile(r"\bembedding", re.I), "embeddings experience"),
        (re.compile(r"\brelevance\b", re.I), "relevance/search-quality work"),
        (re.compile(r"\bsearch (backend|infrastructure|product|result)", re.I), "search systems"),
        (re.compile(r"\bA/?B test", re.I), "A/B testing / online eval"),
        (re.compile(r"\boffline.{0,15}online", re.I), "offline-to-online eval correlation"),
        (re.compile(r"\bclick-through|dwell time|conversion\b", re.I), "engagement-metric optimization"),
        (re.compile(r"\bvector (search|database|index)", re.I), "vector search/database work"),
        (re.compile(r"\bhybrid (search|retrieval)", re.I), "hybrid retrieval work"),
        (re.compile(r"\bfeature pipeline|feature engineering\b", re.I), "feature engineering"),
        (re.compile(r"\blearning.to.rank\b", re.I), "learning-to-rank experience"),
        (re.compile(r"\bfine.tun(e|ing|ed)\b.{0,20}\b(model|llm)", re.I), "LLM fine-tuning experience"),
        (re.compile(r"\bmatching layer\b", re.I), "matching-system experience"),
        (re.compile(r"\bdiscovery feed\b", re.I), "discovery/recommendation feed work"),
    ]
    text = " ".join(j.get("description", "") for j in career_history)
    hits = []
    for pattern, label in pattern_groups:
        if pattern.search(text):
            hits.append(label)
    score = min(1.0, len(hits) / 8.0)
    return score, hits


def product_company_score(profile: dict, career_history: list[dict]) -> tuple[float, str]:
    """
    JD explicit disqualifier: entire career at consulting/IT-services firms (TCS,
    Infosys, Wipro, Accenture, Cognizant, Capgemini, etc.) with no product-company
    experience. Exception: currently there but has prior product-company history.

    Returns (multiplier in (0,1], explanation tag).
    """
    cur_industry = profile.get("current_industry")
    history_industries = {j.get("industry") for j in career_history}
    all_industries = history_industries | {cur_industry}

    has_product_exp = bool(all_industries & PRODUCT_INDUSTRIES)
    is_consulting_now = cur_industry in CONSULTING_INDUSTRIES
    all_consulting = all_industries.issubset(CONSULTING_INDUSTRIES | {None})

    if all_consulting and not has_product_exp:
        return 0.35, "entire_career_consulting_only"
    if is_consulting_now and has_product_exp:
        return 0.95, "consulting_now_product_before"  # JD's explicit exception
    return 1.0, "product_company_background"


def title_chaser_penalty(career_history: list[dict]) -> tuple[float, bool]:
    """JD: penalize career trajectories that look like title-hopping every ~1.5
    years across many employers chasing Senior->Staff->Principal."""
    if len(career_history) < 4:
        return 1.0, False
    durations = [j.get("duration_months", 0) or 0 for j in career_history]
    avg_tenure = sum(durations) / len(durations)
    if avg_tenure <= TITLE_CHASER_AVG_TENURE_MONTHS:
        return 0.7, True
    return 1.0, False


def recent_code_score(profile: dict, career_history: list[dict]) -> float:
    """JD: senior engineers who've moved into pure architecture/tech-lead roles
    with no code in 18+ months are not a fit -- 'this role writes code.'
    Proxy: does the current/most-recent role's title or description still read
    as hands-on IC work (vs. pure people-management / architecture)."""
    current_title = (profile.get("current_title") or "").lower()
    mgmt_markers = ("manager", "director", "head of", "vp ", "architect")
    if any(m in current_title for m in mgmt_markers):
        return 0.6
    return 1.0


def location_score(location: str, country: str, willing_to_relocate: bool) -> tuple[float, str]:
    if country == "India":
        loc_lower = (location or "").lower()
        for key, val in LOCATION_TIERS.items():
            if key in loc_lower:
                return val, f"india_{key}"
        return OTHER_INDIA_SCORE, "india_other_city"
    # outside India: JD is case-by-case, no visa sponsorship. Relocation
    # willingness matters a lot here since there's no remote option implied.
    base = NON_INDIA_NO_VISA_SCORE
    if willing_to_relocate:
        base = min(1.0, base + 0.25)
    return base, "outside_india"


def notice_period_score(notice_days: int) -> float:
    """Smooth penalty; JD: sub-30 ideal, buyout up to 30 possible, 30+ 'bar gets
    higher' but still in scope."""
    if notice_days <= NOTICE_IDEAL_DAYS:
        return 1.0
    # gentle decay from 30 to 180 days
    return max(0.5, 1.0 - 0.5 * (notice_days - NOTICE_IDEAL_DAYS) / (180 - NOTICE_IDEAL_DAYS))


def availability_modifier(signals: dict) -> tuple[float, dict]:
    """
    The JD's explicit instruction: a perfect-on-paper candidate who hasn't logged
    in for 6 months and has a 5% recruiter response rate is, for hiring purposes,
    not actually available -- down-weight appropriately. This is a MULTIPLIER on
    top of the fit score, not an additive feature, by design: it should not be
    able to rescue a bad fit, but it should meaningfully discount an unreachable
    good fit.
    """
    last_active = _parse_date(signals.get("last_active_date"))
    days_inactive = (REFERENCE_DATE - last_active).days if last_active else 365

    recency = 1.0
    if days_inactive > 180:
        recency = 0.5
    elif days_inactive > 90:
        recency = 0.75
    elif days_inactive > 30:
        recency = 0.9

    response_rate = signals.get("recruiter_response_rate", 0.0) or 0.0
    open_flag = signals.get("open_to_work_flag", False)
    interview_completion = signals.get("interview_completion_rate", 1.0)
    if interview_completion is None:
        interview_completion = 1.0

    open_factor = 1.0 if open_flag else 0.85  # not flagged isn't disqualifying, just a soft signal
    response_factor = 0.6 + 0.4 * response_rate          # 0.6 floor at rate=0, 1.0 at rate=1
    interview_factor = 0.7 + 0.3 * interview_completion   # 0.7 floor, 1.0 at full completion

    multiplier = recency * open_factor * response_factor * interview_factor
    detail = {
        "days_inactive": days_inactive,
        "recruiter_response_rate": response_rate,
        "open_to_work_flag": open_flag,
        "interview_completion_rate": interview_completion,
        "multiplier": round(multiplier, 3),
    }
    return multiplier, detail


def skill_assessment_credibility(skills: list[dict], signals: dict) -> float:
    """
    Cross-checks self-reported proficiency against Redrob's own platform
    assessment scores where available. A candidate claiming 'expert'/'advanced'
    on a skill the platform assessed at <30/100 is over-claiming; we apply a
    mild credibility discount proportional to how many such mismatches exist.
    This is intentionally gentle -- assessments only exist for a few skills
    per candidate, so we don't want one low score to dominate.
    """
    assess = signals.get("skill_assessment_scores") or {}
    if not assess:
        return 1.0
    skill_prof = {s["name"]: s.get("proficiency") for s in skills}
    mismatches = 0
    checked = 0
    for name, score in assess.items():
        prof = skill_prof.get(name)
        if prof is None:
            continue
        checked += 1
        if prof in ("expert", "advanced") and score < 30:
            mismatches += 1
    if checked == 0:
        return 1.0
    mismatch_ratio = mismatches / checked
    return 1.0 - 0.3 * mismatch_ratio  # at most a 30% credibility haircut


def honeypot_flags(skills: list[dict]) -> tuple[bool, int]:
    """
    Detects the dataset's confirmed honeypot fingerprint: 'expert' proficiency
    paired with 0 duration_months. Verified against the released candidate pool:
    this pattern occurs in exactly 21 candidates, all with occupations completely
    unrelated to AI/ML engineering, and never occurs as an isolated 1-2 count --
    it's a clean bimodal signal (0 or 3+), strongly suggesting deliberate
    construction rather than natural data noise.
    """
    n = sum(1 for s in skills if s.get("proficiency") == "expert" and (s.get("duration_months", 0) or 0) == 0)
    return n >= 2, n


def experience_consistency_penalty(profile: dict, career_history: list[dict]) -> float:
    """Soft penalty when stated years_of_experience is wildly inconsistent with
    the sum of career_history durations (career_history is capped at 10 entries
    per schema, so some gap is expected for long careers -- only penalize large,
    suspicious gaps)."""
    yoe_months = (profile.get("years_of_experience") or 0) * 12
    hist_months = sum(j.get("duration_months", 0) or 0 for j in career_history)
    if yoe_months <= 0:
        return 1.0
    gap_ratio = abs(hist_months - yoe_months) / yoe_months
    if gap_ratio > 1.5:
        return 0.85
    return 1.0
