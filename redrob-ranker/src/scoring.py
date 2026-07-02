"""
Combines semantic similarity + structured features into one composite score
per candidate, and generates a human-readable reasoning string.

Architecture (see docs/methodology.md for the full writeup):

  composite = fit_score * availability_multiplier

  fit_score = weighted sum of:
    - title_score                  (0.20)
    - semantic_similarity          (0.20)   <- TF-IDF JD<->narrative
    - skills_match_score           (0.20)
    - career_narrative_signal      (0.15)   <- "plain language Tier 5" detector
    - experience_fit_score         (0.10)
    - product_company_score        (0.08)   <- consulting-only disqualifier lives here
    - location_score               (0.05)
    - notice_period_score          (0.02)

  then fit_score is further adjusted by penalty multipliers that are NOT part
  of the weighted sum, because they represent reasons to distrust the profile
  rather than reasons to like it more or less along a spectrum:
    - keyword_stuffing_penalty
    - cv_speech_without_nlp_penalty
    - title_chaser_penalty
    - recent_code_score
    - skill_assessment_credibility
    - experience_consistency_penalty
    - honeypot hard suppression

  availability_multiplier (behavioral signals) is applied LAST and multiplicatively,
  by design: it should discount an otherwise-great candidate's *visible rank*
  (because they're hard to actually hire right now) without ever being able to
  rescue a candidate who is a poor fit on substance. A 0.99 fit score with a 0.5
  availability multiplier still beats a 0.3 fit score with a 1.0 multiplier.
"""
from __future__ import annotations

from datetime import date

import features as feat

WEIGHTS = {
    "title": 0.20,
    "semantic": 0.20,
    "skills": 0.20,
    "narrative": 0.15,
    "experience": 0.10,
    "product_company": 0.08,
    "location": 0.05,
    "notice": 0.02,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

HONEYPOT_SUPPRESSION_SCORE = 0.001  # forces honeypots to the bottom, never top 100


def score_candidate(candidate: dict, semantic_score: float) -> dict:
    profile = candidate["profile"]
    career_history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})

    # ---- honeypot hard suppression -------------------------------------
    is_honeypot, honeypot_count = feat.honeypot_flags(skills)
    if is_honeypot:
        return {
            "candidate_id": candidate["candidate_id"],
            "score": HONEYPOT_SUPPRESSION_SCORE,
            "is_honeypot": True,
            "honeypot_signal_count": honeypot_count,
            "components": {},
            "reasoning_facts": {},
        }

    # ---- weighted fit components ----------------------------------------
    t_score, t_tier = feat.title_score(profile["current_title"])
    s_score, s_detail = feat.skills_match_score(skills)
    n_score, n_hits = feat.career_narrative_signal(career_history)
    e_score = feat.experience_fit_score(profile["years_of_experience"])
    pc_score, pc_tag = feat.product_company_score(profile, career_history)
    loc_score, loc_tag = feat.location_score(
        profile["location"], profile["country"], signals.get("willing_to_relocate", False)
    )
    notice_score = feat.notice_period_score(signals.get("notice_period_days", 30))

    fit_score = (
        WEIGHTS["title"] * t_score
        + WEIGHTS["semantic"] * semantic_score
        + WEIGHTS["skills"] * s_score
        + WEIGHTS["narrative"] * n_score
        + WEIGHTS["experience"] * e_score
        + WEIGHTS["product_company"] * pc_score
        + WEIGHTS["location"] * loc_score
        + WEIGHTS["notice"] * notice_score
    )

    # ---- multiplicative penalty/credibility adjustments -----------------
    stuffing_mult, is_stuffer = feat.keyword_stuffing_penalty(skills)
    cv_mult = feat.cv_speech_without_nlp_penalty(skills)
    chaser_mult, is_chaser = feat.title_chaser_penalty(career_history)
    code_mult = feat.recent_code_score(profile, career_history)
    credibility_mult = feat.skill_assessment_credibility(skills, signals)
    consistency_mult = feat.experience_consistency_penalty(profile, career_history)

    fit_score *= stuffing_mult * cv_mult * chaser_mult * code_mult * credibility_mult * consistency_mult
    fit_score = max(0.0, min(1.0, fit_score))

    # ---- availability multiplier (behavioral signals) -------------------
    avail_mult, avail_detail = feat.availability_modifier(signals)

    final_score = fit_score * avail_mult

    components = {
        "title_score": round(t_score, 3),
        "title_tier": t_tier,
        "semantic_score": round(semantic_score, 3),
        "skills_score": round(s_score, 3),
        "narrative_score": round(n_score, 3),
        "experience_score": round(e_score, 3),
        "product_company_score": round(pc_score, 3),
        "product_company_tag": pc_tag,
        "location_score": round(loc_score, 3),
        "notice_score": round(notice_score, 3),
        "fit_score_pre_penalty": None,  # filled below for transparency if needed
        "fit_score": round(fit_score, 4),
        "is_keyword_stuffer": is_stuffer,
        "is_title_chaser": is_chaser,
        "availability_multiplier": round(avail_mult, 3),
        "final_score": round(final_score, 4),
    }

    reasoning_facts = {
        "matched_skills": s_detail["matched_skills"],
        "narrative_hits": n_hits,
        "avail_detail": avail_detail,
        "loc_tag": loc_tag,
        "pc_tag": pc_tag,
        "is_stuffer": is_stuffer,
        "is_chaser": is_chaser,
    }

    return {
        "candidate_id": candidate["candidate_id"],
        "score": round(final_score, 4),
        "is_honeypot": False,
        "components": components,
        "reasoning_facts": reasoning_facts,
    }
