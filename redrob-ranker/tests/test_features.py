"""
Unit tests for the ranking pipeline's feature extraction and scoring logic.

Run with: python -m pytest tests/ -v
(or: python tests/test_features.py for a plain run without pytest)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import features as feat
import scoring


def make_candidate(**overrides):
    """Minimal valid candidate dict for testing, with sensible defaults."""
    base = {
        "candidate_id": "CAND_0000000",
        "profile": {
            "anonymized_name": "Test Candidate",
            "headline": "Test headline",
            "summary": "Test summary",
            "location": "Pune, Maharashtra",
            "country": "India",
            "years_of_experience": 6.0,
            "current_title": "Senior AI Engineer",
            "current_company": "TestCo",
            "current_company_size": "201-500",
            "current_industry": "Software",
        },
        "career_history": [
            {
                "company": "TestCo", "title": "Senior AI Engineer",
                "start_date": "2023-01-01", "end_date": None, "duration_months": 36,
                "is_current": True, "industry": "Software", "company_size": "201-500",
                "description": "Built and shipped ranking models for our retrieval pipeline.",
            }
        ],
        "education": [],
        "skills": [],
        "certifications": [],
        "languages": [],
        "redrob_signals": {
            "profile_completeness_score": 90, "signup_date": "2026-01-01",
            "last_active_date": "2026-06-20", "open_to_work_flag": True,
            "profile_views_received_30d": 10, "applications_submitted_30d": 1,
            "recruiter_response_rate": 0.8, "avg_response_time_hours": 24,
            "skill_assessment_scores": {}, "connection_count": 100,
            "endorsements_received": 10, "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 30, "max": 50},
            "preferred_work_mode": "hybrid", "willing_to_relocate": True,
            "github_activity_score": 50, "search_appearance_30d": 100,
            "saved_by_recruiters_30d": 5, "interview_completion_rate": 0.9,
            "offer_acceptance_rate": 0.5, "verified_email": True,
            "verified_phone": True, "linkedin_connected": True,
        },
    }
    for key, val in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            base[key] = {**base[key], **val}
        else:
            base[key] = val
    return base


def test_honeypot_detection_fires_on_expert_zero_duration():
    skills = [
        {"name": "MLflow", "proficiency": "expert", "endorsements": 2, "duration_months": 0},
        {"name": "Photoshop", "proficiency": "expert", "endorsements": 2, "duration_months": 0},
        {"name": "Docker", "proficiency": "beginner", "endorsements": 0, "duration_months": 5},
    ]
    is_honeypot, count = feat.honeypot_flags(skills)
    assert is_honeypot is True
    assert count == 2


def test_honeypot_detection_does_not_fire_on_legitimate_profile():
    skills = [
        {"name": "Python", "proficiency": "expert", "endorsements": 50, "duration_months": 60},
        {"name": "FAISS", "proficiency": "advanced", "endorsements": 10, "duration_months": 24},
    ]
    is_honeypot, count = feat.honeypot_flags(skills)
    assert is_honeypot is False


def test_honeypot_single_occurrence_does_not_trigger():
    # A single expert+0duration skill could plausibly be a data-entry quirk;
    # the dataset's real honeypot fingerprint is always >=3 in practice, so we
    # require >=2 to fire, giving a small safety margin against false positives.
    skills = [{"name": "MLflow", "proficiency": "expert", "endorsements": 2, "duration_months": 0}]
    is_honeypot, count = feat.honeypot_flags(skills)
    assert is_honeypot is False


def test_title_score_core_ai_title_high():
    score, tier = feat.title_score("Senior AI Engineer")
    assert score >= 0.9
    assert tier == "core_ai_title"


def test_title_score_unrelated_title_low():
    score, tier = feat.title_score("Accountant")
    assert score < 0.2
    assert tier == "unrelated_title"


def test_experience_fit_inside_band_is_max():
    assert feat.experience_fit_score(6.0) == 1.0
    assert feat.experience_fit_score(5.0) == 1.0
    assert feat.experience_fit_score(9.0) == 1.0


def test_experience_fit_decays_outside_band():
    assert feat.experience_fit_score(1.0) < feat.experience_fit_score(4.0) < 1.0
    assert feat.experience_fit_score(16.0) < 1.0


def test_consulting_only_career_penalized():
    profile = {"current_industry": "IT Services"}
    history = [
        {"industry": "IT Services"},
        {"industry": "Consulting"},
    ]
    mult, tag = feat.product_company_score(profile, history)
    assert mult < 0.5
    assert tag == "entire_career_consulting_only"


def test_consulting_now_with_product_history_not_penalized_hard():
    profile = {"current_industry": "IT Services"}
    history = [
        {"industry": "IT Services"},
        {"industry": "Fintech"},  # prior product-company experience
    ]
    mult, tag = feat.product_company_score(profile, history)
    assert mult > 0.9
    assert tag == "consulting_now_product_before"


def test_keyword_stuffing_detected():
    # All five names below are in SURFACE_AI_SKILLS (flashy CV/speech tags) and
    # none are in MUST_HAVE_SKILLS, with short duration_months -- the exact
    # "many AI keyword tags, little substance" pattern the JD warns about.
    skills = [
        {"name": s, "proficiency": "advanced", "endorsements": 5, "duration_months": 3}
        for s in ["GANs", "TTS", "Speech Recognition", "Image Classification", "YOLO"]
    ]
    mult, flagged = feat.keyword_stuffing_penalty(skills)
    assert flagged is True
    assert mult < 1.0


def test_keyword_stuffing_not_triggered_with_substance():
    skills = [
        {"name": "FAISS", "proficiency": "expert", "endorsements": 20, "duration_months": 40},
        {"name": "Information Retrieval", "proficiency": "expert", "endorsements": 15, "duration_months": 50},
        {"name": "Embeddings", "proficiency": "advanced", "endorsements": 10, "duration_months": 30},
    ]
    mult, flagged = feat.keyword_stuffing_penalty(skills)
    assert flagged is False
    assert mult == 1.0


def test_location_score_pune_and_noida_highest():
    score_pune, _ = feat.location_score("Pune, Maharashtra", "India", False)
    score_other, _ = feat.location_score("Bhubaneswar, Odisha", "India", False)
    assert score_pune > score_other


def test_location_score_outside_india_lower_than_india():
    score_india, _ = feat.location_score("Pune, Maharashtra", "India", False)
    score_abroad, _ = feat.location_score("Toronto", "Canada", False)
    assert score_india > score_abroad


def test_availability_modifier_penalizes_inactive_candidate():
    active_signals = {"last_active_date": "2026-06-26", "recruiter_response_rate": 0.9,
                       "open_to_work_flag": True, "interview_completion_rate": 1.0}
    inactive_signals = {"last_active_date": "2025-01-01", "recruiter_response_rate": 0.05,
                         "open_to_work_flag": False, "interview_completion_rate": 0.1}
    mult_active, _ = feat.availability_modifier(active_signals)
    mult_inactive, _ = feat.availability_modifier(inactive_signals)
    assert mult_active > mult_inactive
    assert mult_active > 0.9
    assert mult_inactive < 0.5


def test_score_candidate_end_to_end_strong_profile_scores_high():
    candidate = make_candidate(skills=[
        {"name": "FAISS", "proficiency": "expert", "endorsements": 30, "duration_months": 40},
        {"name": "Information Retrieval", "proficiency": "expert", "endorsements": 20, "duration_months": 50},
        {"name": "Pinecone", "proficiency": "advanced", "endorsements": 15, "duration_months": 24},
    ])
    result = scoring.score_candidate(candidate, semantic_score=0.8)
    assert result["is_honeypot"] is False
    assert result["score"] > 0.5


def test_score_candidate_honeypot_forced_to_bottom():
    candidate = make_candidate(
        profile={"current_title": "Accountant", "current_industry": "IT Services"},
        skills=[
            {"name": "MLflow", "proficiency": "expert", "endorsements": 2, "duration_months": 0},
            {"name": "Photoshop", "proficiency": "expert", "endorsements": 2, "duration_months": 0},
        ],
    )
    result = scoring.score_candidate(candidate, semantic_score=0.9)
    assert result["is_honeypot"] is True
    assert result["score"] < 0.01


def test_score_candidate_irrelevant_profile_scores_low():
    candidate = make_candidate(
        profile={"current_title": "Accountant", "current_industry": "IT Services",
                 "current_company": "Infosys"},
        career_history=[{
            "company": "Infosys", "title": "Accountant",
            "start_date": "2020-01-01", "end_date": None, "duration_months": 60,
            "is_current": True, "industry": "IT Services", "company_size": "10001+",
            "description": "Managed accounts payable and receivable processes.",
        }],
        skills=[{"name": "Excel", "proficiency": "expert", "endorsements": 5, "duration_months": 60}],
    )
    result = scoring.score_candidate(candidate, semantic_score=0.05)
    assert result["score"] < 0.3


def _run_all_tests():
    test_fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed, failed = 0, 0
    for fn in test_fns:
        try:
            fn()
            passed += 1
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{passed} passed, {failed} failed out of {len(test_fns)} tests")
    return failed == 0


if __name__ == "__main__":
    ok = _run_all_tests()
    sys.exit(0 if ok else 1)
