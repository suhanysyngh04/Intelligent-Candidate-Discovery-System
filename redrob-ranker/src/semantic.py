"""
Lexical-semantic similarity between the JD and each candidate's narrative text.

We use TF-IDF + cosine similarity rather than a neural embedding model. This is
a deliberate architecture choice given the competition's compute constraints
(CPU-only, no network, 5-minute wall clock, 100K candidates):

  - A neural sentence embedding model would need to be either downloaded at
    ranking time (violates "no network during ranking") or shipped as a
    multi-hundred-MB artifact in the repo, then loaded and run for 100K
    candidates on CPU within the time budget -- risky for Stage 3 reproduction
    on an unknown sandbox.
  - TF-IDF + cosine over the full 100K pool fits and scores in ~15 seconds on a
    single CPU core (benchmarked during development), using only scikit-learn,
    which is a near-universal dependency with no model weights to manage.
  - On a fixed, mostly-templated synthetic vocabulary like this dataset's
    (133 distinct skill tags, 47 distinct titles, formulaic description
    sentences), TF-IDF captures the relevant signal well: candidates whose
    narrative text shares vocabulary with the JD's retrieval/ranking/embeddings
    framing score higher, including candidates who never use the JD's exact
    skill-tag nouns but describe the same work in different words (bigrams help
    here, e.g. "ranking model", "offline online", "click through").

This is intentionally the SECOND signal in the composite score, not the only
one -- see scoring.py. Pure lexical similarity is exactly the "keyword search"
behavior the JD challenge directly tells us to avoid if used alone; here it is
one input among many structured features, and the keyword-stuffing penalty in
features.py specifically guards against gaming it.
"""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def build_candidate_text(candidate: dict) -> str:
    """Concatenate the narrative fields we want the JD compared against:
    headline, summary, and all career_history descriptions. Skills are
    deliberately EXCLUDED here -- skill-tag matching is handled by the
    structured skills_match_score in features.py, so we don't double-count
    keyword presence inside the similarity score too (that would just
    recreate the keyword-stuffing trap inside our own "semantic" layer)."""
    profile = candidate.get("profile", {})
    parts = [
        profile.get("headline", "") or "",
        profile.get("summary", "") or "",
    ]
    for job in candidate.get("career_history", []):
        parts.append(job.get("title", "") or "")
        parts.append(job.get("description", "") or "")
    return " ".join(parts)


class SemanticSimilarityScorer:
    def __init__(self, jd_text: str):
        self.jd_text = jd_text
        self.vectorizer = TfidfVectorizer(
            max_features=40000,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )
        self._fitted = False
        self._candidate_matrix = None
        self._candidate_ids: list[str] = []

    def fit(self, candidates: list[dict]):
        texts = [build_candidate_text(c) for c in candidates]
        self._candidate_ids = [c["candidate_id"] for c in candidates]
        # Fit on candidate corpus + JD text together so JD vocabulary that
        # doesn't appear often in candidate text still gets a sane idf weight.
        self._candidate_matrix = self.vectorizer.fit_transform(texts + [self.jd_text])
        # Drop the JD row after fitting; keep only candidate vectors.
        self._candidate_matrix = self._candidate_matrix[:-1]
        self._fitted = True
        return self

    def scores(self) -> dict[str, float]:
        """Returns candidate_id -> raw cosine similarity in [0, 1] against the JD."""
        if not self._fitted:
            raise RuntimeError("call fit() first")
        jd_vec = self.vectorizer.transform([self.jd_text])
        sims = (self._candidate_matrix @ jd_vec.T).toarray().ravel()
        # TF-IDF vectors are L2-normalized by default in scikit-learn, so the
        # dot product IS the cosine similarity already; clip for safety.
        sims = np.clip(sims, 0.0, 1.0)
        # Min-max normalize across the pool so the score uses the full 0-1
        # range regardless of absolute similarity magnitudes (which depend on
        # vocabulary overlap and shrink as corpus size grows).
        if sims.max() > sims.min():
            sims = (sims - sims.min()) / (sims.max() - sims.min())
        return dict(zip(self._candidate_ids, sims.tolist()))
