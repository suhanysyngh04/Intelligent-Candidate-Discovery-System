"""
Structured representation of the target Job Description.

This is NOT the raw JD text — it's our interpretation of what the JD means,
written down explicitly so the scoring logic is auditable and so we are not
silently re-deriving "what the JD wants" via a black-box prompt at ranking time.

Source: job_description.docx (Senior AI Engineer - Founding Team, Redrob AI)

Every field below is justified by a specific passage in the JD. See docs/jd_analysis.md
for the full reasoning trail from JD text -> structured field.
"""

# Free-text JD content used ONLY for semantic similarity (TF-IDF) against candidate
# narratives. Distilled to focus the vector on substance, not boilerplate.
JD_TEXT = """
Senior AI Engineer, founding AI engineering team at a Series A AI-native talent
intelligence platform. Owns the intelligence layer: ranking, retrieval, and matching
systems that decide what recruiters see when they search for candidates.

Core mandate: build and ship a v2 ranking system improving on an existing BM25 plus
rule-based scorer. Work involves embeddings, hybrid retrieval, dense and sparse search,
and LLM-based re-ranking. Sets up evaluation infrastructure: offline benchmarks,
NDCG, MRR, MAP, online A/B testing, recruiter feedback loops.

Required: production experience with embeddings-based retrieval systems deployed to
real users -- sentence-transformers, OpenAI embeddings, BGE, E5 or similar -- including
handling embedding drift, index refresh, retrieval quality regression in production.
Production experience with vector databases or hybrid search infrastructure --
Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS or similar.
Strong Python and code quality. Hands-on experience designing evaluation frameworks
for ranking systems -- NDCG, MRR, MAP, offline-to-online correlation, A/B test
interpretation.

Nice to have: LLM fine-tuning (LoRA, QLoRA, PEFT), learning-to-rank models
(XGBoost-based or neural), prior HR-tech or recruiting-tech or marketplace exposure,
distributed systems or large-scale inference optimization, open-source contributions.

Ideal candidate has shipped at least one end-to-end ranking, search, or recommendation
system to real users at meaningful scale, at a product company (not pure services).
Has opinions about hybrid vs dense retrieval, offline vs online evaluation, when to
fine-tune versus prompt an LLM, grounded in systems they actually built. Comfortable
being a hands-on senior IC who writes production code, not someone who has moved
purely into architecture or tech-lead work with no recent code. Scrappy
product-engineering attitude: willing to ship a working v1 even if the underlying ML
is imperfect, learns from real users, then iterates.
""".strip()

# ---------------------------------------------------------------------------
# Structured requirements (drives rule-based feature scoring)
# ---------------------------------------------------------------------------

# Experience: JD frames 5-9 years as "a range, not a requirement" -- candidates
# outside the band are explicitly still in scope if other signals are strong.
# We use a soft penalty curve, not a hard filter.
EXPERIENCE_BAND = (5.0, 9.0)
EXPERIENCE_SOFT_MIN = 3.0   # below this, steep penalty (pure-research-fresh-grad territory)
EXPERIENCE_SOFT_MAX = 13.0  # above this, mild penalty (likely over-senior / architecture-only risk)

# Titles that map directly onto "applied ML / AI engineering at a product company".
# Tiered because a "Staff"/"Lead"/"Senior" title in this space is a stronger signal
# than a plain "ML Engineer" -- but ALL of these are in-domain.
CORE_AI_TITLES = {
    "Senior AI Engineer": 1.00,
    "Lead AI Engineer": 1.00,
    "Staff Machine Learning Engineer": 1.00,
    "Senior Applied Scientist": 0.97,
    "Senior Machine Learning Engineer": 0.97,
    "Senior NLP Engineer": 0.95,
    "Senior Data Scientist": 0.90,
    "Machine Learning Engineer": 0.85,
    "AI Engineer": 0.85,
    "Applied ML Engineer": 0.85,
    "NLP Engineer": 0.83,
    "Recommendation Systems Engineer": 0.90,  # exactly the JD's "ranking/retrieval" mandate
    "Search Engineer": 0.90,                  # exactly the JD's "search" mandate
    "AI Research Engineer": 0.70,             # JD is wary of pure-research framing
    "Data Scientist": 0.65,
    "Computer Vision Engineer": 0.55,         # JD explicitly de-prioritizes CV-only without NLP/IR
    "AI Specialist": 0.55,
    "Junior ML Engineer": 0.45,               # junior framing cuts against "senior IC" need
    "Senior Software Engineer (ML)": 0.80,
}

# Adjacent titles: not AI-titled, but the JD explicitly says title isn't the point --
# career narrative is. These get a smaller baseline bump; the real signal comes from
# the semantic-similarity score and the career-history skill scan, not from this dict.
ADJACENT_TITLES = {
    "Senior Software Engineer": 0.35,
    "Senior Data Engineer": 0.30,
    "Software Engineer": 0.25,
    "Data Engineer": 0.25,
    "Data Analyst": 0.15,
    "Analytics Engineer": 0.20,
    "Backend Engineer": 0.20,
    "DevOps Engineer": 0.10,
    "Cloud Engineer": 0.10,
    "Full Stack Developer": 0.12,
}

# Skills the JD calls out by name as "things you absolutely need" (retrieval/embeddings/
# vector-db side) and "things we'd like" (fine-tuning, LTR). Weighted by how central
# the JD makes them, not by how rare they are in the dataset.
MUST_HAVE_SKILLS = {
    # embeddings-based retrieval
    "Embeddings": 1.0, "Sentence Transformers": 1.0, "Vector Search": 1.0,
    "Semantic Search": 0.9, "Hugging Face Transformers": 0.7,
    # vector DBs / hybrid search infra
    "FAISS": 1.0, "Pinecone": 1.0, "Qdrant": 1.0, "Milvus": 1.0,
    "Weaviate": 1.0, "OpenSearch": 1.0, "Elasticsearch": 0.9, "pgvector": 0.8,
    # retrieval / IR / ranking
    "Information Retrieval": 1.0, "BM25": 0.9, "Learning to Rank": 1.0,
    "Recommendation Systems": 0.9, "RAG": 0.8,
    # LLM
    "LLMs": 0.8, "Prompt Engineering": 0.5, "Fine-tuning LLMs": 0.7,
    "LoRA": 0.6, "QLoRA": 0.6, "PEFT": 0.6, "LangChain": 0.3,  # JD is wary of LangChain-only profiles
    # core
    "Python": 0.6, "Machine Learning": 0.5, "Deep Learning": 0.5,
    "NLP": 0.6, "MLOps": 0.4, "Statistical Modeling": 0.3,
    # rare hand-placed "great fit" skills observed in the data
    "Information Retrieval Systems": 1.0, "Search Backend": 1.0, "Text Encoders": 0.9,
    "Vector Representations": 0.9, "Content Matching": 0.8, "Model Adaptation": 0.7,
    "Ranking Systems": 1.0, "Search & Discovery": 1.0, "Search Infrastructure": 1.0,
    "Indexing Algorithms": 0.9, "Natural Language Processing": 0.6,
}

# Skills that signal "framework tourist" / keyword-stuffer if they show up WITHOUT
# the substantive retrieval/IR skills above. Not penalized on their own -- penalized
# only via the keyword-stuffing ratio check in scoring.py.
SURFACE_AI_SKILLS = {
    "LangChain", "Prompt Engineering", "Diffusion Models", "GANs", "YOLO",
    "Object Detection", "Image Classification", "Speech Recognition", "TTS", "ASR",
    "Computer Vision", "OpenCV", "CNN", "Reinforcement Learning",
}

# Pure CV/speech/robotics skills with no NLP/IR -- JD explicitly de-prioritizes
# candidates whose primary expertise is here without NLP/IR exposure.
CV_SPEECH_ROBOTICS_SKILLS = {
    "Computer Vision", "OpenCV", "CNN", "Image Classification", "Object Detection",
    "YOLO", "GANs", "Diffusion Models", "Speech Recognition", "TTS", "ASR",
}
NLP_IR_SKILLS = {
    "NLP", "Natural Language Processing", "Information Retrieval",
    "Information Retrieval Systems", "Embeddings", "Sentence Transformers",
    "Semantic Search", "Vector Search", "RAG", "LLMs", "Hugging Face Transformers",
    "BM25", "Learning to Rank", "Recommendation Systems", "Text Encoders",
    "Search Backend", "Search & Discovery", "Search Infrastructure",
}

# Companies whose `current_industry` is IT-Services / pure consulting. The JD
# disqualifies candidates whose ENTIRE career has been at firms like these,
# but explicitly carves out an exception for people currently there who have
# prior product-company experience.
CONSULTING_INDUSTRIES = {"IT Services", "Consulting"}

# Industries that count as "product company" experience for the JD's framing.
PRODUCT_INDUSTRIES = {
    "Software", "Fintech", "Food Delivery", "E-commerce", "EdTech", "SaaS",
    "AI/ML", "AdTech", "Transportation", "Insurance Tech", "Gaming", "HealthTech",
    "HealthTech AI", "Conversational AI", "AI Services", "Voice AI", "Internet",
    "Media", "Consumer Electronics",
}

# JD location framing: Pune/Noida preferred, but Hyderabad/Pune/Mumbai/Delhi-NCR
# explicitly welcomed; broader India + relocation-willing is in scope; outside
# India is "case-by-case" with no visa sponsorship (so heavily down-weighted,
# not disqualified outright -- the JD doesn't say "rejected").
LOCATION_TIERS = {
    # (city fragment match, case-insensitive) -> score
    "pune": 1.00,
    "noida": 1.00,
    "hyderabad": 0.90,
    "mumbai": 0.90,
    "delhi": 0.90,
    "gurgaon": 0.85,   # Delhi NCR
    "bangalore": 0.80,  # Tier-1, not named but clearly "Tier-1 Indian city"
}
OTHER_INDIA_SCORE = 0.65       # any other Indian city: Tier-1-ish framing, JD open to relocation
NON_INDIA_NO_VISA_SCORE = 0.20  # JD: "case-by-case... we don't sponsor work visas"

# Notice period framing: JD wants sub-30-day, can buy out up to 30 days, 30+ is
# "still in scope but the bar gets higher" -- so smooth penalty, not a hard cut.
NOTICE_IDEAL_DAYS = 30

# Title-chaser detection: 1.5-year-or-less average tenure across a multi-employer
# career, especially when paired with monotonically increasing seniority titles.
TITLE_CHASER_AVG_TENURE_MONTHS = 15
