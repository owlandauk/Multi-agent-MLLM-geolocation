import os

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_PATH = os.environ.get("MODEL_PATH", "/cvhci/temp/szuo/models/qwen2.5-vl-7b")
DEVICE = "cuda"

# ── Dataset ────────────────────────────────────────────────────────────────────
YFCC4K_IMG_DIR  = os.environ.get("YFCC4K_IMG_DIR",  "/cvhci/temp/szuo/yfcc4k/yfcc4k")
YFCC4K_GPS_CSV  = os.environ.get("YFCC4K_GPS_CSV",  "/cvhci/temp/szuo/yfcc4k/yfcc4k_gps.csv")
RESULTS_DIR     = os.environ.get("RESULTS_DIR",     "/cvhci/temp/szuo/geo_results")

# ── GeoBayes hyperparams (defaults kept stable; env overrides for ablations) ──
PRIOR_TEMP      = float(os.environ.get("PRIOR_TEMP", "1.5"))    # T  in Eq.5
PRIOR_CUTOFF    = float(os.environ.get("PRIOR_CUTOFF", "0.6"))  # τp in Eq.5
TRANSITION_THR  = float(os.environ.get("TRANSITION_THR", "0.55"))
ENHANCE_THR     = 0.05   # τ_enhance  (ΔP threshold)
BETA            = 0.693  # ln2

# ── Hierarchical diagnostics/control ──────────────────────────────────────────
# A level is considered stable for descent when either the top posterior is
# strong, or it clears the transition threshold with enough top1-top2 margin / low
# normalized entropy. This prevents flat country posteriors from blindly driving
# city/street prompts.
STRONG_POSTERIOR_THR = 0.60
STABLE_MARGIN_THR    = 0.04
STABLE_ENTROPY_THR   = 0.98
GUARDED_DESCENT_THR  = 0.40
COUNTRY_REPLACE_TOP_THR    = 0.45
COUNTRY_REPLACE_MARGIN_THR = 0.02
COUNTRY_REPLACE_ATTEMPTS = int(os.environ.get("COUNTRY_REPLACE_ATTEMPTS", "0"))
COUNTRY_CUE_ENSEMBLE = os.environ.get("COUNTRY_CUE_ENSEMBLE", "0").lower() in {
    "1", "true", "yes", "on"
}
BALANCED_COUNTRY_GUARD = os.environ.get("BALANCED_COUNTRY_GUARD", "0").lower() in {
    "1", "true", "yes", "on"
}
COUNTRY_GEOREASONER_SEED = os.environ.get("COUNTRY_GEOREASONER_SEED", "0").lower() in {
    "1", "true", "yes", "on"
}
GEOREASONER_COUNTRY_BOOST = float(os.environ.get("GEOREASONER_COUNTRY_BOOST", "0.2"))
GEOREASONER_REQUIRE_DIRECT_CLUE = os.environ.get(
    "GEOREASONER_REQUIRE_DIRECT_CLUE", "1"
).lower() not in {"0", "false", "no", "off"}
CITY_COUNTRY_FACTCHECK = os.environ.get("CITY_COUNTRY_FACTCHECK", "0").lower() in {
    "1", "true", "yes", "on"
}
CITY_COUNTRY_FACTCHECK_MIN_COUNTRY_TOP = float(
    os.environ.get("CITY_COUNTRY_FACTCHECK_MIN_COUNTRY_TOP", "0.55")
)
CHILD_BACKTRACK_PROMOTE = os.environ.get("CHILD_BACKTRACK_PROMOTE", "0").lower() in {
    "1", "true", "yes", "on"
}
CHILD_BACKTRACK_MAX_COUNTRY_TOP = float(
    os.environ.get("CHILD_BACKTRACK_MAX_COUNTRY_TOP", "0.55")
)
CHILD_BACKTRACK_MIN_CHILD_TOP = float(
    os.environ.get("CHILD_BACKTRACK_MIN_CHILD_TOP", "0.50")
)

# ── Retrieval / GeoCLIP prior ────────────────────────────────────────────────
# Optional external geo prior. A retrieval backend can write top-k similar-image
# country evidence to JSON/JSONL/CSV; the pipeline softly mixes it into the
# country prior before normal SL/DST/POMDP verification.
RETRIEVAL_PRIOR_ENABLED = os.environ.get("RETRIEVAL_PRIOR_ENABLED", "0").lower() in {
    "1", "true", "yes", "on"
}
RETRIEVAL_PRIOR_PATH = os.environ.get("RETRIEVAL_PRIOR_PATH", "")
RETRIEVAL_PRIOR_WEIGHT = float(os.environ.get("RETRIEVAL_PRIOR_WEIGHT", "0.25"))
RETRIEVAL_PRIOR_ADAPTIVE = os.environ.get("RETRIEVAL_PRIOR_ADAPTIVE", "0").lower() in {
    "1", "true", "yes", "on"
}
RETRIEVAL_PRIOR_SAME_CONTINENT_WEIGHT = float(
    os.environ.get("RETRIEVAL_PRIOR_SAME_CONTINENT_WEIGHT", "0.10")
)
RETRIEVAL_PRIOR_CROSS_CONTINENT_WEIGHT = float(
    os.environ.get("RETRIEVAL_PRIOR_CROSS_CONTINENT_WEIGHT", "0.05")
)
RETRIEVAL_COUNTRY_ANCHOR_ENABLED = os.environ.get(
    "RETRIEVAL_COUNTRY_ANCHOR_ENABLED", "0"
).lower() in {"1", "true", "yes", "on"}
RETRIEVAL_COUNTRY_ANCHOR_MAX_COUNTRY_TOP = float(
    os.environ.get("RETRIEVAL_COUNTRY_ANCHOR_MAX_COUNTRY_TOP", "0.55")
)
RETRIEVAL_COUNTRY_ANCHOR_MIN_PRIOR_TOP = float(
    os.environ.get("RETRIEVAL_COUNTRY_ANCHOR_MIN_PRIOR_TOP", "0.15")
)
RETRIEVAL_COUNTRY_ANCHOR_WEIGHT = float(
    os.environ.get("RETRIEVAL_COUNTRY_ANCHOR_WEIGHT", "1.0")
)
RETRIEVAL_VERIFY_ACTION_ENABLED = os.environ.get(
    "RETRIEVAL_VERIFY_ACTION_ENABLED", "0"
).lower() in {"1", "true", "yes", "on"}
RETRIEVAL_VERIFY_MIN_PRIOR_TOP = float(
    os.environ.get("RETRIEVAL_VERIFY_MIN_PRIOR_TOP", "0.55")
)
RETRIEVAL_VERIFY_RELATIONS = tuple(
    relation.strip()
    for relation in os.environ.get(
        "RETRIEVAL_VERIFY_RELATIONS",
        "same_continent_conflict,cross_continent_conflict",
    ).split(",")
    if relation.strip()
)
RETRIEVAL_VERIFY_TOP_K = int(os.environ.get("RETRIEVAL_VERIFY_TOP_K", "3"))

# ── Continent-first country calibration ──────────────────────────────────────
# The continent stage is a weak prior for country inference. It reduces obvious
# cross-continent country defaults without hard-blocking North America or child
# descent, avoiding the over-conservative v9 behavior.
# The standalone continent pass over-regularized country predictions on the
# CVHCI/YFCC checks. Keep it available for ablations, but default to the
# stronger v5-style country -> city -> street path.
ENABLE_CONTINENT_LEVEL = os.environ.get("ENABLE_CONTINENT_LEVEL", "0").lower() not in {
    "0", "false", "no", "off"
}
CONTINENT_REG_MIN_TOP = float(os.environ.get("CONTINENT_REG_MIN_TOP", "0.45"))
CONTINENT_REG_STRENGTH = float(os.environ.get("CONTINENT_REG_STRENGTH", "0.35"))
CONTINENT_REG_FLOOR = float(os.environ.get("CONTINENT_REG_FLOOR", "0.15"))

# Optional GeoBayes-style web evidence enhancement. Disabled by default because
# HPC compute nodes may not have outbound network access and web search can be slow.
WEB_SEARCH_TOP_THR = 0.60
WEB_SEARCH_MARGIN_THR = ENHANCE_THR
WEB_SEARCH_MAX_RESULTS = 3
WEB_SEARCH_TIMEOUT = 8
WEB_SEARCH_REQUIRE_ENTITY = True
WEB_SEARCH_LEVELS = tuple(
    level.strip()
    for level in os.environ.get("WEB_SEARCH_LEVELS", "country,city,street").split(",")
    if level.strip()
)
WEB_SEARCH_UPDATE_MODE = os.environ.get("WEB_SEARCH_UPDATE_MODE", "verify").lower()
WEB_SEARCH_ACCEPT_MODE = os.environ.get("WEB_SEARCH_ACCEPT_MODE", "any").lower()
WEB_SEARCH_VERIFY_MAX_NEW_TOKENS = int(os.environ.get("WEB_SEARCH_VERIFY_MAX_NEW_TOKENS", "160"))

# Optional GeoBayes-style ImageSearch enhancement. When enabled, Google Vision
# Web Detection extracts concrete web entities from the image first; Tavily is
# kept as a fallback for concrete text clues when image search is uninformative.
IMAGE_SEARCH_MAX_ENTITIES = int(os.environ.get("IMAGE_SEARCH_MAX_ENTITIES", "5"))
IMAGE_SEARCH_TIMEOUT = int(os.environ.get("IMAGE_SEARCH_TIMEOUT", "15"))
IMAGE_SEARCH_STRICT_TEXT_QUERY = os.environ.get(
    "IMAGE_SEARCH_STRICT_TEXT_QUERY", "1"
).lower() not in {"0", "false", "no", "off"}

# ── SL (single-source uncertainty) ────────────────────────────────────────────
SL_N_SAMPLES    = 5      # samples per hypothesis for uncertainty estimation in SLModule
SL_TEMPERATURE  = 0.8    # sampling temperature
SL_SUPPORT_ALPHA = float(os.environ.get("SL_SUPPORT_ALPHA", "0.7"))

# ── DST (Dempster-Shafer fusion) ───────────────────────────────────────────────
DST_CONFLICT_THR = 0.5   # K > this → treat as high-conflict, apply cautious rule

# ── POMDP ─────────────────────────────────────────────────────────────────────
POMDP_MAX_STEPS = 8      # full experiments
POMDP_GAMMA     = 0.95   # discount factor (used if computing cumulative reward)
POMDP_POLICY = os.environ.get("POMDP_POLICY", "llm").lower()
POMDP_EIG_LEVELS = tuple(
    level.strip()
    for level in os.environ.get("POMDP_EIG_LEVELS", "continent,country,city,street").split(",")
    if level.strip()
)
POMDP_TOP_HYPOTHESES = int(os.environ.get("POMDP_TOP_HYPOTHESES", "5"))
POMDP_MAX_ACTIONS = int(os.environ.get("POMDP_MAX_ACTIONS", "6"))
POMDP_OBS_SAMPLES = int(os.environ.get("POMDP_OBS_SAMPLES", "3"))
POMDP_OBS_SMOOTHING = float(os.environ.get("POMDP_OBS_SMOOTHING", "0.25"))
POMDP_ACTION_COST = float(os.environ.get("POMDP_ACTION_COST", "0.0"))
POMDP_REWARD_MODE = os.environ.get("POMDP_REWARD_MODE", "entropy").lower()

# ── Evaluation thresholds (km) ─────────────────────────────────────────────────
EVAL_THRESHOLDS = [1, 25, 200, 750, 2500]

# ── Generation ────────────────────────────────────────────────────────────────
MAX_NEW_TOKENS         = 384    # default cap (used by hypothesize)
SL_MAX_NEW_TOKENS      = 48     # SL responses are tiny: "Rating: X / Confidence: Y / one sentence"
VERIFY_MAX_NEW_TOKENS  = 160    # verify observation: a paragraph is plenty
VERIFY_SUPPORT_FORMAT = os.environ.get("VERIFY_SUPPORT_FORMAT", "0").lower() in {
    "1", "true", "yes", "on"
}
POMDP_MAX_NEW_TOKENS   = 64     # policy returns {"task_index": N, "reason": "..."}
FACTCHECK_MAX_NEW_TOKENS = 96   # text-only city/country consistency check

# ── Batch inference ────────────────────────────────────────────────────────────
MAX_SL_BATCH_SIZE = 8    # original prompts per SL batch; multiplied by SL_N_SAMPLES → actual GPU batch. Reduce if OOM.
