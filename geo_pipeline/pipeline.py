"""
Main pipeline: SL + DST + POMDP on YFCC4K.

Flow (one image):
  1. Hypothesize  — MLLM global analysis → hypothesis set H_0 + verification plan V_0
  2. Per level (continent → country → city → street):
       a. SL: score each pending evidence against each hypothesis (uncertainty-aware)
       b. DST: fuse all evidence BBAs into updated posterior
       c. POMDP: select next verification task (expected information gain or LLM policy)
       d. Repeat until POMDP stopping condition
       e. Continent posterior weakly regularizes country posterior; finer descent stays unconstrained
  3. Output MAP location → geocode → (lat, lon)
"""

from __future__ import annotations

import json
import re
import math
from PIL import Image

from models.mllm_client import MLLMClient
from modules.sl import SLModule
from modules.dst import DSTModule
from modules.pomdp import POMDPModule
from country_aliases import canonicalize_country, continent_of
from web_search import WebSearchClient, format_search_evidence
from config import (
    PRIOR_TEMP, PRIOR_CUTOFF, TRANSITION_THR, ENHANCE_THR,
    VERIFY_MAX_NEW_TOKENS, POMDP_MAX_NEW_TOKENS, VERIFY_SUPPORT_FORMAT,
    STRONG_POSTERIOR_THR, STABLE_MARGIN_THR, STABLE_ENTROPY_THR,
    GUARDED_DESCENT_THR, COUNTRY_REPLACE_TOP_THR,
    COUNTRY_REPLACE_MARGIN_THR, COUNTRY_REPLACE_ATTEMPTS, COUNTRY_CUE_ENSEMBLE,
    BALANCED_COUNTRY_GUARD,
    COUNTRY_GEOREASONER_SEED, GEOREASONER_COUNTRY_BOOST,
    GEOREASONER_REQUIRE_DIRECT_CLUE,
    CITY_COUNTRY_FACTCHECK, CITY_COUNTRY_FACTCHECK_MIN_COUNTRY_TOP,
    CHILD_BACKTRACK_PROMOTE, CHILD_BACKTRACK_MAX_COUNTRY_TOP,
    CHILD_BACKTRACK_MIN_CHILD_TOP,
    ENABLE_CONTINENT_LEVEL,
    CONTINENT_REG_MIN_TOP, CONTINENT_REG_STRENGTH, CONTINENT_REG_FLOOR,
    WEB_SEARCH_TOP_THR, WEB_SEARCH_MARGIN_THR, WEB_SEARCH_REQUIRE_ENTITY,
    WEB_SEARCH_LEVELS,
    FACTCHECK_MAX_NEW_TOKENS,
)

LEVELS = ["continent", "country", "city", "street"] if ENABLE_CONTINENT_LEVEL else [
    "country", "city", "street"
]

_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_SEARCH_ENTITY_RE = re.compile(
    r"\b(?:[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,}|[A-Z]{2,}|[A-Z]?\d[A-Z0-9 -]{2,})\b"
)

_COUNTRY_CUE_PROMPTS = [
    (
        "language_script_traffic",
        "Focus on visible language, scripts, road signs, license plates, lane markings, "
        "traffic direction, and public transport clues. Return likely countries only.",
    ),
    (
        "climate_landscape_hemisphere",
        "Focus on climate, vegetation, terrain, sunlight, seasons, hemisphere, and "
        "rural or coastal landscape clues. Return likely countries only.",
    ),
    (
        "architecture_urban_form",
        "Focus on architecture, urban layout, road furniture, utilities, building "
        "materials, and regional infrastructure style. Return likely countries only.",
    ),
]

_COUNTRY_FIXED_TASKS = [
    {"desc": "Check visible language, script, road signs, and storefront text", "bbox": None},
    {"desc": "Check road layout, traffic direction, lane markings, and license plates", "bbox": None},
    {"desc": "Check vegetation, climate, terrain, season, and hemisphere cues", "bbox": None},
    {"desc": "Check architecture, building materials, utilities, and street furniture", "bbox": None},
    {"desc": "Check landscape context such as coast, mountains, rural setting, or urban density", "bbox": None},
]


def _try_parse_json(text: str):
    """Parse the first JSON object/array from raw model text."""
    if isinstance(text, (dict, list)):
        return text
    if not isinstance(text, str):
        return None

    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    fenced = _FENCED_JSON_RE.search(stripped)
    if fenced:
        parsed = _try_parse_json(fenced.group(1))
        if parsed is not None:
            return parsed

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(stripped):
        if ch not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[idx:])
            return value
        except json.JSONDecodeError:
            continue
    return None


def _parse_hypothesis_payload(text: str) -> dict | None:
    """Normalize model/wrapper outputs to {hypotheses, verification_plan}."""
    parsed = _try_parse_json(text)
    if parsed is None:
        return None
    if isinstance(parsed, list):
        return {"hypotheses": parsed, "verification_plan": []}
    if not isinstance(parsed, dict):
        return None
    if "hypotheses" in parsed:
        return parsed

    hypotheses: list[dict] = []
    verification_plan: list[dict] = []

    general = _parse_hypothesis_payload(parsed.get("general", ""))
    if general:
        hypotheses.extend(general.get("hypotheses", []))
        verification_plan = general.get("verification_plan", []) or []

    for cue_text in parsed.get("cue_responses", []) or []:
        cue = _parse_hypothesis_payload(cue_text)
        if cue:
            hypotheses.extend(cue.get("hypotheses", []))

    if hypotheses:
        return {"hypotheses": hypotheses, "verification_plan": verification_plan}
    return None


def _softmax_prior(scores: dict[str, float]) -> dict[str, float]:
    """Eq.5 from GeoBayes: temperature-scaled softmax with score cutoff."""
    import math
    clipped = {h: min(s, PRIOR_CUTOFF) for h, s in scores.items()}
    exps = {h: math.exp(s / PRIOR_TEMP) for h, s in clipped.items()}
    total = sum(exps.values())
    return {h: v / total for h, v in exps.items()}


def _canonicalize_continent(raw: str) -> str | None:
    if not raw:
        return None
    low = raw.strip().lower()
    aliases = {
        "africa": "Africa",
        "asia": "Asia",
        "europe": "Europe",
        "north america": "North America",
        "northern america": "North America",
        "south america": "South America",
        "oceania": "Oceania",
        "australia/oceania": "Oceania",
        "australasia": "Oceania",
    }
    if low in aliases:
        return aliases[low]
    for alias, continent in aliases.items():
        if alias in low:
            return continent
    return None


def _renormalize_posterior(posterior: dict[str, float]) -> dict[str, float]:
    total = sum(float(v) for v in (posterior or {}).values())
    if total <= 0:
        return dict(posterior or {})
    return {k: float(v) / total for k, v in posterior.items()}


def _regularize_country_by_continent(
    country_posterior: dict[str, float],
    continent_posterior: dict[str, float],
) -> tuple[dict[str, float], bool]:
    if not country_posterior or not continent_posterior:
        return country_posterior, False
    if max(continent_posterior.values(), default=0.0) < CONTINENT_REG_MIN_TOP:
        return country_posterior, False

    adjusted = {}
    changed = False
    for country, prob in country_posterior.items():
        continent = continent_of(country)
        if not continent:
            adjusted[country] = prob
            continue
        continent_mass = max(
            CONTINENT_REG_FLOOR,
            float(continent_posterior.get(continent, 0.0)),
        )
        multiplier = (1.0 - CONTINENT_REG_STRENGTH) + CONTINENT_REG_STRENGTH * continent_mass
        adjusted[country] = float(prob) * multiplier
        changed = changed or abs(multiplier - 1.0) > 1e-9
    return _renormalize_posterior(adjusted), changed


def _collect_scores(hypotheses: list, level: str) -> dict[str, float]:
    """Collect {location: confidence} from parsed hypotheses.

    At the country level we canonicalize location names via the alias map
    before scoring, so "USA"/"California, USA"/"Southeast Asia" don't leak
    through as distinct entries (see full_v4 diagnosis: raw MLLM strings
    that didn't match a country killed 60% of records to Unknown). When
    multiple candidates map to the same canonical country, keep the max
    confidence — we don't want to double-count "USA" and "United States".
    Non-country levels keep the raw string.
    """
    scores: dict[str, float] = {}
    for h in hypotheses:
        loc = h.get("location")
        if not loc:
            continue
        conf = h.get("confidence", 0.5)
        if level == "continent":
            canon_continent = _canonicalize_continent(loc)
            if canon_continent is None:
                continue
            loc = canon_continent
        elif level == "country":
            canon = canonicalize_country(loc)
            if canon is None:
                continue  # drop non-country strings like "Southeast Asia"
            loc = canon
        scores[loc] = max(scores.get(loc, 0.0), conf)
    return scores


def _merge_country_scores(parsed_responses: list[dict | None], top_k: int = 8) -> dict[str, float]:
    """Merge country hypotheses from multiple focused country prompts."""
    max_scores: dict[str, float] = {}
    source_counts: dict[str, int] = {}
    for parsed in parsed_responses:
        if not parsed or "hypotheses" not in parsed:
            continue
        for country, score in _collect_scores(parsed["hypotheses"], "country").items():
            max_scores[country] = max(max_scores.get(country, 0.0), score)
            source_counts[country] = source_counts.get(country, 0) + 1
    merged = {
        country: min(score + 0.05 * (source_counts.get(country, 1) - 1), 0.95)
        for country, score in max_scores.items()
    }
    return dict(sorted(merged.items(), key=lambda kv: -kv[1])[:top_k])


def _prepend_country_tasks(plan: list[dict]) -> list[dict]:
    seen = set()
    out: list[dict] = []
    for task in [*_COUNTRY_FIXED_TASKS, *plan]:
        desc = task.get("desc")
        if not desc or desc in seen:
            continue
        seen.add(desc)
        out.append({"desc": desc, "bbox": task.get("bbox")})
    return out


def _format_top_candidates(posterior: dict[str, float], k: int = 3) -> str:
    items = sorted((posterior or {}).items(), key=lambda x: -x[1])[:k]
    return ", ".join(f"{loc} ({prob:.2f})" for loc, prob in items)


def _posterior_stats(posterior: dict[str, float]) -> dict[str, float]:
    """Return top mass, top1-top2 margin, and normalized entropy."""
    vals = sorted((float(v) for v in (posterior or {}).values()), reverse=True)
    if not vals:
        return {"top": 0.0, "margin": 0.0, "entropy": 1.0}
    top = vals[0]
    margin = top - (vals[1] if len(vals) > 1 else 0.0)
    if len(vals) <= 1:
        entropy = 0.0
    else:
        entropy = -sum(v * math.log(max(v, 1e-12)) for v in vals) / math.log(len(vals))
    return {"top": top, "margin": margin, "entropy": entropy}


def _stable_for_descent(posterior: dict[str, float]) -> bool:
    """Multi-signal gate for hierarchical descent.

    Top probability alone is not enough: v8 showed that flat country posteriors
    around 0.51 can still drive noisy city/street guesses. We descend only when
    the top candidate is strong, or when it clears the transition threshold with
    a meaningful margin or low normalized entropy.
    """
    stats = _posterior_stats(posterior)
    if stats["top"] >= STRONG_POSTERIOR_THR:
        return True
    if stats["top"] < TRANSITION_THR:
        return False
    return stats["margin"] >= STABLE_MARGIN_THR or stats["entropy"] <= STABLE_ENTROPY_THR


def _allow_guarded_descent(posterior: dict[str, float]) -> bool:
    """Allow child reasoning with conflict filtering when country is plausible."""
    return _posterior_stats(posterior)["top"] >= GUARDED_DESCENT_THR


def _descent_block_reason(posterior: dict[str, float]) -> str | None:
    return None


def _should_replace_country(posterior: dict[str, float]) -> bool:
    """Replace only when country belief is both weak and nearly tied."""
    stats = _posterior_stats(posterior)
    return (
        stats["top"] < COUNTRY_REPLACE_TOP_THR
        and stats["margin"] < COUNTRY_REPLACE_MARGIN_THR
    )


def _posterior_web_trigger(posterior: dict[str, float], visual_delta: float) -> bool:
    """Return whether posterior dynamics justify external evidence search."""
    stats = _posterior_stats(posterior)
    return (
        visual_delta < ENHANCE_THR
        and stats["top"] < WEB_SEARCH_TOP_THR
        and stats["margin"] < WEB_SEARCH_MARGIN_THR
    )


def _has_searchable_web_entity(evidence: list[str]) -> bool:
    """Only search when visual evidence contains a concrete searchable clue."""
    if not WEB_SEARCH_REQUIRE_ENTITY:
        return True
    recent = " ".join(_clean_search_clue(item) for item in evidence[-3:])
    low = recent.lower()
    if not low.strip():
        return False
    negative_markers = (
        "no visible landmark", "no recognizable landmark", "no specific",
        "no visible signs", "no visible text", "not available", "placeholder",
        "does not provide", "cannot provide", "generic", "common sight",
    )
    if any(marker in low for marker in negative_markers):
        return False
    concrete_markers = (
        "sign", "text", "logo", "license", "plate", "reads", "named",
        "landmark", "bridge", "cathedral", "temple", "station", "museum",
        "monument", "statue", "storefront", "street sign",
    )
    if any(marker in low for marker in concrete_markers):
        return True
    return bool(_SEARCH_ENTITY_RE.search(recent))


def _should_web_enhance_level(
    level: str,
    posterior: dict[str, float],
    visual_delta: float,
    search_evidence: list[str],
) -> bool:
    """GeoBayes-style enhance gate for country/city/street reasoning.

    Search is opt-in at the client layer. This gate only decides whether the
    current inference state is a good candidate for external evidence.
    """
    if level not in WEB_SEARCH_LEVELS:
        return False
    if not _posterior_web_trigger(posterior, visual_delta):
        return False
    return _has_searchable_web_entity(search_evidence)


def _should_web_enhance_country(posterior: dict[str, float], visual_delta: float) -> bool:
    """Backward-compatible country-only posterior gate."""
    return _posterior_web_trigger(posterior, visual_delta)


def _clean_search_clue(item: str) -> str:
    text = str(item or "")
    obs_match = re.search(r"(?is)\bobservation\s*:\s*(.*?)(?:\n\s*support\s*:|$)", text)
    if obs_match:
        text = obs_match.group(1)
    text = re.sub(r"(?is)\bsupport\s*:.*", " ", text)
    text = re.sub(r"(?i)<\/?observation text>", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" ;")
    return text


def _compact_search_evidence(evidence: list[str], max_chars: int = 180) -> str:
    cleaned = [_clean_search_clue(item) for item in evidence[-3:]]
    clues = "; ".join(item for item in cleaned if item)
    if not clues:
        return "visual clue"
    if len(clues) <= max_chars:
        return clues
    trimmed = clues[:max_chars].rsplit(" ", 1)[0].strip()
    return trimmed or clues[:max_chars]


def _build_web_search_query(
    level: str,
    posterior: dict[str, float],
    evidence: list[str],
    parent_context: str = "",
) -> str:
    clue = _compact_search_evidence(evidence)
    candidates = _format_top_candidates(posterior, 5)
    if level == "country":
        query = f"{clue} in which country?"
    elif level == "city":
        if parent_context:
            query = f"{clue} in which cities of {parent_context}?"
        else:
            query = f"{clue} in which city?"
    elif level == "street":
        if parent_context:
            query = f"{clue} near which street district or landmark in {parent_context}?"
        else:
            query = f"{clue} near which street district or landmark?"
    else:
        query = f"{clue} geolocation evidence"
    if candidates:
        query += f" Candidate locations: {candidates}"
    return query[:280]


def _web_enhance_context(
    level: str,
    posterior: dict[str, float],
    query: str,
    evidence: str,
    parent_context: str = "",
) -> str:
    stats = _posterior_stats(posterior)
    level_rules = {
        "country": "Return country names only.",
        "city": "Return city or locality names only.",
        "street": "Return street, district, or landmark names only.",
    }.get(level, "Return location names only.")
    parent = f" Parent location context: {parent_context}." if parent_context else ""
    return (
        f"External web search fallback was triggered at the {level} level because the posterior "
        f"remained ambiguous and visual verification stagnated. "
        f"top={stats['top']:.2f}, margin={stats['margin']:.2f}, "
        f"entropy={stats['entropy']:.2f}. Previous top candidates: "
        f"{_format_top_candidates(posterior, 5)}.{parent} Search query: {query}. "
        f"{level_rules} "
        "Use the search snippets only as supporting evidence; visual evidence still has priority. "
        "Avoid inventing a location not supported by either the image or snippets. "
        f"Search snippets:\n{evidence}"
    )


def _parent_context_for_web(level: str, result: dict) -> str:
    if level == "country":
        return result.get("continent", "")
    if level == "city":
        return result.get("country", "")
    if level == "street":
        return ", ".join(x for x in (result.get("city", ""), result.get("country", "")) if x)
    return ""


def _country_candidate_set(country_posterior: dict[str, float], k: int = 3) -> set[str]:
    return {
        country
        for country, _ in sorted((country_posterior or {}).items(), key=lambda x: -x[1])[:k]
    }


def _child_country_conflict(location: str, country_posterior: dict[str, float]) -> bool:
    child_country = canonicalize_country(location or "")
    if not child_country:
        return False
    return child_country not in _country_candidate_set(country_posterior)


def _filter_child_posterior(
    posterior: dict[str, float],
    country_posterior: dict[str, float],
) -> tuple[dict[str, float], list[str]]:
    """Remove child hypotheses that explicitly contradict country top-3."""
    if not posterior:
        return posterior, []
    filtered = {}
    conflicts = []
    for loc, prob in posterior.items():
        if _child_country_conflict(loc, country_posterior):
            conflicts.append(loc)
        else:
            filtered[loc] = prob
    if not conflicts:
        return posterior, []
    if not filtered:
        return posterior, conflicts
    total = sum(filtered.values())
    return ({k: v / total for k, v in filtered.items()} if total > 0 else filtered), conflicts


def _maybe_promote_country_from_child(
    result: dict,
    level: str,
    child_posterior: dict[str, float],
) -> bool:
    """Backtrack a weak parent country to a confident child-embedded country."""
    if not CHILD_BACKTRACK_PROMOTE or level not in ("city", "street"):
        return False

    parent_country = canonicalize_country(result.get("country") or "")
    child_country = canonicalize_country(result.get(level) or "")
    if not parent_country or not child_country or child_country == parent_country:
        return False

    country_posterior = result.get("country_posterior") or {}
    country_stats = _posterior_stats(country_posterior)
    if country_stats["top"] > CHILD_BACKTRACK_MAX_COUNTRY_TOP:
        return False
    if child_country not in _country_candidate_set(country_posterior):
        return False
    if _posterior_stats(child_posterior)["top"] < CHILD_BACKTRACK_MIN_CHILD_TOP:
        return False

    if level == "street":
        city_country = canonicalize_country(result.get("city") or "")
        if city_country and city_country != child_country:
            return False

    result["country_before_child_backtrack"] = result.get("country")
    result["country_child_backtrack_level"] = level
    result["country_child_backtrack_country"] = child_country
    result["country"] = child_country
    result["country_child_backtracked"] = True
    return True


def _text_prompt(text: str) -> list:
    return [{"role": "user", "content": [{"type": "text", "text": text}]}]


def _city_country_factcheck_prompt(city: str, country: str) -> list:
    text = (
        "You are checking geographic consistency only; do not infer from an image.\n"
        f"Question: Is the named city/locality '{city}' located in the country '{country}'?\n"
        "Use the usual primary geographic meaning of the city name unless the phrase explicitly "
        "names a smaller locality in that country. Ignore capitalization and spelling style.\n"
        "Answer JSON only: "
        '{"consistent": true|false, "true_country": "country name or null", "reason": "short"}'
    )
    return _text_prompt(text)


def _parse_city_country_factcheck(raw: str) -> dict:
    parsed = _try_parse_json(raw)
    if not isinstance(parsed, dict):
        return {"consistent": None, "true_country": None, "reason": "parse_failed"}
    consistent = parsed.get("consistent")
    if isinstance(consistent, str):
        low = consistent.strip().lower()
        if low in {"true", "yes", "consistent"}:
            consistent = True
        elif low in {"false", "no", "inconsistent"}:
            consistent = False
        else:
            consistent = None
    elif not isinstance(consistent, bool):
        consistent = None

    true_country = parsed.get("true_country")
    if true_country is None:
        true_country = parsed.get("country")
    if isinstance(true_country, str) and true_country.strip().lower() in {"", "null", "none", "unknown"}:
        true_country = None
    return {
        "consistent": consistent,
        "true_country": true_country if isinstance(true_country, str) else None,
        "reason": str(parsed.get("reason") or ""),
    }


def _should_factcheck_city_country(result: dict) -> bool:
    if not CITY_COUNTRY_FACTCHECK:
        return False
    city = result.get("city")
    country = canonicalize_country(result.get("country") or "")
    if not city or city == "Unknown" or not country:
        return False
    if canonicalize_country(city) == country:
        return False
    country_top = max((result.get("country_posterior") or {}).values(), default=0.0)
    return float(country_top) >= CITY_COUNTRY_FACTCHECK_MIN_COUNTRY_TOP


def _apply_city_country_factcheck_result(result: dict, raw: str) -> bool:
    parsed = _parse_city_country_factcheck(raw)
    result["city_country_factcheck_raw"] = raw
    result["city_country_factcheck_consistent"] = parsed["consistent"]
    result["city_country_factcheck_true_country"] = parsed["true_country"]
    result["city_country_factcheck_reason"] = parsed["reason"]
    if parsed["consistent"] is not False:
        result["city_country_factcheck_rejected"] = False
        return False
    predicted_country = canonicalize_country(result.get("country") or "")
    checked_country = canonicalize_country(parsed.get("true_country") or "")
    if checked_country and predicted_country and checked_country == predicted_country:
        result["city_country_factcheck_rejected"] = False
        return False

    result["city_country_factcheck_rejected"] = True
    result["city_before_factcheck"] = result.get("city")
    result["city"] = "Unknown"
    result["city_posterior"] = {}
    result["city_stable"] = False
    result["street"] = "Unknown"
    result["street_posterior"] = {}
    result["street_stable"] = False
    return True


def _replace_context(level: str, posterior: dict[str, float], key_evidence: list[str]) -> str:
    stats = _posterior_stats(posterior)
    clues = "; ".join(key_evidence[-3:])
    return (
        "Previous country candidates remained unstable after verification. "
        f"Top candidates were {_format_top_candidates(posterior, 5)}; "
        f"top={stats['top']:.2f}, margin={stats['margin']:.2f}, entropy={stats['entropy']:.2f}. "
        "Re-analyze from scratch and use confidence to reflect the visual evidence. "
        "Keep alternatives only when the image is ambiguous. "
        f"Previous useful clues: {clues}"
    )


def _context_for_level(level: str, result: dict, key_evidence: list[str]) -> str:
    clues = "; ".join(key_evidence[-3:])
    if level == "country":
        continents = _format_top_candidates(result.get("continent_posterior", {}), 3)
        if continents:
            return (
                f"Continent candidates from a coarse pass: {continents}. Treat them as weak priors, "
                "not hard constraints. Return country names only. If visual evidence is generic, "
                "keep plausible countries across the candidate continents instead of defaulting to a familiar country. "
                f"Key clues: {clues}"
            )
        return f"Key clues: {clues}"
    if level == "city":
        parent = result.get("country", "")
        if parent:
            return (
                f"Previous country estimate: {parent}. Treat it as a weak prior, "
                "not a hard constraint; include another country/city if visual evidence supports it. "
                f"Key clues: {clues}"
            )
        return f"Key clues: {clues}"
    if level == "street":
        city = result.get("city", "")
        country = result.get("country", "")
        parent = ", ".join(x for x in (city, country) if x)
        if parent:
            return (
                f"Previous coarse estimate: {parent}. Treat it as a weak prior, "
                "not a hard constraint; prefer the visible street/district/landmark evidence. "
                f"Key clues: {clues}"
            )
        return f"Key clues: {clues}"
    return ""


# ── Prompt builders ────────────────────────────────────────────────────────────

def _hypothesize_prompt(image: Image.Image, level: str, context: str = "") -> list:
    level_hint = {
        "continent": "Identify the most likely continents and generate a plan to verify.",
        "country": "Identify the most likely countries and generate a plan to verify.",
        "city":    "Identify the most likely cities and generate a plan to verify.",
        "street":  "Identify the most likely streets/districts and generate a plan to verify.",
    }[level]
    country_rule = (
        "For country-level reasoning, return country names only, not continents or regions. "
        "Separate direct localizing evidence such as road signs, plates, addresses, named places, landmarks, and place-specific scripts "
        "from generic cues such as English text, brands, food, ordinary roads, vegetation, indoor objects, or common architecture. "
        "Generic cues alone are weak evidence: keep alternatives instead of ruling countries out. "
        "United States, Canada, United Kingdom, Japan, and other familiar countries can receive high confidence when direct localizing evidence supports them. "
    ) if BALANCED_COUNTRY_GUARD else (
        "For country-level reasoning, return country names only, not continents or regions. "
        "Separate direct localizing evidence such as road signs, plates, addresses, named places, landmarks, and place-specific scripts "
        "from generic cues such as English text, brands, food, ordinary roads, vegetation, indoor objects, or common architecture. "
        "Generic cues alone must not give high confidence to any familiar country, including United States, Canada, United Kingdom, or Japan. "
    )
    level_rules = {
        "continent": (
            "For continent-level reasoning, return continent names only: "
            "Africa, Asia, Europe, North America, Oceania, or South America. "
            "Do not return countries, regions, or hemispheres. "
        ),
        "country": country_rule,
        "city": "For city-level reasoning, return city or locality names. ",
        "street": "For street-level reasoning, return street, district, or landmark names. ",
    }[level]
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": (
                    f"You are a geolocation expert. {level_hint}\n"
                    + level_rules
                    + "Use confidence to reflect direct visual evidence; keep alternatives when the image is ambiguous. "
                    + (f"Prior context: {context}\n" if context else "")
                    + "\nAnalyze this image and respond with valid JSON only, no markdown fences:\n"
                    '{\n'
                    '  "hypotheses": [{"location": "<name>", "confidence": <0-1>}, ...],\n'
                    '  "verification_plan": [{"desc": "<what to check>", "bbox": [x,y,w,h] or null}, ...]\n'
                    '}'
                )},
            ],
        }
    ]


def _country_cue_prompt(image: Image.Image, cue_instruction: str) -> list:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": (
                    "You are a geolocation expert. Identify likely countries. "
                    f"{cue_instruction}\n\n"
                    "Analyze this image and respond with JSON only, no markdown fences:\n"
                    '{"hypotheses": [{"location": "<country>", "confidence": <0-1>}, ...]}'
                )},
            ],
        }
    ]


def _georeasoner_country_prompt(image: Image.Image) -> list:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": (
                    "According to the content of the image, please think step by step and deduce "
                    "in which country and city the image is most likely located and give the "
                    "most important reason. Output in JSON format, e.g. "
                    '{"country":"", "city":"", "reasons":""}.'
                )},
            ],
        }
    ]


def _parse_georeasoner_country(text: str) -> str | None:
    parsed = _try_parse_json(text)
    raw_country = ""
    if isinstance(parsed, dict):
        raw_country = str(parsed.get("country") or parsed.get("Country") or "")
    if not raw_country:
        match = re.search(r'"?country"?\s*[:=]\s*"?([^"\n,}]+)', text or "", re.IGNORECASE)
        raw_country = match.group(1).strip() if match else ""
    return canonicalize_country(raw_country)


_GEOREASONER_DIRECT_CLUE_MARKERS = (
    "address", "area code", "cathedral", "chain", "currency", "domain", "flag",
    "landmark", "language", "license", "marked", "measurement", "plate",
    "reads", "road sign", "script", "signage", "station", "store", "storefront",
    "street sign", "temple", "text", "zip code",
)

_GEOREASONER_STRONG_US_MARKERS = (
    "american flag", "area code", "interstate", "license plate", "mph",
    "state route", "u.s. route", "us route", "zip code",
)


def _georeasoner_has_direct_clue(text: str, country: str) -> bool:
    if not GEOREASONER_REQUIRE_DIRECT_CLUE:
        return True
    low = (text or "").lower()
    if country == "united states":
        return any(marker in low for marker in _GEOREASONER_STRONG_US_MARKERS)
    return any(marker in low for marker in _GEOREASONER_DIRECT_CLUE_MARKERS)


def _seed_country_prior(
    prior: dict[str, float],
    country: str | None,
    seed_text: str = "",
    boost: float = GEOREASONER_COUNTRY_BOOST,
) -> tuple[dict[str, float], bool]:
    if not country or boost <= 0:
        return prior, False
    if not _georeasoner_has_direct_clue(seed_text, country):
        return prior, False
    if not prior or set(prior) == {"Unknown"}:
        return {country: 1.0}, True
    boost = min(float(boost), 0.95)
    seeded = dict(prior)
    seeded[country] = seeded.get(country, 0.0) + boost
    return _renormalize_posterior(seeded), True


def _verify_prompt(image: Image.Image, task: dict, hypotheses: list[str], level: str) -> list:
    hyp_str = ", ".join(hypotheses[:5])
    bbox = task.get("bbox")
    desc = task.get("desc", "")
    region_note = f" Focus on region [x,y,w,h]={bbox}." if bbox else ""
    if not VERIFY_SUPPORT_FORMAT:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": (
                        f"Task: {desc}.{region_note}\n"
                        f"Current hypotheses: {hyp_str}\n"
                        f"Reasoning level: {level}\n\n"
                        "Describe what you observe and how it relates to the hypotheses.\n"
                        "Respond with: <observation text>"
                    )},
                ],
            }
        ]

    hyp_lines = "\n".join(f"  - {hyp}" for hyp in hypotheses[:5])
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": (
                    f"Task: {desc}.{region_note}\n"
                    f"Current hypotheses: {hyp_str}\n"
                    f"Reasoning level: {level}\n\n"
                    "Candidate hypotheses:\n"
                    f"{hyp_lines}\n\n"
                    "Step 1 - Describe only visible evidence in 1-2 sentences.\n"
                    "Step 2 - For each candidate above, mark S if the visible evidence supports it, "
                    "C if it contradicts it, or N if it is neutral/uncertain. Most weak clues should be N.\n\n"
                    "Respond exactly in this format:\n"
                    "Observation: <what is visible>\n"
                    "Support: <hypothesis_1>=S/C/N; <hypothesis_2>=S/C/N; ..."
                )},
            ],
        }
    ]


# ── Main pipeline class ────────────────────────────────────────────────────────

BATCH_SIZE = 20  # number of images to process in parallel; reduce if OOM


class GeoPipeline:
    def __init__(self, mllm: MLLMClient):
        self.mllm  = mllm
        self.sl    = SLModule(mllm)
        self.dst   = DSTModule()
        self.pomdp = POMDPModule(mllm)
        self.web_search = WebSearchClient()

    def _hypothesize(self, image: Image.Image, level: str, context: str = "") -> tuple[dict, list, str]:
        """Returns (prior_dict, verification_plan_list, raw_response)."""
        messages = _hypothesize_prompt(image, level, context)
        response = self.mllm.generate(messages)
        parsed = _parse_hypothesis_payload(response)
        if parsed is None or "hypotheses" not in parsed:
            if level == "country" and COUNTRY_GEOREASONER_SEED:
                seed_response = self.mllm.generate(_georeasoner_country_prompt(image))
                prior, seeded = _seed_country_prior(
                    {"Unknown": 1.0}, _parse_georeasoner_country(seed_response), seed_response
                )
                if seeded:
                    raw_bundle = json.dumps(
                        {
                            "general": response,
                            "georeasoner_response": seed_response,
                            "georeasoner_seeded": True,
                        },
                        ensure_ascii=True,
                    )
                    return prior, [], raw_bundle
            # fallback: single hypothesis with uniform prior
            return {"Unknown": 1.0}, [], response

        if level == "country" and COUNTRY_CUE_ENSEMBLE:
            cue_responses = [
                self.mllm.generate(_country_cue_prompt(image, cue_text))
                for _, cue_text in _COUNTRY_CUE_PROMPTS
            ]
            cue_parsed = [_parse_hypothesis_payload(resp) for resp in cue_responses]
            parsed_sources = [parsed] if parsed and "hypotheses" in parsed else []
            raw_scores = _merge_country_scores([*parsed_sources, *cue_parsed])
            prior = _softmax_prior(raw_scores) if raw_scores else {"Unknown": 1.0}
            seed_response = ""
            seeded = False
            if COUNTRY_GEOREASONER_SEED:
                seed_response = self.mllm.generate(_georeasoner_country_prompt(image))
                prior, seeded = _seed_country_prior(
                    prior, _parse_georeasoner_country(seed_response), seed_response
                )
            plan = _prepend_country_tasks(parsed.get("verification_plan", []))
            raw_bundle = json.dumps(
                {
                    "general": response,
                    "cue_responses": cue_responses,
                    "georeasoner_response": seed_response,
                    "georeasoner_seeded": seeded,
                },
                ensure_ascii=True,
            )
            return prior, plan, raw_bundle

        raw_scores = _collect_scores(parsed["hypotheses"], level)
        prior = _softmax_prior(raw_scores) if raw_scores else {"Unknown": 1.0}
        plan  = parsed.get("verification_plan", [])
        if level == "country" and COUNTRY_GEOREASONER_SEED:
            seed_response = self.mllm.generate(_georeasoner_country_prompt(image))
            prior, seeded = _seed_country_prior(
                prior, _parse_georeasoner_country(seed_response), seed_response
            )
            response = json.dumps(
                {
                    "general": response,
                    "georeasoner_response": seed_response,
                    "georeasoner_seeded": seeded,
                },
                ensure_ascii=True,
            )
        return prior, plan, response

    def _run_level(
        self,
        image: Image.Image,
        level: str,
        initial_posterior: dict[str, float],
        initial_plan: list[dict],
        key_evidence: list[str],
    ) -> tuple[dict, list[str], float, list[str], int]:
        """
        Run one hierarchy level.

        Returns (final_posterior, updated_key_evidence, last_delta_p, observed_evidence), where
        last_delta_p is the top-posterior change after the latest visual
        verification evidence. GeoBayes uses this kind of posterior gain to
        decide when external evidence enhancement is useful.
        """
        posterior = dict(initial_posterior)
        pending   = list(initial_plan)
        step      = 0
        visual_delta = 0.0
        evidence_scores_all: list[dict[str, float]] = []
        observed_evidence: list[str] = []

        while True:
            exhausted = len(pending) == 0
            if self.pomdp.should_stop(posterior, step, level, exhausted):
                break

            # POMDP: select best action (skip if only one task)
            if len(pending) == 1:
                task_idx = 0
            elif self.pomdp.use_expected_gain_for(level):
                task_idx = self.pomdp.select_action_by_expected_gain(
                    image, posterior, pending, level, step
                )
            else:
                task_idx = self.pomdp.select_action(posterior, pending, level, step)
            task = pending.pop(task_idx)

            # Verify: get evidence description from MLLM
            hyps = list(posterior.keys())
            v_messages = _verify_prompt(image, task, hyps, level)
            evidence_desc = self.mllm.generate(v_messages, max_new_tokens=VERIFY_MAX_NEW_TOKENS)
            observed_evidence.append(evidence_desc[:240])

            # SL: uncertainty-aware per-hypothesis scores
            w_scores = self.sl.score(evidence_desc, hyps, level)
            evidence_scores_all.append(w_scores)

            # DST: fuse all evidence so far into new posterior
            prev_top = max(posterior.values(), default=0.0)
            posterior = self.dst.fuse(initial_posterior, evidence_scores_all)
            visual_delta = max(0.0, max(posterior.values(), default=0.0) - prev_top)

            # track key evidence (high-information clues)
            max_w = max(w_scores.values(), default=1.0)
            if max_w > 1.5:
                key_evidence.append(evidence_desc[:120])

            step += 1

        return posterior, key_evidence, visual_delta, observed_evidence, step

    def _web_enhance_level(
        self,
        image: Image.Image,
        level: str,
        posterior: dict[str, float],
        key_evidence: list[str],
        visual_delta: float,
        search_evidence: list[str],
        parent_context: str = "",
    ) -> tuple[dict[str, float], list[str], str, str, float, list[str]] | None:
        """Use optional web search snippets to re-run an ambiguous level."""
        if not _should_web_enhance_level(level, posterior, visual_delta, search_evidence):
            return None

        query = _build_web_search_query(level, posterior, search_evidence, parent_context)
        search_data = self.web_search.search(query)
        search_evidence = format_search_evidence(search_data)
        if not search_evidence:
            return None

        context = _web_enhance_context(level, posterior, query, search_evidence, parent_context)
        prior, plan, raw_resp = self._hypothesize(image, level, context)
        enhanced_posterior, enhanced_evidence, web_delta, observed_evidence, _ = self._run_level(
            image, level, prior, plan, key_evidence
        )
        enhanced_evidence.append(f"web search ({level}): {search_evidence[:120]}")
        return enhanced_posterior, enhanced_evidence, raw_resp, query, web_delta, observed_evidence

    def _web_enhance_country(
        self,
        image: Image.Image,
        posterior: dict[str, float],
        key_evidence: list[str],
        visual_delta: float,
    ) -> tuple[dict[str, float], list[str], str, str, float] | None:
        """Backward-compatible country-only web enhancement wrapper."""
        enhanced = self._web_enhance_level(
            image, "country", posterior, key_evidence, visual_delta, key_evidence
        )
        if enhanced is None:
            return None
        posterior, evidence, raw_resp, query, web_delta, _ = enhanced
        return posterior, evidence, raw_resp, query, web_delta

    def predict(self, image: Image.Image) -> dict:
        """
        Full coarse-to-fine inference for one image.
        Returns {level: best_location_name, "posterior": final_posterior_dict}.
        """
        result       = {"pomdp_policy": self.pomdp.policy_label}
        key_evidence = []
        context      = ""

        for level in LEVELS:
            # at city/street level, seed hypotheses from prior level result
            if level != "continent" and result:
                context = _context_for_level(level, result, key_evidence)

            prior, plan, raw_resp = self._hypothesize(image, level, context)
            result[f"{level}_raw_response"] = raw_resp

            posterior, key_evidence, visual_delta, level_evidence, level_steps = self._run_level(
                image, level, prior, plan, key_evidence
            )

            if level == "country" and _should_replace_country(posterior):
                for _ in range(COUNTRY_REPLACE_ATTEMPTS):
                    replace_context = _replace_context(level, posterior, key_evidence)
                    prior, plan, raw_resp = self._hypothesize(image, level, replace_context)
                    result[f"{level}_raw_response"] = raw_resp
                    posterior, key_evidence, visual_delta, level_evidence, level_steps = self._run_level(
                        image, level, prior, plan, key_evidence
                    )
                    result["country_replaced"] = True
                    if _stable_for_descent(posterior):
                        break

            if level == "country":
                result["country_visual_delta"] = visual_delta

            search_clues = (level_evidence or []) + key_evidence[-3:]
            enhanced = self._web_enhance_level(
                image,
                level,
                posterior,
                key_evidence,
                visual_delta,
                search_clues,
                _parent_context_for_web(level, result),
            )
            if enhanced is not None:
                posterior, key_evidence, raw_resp, web_query, web_delta, level_evidence = enhanced
                result[f"{level}_web_enhanced"] = True
                result[f"{level}_web_search_query"] = web_query
                result[f"{level}_web_delta"] = web_delta
                result[f"{level}_raw_response"] = raw_resp

            if level == "country":
                posterior, regularized = _regularize_country_by_continent(
                    posterior, result.get("continent_posterior", {})
                )
                result["country_continent_regularized"] = regularized
            elif level in ("city", "street"):
                posterior, conflicts = _filter_child_posterior(
                    posterior, result.get("country_posterior", {})
                )
                result[f"{level}_backtrack_conflicts"] = conflicts

            best = max(posterior, key=posterior.get)
            result[level] = best
            result[f"{level}_posterior"] = posterior
            result[f"{level}_stable"] = _stable_for_descent(posterior)
            result[f"{level}_steps"] = level_steps

            if level in ("city", "street"):
                _maybe_promote_country_from_child(result, level, posterior)

            if level == "city" and _should_factcheck_city_country(result):
                raw_fc = self.mllm.generate(
                    _city_country_factcheck_prompt(result["city"], result["country"]),
                    max_new_tokens=FACTCHECK_MAX_NEW_TOKENS,
                )
                if _apply_city_country_factcheck_result(result, raw_fc):
                    posterior = {}
                    best = "Unknown"

            if level == "country" and (block_reason := _descent_block_reason(posterior)):
                # Even guarded descent would be too noisy. Avoid propagating a
                # very weak parent posterior into child prompts.
                result["country_descent_blocked_reason"] = block_reason
                for remaining in LEVELS[LEVELS.index(level) + 1:]:
                    result[remaining] = "Unknown"
                    result[f"{remaining}_posterior"] = {}
                break
            if level == "city" and best == "Unknown":
                result["street"] = "Unknown"
                result["street_posterior"] = {}
                break

        result["posterior"] = posterior
        return result

    def _run_level_batch(
        self,
        images: list,
        level: str,
        contexts: list[str],
        key_evidence: list[list[str]],
    ) -> tuple[list[str], list[dict[str, float]], list[float], list[list[str]], list[int]]:
        """Run one hierarchy level for a batch and update key_evidence in place."""
        n = len(images)
        hyp_messages = [_hypothesize_prompt(images[i], level, contexts[i]) for i in range(n)]
        hyp_responses = self.mllm.batch_generate(hyp_messages)
        cue_responses_by_cue = []
        if level == "country" and COUNTRY_CUE_ENSEMBLE:
            for _, cue_text in _COUNTRY_CUE_PROMPTS:
                cue_messages = [_country_cue_prompt(images[i], cue_text) for i in range(n)]
                cue_responses_by_cue.append(self.mllm.batch_generate(cue_messages))
        georeasoner_responses = []
        if level == "country" and COUNTRY_GEOREASONER_SEED:
            seed_messages = [_georeasoner_country_prompt(images[i]) for i in range(n)]
            georeasoner_responses = self.mllm.batch_generate(seed_messages)

        priors = []
        plans = []
        for idx, resp in enumerate(hyp_responses):
            parsed = _parse_hypothesis_payload(resp)
            seed_resp = georeasoner_responses[idx] if georeasoner_responses else ""
            seeded = False
            if level == "country" and COUNTRY_CUE_ENSEMBLE:
                cue_resps = [cue_batch[idx] for cue_batch in cue_responses_by_cue]
                cue_parsed = [_parse_hypothesis_payload(cue_resp) for cue_resp in cue_resps]
                parsed_sources = [parsed] if parsed and "hypotheses" in parsed else []
                raw_scores = _merge_country_scores([*parsed_sources, *cue_parsed])
                prior = _softmax_prior(raw_scores) if raw_scores else {"Unknown": 1.0}
                if seed_resp:
                    prior, seeded = _seed_country_prior(
                        prior, _parse_georeasoner_country(seed_resp), seed_resp
                    )
                priors.append(prior)
                plan = parsed.get("verification_plan", []) if parsed else []
                plans.append(_prepend_country_tasks(plan))
                hyp_responses[idx] = json.dumps(
                    {
                        "general": resp,
                        "cue_responses": cue_resps,
                        "georeasoner_response": seed_resp,
                        "georeasoner_seeded": seeded,
                    },
                    ensure_ascii=True,
                )
            elif parsed is None or "hypotheses" not in parsed:
                prior = {"Unknown": 1.0}
                if seed_resp:
                    prior, seeded = _seed_country_prior(
                        prior, _parse_georeasoner_country(seed_resp), seed_resp
                    )
                priors.append(prior)
                plans.append([])
                if seed_resp:
                    hyp_responses[idx] = json.dumps(
                        {
                            "general": resp,
                            "georeasoner_response": seed_resp,
                            "georeasoner_seeded": seeded,
                        },
                        ensure_ascii=True,
                    )
            else:
                raw_scores = _collect_scores(parsed["hypotheses"], level)
                prior = _softmax_prior(raw_scores) if raw_scores else {"Unknown": 1.0}
                if seed_resp:
                    prior, seeded = _seed_country_prior(
                        prior, _parse_georeasoner_country(seed_resp), seed_resp
                    )
                priors.append(prior)
                plans.append(parsed.get("verification_plan", []))
                if seed_resp:
                    hyp_responses[idx] = json.dumps(
                        {
                            "general": resp,
                            "georeasoner_response": seed_resp,
                            "georeasoner_seeded": seeded,
                        },
                        ensure_ascii=True,
                    )

        posteriors = [dict(p) for p in priors]
        pending = [list(pl) for pl in plans]
        steps = [0] * n
        ev_scores_all = [[] for _ in range(n)]
        visual_deltas = [0.0] * n
        observed_evidence = [[] for _ in range(n)]

        while True:
            active = [
                i for i in range(n)
                if not self.pomdp.should_stop(
                    posteriors[i], steps[i], level, len(pending[i]) == 0
                )
            ]
            if not active:
                break

            task_choices = {}
            single_task = [i for i in active if len(pending[i]) == 1]
            for i in single_task:
                task_choices[i] = 0

            multi_task = [i for i in active if len(pending[i]) > 1]
            if multi_task and self.pomdp.use_expected_gain_for(level):
                eig_choices = self.pomdp.select_actions_by_expected_gain(
                    [images[i] for i in multi_task],
                    [posteriors[i] for i in multi_task],
                    [pending[i] for i in multi_task],
                    level,
                    [steps[i] for i in multi_task],
                )
                for i, idx in zip(multi_task, eig_choices):
                    task_choices[i] = min(idx, len(pending[i]) - 1)
            elif multi_task:
                policy_msgs = [
                    self.pomdp._make_policy_prompt(
                        posteriors[i], pending[i], level, steps[i]
                    )
                    for i in multi_task
                ]
                policy_resps = self.mllm.batch_generate(
                    policy_msgs, max_new_tokens=POMDP_MAX_NEW_TOKENS
                )
                for i, resp in zip(multi_task, policy_resps):
                    match = re.search(r'"?task_index"?\s*:\s*(\d+)', resp)
                    idx = int(match.group(1)) if match else 0
                    task_choices[i] = min(idx, len(pending[i]) - 1)

            tasks = {i: pending[i].pop(task_choices[i]) for i in active}

            verify_msgs = [
                _verify_prompt(images[i], tasks[i], list(posteriors[i].keys()), level)
                for i in active
            ]
            verify_resps = self.mllm.batch_generate(
                verify_msgs, max_new_tokens=VERIFY_MAX_NEW_TOKENS
            )
            evidence_descs = {i: resp for i, resp in zip(active, verify_resps)}
            for i, resp in evidence_descs.items():
                observed_evidence[i].append(resp[:240])

            sl_items = [
                (evidence_descs[i], list(posteriors[i].keys()))
                for i in active
            ]
            sl_results = self.sl.score_many(sl_items, level)

            for k, i in enumerate(active):
                w_scores = sl_results[k]
                ev_scores_all[i].append(w_scores)
                prev_top = max(posteriors[i].values(), default=0.0)
                posteriors[i] = self.dst.fuse(priors[i], ev_scores_all[i])
                visual_deltas[i] = max(0.0, max(posteriors[i].values(), default=0.0) - prev_top)

                max_w = max(w_scores.values(), default=1.0)
                if max_w > 1.5:
                    key_evidence[i].append(evidence_descs[i][:120])

                steps[i] += 1

        return hyp_responses, posteriors, visual_deltas, observed_evidence, steps

    def predict_batch(self, images: list) -> list[dict]:
        """
        Process a batch of images together, grouping MLLM calls across images
        at each pipeline step to maximise GPU utilisation.
        Returns a list of result dicts in the same order as images.
        """
        n = len(images)
        # per-image state
        results = [{"pomdp_policy": self.pomdp.policy_label} for _ in range(n)]
        key_evidence = [[] for _ in range(n)]
        contexts = [""] * n
        skip_finer = [False] * n

        for level in LEVELS:
            level_indices = [i for i in range(n) if not skip_finer[i]]
            if not level_indices:
                break

            # seed context from parent level before hypothesizing the next level
            if level != "continent":
                for i in level_indices:
                    contexts[i] = _context_for_level(level, results[i], key_evidence[i])

            subset_images = [images[i] for i in level_indices]
            subset_contexts = [contexts[i] for i in level_indices]
            subset_key_evidence = [key_evidence[i] for i in level_indices]
            raw_responses, posteriors_subset, deltas_subset, evidence_subset, steps_subset = self._run_level_batch(
                subset_images, level, subset_contexts, subset_key_evidence
            )
            posteriors_by_idx = {
                idx: post for idx, post in zip(level_indices, posteriors_subset)
            }
            raw_by_idx = {
                idx: raw for idx, raw in zip(level_indices, raw_responses)
            }
            visual_delta_by_idx = {
                idx: delta for idx, delta in zip(level_indices, deltas_subset)
            }
            level_evidence_by_idx = {
                idx: evidence for idx, evidence in zip(level_indices, evidence_subset)
            }
            steps_by_idx = {
                idx: step_count for idx, step_count in zip(level_indices, steps_subset)
            }

            # Replace: only regenerate the country candidate set when belief is
            # genuinely weak and nearly tied. Marginally unstable but plausible
            # country distributions are allowed to descend with child filtering.
            if level == "country" and COUNTRY_REPLACE_ATTEMPTS > 0:
                unstable = [idx for idx in level_indices if _should_replace_country(posteriors_by_idx[idx])]
                for _ in range(COUNTRY_REPLACE_ATTEMPTS):
                    if not unstable:
                        break
                    replace_images = [images[i] for i in unstable]
                    replace_contexts = [
                        _replace_context(level, posteriors_by_idx[i], key_evidence[i])
                        for i in unstable
                    ]
                    replace_key_evidence = [key_evidence[i] for i in unstable]
                    repl_raw, repl_posts, repl_deltas, repl_evidence, repl_steps = self._run_level_batch(
                        replace_images, level, replace_contexts, replace_key_evidence
                    )
                    for idx, raw, post, delta, evidence in zip(
                        unstable, repl_raw, repl_posts, repl_deltas, repl_evidence
                    ):
                        raw_by_idx[idx] = raw
                        posteriors_by_idx[idx] = post
                        visual_delta_by_idx[idx] = delta
                        level_evidence_by_idx[idx] = evidence
                        results[idx]["country_replaced"] = True
                    for idx, step_count in zip(unstable, repl_steps):
                        steps_by_idx[idx] = step_count
                    unstable = [idx for idx in unstable if _should_replace_country(posteriors_by_idx[idx])]

            if level in WEB_SEARCH_LEVELS:
                web_unstable = [
                    idx for idx in level_indices
                    if _should_web_enhance_level(
                        level,
                        posteriors_by_idx[idx],
                        visual_delta_by_idx.get(idx, 0.0),
                        (level_evidence_by_idx.get(idx, []) or []) + key_evidence[idx][-3:],
                    )
                ]
                for idx in web_unstable:
                    enhanced = self._web_enhance_level(
                        images[idx],
                        level,
                        posteriors_by_idx[idx],
                        key_evidence[idx],
                        visual_delta_by_idx.get(idx, 0.0),
                        (level_evidence_by_idx.get(idx, []) or []) + key_evidence[idx][-3:],
                        _parent_context_for_web(level, results[idx]),
                    )
                    if enhanced is None:
                        continue
                    post, enhanced_key_evidence, raw, web_query, web_delta, observed_evidence = enhanced
                    posteriors_by_idx[idx] = post
                    key_evidence[idx] = enhanced_key_evidence
                    raw_by_idx[idx] = raw
                    level_evidence_by_idx[idx] = observed_evidence
                    steps_by_idx[idx] = None
                    results[idx][f"{level}_web_enhanced"] = True
                    results[idx][f"{level}_web_search_query"] = web_query
                    results[idx][f"{level}_web_delta"] = web_delta

            # ── Collect level results ───────────────────────────────────────────
            for i in level_indices:
                posterior = posteriors_by_idx[i]
                results[i][f"{level}_raw_response"] = raw_by_idx[i]

                if level == "country":
                    posterior, regularized = _regularize_country_by_continent(
                        posterior, results[i].get("continent_posterior", {})
                    )
                    results[i]["country_continent_regularized"] = regularized
                    posteriors_by_idx[i] = posterior
                elif level in ("city", "street"):
                    posterior, conflicts = _filter_child_posterior(
                        posterior, results[i].get("country_posterior", {})
                    )
                    results[i][f"{level}_backtrack_conflicts"] = conflicts
                    posteriors_by_idx[i] = posterior

                best = max(posterior, key=posterior.get)
                results[i][level] = best
                results[i][f"{level}_posterior"] = posterior
                results[i][f"{level}_stable"] = _stable_for_descent(posterior)
                results[i][f"{level}_steps"] = steps_by_idx.get(i)

                if level == "country":
                    results[i]["country_visual_delta"] = visual_delta_by_idx.get(i, 0.0)

                if level in ("city", "street"):
                    _maybe_promote_country_from_child(results[i], level, posterior)

                if level == "country" and (block_reason := _descent_block_reason(posterior)):
                    # Even guarded descent would be too noisy. Avoid propagating a
                    # very weak parent into child prompts.
                    results[i]["country_descent_blocked_reason"] = block_reason
                    for remaining in LEVELS[LEVELS.index(level) + 1:]:
                        results[i][remaining] = "Unknown"
                        results[i][f"{remaining}_posterior"] = {}
                    skip_finer[i] = True
                elif level == "city" and best == "Unknown":
                    results[i]["street"] = "Unknown"
                    results[i]["street_posterior"] = {}
                    skip_finer[i] = True

            if level == "city" and CITY_COUNTRY_FACTCHECK:
                factcheck_indices = [
                    i for i in level_indices
                    if not skip_finer[i] and _should_factcheck_city_country(results[i])
                ]
                if factcheck_indices:
                    messages = [
                        _city_country_factcheck_prompt(results[i]["city"], results[i]["country"])
                        for i in factcheck_indices
                    ]
                    raw_checks = self.mllm.batch_generate(
                        messages, max_new_tokens=FACTCHECK_MAX_NEW_TOKENS
                    )
                    for idx, raw_fc in zip(factcheck_indices, raw_checks):
                        if _apply_city_country_factcheck_result(results[idx], raw_fc):
                            skip_finer[idx] = True

        for i in range(n):
            final_level = next(
                (lv for lv in reversed(LEVELS) if results[i].get(lv, "Unknown") != "Unknown"),
                LEVELS[-1]
            )
            results[i]["posterior"] = results[i].get(f"{final_level}_posterior", {})

        return results
