"""Optional image-search evidence for GeoBayes enhancement."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from PIL import Image

from config import IMAGE_SEARCH_MAX_ENTITIES, IMAGE_SEARCH_TIMEOUT


_GEO_HINT_RE = re.compile(
    r"\b(?:city|town|village|street|road|square|plaza|placa|plaça|avenue|station|"
    r"cathedral|church|basilica|temple|mosque|museum|monument|statue|bridge|airport|"
    r"university|hotel|restaurant|park|landmark|river|waterway|lake|beach|"
    r"castle|palace|stadium|market|harbou?r|port|county|province|prefecture|dori)\b",
    re.IGNORECASE,
)
_PHONE_OR_CODE_RE = re.compile(r"\b(?:\+?\d[\d ()-]{5,}|\d{4,})\b")
_TITLEISH_RE = re.compile(r"\b[A-Z][a-zA-ZÀ-ÖØ-öø-ÿ]+(?:\s+[A-Z][a-zA-ZÀ-ÖØ-öø-ÿ]+){1,}\b")
_GENERIC_PRODUCT_TERMS = (
    "audi", "honda", "toyota", "ford", "bmw", "mercedes", "volkswagen",
    "minivan", "vehicle", "automobile", "compact mpv", "mini mpv", "motor company",
    "camera", "lens", "shoe", "clothing", "product", "brand", "electronics",
    "amazon", "etsy", "plaque", "engraved brass", "custom plaques", "personalized",
)
_GENERIC_ENTITY_TERMS = _GENERIC_PRODUCT_TERMS + (
    "aegean cat", "arch bridge", "architecture", "art", "bed frame", "bedroom",
    "beam bridge", "beverage", "biology", "bird", "black swallowtail", "bookcase",
    "box girder bridge", "bridge", "brunch", "cabinetry", "caipirinha", "car",
    "cat", "ceiling fixture", "chinese new year", "city car", "concrete bridge",
    "corgi", "creeping thistle", "curb", "dagens nyheter", "dirt road", "dog",
    "domestic short-haired cat", "drink", "earth", "eclipse", "eyelash", "fireworks", "flickr",
    "floating shelf", "flower", "fun", "furniture", "giant swallowtail", "girder bridge",
    "hair extension", "hatchback", "hibiscus", "image", "image sharing", "institute",
    "jellyfish", "leisure", "light-hearted", "lunar", "lunar eclipse",
    "lunar phase", "lunch", "macaque", "marine biology", "marine mammal",
    "marsh thistle", "meter", "moggy", "moon", "new year", "new year's day",
    "new year's eve", "news", "nightstand", "non-alcoholic", "ojos azules",
    "old world swallowtail", "overhead power line", "papilio", "parish", "pedestrian",
    "photograph", "photography", "rhesus macaque", "recreation", "religious institute",
    "riba", "rural area", "sea-m", "shelf", "sidewalk", "snow", "soft drink",
    "spear thistle", "street light", "summit", "swallowtail", "table", "thistle",
    "tie", "tourism", "tree", "truss bridge", "underwater", "visual arts", "what is a tree",
    "working dog", "youtube",
)
_GENERIC_ONLY_LABELS = {
    "building", "car", "city", "concert", "flower", "food", "indoor", "landscape",
    "motor vehicle", "ocean", "plant", "road", "sea", "sky", "stage", "street",
    "town", "traffic", "tree", "urban area", "vehicle", "village",
}
_WEB_NOISE_RE = re.compile(
    r"\b(?:stack overflow|direct terminal output|mysql|stock photo(?:s)?|high-res pictures|"
    r"bride groom|wedding photo|what to shoot|typical wedding|learning with|deacon|journal|"
    r"can someone tell me|r/|for sale|ebay|etsy|amazon|zillow|\bmls\b|discogs|imdb|"
    r"facebook|instagram|youtube|flickr|vinyl|cds|birthday|tattoo|sneakers?|shoes?|"
    r"cats?|kittens?|adoption|fundraiser|flycatcher|muscicapa|wallpaper|background|"
    r"download|royalty-free|stock video|stock footage|photos?)\b",
    re.IGNORECASE,
)
_CAMERA_FILE_RE = re.compile(r"^(?:dsc|img|pict|photo)[-_ ]?\d{3,}\b", re.IGNORECASE)
_STRONG_GENERIC_RE = re.compile(
    r"\b(?:hibiscus|zephyranthes|flycatcher|muscicapa|cats?|kittens?|lamb|"
    r"shoes?|sneakers?|plaque|flowers?|plants?)\b",
    re.IGNORECASE,
)


class ImageSearchClient:
    """Opt-in wrapper around Google Vision or SerpAPI Google Lens."""

    def __init__(self):
        self.enabled = os.environ.get("IMAGE_SEARCH_ENABLED", "0").lower() in {
            "1", "true", "yes", "on"
        }
        self.provider = os.environ.get("IMAGE_SEARCH_PROVIDER", "google_vision").lower()
        self.credentials = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
        self.serpapi_key = self._read_serpapi_key()
        self.serpapi_type = os.environ.get("SERPAPI_LENS_TYPE", "all").strip() or "all"
        self.image_url_template = os.environ.get("SERPAPI_IMAGE_URL_TEMPLATE", "").strip()
        self.cache_path = os.environ.get("IMAGE_SEARCH_CACHE_PATH", "").strip()
        self.max_uncached_calls = self._read_int_env("IMAGE_SEARCH_MAX_UNCACHED_CALLS", 0)
        self._uncached_calls = 0
        self._cap_warned = False
        self._cache = self._load_cache(self.cache_path)
        self._client = None

        if self.enabled and self.provider not in {"google_vision", "serpapi_lens"}:
            print(f"[IMAGE] Unsupported IMAGE_SEARCH_PROVIDER={self.provider}; disabled.")
            self.enabled = False
        if self.enabled and self.provider == "google_vision" and not self.credentials:
            print("[IMAGE] IMAGE_SEARCH_ENABLED=1 but GOOGLE_APPLICATION_CREDENTIALS is missing; disabled.")
            self.enabled = False
        if (
            self.enabled
            and self.provider == "google_vision"
            and self.credentials
            and not Path(self.credentials).expanduser().exists()
        ):
            print(f"[IMAGE] credentials file not found: {self.credentials}; disabled.")
            self.enabled = False
        if self.enabled and self.provider == "serpapi_lens" and not self.serpapi_key:
            print("[IMAGE] IMAGE_SEARCH_PROVIDER=serpapi_lens but SERPAPI_API_KEY is missing; disabled.")
            self.enabled = False
        if self.enabled:
            label = "Google Vision Web Detection" if self.provider == "google_vision" else "SerpAPI Google Lens"
            print(f"[IMAGE] {label} enabled.")

    def search_image(self, image: Image.Image) -> dict | None:
        if not self.enabled:
            return None
        image_path = self._image_path(image)
        if not image_path:
            return None
        image_url = self._image_url(image, image_path) if self.provider == "serpapi_lens" else ""
        if self.provider == "serpapi_lens" and not image_url:
            print("[IMAGE] SerpAPI Google Lens needs a public image URL; skipped.")
            return None

        cache_key = f"{self.provider}:{image_url or image_path}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        if image_path in self._cache:
            return self._cache[image_path]
        if self.max_uncached_calls and self._uncached_calls >= self.max_uncached_calls:
            if not self._cap_warned:
                print(f"[IMAGE] uncached search cap reached ({self.max_uncached_calls}); skipping more calls.")
                self._cap_warned = True
            return None
        try:
            self._uncached_calls += 1
            if self.provider == "serpapi_lens":
                data = self._serpapi_lens_search(image_url)
            else:
                data = self._google_vision_web_detection(image_path)
            self._cache[cache_key] = data
            self._save_cache()
            return data
        except Exception as exc:
            print(f"[IMAGE] search failed: {exc}")
            return None

    def _read_serpapi_key(self) -> str:
        key = os.environ.get("SERPAPI_API_KEY") or os.environ.get("SERPAPI_KEY")
        if key:
            return key.strip()
        key_file = os.environ.get("SERPAPI_API_KEY_FILE", "~/.serpapi_key")
        path = Path(key_file).expanduser()
        if not path.exists():
            return ""
        try:
            return path.read_text().strip()
        except OSError:
            return ""

    def _read_int_env(self, name: str, default: int) -> int:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return default
        try:
            return max(0, int(raw))
        except ValueError:
            print(f"[IMAGE] invalid {name}={raw!r}; using {default}.")
            return default

    def _load_cache(self, cache_path: str) -> dict:
        if not cache_path:
            return {}
        path = Path(cache_path).expanduser()
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text())
        except Exception as exc:
            print(f"[IMAGE] cache ignored: {exc}")
            return {}
        return data if isinstance(data, dict) else {}

    def _save_cache(self) -> None:
        if not self.cache_path:
            return
        path = Path(self.cache_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._cache, ensure_ascii=True))
        tmp.replace(path)

    def _image_path(self, image: Image.Image) -> str | None:
        if hasattr(image, "info") and image.info.get("img_path"):
            return str(image.info["img_path"])
        if getattr(image, "img_path", None):
            return str(image.img_path)
        if getattr(image, "filename", None):
            return str(image.filename)
        return None

    def _image_url(self, image: Image.Image, image_path: str) -> str:
        if hasattr(image, "info") and image.info.get("image_url"):
            return str(image.info["image_url"]).strip()
        if getattr(image, "image_url", None):
            return str(image.image_url).strip()
        if not self.image_url_template:
            return ""

        path = Path(image_path)
        photo_id = ""
        if hasattr(image, "info") and image.info.get("photo_id"):
            photo_id = str(image.info["photo_id"])
        if not photo_id:
            photo_id = path.stem
        try:
            return self.image_url_template.format(
                photo_id=photo_id,
                filename=path.name,
                basename=path.name,
                stem=path.stem,
                img_path=image_path,
            ).strip()
        except KeyError as exc:
            print(f"[IMAGE] invalid SERPAPI_IMAGE_URL_TEMPLATE placeholder: {exc}")
            return ""

    def _vision_client(self):
        if self._client is None:
            from google.cloud import vision

            self._client = vision.ImageAnnotatorClient()
        return self._client

    def _google_vision_web_detection(self, image_path: str) -> dict | None:
        from google.cloud import vision

        with open(image_path, "rb") as f:
            image = vision.Image(content=f.read())

        response = self._vision_client().web_detection(image=image, timeout=IMAGE_SEARCH_TIMEOUT)
        if getattr(response, "error", None) and response.error.message:
            raise RuntimeError(response.error.message)
        web = response.web_detection
        if not web:
            return None

        return {
            "entities": [
                {"description": entity.description, "score": float(entity.score or 0.0)}
                for entity in web.web_entities[:IMAGE_SEARCH_MAX_ENTITIES]
                if entity.description
            ],
            "best_guess_labels": [label.label for label in web.best_guess_labels if label.label],
            "pages": [
                {"title": page.page_title, "url": page.url}
                for page in web.pages_with_matching_images[:IMAGE_SEARCH_MAX_ENTITIES]
                if page.url or page.page_title
            ],
        }

    def _serpapi_lens_search(self, image_url: str) -> dict | None:
        import requests

        params = {
            "engine": "google_lens",
            "url": image_url,
            "api_key": self.serpapi_key,
            "type": self.serpapi_type,
            "hl": os.environ.get("SERPAPI_LENS_HL", "en"),
        }
        country = os.environ.get("SERPAPI_LENS_COUNTRY", "").strip()
        if country:
            params["country"] = country

        response = requests.get("https://serpapi.com/search.json", params=params, timeout=IMAGE_SEARCH_TIMEOUT)
        if response.status_code != 200:
            raise RuntimeError(f"SerpAPI HTTP {response.status_code}")
        payload = response.json()
        if payload.get("error"):
            raise RuntimeError(str(payload["error"]))
        return serpapi_lens_results_to_data(payload)


def serpapi_lens_results_to_data(payload: dict | None) -> dict | None:
    if not payload:
        return None

    titles: list[str] = []
    pages: list[dict[str, str]] = []

    def add_result(item: dict) -> None:
        title = str(item.get("title") or item.get("name") or "").strip()
        link = str(item.get("link") or item.get("url") or "").strip()
        if title and title not in titles:
            titles.append(title)
        if title or link:
            page = {"title": title, "url": link}
            if page not in pages:
                pages.append(page)

    knowledge = payload.get("knowledge_graph")
    if isinstance(knowledge, dict):
        add_result(knowledge)
    elif isinstance(knowledge, list):
        for item in knowledge:
            if isinstance(item, dict):
                add_result(item)

    for section in ("visual_matches", "exact_matches", "image_sources"):
        for item in payload.get(section) or []:
            if isinstance(item, dict):
                add_result(item)

    if not titles and not pages:
        return None
    titles = titles[:IMAGE_SEARCH_MAX_ENTITIES]
    return {
        "best_guess_labels": titles[:3],
        "entities": [{"description": title} for title in titles],
        "pages": pages[:IMAGE_SEARCH_MAX_ENTITIES],
    }


def format_image_search_evidence(data: dict | None, max_chars: int = 1200) -> str:
    if not data:
        return ""
    parts: list[str] = []
    labels = [str(label).strip() for label in data.get("best_guess_labels") or [] if str(label).strip()]
    if labels:
        parts.append("Best guess: " + "; ".join(labels[:3]))
    for idx, entity in enumerate(data.get("entities") or [], start=1):
        if not isinstance(entity, dict):
            continue
        desc = str(entity.get("description") or "").strip()
        if not desc:
            continue
        score = entity.get("score")
        if isinstance(score, (int, float)):
            parts.append(f"Entity {idx}: {desc} ({score:.2f})")
        else:
            parts.append(f"Entity {idx}: {desc}")
    for idx, page in enumerate(data.get("pages") or [], start=1):
        if not isinstance(page, dict):
            continue
        title = str(page.get("title") or "").strip()
        url = str(page.get("url") or "").strip()
        if title or url:
            parts.append(f"Page {idx}: {title} {url}".strip())
    return "\n".join(parts)[:max_chars]


def _strip_score(text: str) -> str:
    return re.sub(r"\s*\([0-9.]+\)\s*$", "", text).strip()


def _image_evidence_terms(evidence: str) -> list[str]:
    terms: list[str] = []
    for line in str(evidence or "").splitlines():
        line = line.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith("best guess:"):
            values = line.split(":", 1)[1]
            terms.extend(part.strip() for part in values.split(";") if part.strip())
        elif low.startswith("entity ") and ":" in line:
            terms.append(_strip_score(line.split(":", 1)[1]))
        elif low.startswith("page ") and ":" in line:
            title = line.split(":", 1)[1].strip()
            title = re.sub(r"\s+https?://\S+", "", title).strip()
            if title:
                terms.append(title)

    unique_terms: list[str] = []
    seen: set[str] = set()
    for term in terms:
        clean = re.sub(r"\s+", " ", _strip_score(term)).strip(" ;")
        if not clean:
            continue
        key = clean.lower()
        if key not in seen:
            seen.add(key)
            unique_terms.append(clean)
    return unique_terms


def _generic_only_image_evidence(evidence: str) -> bool:
    terms = _image_evidence_terms(evidence)
    if not terms:
        return False
    return all(_is_generic_image_term(term) for term in terms)


def _is_generic_image_term(term: str) -> bool:
    low = term.lower().strip(" .:-")
    if low in _GENERIC_ONLY_LABELS:
        return True
    if _STRONG_GENERIC_RE.search(low):
        return True
    if _GEO_HINT_RE.search(term) and _TITLEISH_RE.search(term):
        return False
    return any(re.search(rf"\b{re.escape(generic)}\b", low) for generic in _GENERIC_ENTITY_TERMS)


def _image_query_clue(image_evidence: str, max_chars: int = 220) -> str:
    terms = []
    for idx, term in enumerate(_image_evidence_terms(image_evidence)):
        if _is_generic_image_term(term) or _is_web_noise_term(term):
            continue
        terms.append((idx, term))
    terms.sort(key=lambda item: (-_location_term_rank(item[1]), item[0]))
    terms = [term for _, term in terms]
    clue = "; ".join(terms[:5])
    return clue[:max_chars]


def is_location_worthy_image_evidence(evidence: str) -> bool:
    text = str(evidence or "").strip()
    if not text:
        return False
    if _PHONE_OR_CODE_RE.search(text):
        return True
    specific_terms = [term for term in _image_evidence_terms(text) if not _is_generic_image_term(term)]
    if not specific_terms:
        return False
    specific_text = "\n".join(specific_terms)
    if _GEO_HINT_RE.search(specific_text):
        return True
    return any(_TITLEISH_RE.search(term) for term in specific_terms)


def _is_web_noise_term(term: str) -> bool:
    low = term.lower()
    if "@" in term:
        return True
    if "tattoo" in low:
        return True
    if _WEB_NOISE_RE.search(low):
        return True
    if _CAMERA_FILE_RE.search(term):
        return True
    if "| flickr" in low:
        return True
    return False


def _location_term_rank(term: str) -> int:
    score = 0
    if _PHONE_OR_CODE_RE.search(term):
        score += 100
    if _GEO_HINT_RE.search(term):
        score += 90
    if "," in term:
        score += 10
    if _TITLEISH_RE.search(term):
        score += 5
    return score


def image_evidence_to_text_query(level: str, image_evidence: str, parent_context: str = "") -> str:
    clue = _image_query_clue(image_evidence)
    if not clue:
        return ""
    if level == "country":
        return f"{clue} location country"
    if level == "city" and parent_context:
        return f"{clue} location city in {parent_context}"
    if level == "city":
        return f"{clue} location city"
    if level == "street" and parent_context:
        return f"{clue} location landmark or street near {parent_context}"
    return f"{clue} location landmark or street"
