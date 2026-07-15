"""Optional web search fallback for GeoBayes-style evidence enhancement.

The pipeline must remain runnable on offline HPC nodes, so this module is
strictly opt-in. Set WEB_SEARCH_ENABLED=1 and TAVILY_API_KEY to enable Tavily.
Network/API failures return no evidence instead of failing the evaluation.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from config import WEB_SEARCH_MAX_RESULTS, WEB_SEARCH_TIMEOUT


class WebSearchClient:
    def __init__(self):
        self.enabled = os.environ.get("WEB_SEARCH_ENABLED", "0").lower() in {
            "1", "true", "yes", "on"
        }
        self.api_key = os.environ.get("TAVILY_API_KEY", "")
        self.provider = os.environ.get("WEB_SEARCH_PROVIDER", "tavily").lower()
        if self.enabled and self.provider != "tavily":
            print(f"[WEB] Unsupported WEB_SEARCH_PROVIDER={self.provider}; disabled.")
            self.enabled = False
        if self.enabled and not self.api_key:
            print("[WEB] WEB_SEARCH_ENABLED=1 but TAVILY_API_KEY is missing; disabled.")
            self.enabled = False
        if self.enabled:
            print("[WEB] Tavily search fallback enabled.")

    def search(self, query: str) -> dict | None:
        if not self.enabled or not query.strip():
            return None
        try:
            return self._tavily_search(query)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            print(f"[WEB] search failed: {exc}")
            return None

    def _tavily_search(self, query: str) -> dict | None:
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": WEB_SEARCH_MAX_RESULTS,
            "include_answer": True,
            "include_raw_content": False,
        }
        req = urllib.request.Request(
            "https://api.tavily.com/search",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=WEB_SEARCH_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data if isinstance(data, dict) else None


def format_search_evidence(data: dict | None, max_chars: int = 1200) -> str:
    if not data:
        return ""
    parts: list[str] = []
    answer = (data.get("answer") or "").strip()
    if answer:
        parts.append(f"Answer: {answer}")
    for idx, item in enumerate(data.get("results") or [], start=1):
        if not isinstance(item, dict):
            continue
        title = (item.get("title") or "").strip()
        content = (item.get("content") or "").strip()
        url = (item.get("url") or "").strip()
        line = f"{idx}. {title}: {content}"
        if url:
            line += f" ({url})"
        parts.append(line)
    text = "\n".join(p for p in parts if p)
    return text[:max_chars]
