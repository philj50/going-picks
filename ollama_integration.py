"""
Local-LLM helpers for Ollama on Lucy (mistral:latest by default).

Used by overnight IDEATE (via overnight_runner) and optional desktop callers.
Daily race analysis goes through ai_daily_analysis.call_ollama with hard caps
(GOING_AI_OLLAMA / GOING_OLLAMA_MAX_RACES) — see docs/HOBBY_OPS.md.

Read-only and best-effort: if Ollama is unreachable, callers get None back
rather than an exception — this must never block a page render.
"""
from __future__ import annotations
import hashlib
import json
import os
import time
from pathlib import Path

import requests

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
TIMEOUT_SECONDS = 30   # cold-loads in ~20-25s when Ollama has unloaded the
                       # model from memory after idling; warm calls are ~1s
CACHE_PATH = Path("/mnt/nas/going/cache/opinions.json")
CACHE_TTL_SECONDS = 24 * 60 * 60


def _model() -> str:
    try:
        import ai_config
        return ai_config.ollama_model()
    except Exception:
        return (os.getenv("GOING_OLLAMA_MODEL") or "mistral:latest").strip() or "mistral:latest"


# Back-compat alias for callers that imported MODEL
MODEL = "mistral:latest"


def _cache_key(context: dict) -> str:
    """Stable hash of the context dict + model so identical race/horse lookups
    reuse the same cached opinion regardless of key order, but switching
    model never serves a stale opinion from a different model."""
    blob = json.dumps(
        {"model": _model(), "context": context}, sort_keys=True, ensure_ascii=False
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _load_cache() -> dict:
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except Exception:
        pass    # NAS cache unreachable — caller still gets their result


def _cached_opinion(key: str) -> str | None:
    entry = _load_cache().get(key)
    if not entry:
        return None
    if time.time() - entry.get("ts", 0) > CACHE_TTL_SECONDS:
        return None
    return entry.get("opinion")


def _store_opinion(key: str, opinion: str) -> None:
    cache = _load_cache()
    cache[key] = {"opinion": opinion, "ts": time.time()}
    _save_cache(cache)


def _build_prompt(context: dict) -> str:
    lines = [f"{k}: {v}" for k, v in context.items() if v not in (None, "")]
    return (
        "You are a concise horse racing analyst. Given this runner's context, "
        "give a short (2-3 sentence) opinion on its chances. Be specific, not generic.\n\n"
        + "\n".join(lines)
    )


def is_cached(context: dict) -> bool:
    """True if a fresh (non-expired) cached opinion exists for this context."""
    return _cached_opinion(_cache_key(context)) is not None


def warm() -> bool:
    """Trigger a trivial generate call so Ollama loads the model into memory.
    Cold load takes ~20-25s; this lets that cost happen in the background
    (fired on page load) instead of on the user's first real request. Returns
    whether Ollama responded — never raises."""
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": _model(), "prompt": "hi", "stream": False},
            timeout=TIMEOUT_SECONDS,
        )
        return resp.ok
    except requests.exceptions.RequestException:
        return False


def get_ai_opinion(context: dict) -> str | None:
    """Cached local-LLM opinion on a runner. Returns the opinion text, or
    None if Ollama is unavailable/times out — never raises."""
    key = _cache_key(context)
    cached = _cached_opinion(key)
    if cached is not None:
        return cached

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": _model(), "prompt": _build_prompt(context), "stream": False},
            timeout=TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        opinion = resp.json().get("response", "").strip()
    except requests.exceptions.RequestException:
        return None
    except (ValueError, json.JSONDecodeError):
        return None

    if not opinion:
        return None
    _store_opinion(key, opinion)
    return opinion
