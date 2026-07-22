"""
ai_config.py — shared AI pipeline settings (token control, provider filter, cache).

Environment variables (also documented in docs/HOBBY_OPS.md):

  GOING_AI_LIGHT_BRIEF=1       — compact race briefs (~60–80% fewer input tokens)
  GOING_AI_MAX_TOKENS=768      — cap LLM output tokens (default 768, was 2048)
  GOING_AI_PROVIDERS=groq      — comma list; if set, overrides tier filter
  GOING_AI_TIER=paid           — free | paid (default paid = free + subscription voices)
  GOING_AI_DEV=1               — dev mode: groq-only + rule-based credits + cache on
  GOING_AI_RULE_CREDITS=1      — skip LLM credit-allocation calls (deterministic stakes)
  GOING_AI_CACHE=1             — cache API responses by (provider, brief hash)
  GOING_AI_RESUME=1            — skip race+provider already in day's report (default on; --fresh to override)
  GOING_AI_SKIP_NAP=1          — skip the day-level NAP LLM call
  GOING_AI_OLLAMA=1            — Lucy local Ollama voice (default on; sequential + race-capped)
  GOING_OLLAMA_MODEL=mistral:latest
  GOING_OLLAMA_MAX_RACES=8     — max new Ollama analyses per run (protect Jetson)
  GOING_OLLAMA_TIMEOUT=180     — seconds per race; soft-fail on hang
  GOING_OLLAMA_PAUSE_SECS=60   — cool-down between Ollama races (protect Jetson)
  CURSOR_API_KEY=…             — Cursor Dashboard API key (paid / subscription usage)
  GOING_AI_CURSOR_MODEL=composer-2.5
  GOING_CURSOR_BASE_URL=…      — optional OpenAI-compatible proxy instead of SDK
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

REPO = Path(__file__).parent
CACHE_DIR = REPO / "ai_reports" / "cache"

# Provider registry keys — must match ai_daily_analysis.PROVIDERS
ALL_PROVIDER_KEYS = (
    "claude", "openai", "gemini", "groq", "cerebras", "nvidia", "cursor", "ollama",
)

# Free-tier / hobby keys (no subscription bill beyond free quotas)
# gemini parked — idle in practice; re-add to re-enable
# ollama (Lucy local) is gated by GOING_AI_OLLAMA — see providers_for_tier()
FREE_PROVIDERS = frozenset({"groq", "cerebras", "nvidia"})
# Enabled automatically when GOING_AI_TIER=paid (Cursor uses your Cursor subscription)
PAID_PROVIDERS = frozenset({"cursor"})
# Also paid, but only when listed in GOING_AI_PROVIDERS (avoid surprise Anthropic/OpenAI bills)
EXTRA_PAID_PROVIDERS = frozenset({"claude", "openai"})

PROVIDER_ENV = {
    "claude": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
    "cursor": "CURSOR_API_KEY",
    # ollama: no API key — see ollama_enabled() / provider_key_present()
}


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def dev_mode() -> bool:
    return _env_bool("GOING_AI_DEV")


def light_brief() -> bool:
    if dev_mode():
        return True
    return _env_bool("GOING_AI_LIGHT_BRIEF", default=True)


def max_output_tokens(provider: str | None = None) -> int:
    try:
        if provider == "cerebras":
            v = os.getenv("GOING_AI_MAX_TOKENS_CEREBRAS", "1024")
            return max(256, int(v))
        if provider == "cursor":
            v = os.getenv("GOING_AI_MAX_TOKENS_CURSOR", "1024")
            return max(256, int(v))
        return max(256, int(os.getenv("GOING_AI_MAX_TOKENS", "768")))
    except ValueError:
        return 1024 if provider in ("cerebras", "cursor") else 768


def use_cache() -> bool:
    if dev_mode():
        return True
    return _env_bool("GOING_AI_CACHE", default=True)


def resume_reports() -> bool:
    """If True, reuse {provider}_analysis already stored in the day's report JSON."""
    if _env_bool("GOING_AI_NO_RESUME"):
        return False
    return _env_bool("GOING_AI_RESUME", default=True)


def rule_based_credits() -> bool:
    if dev_mode():
        return True
    return _env_bool("GOING_AI_RULE_CREDITS", default=False)


def skip_nap() -> bool:
    return _env_bool("GOING_AI_SKIP_NAP", default=False)


def ollama_enabled() -> bool:
    """Local Lucy Ollama as a daily voice. Default on; turn off with GOING_AI_OLLAMA=0."""
    if dev_mode():
        return False
    return _env_bool("GOING_AI_OLLAMA", default=True)


def ollama_model() -> str:
    return (os.getenv("GOING_OLLAMA_MODEL") or "mistral:latest").strip() or "mistral:latest"


def ollama_max_races() -> int:
    """Hard cap on new Ollama race analyses per analyze_today run."""
    try:
        return max(0, int(os.getenv("GOING_OLLAMA_MAX_RACES", "8")))
    except ValueError:
        return 8


def ollama_timeout() -> int:
    """Seconds per Ollama generate call; soft-fail on timeout (don't block the day)."""
    try:
        return max(30, int(os.getenv("GOING_OLLAMA_TIMEOUT", "180")))
    except ValueError:
        return 180


def ollama_pause_secs() -> int:
    """Seconds to sleep after each Ollama race (cool-down for Jetson)."""
    try:
        return max(0, int(os.getenv("GOING_OLLAMA_PAUSE_SECS", "60")))
    except ValueError:
        return 60


def ai_tier() -> str:
    """'free' = free-tier voices only; 'paid' = free + paid/subscription (default)."""
    raw = os.getenv("GOING_AI_TIER", "").strip().lower()
    if raw in ("free", "paid"):
        return raw
    try:
        import settings as user_settings
        t = str(user_settings.load().get("ai_tier", "paid") or "paid").strip().lower()
        if t in ("free", "paid"):
            return t
    except Exception:
        pass
    return "paid"


def providers_for_tier(tier: str | None = None) -> frozenset[str]:
    t = (tier or ai_tier()).lower()
    base = set(FREE_PROVIDERS)
    if ollama_enabled():
        base.add("ollama")
    if t == "free":
        return frozenset(base)
    return frozenset(base | PAID_PROVIDERS)


def provider_filter() -> set[str] | None:
    """None = all configured; else restrict to these keys.

    Precedence:
      1. GOING_AI_PROVIDERS (explicit list) always wins
      2. GOING_AI_DEV → groq only
      3. GOING_AI_TIER / settings.ai_tier (default paid)
    When GOING_AI_OLLAMA=1 and an explicit list omits ollama, ollama is still
    added (local voice is opt-in via its own flag, not the cloud list).
    """
    raw = os.getenv("GOING_AI_PROVIDERS", "").strip()
    if raw:
        allowed = {p.strip().lower() for p in raw.split(",") if p.strip()}
        if ollama_enabled():
            allowed.add("ollama")
        else:
            allowed.discard("ollama")
        try:
            import settings as user_settings
            s = user_settings.load()
            for key in list(allowed):
                if not user_settings._as_bool(s.get(f"voice_{key}", True)):
                    allowed.discard(key)
        except Exception:
            pass
        return allowed
    if dev_mode():
        return {"groq"}
    allowed = set(providers_for_tier())
    try:
        import settings as user_settings
        s = user_settings.load()
        for key in list(allowed):
            if key == "model":
                continue
            if not user_settings._as_bool(s.get(f"voice_{key}", True)):
                allowed.discard(key)
        if not user_settings._as_bool(s.get("voice_ollama", True)):
            allowed.discard("ollama")
    except Exception:
        pass
    return allowed


def provider_key_present(key: str) -> bool:
    if key == "ollama":
        return ollama_enabled()
    env = PROVIDER_ENV.get(key)
    return bool(env and os.getenv(env))


def active_provider_keys() -> list[str]:
    """Keys that pass the filter and have an API key set (display/order helpers)."""
    allowed = provider_filter()
    out = []
    for key in ALL_PROVIDER_KEYS:
        if allowed is not None and key not in allowed:
            continue
        if provider_key_present(key):
            out.append(key)
    return out


def brief_hash(brief: str) -> str:
    return hashlib.sha256(brief.encode("utf-8")).hexdigest()[:16]


def cache_get(provider: str, brief: str) -> str | None:
    if not use_cache():
        return None
    path = CACHE_DIR / f"{provider}_{brief_hash(brief)}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def cache_put(provider: str, brief: str, text: str) -> None:
    if not use_cache() or not text or text.startswith("ERROR"):
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{provider}_{brief_hash(brief)}.txt"
    path.write_text(text, encoding="utf-8")
