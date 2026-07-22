"""
ai_daily_analysis.py — Daily AI evaluation of today's races.

Runs at 7 AM daily on Lucy. For each race today:
1. Compiles comprehensive race + horse data
2. Gets predictions from the model (which horse will win, why)
3. Calls Claude, OpenAI, and Ollama for sophisticated analysis
4. Saves structured results to reports/

Usage:
    python3 ai_daily_analysis.py                # analyze today's races
    python3 ai_daily_analysis.py --dry-run      # test without API calls

Cron (Lucy):
    0 7 * * * cd /mnt/nas/going/repo && python3 ai_daily_analysis.py >> /mnt/nas/going/logs/cron.log 2>&1
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sqlite3
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ai_config
import brief_builder
import going_paths
import race_stats

# Load .env file for API keys
def load_env_file():
    """Load .env file and set environment variables."""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        try:
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        # setdefault: an explicitly-set (even empty) shell var
                        # wins over .env, so providers can be disabled per-run
                        os.environ.setdefault(key.strip(), value.strip())
        except Exception:
            pass

load_env_file()

REPO = Path(__file__).parent
LOG_DIR = REPO / "logs"
REPORT_DIR = REPO / "ai_reports"

def connect_db():
    path = going_paths.db_path()
    if not path.exists():
        print(f"WARNING: DB not found at {path} — corpus stats skipped in briefs")
        return None
    con = sqlite3.connect(str(path))
    con.row_factory = sqlite3.Row
    return con

def load_today_card(day: str = "today") -> dict:
    """Load the racecard for today.json or tomorrow.json."""
    path = REPO / ("tomorrow.json" if day == "tomorrow" else "today.json")
    if not path.exists():
        return {"races": []}
    return json.loads(path.read_text())

def load_model_picks(day: str = "today") -> dict:
    """Fetch the model's actual board picks from the running dashboard.
    Returns {race_name: [pick dicts sorted by win_prob desc]}.
    Falls back to scoring tomorrow.json/today.json locally if serve is down."""
    try:
        import requests
        data = requests.get(f"http://localhost:8011/api/picks.json?day={day}", timeout=180).json()
        by_race = {}
        for p in data.get("picks", []):
            by_race.setdefault(p.get("race"), []).append(p)
        for lst in by_race.values():
            lst.sort(key=lambda p: p.get("win_prob") or 0, reverse=True)
        if by_race:
            return by_race
    except Exception as e:
        print(f"WARNING: could not fetch model picks from serve ({e}) — scoring card locally")

    return _model_picks_from_card(load_today_card(day), day)


def _model_picks_from_card(card: dict, day: str = "today") -> dict:
    """Score each race in a card file without the web server."""
    from fetch import load_card
    from scoring import score_race
    from value import assess

    path = REPO / ("tomorrow.json" if day == "tomorrow" else "today.json")
    by_race = {}
    try:
        loaded = load_card(str(path))
        for race in loaded.races:
            score_race(race)
            picks = []
            for runner in race.runners:
                if not runner.odds_decimal:
                    continue
                v = assess(runner, race)
                picks.append({
                    "horse": runner.name,
                    "win_prob": round(v.decision_prob * 100, 1),
                    "confidence": round(v.confidence * 100),
                    "odds": runner.odds_decimal,
                    "reasons": v.reasons or [],
                })
            picks.sort(key=lambda p: p.get("win_prob") or 0, reverse=True)
            if picks:
                by_race[race.name] = picks
    except Exception as e:
        print(f"WARNING: local model scoring failed ({e})")
    return by_race


def get_model_prediction(picks_by_race: dict, race: dict) -> dict:
    """The model's real prediction for a race, from the live board."""
    picks = picks_by_race.get(race.get("name")) or []
    return {
        "top_3": [
            {"name": p["horse"], "win_prob": p.get("win_prob"),
             "confidence": p.get("confidence"), "odds": p.get("odds"),
             "reasons": p.get("reasons") or []}
            for p in picks[:3]
        ],
    }

def get_trainer_form(con, trainer_name: str, days=30) -> dict:
    """Get recent form for a trainer."""
    # Placeholder - would query actual race results
    return {"wins": 0, "roi": 0.0, "runs": 0}

def get_jockey_form(con, jockey_name: str, days=30) -> dict:
    """Get recent form for a jockey."""
    # Placeholder - would query actual race results
    return {"wins": 0, "roi": 0.0, "runs": 0}

def compile_race_brief(con, race: dict, prediction: dict, light: bool | None = None) -> str:
    """Compile race brief for AI voices. `light=True` (default via ai_config) skips
    per-runner corpus stats and caps the field list — ~60–80% fewer input tokens."""
    if light is None:
        light = ai_config.light_brief()
    runners = race.get("runners", [])
    if light:
        runners = sorted(
            runners,
            key=lambda r: r.get("odds_decimal") or 999,
        )[:8]

    dist = race.get("distance_f")
    dist_str = f"{dist}f" if dist else "?"
    brief = f"""
RACE ANALYSIS BRIEF
==================

Race: {race.get('name', 'Unknown')}
Course: {race.get('course', '?')} | Time: {race.get('time', '?')}
Type: {'Flat' if race.get('is_flat') else 'Jumps'} | Handicap: {race.get('is_handicap', False)}
Distance: {dist_str} | Going: {race.get('going') or '?'} | Class: {race.get('race_class') or '?'}
Field: {len(runners)} runners

MODEL'S PREDICTION (top-rated by blended win probability)
------------------
"""
    top3 = prediction.get("top_3", [])
    if not top3:
        brief += "The model has no qualifying pick in this race.\n"
    for i, pred in enumerate(top3, 1):
        wp = pred.get("win_prob")
        conf = pred.get("confidence")
        odds = pred.get("odds") or 0
        brief += (f"{i}. {pred['name']}: win prob {wp:.0f}%" if wp is not None
                  else f"{i}. {pred['name']}:")
        if conf is not None:
            brief += f", confidence {conf}%"
        if odds:
            brief += f" (odds {odds:.1f})"
        brief += "\n"
        if pred.get("reasons"):
            brief += f"   Model reasoning: {'; '.join(pred['reasons'])}\n"

    brief += "\nRUNNERS\n-------\n"
    # Race-day date for point-in-time stats / connection notes
    race_day = race.get("date") or dt.date.today().isoformat()
    for r in runners:
        odds = r.get('odds_decimal', 0)
        odds_str = f" @ {odds:.1f}" if odds > 0 else ""
        form = r.get('last_positions', [])[:4]
        form_str = "".join(str(p) for p in form) if form else "?"
        if light:
            brief += (f"- {r['name']}{odds_str} | {r.get('trainer', '?')}/{r.get('jockey', '?')} "
                      f"| form {form_str}\n")
            continue
        brief += f"\n{r['name']} {odds_str}\n"
        brief += f"  Trainer: {r.get('trainer', '?')}\n"
        brief += f"  Jockey: {r.get('jockey', '?')}\n"
        weight = r.get('weight_lbs', '?')
        draw = r.get('draw', '?')
        brief += f"  Weight: {weight}lbs | Draw: {draw}\n"
        if r.get('speed_figure') is not None:
            brief += f"  Speed figure: {r['speed_figure']}\n"
        if r.get('days_since_run') is not None:
            brief += f"  Days since last run: {r['days_since_run']}\n"
        if form:
            brief += f"  Recent form (latest first): {', '.join(str(p) for p in form)}\n"
        if r.get('headgear'):
            brief += f"  Headgear: {r['headgear']}\n"
        if con is not None:
            try:
                stats = race_stats.runner_stats(
                    con, r['name'], r.get('trainer'), r.get('jockey'),
                    race.get('course'), race.get('distance_f'), race.get('going'),
                    race_day)
                for line in race_stats.stats_lines(stats):
                    brief += line + "\n"
            except Exception:
                pass

    # Recent AI post-mortems for horses/jockeys on this card (light + full)
    try:
        import ai_horse_postmortem as hpm
        notes = hpm.connection_notes_for_brief(
            runners, before=race_day, days=21, limit=6 if light else 8)
        if notes:
            brief += "\n" + notes
    except Exception:
        pass

    try:
        import day_chronicle
        prev = (dt.date.fromisoformat(race_day) - dt.timedelta(days=1)).isoformat()
        ex = day_chronicle.excerpt_for_prompt(prev, max_chars=800)
        if ex:
            brief += "\n\nRECENT DAY LOG (yesterday's card — context only):\n" + ex
    except Exception:
        pass

    return brief

def _prompt_lessons_block() -> str:
    """Shared lessons snippet for race prompts and day NAP."""
    if os.getenv("GOING_AI_NO_LESSONS", "").strip().lower() in ("1", "true", "yes"):
        return ""   # lessons-efficacy A/B: suppress entirely
    try:
        import ai_learning
        lessons = ai_learning.load_prompt_lessons()
    except Exception:
        return ""
    if not lessons:
        return ""
    return (
        "\n\nRecent lessons from settled post-mortems "
        "(apply if relevant; do not force):\n"
        f"{lessons}\n"
    )


_CALIB_CACHE: dict = {}
_CALIB_V2_START = "2026-07-16"   # first day of win_prob verdicts


def _calibration_records(voice: str, days: int = 12) -> list:
    """(win_prob, won) pairs for the voice's recent settled verdicts."""
    import ai_consensus
    today = dt.date.today()
    win_cache = _CALIB_CACHE.setdefault("_winners", {})
    con = None
    try:
        import ai_track_record as tr
        con = sqlite3.connect(tr.DB_PATH)
    except Exception:
        con = None
    out = []
    for i in range(1, days + 1):
        d = (today - dt.timedelta(days=i)).isoformat()
        if d < _CALIB_V2_START:
            break
        path = REPORT_DIR / f"ai_analysis_{d}.json"
        if not path.exists():
            continue
        try:
            rep = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for race in rep.get("races") or []:
            v = ai_consensus.extract_verdict(race.get(f"{voice}_analysis") or "")
            if not v or v.get("no_bet") or v.get("win_prob") is None:
                continue
            key = (d, race.get("course") or "", race.get("time") or "")
            if key not in win_cache:
                w = None
                if con is not None:
                    try:
                        import ai_track_record as tr
                        w = tr.find_winner(con, d, key[1], key[2])
                    except Exception:
                        w = None
                win_cache[key] = w
            w = win_cache.get(key)
            if not w:
                continue
            out.append((float(v["win_prob"]), _norm(v["pick"]) == _norm(w)))
    if con is not None:
        con.close()
    return out


def _bands_text(records: list) -> str:
    """Per-band hit rates, n-gated — teaches WHERE a voice is overconfident."""
    bands = (("≥0.40", 0.40, 1.01), ("0.25–0.40", 0.25, 0.40),
             ("<0.25", -0.01, 0.25))
    bits = []
    for label, lo, hi in bands:
        sel = [won for p, won in records if lo <= p < hi]
        if len(sel) >= 10:
            bits.append(f"claims {label} won {round(100 * sum(sel) / len(sel))}%"
                        f" (n={len(sel)})")
    return " · ".join(bits)


def _calibration_line(voice: str | None) -> str:
    """Calibration feedback so a voice reports probability, not enthusiasm.
    Banded by the voice's own win_prob claims when enough have settled."""
    if not voice:
        return ""
    ck = ("line", voice, dt.date.today().isoformat())
    if ck in _CALIB_CACHE:
        return _CALIB_CACHE[ck]
    line = ""
    try:
        bands = _bands_text(_calibration_records(voice))
        if bands:
            line = (f"\nYOUR CALIBRATION (recent settled picks): {bands} — "
                    "report your TRUE win probability, not enthusiasm.\n")
    except Exception:
        line = ""
    if not line:
        try:
            data = json.loads((REPORT_DIR / "track_record.json").read_text(encoding="utf-8"))
            rec = (data.get("hit_rates") or {}).get(voice) or {}
            picks, rate = rec.get("picks") or 0, rec.get("rate")
            if picks >= 20 and rate is not None:
                line = (f"\nYOUR RECORD: your staked picks have won {round(rate * 100)}% "
                        f"of the time (n={picks}) — report your TRUE win probability, "
                        "not enthusiasm.\n")
        except Exception:
            line = ""
    _CALIB_CACHE[ck] = line
    return line


_PROSE_INSTR = {
    0: "Output the VERDICT line only — no further text.",
    1: "Then ONE short sentence of reasoning only.",
    2: "Then at most 2 short paragraphs: your winner case, and the biggest risk to it.",
}


def build_prompt(brief: str, voice: str | None = None) -> str:
    """Shared analysis prompt for all AIs. VERDICT comes first so truncation
    still leaves pick + win_prob (Cerebras often hit token limits)."""
    # v2 briefs carry their own race-contextual lessons; only add the global
    # block for briefs (e.g. backtest legacy) that lack one.
    lessons_block = ("" if "essons from settled post-mortems" in brief
                     else _prompt_lessons_block())
    calib = _calibration_line(voice)
    prose = _PROSE_INSTR[brief_builder.prose_level(voice)]
    return f"""You are an expert horse racing analyst. Estimate each runner's TRUE chance of winning; the market odds are context, not the answer.

{brief}{lessons_block}{calib}
Reply format (strict — do not skip):
Line 1 must be ONLY this JSON (one line, no markdown fences):
VERDICT: {{"pick": "<horse name exactly as listed, or NO BET if you would not stake this race>", "win_prob": <0.0-1.0 your probability the pick wins>, "second": "<next-best horse, or none>", "agrees_with_model": <true, false, or null if the model has no pick>, "key_risk": "<one short sentence>", "missing_factors": ["<up to 3 short factors>"]}}
{prose}"""


def call_claude(brief: str, dry_run: bool = False) -> str:
    """Call Claude API for analysis."""
    if dry_run:
        return "[Claude analysis would be called here]"

    return _claude_raw(build_prompt(brief, "claude"))


def _with_retry(fn, tries: int = 3, base_sleep: int = 20) -> str:
    """Retry transient API failures (rate limits, overload, low credit races)."""
    import time
    out = ""
    for attempt in range(tries):
        out = fn()
        if not (isinstance(out, str) and out.startswith("ERROR")):
            return out
        low = out.lower()
        # Daily-quota exhaustion won't recover within a run — don't retry
        # (Groq phrases it "tokens per day (TPD)"; others say quota/billing)
        if any(t in low for t in ("quota", "billing", "tokens per day",
                                  "(tpd)", "requests per day", "(rpd)")):
            return out
        transient = any(t in low for t in ("429", "rate", "overload", "timeout", "529"))
        if not transient:
            return out
        time.sleep(base_sleep * (attempt + 1))
    return out


def _claude_raw(prompt: str) -> str:
    return _with_retry(lambda: _claude_raw_once(prompt))


def _claude_raw_once(prompt: str) -> str:
    try:
        from anthropic import Anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return "ERROR: ANTHROPIC_API_KEY not set"

        client = Anthropic(api_key=api_key)

        message = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=ai_config.max_output_tokens(),
            messages=[{"role": "user", "content": prompt}]
        )
        return "".join(b.text for b in message.content if b.type == "text")
    except Exception as e:
        return f"ERROR calling Claude: {e}"

def call_openai(brief: str, dry_run: bool = False) -> str:
    """Call OpenAI API for analysis."""
    if dry_run:
        return "[OpenAI analysis would be called here]"

    return _openai_raw(build_prompt(brief, "openai"))


def _openai_raw(prompt: str) -> str:
    return _with_retry(lambda: _openai_raw_once(prompt))


def _openai_raw_once(prompt: str) -> str:
    try:
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "ERROR: OPENAI_API_KEY not set"

        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model="gpt-4-turbo",
            max_tokens=ai_config.max_output_tokens(),
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"ERROR calling OpenAI: {e}"

def _call_openai_compatible(prompt: str, base_url: str, model: str, key_env: str,
                            label: str, dry_run: bool = False) -> str:
    """Shared caller for OpenAI-compatible endpoints (NVIDIA NIM, Gemini).
    Takes the FINAL prompt — callers wrap with build_prompt() themselves."""
    if dry_run:
        return f"[{label} analysis would be called here]"
    return _with_retry(lambda: _call_openai_compatible_once(prompt, base_url, model, key_env, label))


def _call_openai_compatible_once(prompt: str, base_url: str, model: str, key_env: str,
                                 label: str, max_tokens: int | None = None) -> str:
    try:
        from openai import OpenAI
        api_key = os.getenv(key_env)
        if not api_key:
            return f"ERROR: {key_env} not set"
        client = OpenAI(api_key=api_key, base_url=base_url)
        tokens = max_tokens if max_tokens is not None else ai_config.max_output_tokens()
        response = client.chat.completions.create(
            model=model,
            max_tokens=tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        return f"ERROR calling {label}: {e}"


def call_gemini(brief: str, dry_run: bool = False) -> str:
    """Google Gemini via its OpenAI-compatible endpoint (free tier)."""
    return _call_openai_compatible(
        build_prompt(brief, "gemini"), "https://generativelanguage.googleapis.com/v1beta/openai/",
        "gemini-2.0-flash", "GEMINI_API_KEY", "Gemini", dry_run)


def call_nvidia(brief: str, dry_run: bool = False) -> str:
    """NVIDIA NIM cloud API (free dev credits)."""
    return _call_openai_compatible(
        build_prompt(brief, "nvidia"), "https://integrate.api.nvidia.com/v1",
        "meta/llama-3.3-70b-instruct", "NVIDIA_API_KEY", "NVIDIA", dry_run)


def call_groq(brief: str, dry_run: bool = False) -> str:
    """Groq free tier — serves Llama 3.3 70B with a daily-renewing quota."""
    return _call_openai_compatible(
        build_prompt(brief, "groq"), "https://api.groq.com/openai/v1",
        "llama-3.3-70b-versatile", "GROQ_API_KEY", "Groq", dry_run)


def call_cerebras(brief: str, dry_run: bool = False) -> str:
    """Cerebras free tier — gpt-oss-120b on fast inference hardware."""
    if dry_run:
        return "[Cerebras analysis would be called here]"
    tok = ai_config.max_output_tokens("cerebras")
    return _with_retry(lambda: _call_openai_compatible_once(
        build_prompt(brief, "cerebras"), "https://api.cerebras.ai/v1",
        "gpt-oss-120b", "CEREBRAS_API_KEY", "Cerebras", max_tokens=tok))


def call_cursor(brief: str, dry_run: bool = False) -> str:
    """Cursor subscription voice via CURSOR_API_KEY (SDK or optional OpenAI proxy)."""
    if dry_run:
        return "[Cursor analysis would be called here]"
    if not os.getenv("CURSOR_API_KEY"):
        return "ERROR: CURSOR_API_KEY not set"
    prompt = build_prompt(brief, "cursor")
    base = (os.getenv("GOING_CURSOR_BASE_URL") or "").strip()
    model = os.getenv("GOING_AI_CURSOR_MODEL", "composer-2.5")
    if base:
        return _call_openai_compatible(
            prompt, base.rstrip("/"), model, "CURSOR_API_KEY", "Cursor")
    return _with_retry(lambda: _cursor_sdk_once(prompt, model))


def _cursor_sdk_once(prompt: str, model: str) -> str:
    try:
        from cursor_sdk import Agent, AgentOptions, LocalAgentOptions
    except ImportError:
        return ("ERROR: cursor-sdk not installed "
                "(pip install cursor-sdk) — or set GOING_CURSOR_BASE_URL")
    # Lucy installs the bridge under ~/.local/bin — ensure it's findable
    local_bin = str(Path.home() / ".local" / "bin")
    path = os.environ.get("PATH", "")
    if local_bin not in path.split(os.pathsep):
        os.environ["PATH"] = local_bin + os.pathsep + path
    text_prompt = (
        prompt
        + "\n\nIMPORTANT: Reply with text only. Include the VERDICT line. "
        "Do not edit files, run shell commands, or use tools."
    )
    try:
        result = Agent.prompt(
            text_prompt,
            AgentOptions(
                api_key=os.environ["CURSOR_API_KEY"],
                model=model,
                local=LocalAgentOptions(cwd=str(REPO)),
            ),
        )
        out = getattr(result, "result", None) or ""
        if isinstance(out, str) and out.strip():
            return out
        # Some SDK builds put the final assistant text elsewhere
        status = getattr(result, "status", None)
        return f"ERROR calling Cursor: empty result (status={status})"
    except Exception as e:
        return f"ERROR calling Cursor: {e}"


def call_nvidia_raw(prompt: str) -> str:
    """NVIDIA with a caller-supplied prompt (used for the day-level NAP pick)."""
    return _call_openai_compatible(
        prompt, "https://integrate.api.nvidia.com/v1",
        "meta/llama-3.3-70b-instruct", "NVIDIA_API_KEY", "NVIDIA")


def _gemini_raw(prompt: str) -> str:
    return _call_openai_compatible(
        prompt, "https://generativelanguage.googleapis.com/v1beta/openai/",
        "gemini-2.0-flash", "GEMINI_API_KEY", "Gemini")


def _groq_raw(prompt: str) -> str:
    return _call_openai_compatible(
        prompt, "https://api.groq.com/openai/v1",
        "llama-3.3-70b-versatile", "GROQ_API_KEY", "Groq")


def _cerebras_raw(prompt: str) -> str:
    return _call_openai_compatible(
        prompt, "https://api.cerebras.ai/v1",
        "gpt-oss-120b", "CEREBRAS_API_KEY", "Cerebras")


def _cursor_raw(prompt: str) -> str:
    base = (os.getenv("GOING_CURSOR_BASE_URL") or "").strip()
    model = os.getenv("GOING_AI_CURSOR_MODEL", "composer-2.5")
    if base:
        return _call_openai_compatible(
            prompt, base.rstrip("/"), model, "CURSOR_API_KEY", "Cursor")
    return _cursor_sdk_once(prompt, model)


def _ollama_raw(prompt: str) -> str:
    """Raw Ollama generate for day-level prompts (credit allocation, etc.)."""
    model = ai_config.ollama_model()
    timeout = ai_config.ollama_timeout()
    try:
        import requests

        max_tok = ai_config.max_output_tokens("ollama")
        response = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tok},
            },
            timeout=timeout,
        )
        response.raise_for_status()
        text = (response.json().get("response") or "").strip()
        if not text:
            return f"ERROR calling Ollama: empty response from {model}"
        return text
    except Exception as e:
        return f"ERROR calling Ollama: {e}"


# Raw-prompt entry point per voice (for day-level prompts like credit allocation)
RAW_CALLERS = {
    "claude": _claude_raw,
    "openai": _openai_raw,
    "gemini": _gemini_raw,
    "nvidia": call_nvidia_raw,
    "groq": _groq_raw,
    "cerebras": _cerebras_raw,
    "cursor": _cursor_raw,
    "ollama": _ollama_raw,
}


def call_ollama(brief: str, dry_run: bool = False) -> str:
    """Call Ollama on Lucy for analysis. Soft-fail on timeout/unreachable."""
    if dry_run:
        return "[Ollama analysis would be called here]"

    model = ai_config.ollama_model()
    timeout = ai_config.ollama_timeout()
    try:
        import requests

        prompt = build_prompt(brief, "ollama")
        max_tok = ai_config.max_output_tokens("ollama")
        response = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tok},
            },
            timeout=timeout,
        )
        response.raise_for_status()
        text = (response.json().get("response") or "").strip()
        if not text:
            return f"ERROR calling Ollama: empty response from {model}"
        return text
    except Exception as e:
        return f"ERROR calling Ollama: {e}"

# (key, call function, required env var — None means always available when filtered in)
PROVIDERS = [
    ("claude", call_claude, "ANTHROPIC_API_KEY"),
    ("openai", call_openai, "OPENAI_API_KEY"),
    ("gemini", call_gemini, "GEMINI_API_KEY"),
    ("groq", call_groq, "GROQ_API_KEY"),
    ("cerebras", call_cerebras, "CEREBRAS_API_KEY"),
    ("nvidia", call_nvidia, "NVIDIA_API_KEY"),
    ("cursor", call_cursor, "CURSOR_API_KEY"),
    ("ollama", call_ollama, None),  # Lucy local — gated by GOING_AI_OLLAMA + race caps
]


def _filtered_providers():
    """PROVIDERS rows allowed by GOING_AI_PROVIDERS / GOING_AI_DEV / Ollama flag."""
    allowed = ai_config.provider_filter()
    out = []
    for key, fn, key_env in PROVIDERS:
        if allowed is not None and key not in allowed:
            continue
        if key == "ollama":
            if not ai_config.ollama_enabled():
                continue
        elif key_env and not os.getenv(key_env):
            continue
        out.append((key, fn, key_env))
    return out


def _norm_race_part(value) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _race_key(race: dict) -> str:
    return "|".join((
        _norm_race_part(race.get("course")),
        _norm_race_part(race.get("time")),
        _norm_race_part(race.get("name")),
    ))


def _race_key_from_entry(entry: dict) -> str:
    return "|".join((
        _norm_race_part(entry.get("course")),
        _norm_race_part(entry.get("time")),
        _norm_race_part(entry.get("race")),
    ))


def _race_key_course_time(course, time) -> str:
    return f"{_norm_race_part(course)}|{_norm_race_part(time)}"


def _analysis_ok(text: str | None, provider: str = "") -> bool:
    """True when stored text is a real opinion we must not re-bill for."""
    if not text or not str(text).strip():
        return False
    s = str(text).strip()
    if s.startswith("ERROR"):
        return False
    if s.startswith("[") and "dry run" in s.lower():
        return False
    # Structured voices: require a VERDICT block so partial/broken replies re-try once
    if provider in ("groq", "cerebras", "cursor", "nvidia", "ollama") and "VERDICT:" not in s:
        return False
    return True


def _provider_from_analysis_key(key: str) -> str:
    return key[:-9] if key.endswith("_analysis") else ""


def _copy_prior_analyses(prior: dict, entry: dict) -> int:
    """Keep every good prior opinion (including voices not in this run). Returns count kept."""
    kept = 0
    for key, text in (prior or {}).items():
        if not key.endswith("_analysis"):
            continue
        provider = _provider_from_analysis_key(key)
        if _analysis_ok(text, provider):
            entry[key] = text
            kept += 1
    return kept


def _load_existing_report(report_path: Path) -> dict:
    """Map race key -> prior race entry from an existing report file."""
    if not ai_config.resume_reports() or not report_path.exists():
        return {}
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
        by_full = {}
        by_ct = {}
        for r in data.get("races", []):
            full = _race_key_from_entry(r)
            by_full[full] = r
            ct = _race_key_course_time(r.get("course"), r.get("time"))
            # Prefer first full match; course|time is a fallback only
            by_ct.setdefault(ct, r)
        # Store both maps under a thin wrapper so callers can resolve flexibly
        return {"_by_full": by_full, "_by_ct": by_ct}
    except Exception:
        return {}


def _prior_for_race(existing: dict, race: dict) -> dict:
    if not existing:
        return {}
    by_full = existing.get("_by_full") or existing
    by_ct = existing.get("_by_ct") or {}
    prior = by_full.get(_race_key(race))
    if prior:
        return prior
    return by_ct.get(_race_key_course_time(race.get("course"), race.get("time"))) or {}


def _call_with_cache(provider: str, brief: str, fn, dry_run: bool) -> str:
    """Invoke one provider; read/write ai_config cache when enabled."""
    if dry_run:
        return fn(brief, dry_run=True)
    cached = ai_config.cache_get(provider, brief)
    if cached is not None:
        return cached
    text = fn(brief, dry_run=False)
    ai_config.cache_put(provider, brief, text)
    return text


def _save_report(report_path: Path, analyses: dict) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = report_path.with_suffix(report_path.suffix + ".tmp")
    tmp.write_text(json.dumps(analyses, indent=2), encoding="utf-8")
    tmp.replace(report_path)


def _course_in_focus(race_course: str, courses: set[str]) -> bool:
    """Match card course against focus set (fuzzy Newmarket / Newmarket July)."""
    import ai_track_record as tr
    rc = race_course or ""
    return any(tr._course_match(rc, c) for c in courses)


def _race_index_in_list(races: list, race: dict) -> int | None:
    key = _race_key(race)
    ct = _race_key_course_time(race.get("course"), race.get("time"))
    for i, r in enumerate(races):
        if _race_key_from_entry(r) == key:
            return i
        if _race_key_course_time(r.get("course"), r.get("time")) == ct:
            return i
    return None


def _picks_from_report(existing_raw: dict) -> dict:
    """Model predictions from a previous report keyed by race name — used when
    re-running a historical day, where the live board can't be asked."""
    by_race = {}
    for r in (existing_raw or {}).get("races") or []:
        top3 = (r.get("model_prediction") or {}).get("top_3") or []
        picks = [{"horse": p.get("name"), "win_prob": p.get("win_prob"),
                  "confidence": p.get("confidence"), "odds": p.get("odds"),
                  "reasons": p.get("reasons") or []} for p in top3]
        if picks and r.get("race"):
            by_race[r["race"]] = picks
    return by_race


def analyze_today(dry_run: bool = False, day: str = "today", force: bool = False,
                  fresh: bool = False, courses: set[str] | list[str] | None = None,
                  card_file: str | None = None, date: str | None = None):
    """Analyze a day's races with every configured AI voice.

    Token guard: each race+provider opinion already in the day's report is reused
    (GOING_AI_RESUME=1, default on). Use --fresh only when you intentionally want
    to re-bill every provider. --force only bypasses the stale-card guard.

    courses: when set, only process matching races (fill-missing). Existing
    non-matching races and day-level meta (POTD / NAP / allocations) are left
    untouched — do not recompute whole-card picks from a meet-only run.
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    focus = {c.strip() for c in (courses or []) if c and str(c).strip()} or None
    meet_only = bool(focus)

    con = connect_db()
    if card_file:
        card = json.loads(Path(card_file).read_text(encoding="utf-8"))
    else:
        card = load_today_card(day)

    # Stale-card guard: if the card's races aren't for the day we're analyzing
    # (e.g. Lucy slept through the overnight rotation), force a refresh first.
    target = date or (dt.date.today()
                      + dt.timedelta(days=1 if day == "tomorrow" else 0)).isoformat()
    card_date = ((card.get("races") or [{}])[0].get("off_dt") or "")[:10]
    if not card_date and card.get("races"):
        card_date = (card.get("races")[0].get("date") or "")[:10]
    if card_file:
        pass  # explicit card — caller owns date correctness
    elif card_date and card_date != target and not force:
        print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] Card is stale "
              f"({card_date} != {target}) — forcing refresh")
        try:
            import requests
            requests.get(f"http://localhost:8011/refresh?day={day}", timeout=300)
            card = load_today_card(day)
            card_date = ((card.get("races") or [{}])[0].get("off_dt") or "")[:10]
        except Exception as e:
            print(f"  WARNING: forced refresh failed: {e}")
        if card_date != target and not force:
            print(f"  ERROR: card still stale after refresh — aborting to avoid "
                  f"analyzing the wrong day (use --force to override)")
            return False

    if not card.get("races"):
        print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] No races found in today.json")
        return False

    report_date = (dt.date.fromisoformat(date) if date
                   else dt.date.today() + dt.timedelta(days=1 if day == "tomorrow" else 0))
    today_str = report_date.isoformat()
    suffix = "_dryrun" if dry_run else ""
    report_path = REPORT_DIR / f"ai_analysis_{today_str}{suffix}.json"

    existing_raw = {}
    if report_path.exists():
        try:
            existing_raw = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            existing_raw = {}

    existing_by_race = {} if fresh else _load_existing_report(report_path)
    if existing_by_race.get("_by_full"):
        print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] Resume: "
              f"{len(existing_by_race['_by_full'])} races already in {report_path.name} "
              f"(skipping assessed providers — use --fresh to re-call)")

    if meet_only:
        analyses = {
            "date": existing_raw.get("date") or today_str,
            "generated": dt.datetime.now().isoformat(timespec="seconds"),
            "races": list(existing_raw.get("races") or []),
        }
        for meta_key in ("day_pick", "allocations", "pick_of_the_day"):
            if meta_key in existing_raw:
                analyses[meta_key] = existing_raw[meta_key]
        print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] Meet-only: "
              f"{sorted(focus)} — fill missing voices; day meta preserved")
    else:
        analyses = {
            "date": today_str,
            "generated": dt.datetime.now().isoformat(timespec="seconds"),
            "races": []
        }

    races_to_run = list(card.get("races") or [])
    if focus:
        races_to_run = [r for r in races_to_run
                        if _course_in_focus(r.get("course") or "", focus)]
        print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] Filtered to "
              f"{len(races_to_run)}/{len(card.get('races') or [])} races")
        if not races_to_run:
            print(f"  WARNING: no races matched courses={sorted(focus)}")
            if con:
                con.close()
            return False

    if date and date != dt.date.today().isoformat():
        # Historical re-run: reuse the model's original predictions from the
        # prior report — the live board only knows about today/tomorrow.
        picks_by_race = _picks_from_report(existing_raw)
        if picks_by_race:
            print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] Historical model "
                  f"picks reused from prior report ({len(picks_by_race)} races)")
    else:
        picks_by_race = load_model_picks(day)
    consec_fail = {}   # circuit breaker: bench a provider after 3 straight errors
    n_skipped = 0
    n_called = 0
    ollama_new = 0
    ollama_cap = ai_config.ollama_max_races() if ai_config.ollama_enabled() else 0

    for i, race in enumerate(races_to_run, 1):
        print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] Analyzing race "
              f"{i}/{len(races_to_run)}: {race.get('name')}")

        try:
            prediction = get_model_prediction(picks_by_race, race)
            prior = _prior_for_race(existing_by_race, race)

            entry = {
                "race": race.get("name"),
                "course": race.get("course"),
                "time": race.get("time"),
                "runners": len(race.get("runners", [])),
                "model_prediction": prediction,
            }
            # Never drop opinions from a previous run (e.g. free re-run after paid)
            _copy_prior_analyses(prior, entry)

            active = [(k, fn) for k, fn, key_env in _filtered_providers()
                      if consec_fail.get(k, 0) < 3]
            to_call = []
            for k, fn in active:
                prior_text = entry.get(f"{k}_analysis")
                if _analysis_ok(prior_text, k):
                    n_skipped += 1
                    print(f"  skip {k} (already assessed)")
                else:
                    to_call.append((k, fn))

            cloud_call = [(k, fn) for k, fn in to_call if k != "ollama"]
            ollama_call = [(k, fn) for k, fn in to_call if k == "ollama"]

            if not cloud_call and not ollama_call:
                print("  skip race (all active voices already assessed)")
            else:
                race_day = (race.get("date") or (race.get("off_dt") or "")[:10]
                            or today_str)
                briefs_by_budget: dict = {}
                brief_tiers: dict = {}

                def _brief_for(voice: str) -> str:
                    """Budget-matched brief for a voice; voices sharing a budget share text."""
                    budget = brief_builder.voice_budget(voice)[0]
                    if budget not in briefs_by_budget:
                        try:
                            briefs_by_budget[budget] = brief_builder.build_brief(
                                con, race, prediction, budget, date=race_day)
                        except Exception as be:
                            print(f"  WARNING: brief_builder failed ({be}) — legacy brief")
                            briefs_by_budget[budget] = (
                                compile_race_brief(con, race, prediction), "legacy")
                    brief_tiers[voice] = briefs_by_budget[budget][1]
                    return briefs_by_budget[budget][0]

                # Canonical (compact) brief kept for display/tooltips
                entry["race_brief"] = _brief_for("_default")
                brief_tiers.pop("_default", None)
                entry["analysis_prompt"] = build_prompt(entry["race_brief"])
                # Cloud voices in parallel first — never share a worker with Ollama
                if cloud_call:
                    with ThreadPoolExecutor(max_workers=max(1, len(cloud_call))) as ex:
                        futures = {k: ex.submit(_call_with_cache, k, _brief_for(k), fn, dry_run)
                                   for k, fn in cloud_call}
                        for k, fut in futures.items():
                            try:
                                entry[f"{k}_analysis"] = fut.result()
                            except Exception as e:
                                entry[f"{k}_analysis"] = f"ERROR: {e}"
                            n_called += 1
                            if str(entry[f"{k}_analysis"]).startswith("ERROR"):
                                consec_fail[k] = consec_fail.get(k, 0) + 1
                                if consec_fail[k] == 3:
                                    print(f"  NOTE: {k} benched after 3 consecutive errors")
                            else:
                                consec_fail[k] = 0
                # Ollama last, sequential, race-capped (protect Lucy)
                if ollama_call:
                    if ollama_new >= ollama_cap:
                        print(f"  skip ollama (cap {ollama_cap} new races this run; "
                              f"resume will fill later)")
                    else:
                        k, fn = ollama_call[0]
                        print(f"  ollama ({ai_config.ollama_model()}) "
                              f"[{ollama_new + 1}/{ollama_cap}] …")
                        try:
                            entry[f"{k}_analysis"] = _call_with_cache(
                                k, _brief_for(k), fn, dry_run)
                        except Exception as e:
                            entry[f"{k}_analysis"] = f"ERROR: {e}"
                        n_called += 1
                        ollama_new += 1
                        if str(entry[f"{k}_analysis"]).startswith("ERROR"):
                            consec_fail[k] = consec_fail.get(k, 0) + 1
                            if consec_fail[k] == 3:
                                print(f"  NOTE: {k} benched after 3 consecutive errors")
                        else:
                            consec_fail[k] = 0
                        pause = ai_config.ollama_pause_secs()
                        if pause > 0 and ollama_new < ollama_cap:
                            print(f"  ollama pause {pause}s …")
                            time.sleep(pause)
                if brief_tiers:
                    entry["brief_tiers"] = brief_tiers
            if meet_only:
                idx = _race_index_in_list(analyses["races"], race)
                if idx is not None:
                    analyses["races"][idx] = entry
                else:
                    analyses["races"].append(entry)
            else:
                analyses["races"].append(entry)
            # Checkpoint after each race so a crash never forces a full re-bill
            if not dry_run:
                _save_report(report_path, analyses)
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()

    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] Provider calls: "
          f"{n_called} new, {n_skipped} reused from report"
          + (f", ollama new={ollama_new}/{ollama_cap}" if ai_config.ollama_enabled() else ""))

    if meet_only:
        print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] Meet-only: "
              f"skipping day_pick / allocations / pick_of_the_day recompute")
    else:
        # Day-level NAP pick — optional (GOING_AI_SKIP_NAP=1 to disable)
        if not dry_run and not ai_config.skip_nap():
            try:
                day_pick = pick_day_nap(analyses)
                if day_pick:
                    analyses["day_pick"] = day_pick
                    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] NAP of the day: "
                          f"{day_pick.get('nap', {}).get('horse')}")
            except Exception as e:
                print(f"  WARNING: day pick failed: {e}")

        # Credit game — independent of NAP (rule-based by default; no extra LLM calls)
        if not dry_run:
            try:
                if ai_config.rule_based_credits():
                    allocations = allocate_credits_rule(analyses, card)
                else:
                    allocations = allocate_credits(analyses, card)
                model_alloc = allocate_model_credits(analyses, card)
                if model_alloc.get("entries"):
                    allocations["model"] = model_alloc
                if allocations:
                    # Keep books of voices this run didn't reallocate (e.g. a
                    # filtered-out Lucy) — replace only what was recomputed.
                    for k, v in (analyses.get("allocations") or {}).items():
                        allocations.setdefault(k, v)
                    analyses["allocations"] = allocations
                    for ai, alloc in allocations.items():
                        n_bets = len(alloc.get("entries") or [])
                        n_picks = alloc.get("picks")
                        of_bit = f" of {n_picks} picks" if n_picks else ""
                        verb = (f"staked {alloc['total_staked']} credits on "
                                f"{n_bets}{of_bit}" if n_bets else
                                f"sat the day out (0{of_bit} staked)")
                        print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {ai} {verb}")
            except Exception as e:
                print(f"  WARNING: credit allocation failed: {e}")

        import ai_consensus as _consensus
        potd = _consensus.pick_of_the_day(analyses, card)
        if potd:
            analyses["pick_of_the_day"] = potd
            print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] Pick of the day: "
                  f"{potd.get('label')} - {potd.get('horse')} ({potd.get('race')})")

    _save_report(report_path, analyses)
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] Report saved -> {report_path}")

    if con:
        con.close()
    return True


DAILY_CREDITS = 100


def _odds_lookup(card: dict) -> dict:
    odds_by = {}
    for race in card.get("races", []):
        for r in race.get("runners", []):
            o = r.get("odds_decimal") or 0
            if o > 1:
                odds_by[(race.get("name"), _norm(r["name"]))] = o
    return odds_by


def _scale_credits(entries: list, budget: int = DAILY_CREDITS) -> list:
    total = sum(e["credits"] for e in entries)
    if total > budget and total > 0:
        for e in entries:
            e["credits"] = int(e["credits"] * budget / total)
        entries = [e for e in entries if e["credits"] > 0]
    return entries


def _kelly_entries(weighted: list, budget: int = DAILY_CREDITS) -> list:
    """Kelly staking over picks carrying win_prob + odds.

    Bets nothing when no pick has positive expected value, spreads across
    several small edges, and concentrates when one pick towers — the
    deterministic analogue of "bet as many or as few as you like".
    Stakes are Kelly fractions of the daily budget, capped at the budget.
    """
    staked = []
    for w in weighted:
        p, o = w.get("win_prob"), w.get("odds") or 0
        if p is None or o <= 1:
            continue
        edge = p * o - 1.0
        if edge <= 0:
            continue
        staked.append((edge / (o - 1.0), w))
    if not staked:
        return []
    total_k = sum(k for k, _ in staked)
    scale = 1.0 / total_k if total_k > 1.0 else 1.0
    entries = []
    for k, w in staked:
        credits = int(round(budget * k * scale))
        if credits < 1:
            continue
        entries.append({
            "race": w["race"],
            "race_name": w["race_name"],
            "horse": w["horse"],
            "credits": credits,
            "odds": w["odds"],
            "edge": round(w["win_prob"] * w["odds"] - 1, 3),
        })
    return entries


def allocate_credits_rule(analyses: dict, card: dict) -> dict:
    """Deterministic credit stakes — no extra LLM calls. Kelly on each voice's
    own win_prob vs the market: no-edge days are no-bet days."""
    import ai_consensus

    odds_by = _odds_lookup(card)
    allowed = {k for k, _fn, _env in _filtered_providers()}
    allocations = {}
    for ai in ai_consensus.analysis_voice_keys():
        if ai not in allowed:
            continue  # filtered-out voice (e.g. Lucy excluded) — book untouched
        weighted = []
        n_picks = 0
        for race in analyses.get("races", []):
            v = ai_consensus.extract_verdict(race.get(f"{ai}_analysis") or "")
            if not v or v.get("no_bet"):
                continue
            n_picks += 1
            odds = odds_by.get((race.get("race"), _norm(v["pick"])))
            if not odds:
                continue
            weighted.append({
                "race": f"{race.get('course')} {race.get('time')}",
                "race_name": race.get("race"),
                "horse": v["pick"],
                "odds": odds,
                "win_prob": v.get("win_prob"),
            })
        if not n_picks:
            continue  # voice gave no verdicts — not playing today
        # Filter to top edge picks: global max 40, per-voice max 8 for selectivity
        top_picks = _top_edge_picks(weighted, max_picks=40)[:8]
        entries = _kelly_entries(top_picks)
        allocations[ai] = {
            "total_staked": sum(e["credits"] for e in entries),
            "entries": entries,          # may be [] — a deliberate no-bet day
            "picks": n_picks,
            "method": "kelly",
        }
    return allocations


def _top_edge_picks(verdicts: list, max_picks: int = 40) -> list:
    """Filter verdicts to top N by edge = (win_prob × odds) - 1.
    Ensures selectivity: only best opportunities are considered."""
    ranked = []
    for v in verdicts:
        wp = float(v.get("win_prob") or 0)
        odds = float(v.get("odds") or 0)
        if wp > 0 and odds > 1:
            edge = (wp * odds) - 1
            ranked.append((edge, v))
    ranked.sort(reverse=True, key=lambda x: x[0])
    return [v for _, v in ranked[:max_picks]]


def allocate_model_credits(analyses: dict, card: dict) -> dict:
    """Model credit game: Kelly on the model's calibrated top-pick probabilities
    — the model too may sit a day out when the market offers no edge."""
    odds_by = _odds_lookup(card)
    weighted = []
    n_picks = 0
    for race in analyses.get("races", []):
        top3 = (race.get("model_prediction") or {}).get("top_3") or []
        if not top3:
            continue
        p0 = top3[0]
        n_picks += 1
        odds = odds_by.get((race.get("race"), _norm(p0["name"])))
        if not odds:
            continue
        wp = float(p0.get("win_prob") or 0) / 100.0  # stored as percent
        weighted.append({
            "race": f"{race.get('course')} {race.get('time')}",
            "race_name": race.get("race"),
            "horse": p0["name"],
            "odds": odds,
            "win_prob": wp if wp > 0 else None,
        })
    # Filter to top edge picks: global max 40, per-model max 8 for selectivity
    top_picks = _top_edge_picks(weighted, max_picks=40)[:8]
    entries = _kelly_entries(top_picks)
    return {"total_staked": sum(e["credits"] for e in entries),
            "entries": entries, "picks": n_picks, "method": "kelly"}


def allocate_credits(analyses: dict, card: dict) -> dict:
    """The credit game: each AI gets DAILY_CREDITS to stake across its own
    verdict picks at the current decimal odds. Winning stakes return
    stake × odds at settlement (nightly, in ai_track_record.py)."""
    import ai_consensus

    # odds lookup: (race name, normalized horse) -> decimal odds
    odds_by = _odds_lookup(card)

    allowed = {k for k, _fn, _env in _filtered_providers()}
    allocations = {}
    for ai in ai_consensus.analysis_voice_keys():
        if ai not in allowed:
            continue  # filtered-out voice (e.g. Lucy excluded) — book untouched
        raw_call = RAW_CALLERS.get(ai)
        if raw_call is None:
            continue
        verdicts = []
        n_picks = 0
        for race in analyses.get("races", []):
            v = ai_consensus.extract_verdict(race.get(f"{ai}_analysis") or "")
            if not v or v.get("no_bet"):
                continue
            n_picks += 1
            odds = odds_by.get((race.get("race"), _norm(v["pick"])))
            if not odds:
                continue  # no price — cannot be staked, keep it off the menu
            verdicts.append({
                "race": f"{race.get('course')} {race.get('time')}",
                "race_name": race.get("race"),
                "horse": v["pick"],
                "win_prob": v.get("win_prob"),
                "odds": odds,
            })
        if not n_picks:
            continue  # voice gave no verdicts today

        # Filter to top edge picks: global max 40, per-voice max 8 for selectivity
        top_picks = _top_edge_picks(verdicts, max_picks=40)[:8]

        def _menu_line(v):
            p = v.get("win_prob")
            edge = (p * v["odds"] - 1) if p is not None else None
            bits = f"- {v['race']}: {v['horse']} — odds {v['odds']:.1f}"
            if p is not None:
                bits += f", your win_prob {p:.2f}, edge {edge:+.2f}"
            return bits

        menu = "\n".join(_menu_line(v) for v in top_picks)
        prompt = f"""You are "{ai}", an AI horse racing analyst competing against other AIs in a betting game.

Rules: up to {DAILY_CREDITS} credits are available to you today. A winning stake returns stake × decimal odds; a losing stake is lost; unstaked credits are simply not risked. Highest bankroll over time wins.

YOUR winner predictions from today's races (edge = win_prob × odds − 1; positive means you think the market underrates your pick):
{menu}

Bet on as MANY or as FEW as you like: all of them, a handful, everything on one horse, or nothing at all. Concentration and sitting the day out are both legitimate strategies — a bet only makes sense where you believe the edge is positive. Integer credits; do not exceed {DAILY_CREDITS} total.
{_staking_history_line(ai)}

Reply with EXACTLY one line of valid JSON, nothing else. To sit out today reply {{"allocations": []}}:
{{"allocations": [{{"race": "<course time>", "horse": "...", "credits": N}}, ...]}}"""

        entries = None
        method = "self"
        text = raw_call(prompt) if top_picks else '{"allocations": []}'
        parsed = None
        if not text.startswith("ERROR"):
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end > start:
                try:
                    parsed = json.loads(text[start:end + 1])
                except Exception:
                    parsed = None
        if isinstance(parsed, dict) and isinstance(parsed.get("allocations"), list):
            entries = []
            by_key = {(v["race"], _norm(v["horse"])): v for v in top_picks}
            for a in parsed["allocations"]:
                try:
                    credits = int(a.get("credits") or 0)
                except (TypeError, ValueError, AttributeError):
                    continue
                if credits <= 0:
                    continue
                v = by_key.get((a.get("race"), _norm(a.get("horse") or "")))
                if v is None:
                    continue  # hallucinated pick — not on its own menu
                entries.append({
                    "race": a.get("race"),
                    "race_name": v.get("race_name"),
                    "horse": a.get("horse"),
                    "credits": credits,
                    "odds": v.get("odds"),
                })
            total = sum(e["credits"] for e in entries)
            if total > DAILY_CREDITS and total > 0:
                # scale down proportionally, floor to integers
                for e in entries:
                    e["credits"] = int(e["credits"] * DAILY_CREDITS / total)
                entries = [e for e in entries if e["credits"] > 0]
        else:
            # Allocation call failed — deterministic Kelly keeps the voice in the game
            entries = _kelly_entries(top_picks)
            method = "kelly-fallback"
        allocations[ai] = {
            "total_staked": sum(e["credits"] for e in entries),
            "entries": entries,          # may be [] — the voice sat the day out
            "picks": n_picks,
            "method": method,
            "prompt": prompt,
        }
    return allocations


def _staking_history_line(voice: str) -> str:
    """Day-book feedback from staking_review — how past betting choices fared."""
    try:
        import staking_review
        return staking_review.staking_line(voice)
    except Exception:
        return ""


def _norm(name: str) -> str:
    import re
    name = re.sub(r"\s*\([A-Z]{2,3}\)\s*$", "", name or "")
    return re.sub(r"[^a-z0-9]", "", name.lower())


def pick_day_nap(analyses: dict) -> dict | None:
    """Ask NVIDIA to survey the whole day's verdicts and name a NAP."""
    import ai_consensus

    lines = []
    for race in analyses.get("races", []):
        header = f"{race.get('course')} {race.get('time')} — {race.get('race')}"
        parts = []
        top3 = (race.get("model_prediction") or {}).get("top_3") or []
        if top3:
            p0 = top3[0]
            wp = p0.get("win_prob")
            parts.append(f"Model: {p0['name']}" + (f" (win prob {wp:.0f}%)" if wp is not None else ""))
        for ai in ai_consensus.analysis_voice_keys():
            v = ai_consensus.extract_verdict(race.get(f"{ai}_analysis") or "")
            if v and not v.get("no_bet"):
                conf = v.get("confidence")
                parts.append(f"{ai}: {v['pick']}" + (f" ({conf}%)" if conf is not None else ""))
        if parts:
            lines.append(header + "\n  " + "; ".join(parts))

    if not lines:
        return None

    prompt = f"""You are a professional horse racing tipster. Below are today's races, each with the winner predicted by a statistical model and by several independent AI analysts (with their stated confidence).

{chr(10).join(lines)}
{_prompt_lessons_block()}
Select the single strongest bet of the day (the NAP): the horse most likely to WIN its race. Favour horses where the model and multiple AIs agree with high confidence. Also select a next-best.

Reply with EXACTLY one line of valid JSON, nothing else:
{{"nap": {{"horse": "...", "race": "<course time>", "reason": "<one short sentence>"}}, "next_best": {{"horse": "...", "race": "<course time>", "reason": "<one short sentence>"}}}}"""

    text = call_nvidia_raw(prompt)
    if text.startswith("ERROR"):
        print(f"  WARNING: NAP pick: {text[:120]}")
        return None
    # Parse the JSON object out of the reply (tolerate fences/preamble)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start:end + 1])
    except Exception:
        return None
    if not isinstance(parsed, dict) or not (parsed.get("nap") or {}).get("horse"):
        return None
    parsed["source"] = "nvidia"
    return parsed

def main():
    ap = argparse.ArgumentParser(description="Daily AI analysis of a day's races.")
    ap.add_argument("--dry-run", action="store_true", help="test without API calls")
    ap.add_argument("--day", choices=["today", "tomorrow"], default="today")
    ap.add_argument("--dev", action="store_true", help="dev preset (groq, light, rule credits, cache)")
    ap.add_argument("--providers", metavar="LIST",
                    help="comma list of voices, e.g. groq or claude,openai (overrides tier)")
    ap.add_argument("--tier", choices=("free", "paid"),
                    help="free-only vs paid+free (default paid; ignored if --providers set)")
    ap.add_argument("--rule-credits", action="store_true",
                    help="deterministic credit stakes (no LLM allocation calls)")
    ap.add_argument("--full-brief", action="store_true", help="verbose briefs (more tokens)")
    ap.add_argument("--skip-nap", action="store_true", help="skip day-level NAP LLM call")
    ap.add_argument("--force", action="store_true", help="analyze even if card date looks stale")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore existing report and re-call every provider (costly)")
    ap.add_argument("--course", metavar="NAME",
                    help="meet-only: fill missing voices for this course; leave POTD/NAP/credits alone")
    ap.add_argument("--card", metavar="FILE",
                    help="analyze this card JSON instead of today.json (historical re-runs)")
    ap.add_argument("--date", metavar="YYYY-MM-DD",
                    help="report date for --card runs (also reuses that report's model picks)")
    args = ap.parse_args()

    if args.dev:
        os.environ["GOING_AI_DEV"] = "1"
    if args.tier:
        os.environ["GOING_AI_TIER"] = args.tier
    if args.providers:
        os.environ["GOING_AI_PROVIDERS"] = args.providers
    if args.rule_credits:
        os.environ["GOING_AI_RULE_CREDITS"] = "1"
    if args.full_brief:
        os.environ["GOING_AI_LIGHT_BRIEF"] = "0"
    if args.skip_nap:
        os.environ["GOING_AI_SKIP_NAP"] = "1"

    courses = {args.course} if args.course else None
    analyze_today(dry_run=args.dry_run, day=args.day, force=args.force,
                  fresh=args.fresh, courses=courses,
                  card_file=args.card, date=args.date)

if __name__ == "__main__":
    main()
