"""
ai_consensus.py — Parse AI analyses and extract consensus predictions.

For each race, the daily AI analysis report contains Claude, OpenAI, and Ollama
perspectives. This module extracts which horse each AI recommends as the winner,
and computes a consensus score.

The consensus is used to:
1. Show AI agreement badges next to runners (✓ 2/3 AIs, etc.)
2. Optionally boost verdicts when AIs strongly agree with the model
3. Flag disagreements (model likes X, but 2/3 AIs prefer Y)
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from datetime import date

# The AI voices, in display order. Add a new provider here (plus a call_*
# function in ai_daily_analysis.py) and every surface — consensus, lights,
# panels, notification, track record, backtest — picks it up.
# Gemini parked (idle / unused). Ollama (Lucy local) is gated by GOING_AI_OLLAMA
# via ai_config.providers_for_tier — not parked when enabled.
# NVIDIA + Cursor are optional extras (see analysis_voice_keys / GOING_AI_TIER).
AIS = ["claude", "openai", "groq", "cerebras", "ollama"]
PARKED_VOICES = frozenset({"gemini"})
AI_LABELS = {"claude": "Claude", "openai": "GPT", "gemini": "Gemini",
             "groq": "Groq", "cerebras": "Cerebras", "nvidia": "NVIDIA",
             "cursor": "Cursor", "ollama": "Lucy", "model": "Model"}
CREDIT_GAME_KEYS = ("model", "cursor", "nvidia") + tuple(AIS)
AI_COLORS = {"claude": "#B57BEE", "openai": "#34C8A0", "gemini": "#4285F4",
             "groq": "#F55036", "cerebras": "#FF6B35", "nvidia": "#76B900",
             "cursor": "#F54E00", "ollama": "var(--brass)"}
# Exact model behind each voice — shown in tooltips so a light is attributable
AI_MODELS = {"claude": "Anthropic claude-sonnet-5",
             "openai": "OpenAI gpt-4-turbo",
             "gemini": "Google gemini-2.0-flash",
             "groq": "Groq · llama-3.3-70b-versatile",
             "cerebras": "Cerebras · gpt-oss-120b",
             "nvidia": "NVIDIA NIM · meta/llama-3.3-70b-instruct",
             "cursor": "Cursor · composer-2.5 (subscription)",
             "ollama": "Lucy Ollama · mistral (local, capped)"}


def analysis_voice_keys() -> tuple[str, ...]:
    """Voices stored in ai_analysis JSON (daily AIs + NVIDIA/Cursor extras)."""
    return ("cursor", "nvidia") + tuple(AIS)


def credit_game_keys() -> tuple[str, ...]:
    """Model + live credit voices for strips/bankrolls (no idle Claude/GPT sit-outs)."""
    import ai_config
    allowed = ai_config.provider_filter()
    order = ("cursor", "nvidia", "groq", "cerebras", "claude", "openai", "ollama")
    if allowed is None:
        voices = [k for k in order
                  if k in ai_config.providers_for_tier() and k not in PARKED_VOICES]
    else:
        voices = [k for k in order if k in allowed and k not in PARKED_VOICES]
    return ("model",) + tuple(voices)


def mobile_light_keys() -> tuple[str, ...]:
    """Traffic-light order on /m: first AI → model → remaining voices."""
    import ai_config
    allowed = ai_config.provider_filter()
    order = ("cursor", "nvidia", "groq", "cerebras", "claude", "openai", "ollama")
    if allowed is None:
        voices = [k for k in order
                  if k in ai_config.providers_for_tier() and k not in PARKED_VOICES]
    else:
        voices = [k for k in order if k in allowed and k not in PARKED_VOICES]
    if not voices:
        return ("model",)
    return (voices[0], "model") + tuple(voices[1:])


def load_daily_analyses(day: date = None) -> dict:
    """Load today's AI analyses from ai_reports/ai_analysis_YYYY-MM-DD.json.
    Returns {race_name: {analyses...}} or {} if file doesn't exist."""
    if day is None:
        day = date.today()
    day_str = day.isoformat()

    repo = Path(__file__).parent
    report_path = repo / "ai_reports" / f"ai_analysis_{day_str}.json"

    if not report_path.exists():
        return {}

    try:
        data = json.loads(report_path.read_text())
        result = {}
        for race_data in data.get("races", []):
            race_name = race_data.get("race")
            if race_name:
                entry = {ai: race_data.get(f"{ai}_analysis", "") for ai in analysis_voice_keys()}
                entry["model_prediction"] = race_data.get("model_prediction", {})
                entry["analysis_prompt"] = race_data.get("analysis_prompt") or ""
                entry["race_brief"] = race_data.get("race_brief") or ""
                result[race_name] = entry
        return result
    except Exception:
        return {}


def load_day_pick(day: date = None) -> dict | None:
    """The day-level NAP pick from today's report, or None."""
    if day is None:
        day = date.today()
    report_path = Path(__file__).parent / "ai_reports" / f"ai_analysis_{day.isoformat()}.json"
    if not report_path.exists():
        return None
    try:
        return json.loads(report_path.read_text()).get("day_pick")
    except Exception:
        return None


def load_pick_of_the_day(day: date = None) -> dict | None:
    """Whole-card agreement pick from ai_analysis_YYYY-MM-DD.json."""
    if day is None:
        day = date.today()
    report_path = Path(__file__).parent / "ai_reports" / f"ai_analysis_{day.isoformat()}.json"
    if not report_path.exists():
        return None
    try:
        return json.loads(report_path.read_text()).get("pick_of_the_day")
    except Exception:
        return None


def load_allocations(day: date = None) -> dict:
    """The credit-game allocations from today's report: {ai: {entries: [...]}}."""
    if day is None:
        day = date.today()
    report_path = Path(__file__).parent / "ai_reports" / f"ai_analysis_{day.isoformat()}.json"
    if not report_path.exists():
        return {}
    try:
        return json.loads(report_path.read_text()).get("allocations") or {}
    except Exception:
        return {}


def load_bankrolls() -> dict:
    """Running credit-game balances: {voice: balance}, including 'model'."""
    path = Path(__file__).parent / "ai_reports" / "bankrolls.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        out = {ai: rec.get("balance") for ai, rec in (data.get("ais") or {}).items()}
        # Legacy top-level model block
        if "model" not in out and isinstance(data.get("model"), dict):
            out["model"] = data["model"].get("balance")
        return out
    except Exception:
        return {}


def load_bankroll_file() -> dict:
    """Full bankrolls.json payload (start balance + per-voice history)."""
    path = Path(__file__).parent / "ai_reports" / "bankrolls.json"
    if not path.exists():
        return {"start": 1000, "ais": {}}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {"start": 1000, "ais": {}}


NO_BET_TOKENS = frozenset({"no bet", "nobet", "no pick", "pass", "none", "abstain"})


def is_no_bet(verdict: dict | None) -> bool:
    """True when a verdict explicitly declines to pick a horse."""
    if not verdict:
        return False
    if verdict.get("no_bet"):
        return True
    pick = re.sub(r"[_\-]+", " ", str(verdict.get("pick") or "")).strip().lower()
    return pick in NO_BET_TOKENS


def _normalize_verdict(v: dict) -> dict:
    """Bridge old (confidence 0-100) and new (win_prob 0-1) schemas in place.
    Every returned verdict carries BOTH keys plus a no_bet flag, so downstream
    consumers keep working regardless of which prompt generation produced it."""
    wp = v.get("win_prob")
    conf = v.get("confidence")
    if wp is not None:
        try:
            wp = float(wp)
            if wp > 1.0:          # model answered in percent
                wp = wp / 100.0
            wp = max(0.0, min(1.0, wp))
        except (TypeError, ValueError):
            wp = None
    if wp is None and conf is not None:
        try:
            wp = max(0.0, min(1.0, float(conf) / 100.0))
        except (TypeError, ValueError):
            wp = None
    v["win_prob"] = wp
    if conf is None and wp is not None:
        v["confidence"] = round(wp * 100)
    v["no_bet"] = is_no_bet(v)
    return v


def extract_verdict(analysis_text: str) -> dict | None:
    """Parse the structured VERDICT line an AI appends to its analysis.
    Returns {pick, win_prob, confidence, no_bet, agrees_with_model, key_risk,
    missing_factors} or None if no parseable verdict is present.
    win_prob (0-1) and confidence (0-100) are always both present when either
    was parseable; no_bet marks an explicit NO BET verdict."""
    if not analysis_text:
        return None
    text = analysis_text.strip()
    for line in text.split("\n"):
        line = line.strip()
        idx = line.find("VERDICT:")
        if idx == -1:
            continue
        payload = line[idx + len("VERDICT:"):].strip().strip("`")
        try:
            v = json.loads(payload)
            if isinstance(v, dict) and v.get("pick"):
                return _normalize_verdict(v)
        except Exception:
            pass
    # Truncated mid-JSON (common with Cerebras before token bump)
    idx = text.find("VERDICT:")
    if idx != -1:
        chunk = text[idx + len("VERDICT:"):].strip().strip("`")
        pick_m = re.search(r'"pick"\s*:\s*"([^"]+)"', chunk)
        if pick_m:
            conf_m = re.search(r'"confidence"\s*:\s*(\d+)', chunk)
            wp_m = re.search(r'"win_prob"\s*:\s*([0-9.]+)', chunk)
            v = {
                "pick": pick_m.group(1),
                "confidence": int(conf_m.group(1)) if conf_m else None,
                "agrees_with_model": None,
                "key_risk": "",
                "missing_factors": [],
            }
            if wp_m:
                try:
                    v["win_prob"] = float(wp_m.group(1))
                except ValueError:
                    pass
            return _normalize_verdict(v)
    return None


def extract_horse_picks(analysis_text: str) -> list[str]:
    """Extract horse names from AI analysis text.
    Returns list of candidate horses the AI mentions as picks."""
    if not analysis_text or analysis_text.startswith("["):
        # Placeholder text, not real analysis
        return []

    horses = []
    # Look for patterns like "I'd pick X" or "X is the winner" or just horse names
    # This is a simple heuristic - AI analyses mention winner candidates explicitly
    lines = analysis_text.split("\n")
    for line in lines:
        # Look for lines starting with common recommendation patterns
        if any(phrase in line.lower() for phrase in [
            "pick", "winner", "win", "recommend", "best", "should", "likely", "strongest"
        ]):
            # Extract capitalized words (potential horse names)
            # This is crude but works for most analyses
            words = line.split()
            for i, word in enumerate(words):
                if word[0].isupper() and len(word) > 2 and word not in ["The", "I'd", "I"]:
                    horses.append(word.rstrip(".,;:"))

    return list(set(horses))  # Deduplicate


def compute_consensus(race_name: str, analyses: dict, runners: list[str]) -> dict:
    """Compute AI consensus for a race.

    Args:
        race_name: Race identifier
        analyses: {claude, openai, ollama} analysis texts
        runners: List of runner names in the race

    Returns: {
        "consensus_pick": "Horse Name" or None,
        "ai_votes": {horse_name: count},  # How many AIs voted for this horse
        "agreement_level": 0-1.0,  # How strong the consensus is
        "claude_pick": horse or None,
        "openai_pick": horse or None,
        "ollama_pick": horse or None,
    }
    """
    empty = {"consensus_pick": None, "ai_votes": {}, "agreement_level": 0.0,
             "n_votes": 0, "n_ais": 0, "picks": {}, "confidences": {}}
    for ai in analysis_voice_keys():
        empty[f"{ai}_pick"] = None
    if not analyses:
        return empty

    # Match extracted names to actual runner names (fuzzy match)
    def best_match(candidate: str, runners: list[str]) -> str | None:
        """Find best matching runner name from candidate."""
        candidate_lower = candidate.lower()
        for runner in runners:
            if candidate_lower in runner.lower() or runner.lower() in candidate_lower:
                return runner
        return None

    def pick_for(ai_key: str) -> tuple[str | None, dict | None]:
        """Prefer the structured VERDICT pick; fall back to fuzzy extraction."""
        text = analyses.get(ai_key, "")
        verdict = extract_verdict(text)
        if verdict:
            if is_no_bet(verdict):
                return None, verdict  # explicit abstain — no horse vote
            return best_match(verdict["pick"], runners) or verdict["pick"], verdict
        candidates = extract_horse_picks(text)
        return (best_match(candidates[0], runners) if candidates else None), None

    picks, confidences = {}, {}
    for ai in analysis_voice_keys():
        p, verdict = pick_for(ai)
        picks[ai] = p
        confidences[ai] = (verdict or {}).get("confidence")
    n_ais = sum(1 for p in picks.values() if p)

    # Vote tally
    votes = {}
    for pick in picks.values():
        if pick:
            votes[pick] = votes.get(pick, 0) + 1

    # Consensus: most-voted horse, if 2+ AIs agree
    consensus_pick = None
    n_votes = 0
    if votes:
        consensus_pick = max(votes, key=votes.get)
        n_votes = votes[consensus_pick]

    out = {
        "consensus_pick": consensus_pick,
        "ai_votes": votes,
        "n_votes": n_votes,
        "n_ais": n_ais,
        "agreement_level": (n_votes / n_ais) if n_ais else 0.0,
        "picks": picks,
        "confidences": confidences,
    }
    for ai in analysis_voice_keys():
        out[f"{ai}_pick"] = picks[ai]
    return out


def _norm_horse(name: str) -> str:
    name = re.sub(r"\s*\([A-Z]{2,3}\)\s*$", "", name or "")
    return re.sub(r"[^a-z0-9]", "", name.lower())


def pick_of_the_day(report: dict, card: dict | None = None) -> dict | None:
    """Best whole-card pick: the horse with the strongest model + AI agreement.

    Scores each race's runners by how many voices (model + each AI with a pick)
    back the same horse. Returns the global winner, with an honest label like
    'Everyone agrees (5/5)' or 'Strongest agreement (3/4)'.
    """
    races = report.get("races") or []
    if not races:
        return None

    # runner odds from card if supplied
    odds_by = {}
    if card:
        for race in card.get("races", []):
            for r in race.get("runners", []):
                o = r.get("odds_decimal") or 0
                if o > 1:
                    odds_by[(race.get("name"), _norm_horse(r["name"]))] = o

    candidates = []
    for race in races:
        race_name = race.get("race")
        runners = []
        if card:
            for cr in card.get("races", []):
                if cr.get("name") == race_name:
                    runners = [x["name"] for x in cr.get("runners", [])]
                    break

        top3 = (race.get("model_prediction") or {}).get("top_3") or []
        model_pick = top3[0]["name"] if top3 else None
        model_wp = top3[0].get("win_prob") if top3 else None
        model_conf = top3[0].get("confidence") if top3 else None

        picks = {"model": model_pick}
        confidences = {"model": model_conf}
        active = set(analysis_voice_keys()) | {"model"}
        for ai in analysis_voice_keys():
            text = race.get(f"{ai}_analysis") or ""
            verdict = extract_verdict(text)
            if verdict and not is_no_bet(verdict):
                pick = verdict["pick"]
                if runners:
                    for rn in runners:
                        if _norm_horse(pick) == _norm_horse(rn) or _norm_horse(pick) in _norm_horse(rn):
                            pick = rn
                            break
                picks[ai] = pick
                confidences[ai] = verdict.get("confidence")
            else:
                picks[ai] = None

        # tally votes per horse in this race (active voices only)
        by_horse = {}
        for voice, pick in picks.items():
            if not pick or voice not in active:
                continue
            if voice in PARKED_VOICES:
                continue
            hn = _norm_horse(pick)
            by_horse.setdefault(hn, {"name": pick, "voices": [], "confs": []})
            by_horse[hn]["voices"].append(voice)
            if confidences.get(voice) is not None:
                by_horse[hn]["confs"].append(float(confidences[voice]))

        spoke = sum(1 for v, p in picks.items()
                    if p and v in active and v not in PARKED_VOICES)
        for hn, rec in by_horse.items():
            n = len(rec["voices"])
            avg_conf = sum(rec["confs"]) / len(rec["confs"]) if rec["confs"] else 50.0
            candidates.append({
                "horse": rec["name"],
                "race_name": race_name,
                "course": race.get("course"),
                "time": race.get("time"),
                "race_label": f"{race.get('course')} {race.get('time')}",
                "n_voices": n,
                "total_voices": spoke,
                "model_agrees": "model" in rec["voices"],
                "ai_voices": [v for v in rec["voices"] if v != "model"],
                "avg_confidence": avg_conf,
                "model_win_prob": model_wp,
                "odds": odds_by.get((race_name, hn)),
                "score": n * 1000 + avg_conf + (model_wp or 0),
            })

    if not candidates:
        return None

    candidates.sort(key=lambda c: (c["n_voices"], c["avg_confidence"],
                                   c.get("model_win_prob") or 0), reverse=True)
    best = candidates[0]
    n, tot = best["n_voices"], best["total_voices"]
    if n == tot and n >= 3:
        label = f"Everyone agrees ({n}/{tot})"
    elif n >= 2:
        label = f"Strongest agreement ({n}/{tot})"
    else:
        label = "Best available — weak agreement"

    return {
        "horse": best["horse"],
        "race": best["race_label"],
        "race_name": best["race_name"],
        "course": best["course"],
        "time": best["time"],
        "odds": best.get("odds"),
        "n_voices": n,
        "total_voices": tot,
        "label": label,
        "model_win_prob": best.get("model_win_prob"),
        "avg_confidence": best.get("avg_confidence"),
        "reason": f"{label} on {best['race_label']}.",
    }


def consensus_badge_html(consensus: dict) -> str:
    """Render an HTML badge showing AI consensus.

    E.g., "✓ 3/3 AIs" in green, "⚠ 2/3 AIs" in amber, "✗ Disagree" in gray
    """
    if not consensus.get("consensus_pick"):
        return ""

    n = consensus.get("n_votes", 0)
    total = consensus.get("n_ais", 0)
    if not total:
        return ""

    if n == total and total >= 3:
        color, mark, title = "var(--turf)", "✓", "All AIs agree"
    elif n >= 2:
        color, mark, title = "var(--amber)", "⚠", f"{n} of {total} AIs agree"
    else:
        color, mark, title = "var(--ink-3)", "●", "Only one AI backs this horse"
    text = f"{mark} {n}/{total} AIs"

    return (f'<span class="ai-consensus" style="color:{color}; font-size:11px; '
            f'font-weight:600; margin-left:6px" title="{title}">{text}</span>')
