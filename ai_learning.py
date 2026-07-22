#!/usr/bin/env python3
"""
ai_learning.py — Mine settled AI opinions into model-improvement candidates.

Trawls:
  1. Daily voice reports (ai_reports/ai_analysis_*.json) — VERDICT missing_factors,
     key_risk, picks vs winners
  2. AI forecaster rows (ai_forecasts) — reasons + reason_category after settle
  3. Horse post-mortems (going.db horse_postmortems, JSON fallback) —
     improve_next_time / what_we_missed themes from overnight per-horse reviews
  4. Existing track_record.json tallies when present

Produces:
  * ai_reports/ai_learning.json — ranked themes, horse career notes, next actions
  * feature_registry candidates (source=ai:learning) for themes that map to an
    existing catalogue feature OR recur often enough to deserve a new signal
  * load_prompt_lessons() — compact bullets for next-day race / NAP LLM prompts

Does NOT invent feature functions — catalogue-mapped themes can be evaluated
and auto-promoted when they clear the OOS gate (`--promote-cleared`). Core
scorer WEIGHTS are never auto-edited.

Usage:
    python3 ai_learning.py
    python3 ai_learning.py --since 2026-07-01 --register
    python3 ai_learning.py --register --eval-mapped --promote-cleared

Cron (after ai_track_record):
    40 22 * * * cd /home/pj/going && python3 ai_learning.py --register --eval-mapped --promote-cleared >> logs/cron.log 2>&1
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import ai_consensus
import going_paths

REPO = Path(__file__).parent
REPORT_DIR = REPO / "ai_reports"
OUT_PATH = REPORT_DIR / "ai_learning.json"
OVERNIGHT_LOG = REPORT_DIR / "overnight_improvements.jsonl"
DB_PATH = str(going_paths.db_path())

# Map free-text themes (substring match on normalized factor) → catalogue feature
THEME_TO_FEATURE = {
    "headgear": "first_time_headgear",
    "first time headgear": "first_time_headgear",
    "draw": "draw_bias",
    "draw impact": "draw_bias",
    "stall": "draw_bias",
    "course": "cd_winner",
    "course experience": "cd_winner",
    "course and distance": "cd_winner",
    "c&d": "cd_winner",
    "distance": "cd_winner",
    "going": "going_win_rate",
    "ground": "going_win_rate",
    "trainer": "yard_in_form",
    "yard": "yard_in_form",
    "jockey": "jockey_in_form",
    "combo": "combo_in_form",
    "trainer jockey": "combo_in_form",
    "layoff": "layoff_with_top_jockey",
    "fitness": "layoff_with_top_jockey",
    "well handicapped": "below_winning_mark",
    "handicap mark": "below_winning_mark",
    "official rating": "or_proxy",
    "or ": "or_proxy",
    "rpr": "or_proxy",
    "speed": "or_proxy",
    "recent form": "yard_in_form",  # proxy — form often tracks yard heat
    "market": "drift_pct",
    "drift": "drift_pct",
    "steamer": "steamer_magnitude",
}

MIN_THEME_N = 3          # need this many settled cites before ranking
MIN_REGISTER_N = 5       # need this many to register a feature_registry candidate
MIN_HORSE_RUNS = 2       # horse career notes need 2+ settled AI opinions
PROMPT_LESSONS_MAX_CHARS = 600
PROMPT_LESSONS_LIMIT = 8
MAX_PROMOTE_PER_RUN = 2  # nightly auto-promote cap (catalogue-mapped only)

# Noise from dry-runs / API failures — never promote these into candidates
THEME_BLOCKLIST = frozenset({
    "api unavailable", "risk: testing", "testing", "dry run", "dry-run",
    "error", "n/a", "none", "unknown",
})
HORSE_BLOCKLIST = frozenset({"testhorse", "test_horse", "dummy", "placeholder"})


def load_prompt_lessons(limit: int = PROMPT_LESSONS_LIMIT,
                        max_chars: int = PROMPT_LESSONS_MAX_CHARS) -> str:
    """Compact bullet list of recent learning themes for LLM prompts.

    Reads ai_reports/ai_learning.json. Prefer themes mapped to a catalogue
    feature and/or with higher cite counts. Empty string if nothing useful.
    """
    if not OUT_PATH.exists():
        return ""
    try:
        data = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return ""
    themes = list(data.get("themes") or [])
    if not themes:
        return ""

    def rank_key(t: dict):
        cites = int(t.get("cites") or 0)
        mapped = 1 if t.get("maps_to_feature") else 0
        return (mapped, cites)

    themes.sort(key=rank_key, reverse=True)
    lines = []
    used = 0
    for t in themes:
        theme = (t.get("theme") or "").strip()
        if not theme or len(theme) < 4:
            continue
        if theme in THEME_BLOCKLIST or any(b in theme for b in THEME_BLOCKLIST):
            continue
        feat = t.get("maps_to_feature")
        cites = t.get("cites")
        if feat:
            line = f"- {theme} (→ {feat}"
            if cites:
                line += f", n={cites}"
            line += ")"
        else:
            line = f"- {theme}" + (f" (n={cites})" if cites else "")
        # Keep each bullet short
        if len(line) > 120:
            line = line[:117] + "…"
        if used + len(line) + 1 > max_chars:
            break
        lines.append(line)
        used += len(line) + 1
        if len(lines) >= limit:
            break
    return "\n".join(lines)


def _norm_horse(name: str) -> str:
    name = re.sub(r"\s*\([A-Z]{2,3}\)\s*$", "", name or "")
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _norm_theme(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip().lower())
    s = re.sub(r"[^a-z0-9 &/+.-]", "", s)
    return s[:80]


def _map_feature(theme: str) -> str | None:
    t = _norm_theme(theme)
    # longest key first so "course and distance" beats "course"
    for key in sorted(THEME_TO_FEATURE, key=len, reverse=True):
        if key in t:
            return THEME_TO_FEATURE[key]
    return None


def _find_winner(con, date: str, course: str, time: str) -> str | None:
    try:
        import ai_track_record as tr
        return tr.find_winner(con, date, course, time or "")
    except Exception:
        return None


# --------------------------------------------------------------------------- daily reports

def _iter_daily_settled(con, since: str):
    """Yield dicts from settled daily AI analyses (one per race with a winner)."""
    for path in sorted(REPORT_DIR.glob("ai_analysis_????-??-??.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        date = report.get("date") or ""
        if date < since or date >= dt.date.today().isoformat():
            continue
        for race in report.get("races") or []:
            course = race.get("course") or ""
            time = race.get("time") or ""
            winner = _find_winner(con, date, course, time)
            if not winner:
                continue
            winner_n = _norm_horse(winner)
            top3 = (race.get("model_prediction") or {}).get("top_3") or []
            model_pick = top3[0]["name"] if top3 else None
            voices = {}
            factors = []
            risks = []
            for ai in ai_consensus.analysis_voice_keys():
                text = race.get(f"{ai}_analysis") or ""
                verdict = ai_consensus.extract_verdict(text)
                if not verdict:
                    continue
                pick = verdict.get("pick")
                voices[ai] = {
                    "pick": pick,
                    "hit": bool(pick and _norm_horse(pick) == winner_n),
                    "confidence": verdict.get("confidence"),
                }
                for f in verdict.get("missing_factors") or []:
                    factors.append(_norm_theme(str(f)))
                if verdict.get("key_risk"):
                    risks.append(_norm_theme(str(verdict["key_risk"])))
            yield {
                "source": "daily",
                "date": date,
                "course": course,
                "time": time,
                "race": race.get("race") or "",
                "winner": winner,
                "model_pick": model_pick,
                "model_hit": bool(model_pick and _norm_horse(model_pick) == winner_n),
                "voices": voices,
                "missing_factors": [f for f in factors if f],
                "key_risks": [r for r in risks if r],
            }


# --------------------------------------------------------------------------- forecaster

def _iter_forecast_settled(con, since: str):
    try:
        rows = con.execute(
            "SELECT date, course, off_time, ai_pick, ai_pick_norm, ai_reason, "
            "ai_confidence, hit, beat_model, reason_category, reason_assessment "
            "FROM ai_forecasts WHERE status='settled' AND date>=? "
            "ORDER BY date, off_time",
            (since,),
        ).fetchall()
    except Exception:
        return
    for r in rows:
        yield {
            "source": "forecast",
            "date": r[0],
            "course": r[1] or "",
            "time": r[2] or "",
            "pick": r[3],
            "pick_norm": r[4] or _norm_horse(r[3] or ""),
            "reason": (r[5] or "").strip(),
            "confidence": r[6],
            "hit": bool(r[7]),
            "beat_model": bool(r[8]),
            "reason_category": r[9] or "",
            "reason_assessment": r[10] or "",
        }


# --------------------------------------------------------------------------- theme mining

def _horse_pm_cite_from_race(date: str, race: dict, voice: str, payload: dict):
    """Build one learning cite from a race+voice payload, or None."""
    if not isinstance(payload, dict) or payload.get("error"):
        return None
    texts = []
    for key in ("improve_next_time", "what_we_missed", "what_we_got_right"):
        val = (payload.get(key) or "").strip()
        if val:
            texts.append(val)
    if not texts:
        return None
    result = (race.get("result") or "").lower()
    return {
        "date": date,
        "course": race.get("course") or "",
        "time": race.get("time") or "",
        "race": race.get("race_name") or "",
        "horse": race.get("horse") or "",
        "winner": race.get("winner") or "",
        "hit": result == "won",
        "voice": voice,
        "texts": texts,
    }


def _iter_horse_pm_db(since: str):
    """Yield theme cites from going.db horse_postmortems."""
    try:
        import bets_store
        con = bets_store.connect()
    except Exception:
        return
    today = dt.date.today().isoformat()
    try:
        try:
            rows = con.execute(
                """SELECT date, course, off_time, race_name, horse, winner, result,
                          voice, horse_verdict, what_we_missed, what_we_got_right,
                          improve_next_time, error
                   FROM horse_postmortems
                   WHERE date >= ? AND date < ?
                   ORDER BY date, course, off_time, horse, voice""",
                (since, today),
            ).fetchall()
        except sqlite3.OperationalError:
            return
        for r in rows:
            race = {
                "course": r["course"] or "",
                "time": r["off_time"] or "",
                "race_name": r["race_name"] or "",
                "horse": r["horse"] or "",
                "winner": r["winner"] or "",
                "result": r["result"] or "",
            }
            payload = {
                "horse_verdict": r["horse_verdict"] or "",
                "what_we_missed": r["what_we_missed"] or "",
                "what_we_got_right": r["what_we_got_right"] or "",
                "improve_next_time": r["improve_next_time"] or "",
                "error": r["error"] or "",
            }
            cite = _horse_pm_cite_from_race(r["date"], race, r["voice"], payload)
            if cite:
                yield cite
    finally:
        con.close()


def _iter_horse_pm(since: str):
    """Yield theme cites from overnight horse post-mortems (DB first, JSON fallback)."""
    seen_dates: set[str] = set()
    for cite in _iter_horse_pm_db(since):
        seen_dates.add(cite["date"])
        yield cite
    # JSON only for dates not already covered by DB
    for path in sorted(REPORT_DIR.glob("horse_pm_????-??-??.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        date = report.get("date") or ""
        if date < since or date >= dt.date.today().isoformat():
            continue
        if date in seen_dates:
            continue
        for race in report.get("races") or []:
            for voice, payload in (race.get("voices") or {}).items():
                cite = _horse_pm_cite_from_race(date, race, voice, payload or {})
                if cite:
                    yield cite


def _fold_horse_pm_themes(theme: dict, pm_rows) -> None:
    """Add horse_pm improve/miss lines into the shared theme counter."""
    for row in pm_rows:
        for text in row.get("texts") or []:
            # Whole-line theme (short) + catalogue-key hits inside the sentence
            ntheme = _norm_theme(text)
            candidates = set()
            if 6 <= len(ntheme) <= 80:
                candidates.add(ntheme)
            mapped = _map_feature(text)
            if mapped:
                candidates.add(mapped.replace("_", " "))
            # Soft phrases that match THEME_TO_FEATURE keys
            low = text.lower()
            for key in THEME_TO_FEATURE:
                if key in low:
                    candidates.add(key)
            for name in candidates:
                if name in THEME_BLOCKLIST or any(b in name for b in THEME_BLOCKLIST):
                    continue
                t = theme[name]
                t["cites"] += 1
                if row.get("hit"):
                    t["voice_hits"] += 1
                else:
                    t["voice_misses"] += 1
                if len(t["examples"]) < 5:
                    t["examples"].append({
                        "date": row["date"],
                        "race": f"{row.get('course', '')} {row.get('time', '')}".strip(),
                        "horse": row.get("horse"),
                        "winner": row.get("winner"),
                        "voice": row.get("voice"),
                        "kind": "horse_pm",
                    })


def _rank_themes(theme: dict) -> list[dict]:
    ranked = []
    for name, t in theme.items():
        n = t["cites"]
        if n < MIN_THEME_N:
            continue
        if name in THEME_BLOCKLIST or any(b in name for b in THEME_BLOCKLIST):
            continue
        voice_n = t["voice_hits"] + t["voice_misses"]
        fc_n = t["forecast_hits"] + t["forecast_misses"]
        voice_rate = (t["voice_hits"] / voice_n) if voice_n else None
        fc_rate = (t["forecast_hits"] / fc_n) if fc_n else None
        judged = t["sound_won"] + t["plausible_wrong"] + t["post_hoc"] + t["wrong_reason"]
        sound_rate = (t["sound_won"] / judged) if judged else None
        mapped = _map_feature(name)
        ranked.append({
            "theme": name,
            "cites": n,
            "voice_hit_rate": round(voice_rate, 3) if voice_rate is not None else None,
            "forecast_hit_rate": round(fc_rate, 3) if fc_rate is not None else None,
            "sound_reasoning_rate": round(sound_rate, 3) if sound_rate is not None else None,
            "maps_to_feature": mapped,
            "register": bool(n >= MIN_REGISTER_N),
            "examples": t["examples"],
            **{k: t[k] for k in (
                "voice_hits", "voice_misses", "forecast_hits", "forecast_misses",
                "sound_won", "plausible_wrong", "post_hoc", "wrong_reason")},
        })

    # Prefer themes that map to a feature AND show positive signal
    def score(row):
        s = row["cites"]
        if row.get("voice_hit_rate") is not None:
            s += 20 * row["voice_hit_rate"]
        if row.get("forecast_hit_rate") is not None:
            s += 20 * row["forecast_hit_rate"]
        if row.get("sound_reasoning_rate") is not None:
            s += 15 * row["sound_reasoning_rate"]
        if row.get("maps_to_feature"):
            s += 10
        return s

    ranked.sort(key=score, reverse=True)
    return ranked


def _mine_themes(daily_rows, forecast_rows, pm_rows=None) -> list[dict]:
    """Per-theme stats: how often cited, and how often the citing voice was right."""
    theme = defaultdict(lambda: {
        "cites": 0, "voice_hits": 0, "voice_misses": 0,
        "forecast_hits": 0, "forecast_misses": 0,
        "sound_won": 0, "plausible_wrong": 0, "post_hoc": 0, "wrong_reason": 0,
        "examples": [],
    })

    for row in daily_rows:
        # attribute each missing_factor to whether ANY citing voice hit that race
        any_hit = any(v.get("hit") for v in (row.get("voices") or {}).values())
        for f in row.get("missing_factors") or []:
            t = theme[f]
            t["cites"] += 1
            if any_hit:
                t["voice_hits"] += 1
            else:
                t["voice_misses"] += 1
            if len(t["examples"]) < 5:
                t["examples"].append({
                    "date": row["date"], "race": f"{row['course']} {row['time']}",
                    "winner": row["winner"], "kind": "missing_factor",
                })
        for risk in row.get("key_risks") or []:
            # key risks are warnings — track separately under "risk:" prefix
            key = f"risk: {risk}"
            t = theme[key]
            t["cites"] += 1
            if any_hit:
                t["voice_hits"] += 1
            else:
                t["voice_misses"] += 1

    # Forecast reasons: tokenize lightly into bigrams/keywords from reason text
    for row in forecast_rows:
        reason = row.get("reason") or ""
        if not reason:
            continue
        # pull short noun-ish phrases (2–4 words) from reason as soft themes
        words = re.findall(r"[a-zA-Z][a-zA-Z'-]+", reason.lower())
        phrases = set()
        for n in (2, 3):
            for i in range(len(words) - n + 1):
                phrases.add(" ".join(words[i:i + n]))
        # also map whole reason against catalogue keys
        mapped = _map_feature(reason)
        if mapped:
            phrases.add(mapped.replace("_", " "))
        for p in phrases:
            if len(p) < 6:
                continue
            # only keep phrases that look like racing themes
            if not any(k in p for k in THEME_TO_FEATURE):
                continue
            t = theme[p]
            t["cites"] += 1
            if row.get("hit"):
                t["forecast_hits"] += 1
            else:
                t["forecast_misses"] += 1
            cat = (row.get("reason_category") or "").lower()
            if "sound" in cat:
                t["sound_won"] += 1
            elif "plausible" in cat:
                t["plausible_wrong"] += 1
            elif "post-hoc" in cat or "post hoc" in cat:
                t["post_hoc"] += 1
            elif "wrong reason" in cat:
                t["wrong_reason"] += 1
            if len(t["examples"]) < 5:
                t["examples"].append({
                    "date": row["date"], "race": f"{row['course']} {row['time']}",
                    "pick": row.get("pick"), "hit": row.get("hit"),
                    "category": row.get("reason_category"), "kind": "forecast_reason",
                })

    _fold_horse_pm_themes(theme, pm_rows or [])
    return _rank_themes(theme)


def _horse_careers(daily_rows, forecast_rows) -> list[dict]:
    """Same horse across races — recurring themes when opinions settled."""
    by_horse = defaultdict(lambda: {
        "runs": 0, "hits": 0, "factors": Counter(), "reasons": [], "dates": [],
    })

    for row in daily_rows:
        for ai, v in (row.get("voices") or {}).items():
            pick = v.get("pick")
            if not pick:
                continue
            hn = _norm_horse(pick)
            rec = by_horse[hn]
            rec["runs"] += 1
            rec["name"] = pick
            rec["dates"].append(row["date"])
            if v.get("hit"):
                rec["hits"] += 1
            for f in row.get("missing_factors") or []:
                rec["factors"][f] += 1

    for row in forecast_rows:
        hn = row.get("pick_norm") or _norm_horse(row.get("pick") or "")
        if not hn:
            continue
        rec = by_horse[hn]
        rec["runs"] += 1
        rec["name"] = row.get("pick") or rec.get("name") or hn
        rec["dates"].append(row["date"])
        if row.get("hit"):
            rec["hits"] += 1
        if row.get("reason"):
            rec["reasons"].append({
                "date": row["date"], "hit": row["hit"],
                "category": row.get("reason_category"),
                "reason": row["reason"][:160],
            })

    out = []
    for hn, rec in by_horse.items():
        if rec["runs"] < MIN_HORSE_RUNS:
            continue
        if hn in HORSE_BLOCKLIST or (rec.get("name") and _norm_horse(rec["name"]) in HORSE_BLOCKLIST):
            continue
        top_factors = rec["factors"].most_common(5)
        out.append({
            "horse": rec.get("name") or hn,
            "horse_norm": hn,
            "opinions": rec["runs"],
            "hits": rec["hits"],
            "hit_rate": round(rec["hits"] / rec["runs"], 3) if rec["runs"] else None,
            "recurring_factors": [{"theme": t, "n": n} for t, n in top_factors
                                  if t not in THEME_BLOCKLIST],
            "recent_reasons": rec["reasons"][-5:],
            "dates": sorted(set(rec["dates"]))[-8:],
        })
    out.sort(key=lambda r: (r["opinions"], r["hit_rate"] or 0), reverse=True)
    return out[:40]


def _register_candidates(themes: list[dict], dry_run: bool = False) -> list[dict]:
    """Register high-cite themes into feature_registry as candidates."""
    try:
        import feature_registry as reg
        import features as feat_cat
    except Exception as e:
        return [{"error": str(e)}]

    known = set(feat_cat.names())
    actions = []
    con = reg.connect()
    try:
        for t in themes:
            if not t.get("register"):
                continue
            mapped = t.get("maps_to_feature")
            if mapped and mapped in known:
                name = mapped
                desc = (f"Catalogue feature reinforced by AI learning "
                        f"({t['cites']} cites of '{t['theme']}')")
                source = "ai:learning+catalogue"
            else:
                # new ideation slug — not evaluable until a fn exists
                slug = "ai_" + re.sub(r"[^a-z0-9]+", "_", t["theme"]).strip("_")[:40]
                name = slug
                desc = (f"AI-suggested theme '{t['theme']}' "
                        f"({t['cites']} cites; needs a point-in-time feature fn)")
                source = "ai:learning"
            if dry_run:
                actions.append({"action": "would_register", "name": name,
                                "theme": t["theme"], "source": source})
                continue
            reg.register(con, name, desc, source, status="candidate")
            actions.append({"action": "registered", "name": name,
                            "theme": t["theme"], "source": source,
                            "maps_to_feature": mapped})
    finally:
        con.close()
    return actions


def _eval_mapped(themes: list[dict], d_from: str | None, d_to: str | None) -> list[dict]:
    """Run feature_eval on unique catalogue features that themes map to."""
    import feature_eval
    import feature_registry as reg
    import features as feat_cat

    names = []
    seen = set()
    for t in themes:
        m = t.get("maps_to_feature")
        if m and m not in seen and feat_cat.get(m):
            seen.add(m)
            names.append(m)
    if not names:
        return []

    con = reg.connect()
    results = []
    try:
        for name in names:
            fn = feat_cat.get(name)
            stats = feature_eval.evaluate(con, name, fn, d_from, d_to)
            results.append({"feature": name, "stats": {
                k: stats.get(k) for k in (
                    "sample_size", "bet_count", "brier_delta", "mean_clv",
                    "oos_roi", "lift", "insufficient")
            }})
            print(feature_eval.format_stats(stats))
    finally:
        con.close()
    return results


def _auto_promote_mapped(evals: list[dict], *, max_promote: int = MAX_PROMOTE_PER_RUN,
                         dry_run: bool = False) -> list[dict]:
    """Promote catalogue features that clear the OOS gate (cap per run).

    Only names that already exist in features.catalogue and are not already
    live. Uses promote.promote (weight 0.5) so decisions are logged.
    """
    import feature_registry as reg
    import features as feat_cat
    import promote as prom

    if not evals:
        return []
    known = set(feat_cat.names())
    actions = []
    con = reg.connect()
    promoted = 0
    try:
        for ev in evals:
            if promoted >= max_promote:
                actions.append({
                    "action": "cap_reached",
                    "feature": ev.get("feature"),
                    "max_promote": max_promote,
                })
                break
            name = ev.get("feature")
            stats = ev.get("stats") or {}
            if not name or name not in known:
                continue
            row = reg.get(con, name)
            if row and row["status"] == "live":
                actions.append({"action": "already_live", "feature": name})
                continue
            # Prefer freshly evaluated stats; fall back to registry row
            gate_src = stats if stats else row
            if not gate_src:
                actions.append({"action": "skipped", "feature": name,
                                "reason": "no stats"})
                continue
            ok, reason = reg.clears_gate(gate_src)
            if not ok:
                actions.append({"action": "refused", "feature": name,
                                "reason": reason})
                print(f"  promote refuse {name}: {reason}")
                continue
            if dry_run:
                actions.append({"action": "would_promote", "feature": name,
                                "reason": reason})
                print(f"  would promote {name}: {reason}")
                promoted += 1
                continue
            # Ensure registry has latest stats before set_status reads the row
            if stats:
                reg.update_stats(con, name, {
                    **stats,
                    "insufficient": bool(stats.get("insufficient")),
                })
            ok2, msg = prom.promote(con, name)
            actions.append({
                "action": "promoted" if ok2 else "promote_failed",
                "feature": name,
                "reason": msg,
            })
            print(f"  {'✓' if ok2 else '!'} {msg}")
            if ok2:
                promoted += 1
    finally:
        con.close()
    return actions


def append_overnight_log(out: dict) -> None:
    """Append one durable line for the overnight improvements log on /postmortem."""
    try:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": out.get("generated"),
            "since": out.get("since"),
            "horse_pm_cites": out.get("horse_pm_cites", 0),
            "daily_races_settled": out.get("daily_races_settled", 0),
            "forecasts_settled": out.get("forecasts_settled", 0),
            "themes_top": [
                {"theme": t.get("theme"), "cites": t.get("cites"),
                 "maps_to_feature": t.get("maps_to_feature")}
                for t in (out.get("themes") or [])[:8]
            ],
            "registry_actions": out.get("registry_actions") or [],
            "feature_evals": out.get("feature_evals") or [],
            "promote_actions": out.get("promote_actions") or [],
            "next_actions": [
                {"priority": a.get("priority"), "theme": a.get("theme"),
                 "feature": a.get("feature"), "cmd": a.get("cmd")}
                for a in (out.get("next_actions") or [])[:6]
            ],
        }
        with OVERNIGHT_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def load_overnight_log(limit: int = 14) -> list[dict]:
    """Newest-first overnight improvement entries (JSONL)."""
    if not OVERNIGHT_LOG.exists():
        return []
    try:
        lines = OVERNIGHT_LOG.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    out = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
        if len(out) >= limit:
            break
    return out


def run(since: str, register: bool = False, eval_mapped: bool = False,
        promote_cleared: bool = False, dry_run: bool = False,
        max_promote: int = MAX_PROMOTE_PER_RUN) -> dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    try:
        daily = list(_iter_daily_settled(con, since))
        forecasts = list(_iter_forecast_settled(con, since))
    finally:
        con.close()
    pm_rows = list(_iter_horse_pm(since))

    themes = _mine_themes(daily, forecasts, pm_rows)
    horses = _horse_careers(daily, forecasts)

    # Pull track_record summary if present
    tr_summary = {}
    tr_path = REPORT_DIR / "track_record.json"
    if tr_path.exists():
        try:
            tr = json.loads(tr_path.read_text(encoding="utf-8"))
            tr_summary = {
                "races_scored": tr.get("races_scored"),
                "missing_factors_top20": tr.get("missing_factors_top20"),
                "hit_rates": tr.get("hit_rates"),
            }
        except Exception:
            pass

    next_actions = []
    for t in themes[:15]:
        if t.get("maps_to_feature"):
            next_actions.append({
                "priority": "eval",
                "theme": t["theme"],
                "feature": t["maps_to_feature"],
                "cmd": f"python3 feature_eval.py --feature {t['maps_to_feature']}",
                "why": f"{t['cites']} AI cites; mapped to existing catalogue feature",
            })
        elif t.get("register"):
            next_actions.append({
                "priority": "design",
                "theme": t["theme"],
                "feature": None,
                "cmd": None,
                "why": (f"{t['cites']} cites but no catalogue feature — "
                        "write a point-in-time fn then feature_eval"),
            })

    actions = []
    if register:
        actions = _register_candidates(themes, dry_run=dry_run)

    # promote_cleared implies eval so stats exist for the gate
    if promote_cleared:
        eval_mapped = True

    evals = []
    if eval_mapped:
        evals = _eval_mapped(themes, since, None)

    promote_actions = []
    if promote_cleared:
        print(f"Auto-promote (max {max_promote}) …")
        promote_actions = _auto_promote_mapped(
            evals, max_promote=max_promote, dry_run=dry_run)

    out = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "since": since,
        "daily_races_settled": len(daily),
        "forecasts_settled": len(forecasts),
        "horse_pm_cites": len(pm_rows),
        "themes": themes[:40],
        "horse_careers": horses,
        "track_record": tr_summary,
        "next_actions": next_actions[:20],
        "registry_actions": actions,
        "feature_evals": evals,
        "promote_actions": promote_actions,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    if register or eval_mapped or promote_cleared:
        append_overnight_log(out)
    return out


def _print_summary(out: dict) -> None:
    print(f"=== AI LEARNING (since {out['since']}) ===")
    print(f"  daily races settled: {out['daily_races_settled']}")
    print(f"  forecasts settled:   {out['forecasts_settled']}")
    print(f"  horse_pm cites:      {out.get('horse_pm_cites', 0)}")
    print("\nTop themes:")
    for t in (out.get("themes") or [])[:12]:
        bits = [f"n={t['cites']}"]
        if t.get("voice_hit_rate") is not None:
            bits.append(f"voice_hit={t['voice_hit_rate']:.0%}")
        if t.get("forecast_hit_rate") is not None:
            bits.append(f"fc_hit={t['forecast_hit_rate']:.0%}")
        if t.get("sound_reasoning_rate") is not None:
            bits.append(f"sound={t['sound_reasoning_rate']:.0%}")
        mapped = t.get("maps_to_feature") or "—"
        print(f"  {t['theme'][:40]:40s} {', '.join(bits)}  → {mapped}")
    if out.get("horse_careers"):
        print("\nHorses with recurring AI opinions:")
        for h in out["horse_careers"][:8]:
            fac = ", ".join(f"{x['theme']}×{x['n']}" for x in (h.get("recurring_factors") or [])[:3])
            print(f"  {h['horse'][:28]:28s} opinions={h['opinions']} "
                  f"hit={h['hit_rate']:.0%}  {fac}")
    if out.get("next_actions"):
        print("\nNext actions:")
        for a in out["next_actions"][:8]:
            print(f"  [{a['priority']}] {a['theme'][:36]:36s} {a.get('cmd') or a['why']}")
    if out.get("registry_actions"):
        print(f"\nRegistry: {len(out['registry_actions'])} candidate upserts")
    if out.get("promote_actions"):
        print(f"\nPromote: {len(out['promote_actions'])} decisions")
        for a in out["promote_actions"][:8]:
            print(f"  [{a.get('action')}] {a.get('feature') or '—'} "
                  f"{(a.get('reason') or '')[:80]}")
    print(f"\nSaved -> {OUT_PATH}")


def main():
    ap = argparse.ArgumentParser(description="Mine AI opinions into model-improvement candidates")
    ap.add_argument("--since", default=(dt.date.today() - dt.timedelta(days=90)).isoformat())
    ap.add_argument("--register", action="store_true",
                    help="upsert recurring themes into feature_registry as candidates")
    ap.add_argument("--eval-mapped", action="store_true",
                    help="run feature_eval on catalogue features that themes map to")
    ap.add_argument("--promote-cleared", action="store_true",
                    help="after eval, auto-promote catalogue features that clear the OOS gate "
                         f"(max {MAX_PROMOTE_PER_RUN}/run; implies --eval-mapped)")
    ap.add_argument("--max-promote", type=int, default=MAX_PROMOTE_PER_RUN,
                    help=f"cap auto-promotions per run (default {MAX_PROMOTE_PER_RUN})")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    out = run(since=args.since, register=args.register,
              eval_mapped=args.eval_mapped, promote_cleared=args.promote_cleared,
              dry_run=args.dry_run, max_promote=max(0, args.max_promote))
    _print_summary(out)


if __name__ == "__main__":
    main()
