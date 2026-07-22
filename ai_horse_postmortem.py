#!/usr/bin/env python3
"""
ai_horse_postmortem.py — Per-horse overnight post-mortems for Yesterday's settled picks.

After results settle, each live credit-game voice explains why the /m board pick
won or lost, and one concrete improvement for next time. Output feeds /m cards
and ai_learning theme mining.

Source of truth: going.db table horse_postmortems (via bets_store.connect).
Optional JSON export: ai_reports/horse_pm_{date}.json (cache / backup).

    python3 ai_horse_postmortem.py
    python3 ai_horse_postmortem.py --date 2026-07-12
    python3 ai_horse_postmortem.py --date 2026-07-12 --fresh
    python3 ai_horse_postmortem.py --dry-run
    python3 ai_horse_postmortem.py --migrate-json
    python3 ai_horse_postmortem.py --migrate-json --date 2026-07-12

Idempotent: skips race+voice already in the DB (or JSON fallback) unless --fresh.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

import ai_consensus
import ai_config
import bets_store
import brief_builder
import going_paths

REPO = Path(__file__).parent
REPORT_DIR = REPO / "ai_reports"


def _load_env() -> None:
    env_path = REPO / ".env"
    if not env_path.exists():
        return
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    except Exception:
        pass


def pm_path(date: str) -> Path:
    return REPORT_DIR / f"horse_pm_{date}.json"


def _empty_pm(date: str) -> dict:
    return {"date": date, "generated": None, "races": []}


def _race_id(date: str, course: str, time: str) -> str:
    return bets_store.race_key(date, course or "", time or "")


def load_pm_json(date: str) -> dict:
    path = pm_path(date)
    if not path.exists():
        return _empty_pm(date)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return _empty_pm(date)
        data.setdefault("date", date)
        data.setdefault("races", [])
        return data
    except Exception:
        return _empty_pm(date)


def load_pm_db(date: str) -> dict:
    """Assemble the nested report shape from horse_postmortems rows."""
    con = bets_store.connect()
    try:
        rows = con.execute(
            """SELECT * FROM horse_postmortems
               WHERE date = ?
               ORDER BY course, off_time, horse, voice""",
            (date,),
        ).fetchall()
    finally:
        con.close()
    if not rows:
        return _empty_pm(date)

    races: dict[tuple, dict] = {}
    generated = None
    for r in rows:
        key = (r["race_id"] or "", r["horse_norm"] or "")
        slot = races.get(key)
        if slot is None:
            slot = {
                "race_name": r["race_name"] or "",
                "course": r["course"] or "",
                "time": r["off_time"] or "",
                "horse": r["horse"] or "",
                "winner": r["winner"] or "",
                "result": r["result"] or "",
                "odds": r["odds"],
                "voices": {},
            }
            races[key] = slot
        else:
            if r["winner"]:
                slot["winner"] = r["winner"]
            if r["result"]:
                slot["result"] = r["result"]
            if r["odds"] is not None:
                slot["odds"] = r["odds"]
        voice = r["voice"] or ""
        if not voice:
            continue
        if r["error"] and not (r["horse_verdict"] or "").strip():
            slot["voices"][voice] = {
                "error": r["error"],
                "raw_ok": False,
            }
            if "prompt" in r.keys() and r["prompt"]:
                slot["voices"][voice]["prompt"] = r["prompt"]
        else:
            payload = {
                "horse_verdict": r["horse_verdict"] or "",
                "improve_next_time": r["improve_next_time"] or "",
            }
            if r["what_we_missed"]:
                payload["what_we_missed"] = r["what_we_missed"]
            if r["what_we_got_right"]:
                payload["what_we_got_right"] = r["what_we_got_right"]
            if r["prior_pick"]:
                payload["prior_pick"] = r["prior_pick"]
            if r["prior_confidence"] is not None:
                payload["prior_confidence"] = r["prior_confidence"]
            if r["error"]:
                payload["error"] = r["error"]
            if "prompt" in r.keys() and r["prompt"]:
                payload["prompt"] = r["prompt"]
            slot["voices"][voice] = payload
        ga = r["generated_at"]
        if ga and (generated is None or ga > generated):
            generated = ga
    return {"date": date, "generated": generated, "races": list(races.values())}


def load_pm(date: str) -> dict:
    """DB first (source of truth); JSON only if DB has no rows for the date.

    Prompts are mirrored in the JSON cache (DB column optional) — merge them
    onto DB payloads so /m can show the post-mortem brief accordion.
    """
    try:
        db_pm = load_pm_db(date)
        if db_pm.get("races"):
            json_pm = load_pm_json(date)
            if json_pm.get("races"):
                _merge_pm_prompts(db_pm, json_pm)
            return db_pm
    except Exception:
        pass
    return load_pm_json(date)


def _merge_pm_prompts(db_pm: dict, json_pm: dict) -> None:
    """Copy ``prompt`` from JSON voice payloads onto matching DB slots."""
    by_key: dict[tuple, dict] = {}
    for slot in json_pm.get("races") or []:
        key = (
            (slot.get("race_name") or "").strip().lower(),
            bets_store.norm_name(slot.get("horse") or ""),
        )
        by_key[key] = slot
    for slot in db_pm.get("races") or []:
        key = (
            (slot.get("race_name") or "").strip().lower(),
            bets_store.norm_name(slot.get("horse") or ""),
        )
        jslot = by_key.get(key)
        if not jslot:
            continue
        for voice, payload in (slot.get("voices") or {}).items():
            if not isinstance(payload, dict) or payload.get("prompt"):
                continue
            jp = (jslot.get("voices") or {}).get(voice)
            if isinstance(jp, dict) and (jp.get("prompt") or "").strip():
                payload["prompt"] = jp["prompt"]



def save_pm_db(data: dict) -> int:
    """Upsert every voice row from a nested report. Returns rows written."""
    date = data.get("date") or ""
    generated = data.get("generated") or dt.datetime.now().isoformat(timespec="seconds")
    if not date:
        return 0
    con = bets_store.connect()
    n = 0
    try:
        for slot in data.get("races") or []:
            course = slot.get("course") or ""
            time = slot.get("time") or ""
            horse = slot.get("horse") or ""
            race_name = slot.get("race_name") or ""
            rid = _race_id(date, course, time)
            hn = bets_store.norm_name(horse)
            if not hn or not (slot.get("voices") or {}):
                # still allow empty voices skip
                pass
            for voice, payload in (slot.get("voices") or {}).items():
                if not voice:
                    continue
                payload = payload if isinstance(payload, dict) else {}
                con.execute(
                    """INSERT INTO horse_postmortems(
                         date, race_id, course, off_time, race_name, horse, horse_norm,
                         winner, result, odds, voice, horse_verdict, what_we_missed,
                         what_we_got_right, improve_next_time, prior_pick,
                         prior_confidence, error, generated_at, jockey, trainer, prompt)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(date, race_id, horse_norm, voice) DO UPDATE SET
                         course=excluded.course,
                         off_time=excluded.off_time,
                         race_name=excluded.race_name,
                         horse=excluded.horse,
                         winner=excluded.winner,
                         result=excluded.result,
                         odds=excluded.odds,
                         horse_verdict=excluded.horse_verdict,
                         what_we_missed=excluded.what_we_missed,
                         what_we_got_right=excluded.what_we_got_right,
                         improve_next_time=excluded.improve_next_time,
                         prior_pick=excluded.prior_pick,
                         prior_confidence=excluded.prior_confidence,
                         error=excluded.error,
                         generated_at=excluded.generated_at,
                         jockey=COALESCE(excluded.jockey, horse_postmortems.jockey),
                         trainer=COALESCE(excluded.trainer, horse_postmortems.trainer),
                         prompt=COALESCE(excluded.prompt, horse_postmortems.prompt)
                    """,
                    (
                        date, rid, course, time, race_name, horse, hn,
                        slot.get("winner") or "", slot.get("result") or "",
                        slot.get("odds"), voice,
                        (payload.get("horse_verdict") or "")[:800] or None,
                        (payload.get("what_we_missed") or "")[:300] or None,
                        (payload.get("what_we_got_right") or "")[:300] or None,
                        (payload.get("improve_next_time") or "")[:300] or None,
                        payload.get("prior_pick"),
                        payload.get("prior_confidence"),
                        (payload.get("error") or "")[:200] or None,
                        generated,
                        (slot.get("jockey") or "").strip() or None,
                        (slot.get("trainer") or "").strip() or None,
                        (payload.get("prompt") or "") or None,
                    ),
                )
                n += 1
        con.commit()
    finally:
        con.close()
    return n


def save_pm_json(data: dict) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = pm_path(data["date"])
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def save_pm(data: dict, *, write_json: bool = True) -> Path:
    """Persist to going.db; optionally mirror JSON for backup/export."""
    save_pm_db(data)
    if write_json:
        return save_pm_json(data)
    return pm_path(data.get("date") or "unknown")


def _snip(text: str, n: int = 110) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if len(text) <= n:
        return text
    return text[: n - 1].rsplit(" ", 1)[0] + "…"


def connection_notes_for_brief(runners: list[dict], before: str,
                               *, days: int = 21, limit: int = 6) -> str:
    """Short notes from recent horse_postmortems for horses/jockeys on this card.

    Reuses overnight reviews so a jockey on a new mount (or a horse returning)
    carries prior AI lessons without re-calling the LLM.
    """
    if not runners or not before:
        return ""
    horse_norms: list[str] = []
    jockeys: list[str] = []
    seen_h: set[str] = set()
    seen_j: set[str] = set()
    for r in runners:
        hn = bets_store.norm_name(r.get("name") or "")
        if hn and hn not in seen_h:
            seen_h.add(hn)
            horse_norms.append(hn)
        jk = (r.get("jockey") or "").strip()
        if jk and jk.lower() not in seen_j:
            seen_j.add(jk.lower())
            jockeys.append(jk)
    if not horse_norms and not jockeys:
        return ""

    try:
        con = bets_store.connect()
    except Exception:
        return ""

    voice_pref = ("groq", "cursor", "cerebras", "nvidia")
    lines: list[str] = []
    used_keys: set[str] = set()

    def _pick_rows(rows) -> list:
        best: dict[tuple, object] = {}
        for row in rows:
            key = (row["date"], row["horse_norm"])
            voice = (row["voice"] or "").lower()
            cur = best.get(key)
            if cur is None:
                best[key] = row
                continue
            try:
                better = voice_pref.index(voice) < voice_pref.index(
                    (cur["voice"] or "").lower())
            except ValueError:
                better = (
                    voice in voice_pref
                    and (cur["voice"] or "").lower() not in voice_pref
                )
            if better:
                best[key] = row
        return sorted(best.values(), key=lambda r: r["date"] or "", reverse=True)

    try:
        bets_store.ensure_schema(con)
        since = con.execute(
            "SELECT date(?, ?)", (before, f"-{int(days)} days")
        ).fetchone()[0]

        rows = []
        if horse_norms:
            placeholders = ",".join("?" * len(horse_norms))
            rows.extend(con.execute(
                f"""SELECT date, horse, horse_norm, result, course, voice,
                           horse_verdict, improve_next_time, jockey, trainer
                    FROM horse_postmortems
                    WHERE date < ? AND date >= ?
                      AND horse_norm IN ({placeholders})
                      AND horse_verdict IS NOT NULL AND TRIM(horse_verdict) <> ''
                    ORDER BY date DESC""",
                (before, since, *horse_norms),
            ).fetchall())

        if jockeys:
            placeholders = ",".join("?" * len(jockeys))
            rows.extend(con.execute(
                f"""SELECT date, horse, horse_norm, result, course, voice,
                           horse_verdict, improve_next_time, jockey, trainer
                    FROM horse_postmortems
                    WHERE date < ? AND date >= ?
                      AND jockey IN ({placeholders})
                      AND horse_verdict IS NOT NULL AND TRIM(horse_verdict) <> ''
                    ORDER BY date DESC""",
                (before, since, *jockeys),
            ).fetchall())
            try:
                rows.extend(con.execute(
                    f"""SELECT hpm.date, hpm.horse, hpm.horse_norm, hpm.result,
                               hpm.course, hpm.voice, hpm.horse_verdict,
                               hpm.improve_next_time, r.jockey, r.trainer
                        FROM horse_postmortems hpm
                        JOIN runners r ON r.race_id = hpm.race_id
                                      AND r.horse_norm = hpm.horse_norm
                        WHERE hpm.date < ? AND hpm.date >= ?
                          AND r.jockey IN ({placeholders})
                          AND (hpm.jockey IS NULL OR TRIM(hpm.jockey) = '')
                          AND hpm.horse_verdict IS NOT NULL
                          AND TRIM(hpm.horse_verdict) <> ''
                        ORDER BY hpm.date DESC""",
                    (before, since, *jockeys),
                ).fetchall())
            except Exception:
                pass

        for row in _pick_rows(rows):
            if len(lines) >= limit:
                break
            hn = row["horse_norm"] or ""
            jk = (row["jockey"] or "").strip()
            key = f"{row['date']}|{hn}"
            if key in used_keys:
                continue
            used_keys.add(key)
            res = (row["result"] or "").lower() or "?"
            course = row["course"] or ""
            date_s = (row["date"] or "")[5:]
            verdict = _snip(row["horse_verdict"] or "")
            improve = _snip(row["improve_next_time"] or "", 80)
            horse = row["horse"] or hn
            tag = f"{horse} ({date_s} {res}"
            if course:
                tag += f" @ {course}"
            tag += ")"
            if jk and jk.lower() in seen_j and hn not in seen_h:
                line = f"- Jockey {jk} on {tag}: {verdict}"
            else:
                line = f"- {tag}: {verdict}"
            if improve:
                line += f" Improve: {improve}"
            if len(line) > 200:
                line = line[:197] + "…"
            lines.append(line)
    except Exception:
        return ""
    finally:
        try:
            con.close()
        except Exception:
            pass

    if not lines:
        return ""
    return (
        "CONNECTIONS NOTES (recent AI post-mortems — reuse, do not re-ask)\n"
        + "\n".join(lines) + "\n"
    )


def backfill_pm_connections_from_card(date: str, card: dict | None = None) -> int:
    """Fill jockey/trainer on existing PM rows from a card. Returns rows updated."""
    if card is None:
        card = _load_card(None, date)
    by_horse: dict[str, tuple[str, str]] = {}
    for race in card.get("races") or []:
        for r in race.get("runners") or []:
            hn = bets_store.norm_name(r.get("name") or "")
            if not hn:
                continue
            by_horse[hn] = (
                (r.get("jockey") or "").strip(),
                (r.get("trainer") or "").strip(),
            )
    if not by_horse:
        return 0
    con = bets_store.connect()
    n = 0
    try:
        bets_store.ensure_schema(con)
        for hn, (jk, tr) in by_horse.items():
            if not jk and not tr:
                continue
            cur = con.execute(
                """UPDATE horse_postmortems
                   SET jockey = COALESCE(NULLIF(jockey,''), ?),
                       trainer = COALESCE(NULLIF(trainer,''), ?)
                   WHERE date = ? AND horse_norm = ?""",
                (jk or None, tr or None, date, hn),
            )
            n += cur.rowcount or 0
        con.commit()
    finally:
        con.close()
    return n


def migrate_json_to_db(date: str | None = None) -> dict:
    """Backfill horse_postmortems from ai_reports/horse_pm_*.json."""
    paths = []
    if date:
        p = pm_path(date)
        if p.exists():
            paths = [p]
    else:
        paths = sorted(REPORT_DIR.glob("horse_pm_????-??-??.json"))
    total_files = total_rows = 0
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  skip {path.name}: {e}")
            continue
        if not isinstance(data, dict) or not data.get("date"):
            continue
        n = save_pm_db(data)
        total_files += 1
        total_rows += n
        print(f"  {data['date']}: {n} voice rows ← {path.name}")
    return {"files": total_files, "rows": total_rows}


def live_pm_voices() -> tuple[str, ...]:
    """Credit-game LLMs only — no model strip, no parked Claude/GPT sit-outs."""
    keys = [k for k in ai_consensus.credit_game_keys() if k != "model"]
    return tuple(keys)


def _norm_horse(name: str) -> str:
    name = re.sub(r"\s*\([A-Z]{2,3}\)\s*$", "", name or "")
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _load_card(card_path: Path | None, date: str) -> dict:
    candidates = []
    if card_path:
        candidates.append(Path(card_path))
    candidates.extend([
        REPO / "yesterday.json",
        REPO / "today.json",
    ])
    # Prefer a card whose races match the date if annotated; else first existing
    for p in candidates:
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
    return {"races": []}


def _load_ai_report(date: str) -> dict:
    path = REPORT_DIR / f"ai_analysis_{date}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _winner_map(date: str, card_path: str | None = None) -> dict:
    """(norm_course, time_variant) → winner horse. Prefer corpus, fill from SP."""
    try:
        # Reuse serve helpers when available (same alignment as /m)
        import serve
        path = card_path or str(REPO / "yesterday.json")
        win_map = {}
        try:
            win_map = serve._aligned_winner_map(date, path) or {}
        except Exception:
            win_map = {}
        if not win_map or len(win_map) < 3:
            try:
                for k, w in (serve._sp_winner_map(date, path) or {}).items():
                    win_map.setdefault(k, w)
            except Exception:
                pass
        return win_map
    except Exception:
        pass

    # Fallback without serve: corpus find_winner per race
    import ai_track_record as tr
    import bets_store
    import yesterday_results as yr

    card = _load_card(Path(card_path) if card_path else None, date)
    out: dict[tuple[str, str], str] = {}
    con = sqlite3.connect(tr.DB_PATH)
    try:
        for race in card.get("races") or []:
            course = race.get("course") or ""
            time = race.get("time") or ""
            w = tr.find_winner(con, date, course, time)
            if not w:
                continue
            ck = tr.norm_course(course)
            for tv in tr._time_variants(time):
                out[(ck, tv)] = w
    finally:
        con.close()

    if len(out) >= 3:
        return out

    try:
        won_bsp, _ = yr._sp_maps(date, yr.DEFAULT_REGIONS)
    except Exception:
        won_bsp = None
    if not won_bsp:
        return out
    for race in card.get("races") or []:
        winner = None
        for r in race.get("runners") or []:
            hn = bets_store.norm_name(r.get("name") or "")
            info = won_bsp.get(hn) if hn else None
            if info and info.get("won"):
                winner = r.get("name")
                break
        if not winner:
            continue
        ck = tr.norm_course(race.get("course") or "")
        for tv in tr._time_variants(race.get("time") or ""):
            out.setdefault((ck, tv), winner)
    return out


def _lookup_winner(win_map: dict, course: str, time: str) -> str | None:
    import ai_track_record as tr
    ck = tr.norm_course(course or "")
    for tv in tr._time_variants(time or ""):
        w = win_map.get((ck, tv))
        if w:
            return w
    return None


def _race_entry(report: dict, course: str, time: str, race_name: str) -> dict | None:
    """Match an ai_analysis race row to a card race."""
    def norm(s):
        return re.sub(r"\s+", " ", str(s or "")).strip().lower()

    want_c, want_t, want_n = norm(course), norm(time), norm(race_name)
    best = None
    for race in report.get("races") or []:
        if want_n and norm(race.get("race")) == want_n:
            return race
        if norm(race.get("course")) == want_c and norm(race.get("time")) == want_t:
            best = race
    return best


def _model_pick(entry: dict | None, race: dict) -> str | None:
    if entry:
        top3 = (entry.get("model_prediction") or {}).get("top_3") or []
        if top3 and top3[0].get("name"):
            return top3[0]["name"]
    # Last resort: favourite by shortest odds on the card
    runners = race.get("runners") or []
    priced = []
    for r in runners:
        try:
            o = float(r.get("odds_decimal") or 0)
        except (TypeError, ValueError):
            o = 0
        if o > 1 and r.get("name"):
            priced.append((o, r["name"]))
    if priced:
        priced.sort()
        return priced[0][1]
    return (runners[0].get("name") if runners else None)


def _odds_for(race: dict, horse: str) -> float | None:
    hn = _norm_horse(horse)
    for r in race.get("runners") or []:
        if _norm_horse(r.get("name") or "") == hn:
            try:
                o = float(r.get("odds_decimal") or 0)
                return o if o > 1 else None
            except (TypeError, ValueError):
                return None
    return None


def _voice_prior(entry: dict | None, voice: str) -> dict:
    if not entry:
        return {}
    text = entry.get(f"{voice}_analysis") or ""
    verdict = ai_consensus.extract_verdict(text) or {}
    return {
        "pick": verdict.get("pick"),
        "confidence": verdict.get("confidence"),
        "win_prob": verdict.get("win_prob"),
        "no_bet": bool(verdict.get("no_bet")),
        "key_risk": verdict.get("key_risk") or "",
        "missing_factors": list(verdict.get("missing_factors") or [])[:6],
    }


def _result_context(date: str, race: dict, horse: str, winner: str) -> dict:
    """What actually happened, from the corpus: finishing order, SPs, margins."""
    out = {"lines": [], "pick_pos": None, "pick_beaten": None, "pick_sp": None,
           "winner_sp": None, "field": None}
    try:
        con = bets_store.connect()
    except Exception:
        return out
    try:
        con.row_factory = sqlite3.Row
        ck = brief_builder._norm_course(race.get("course") or "")
        tk = brief_builder._time24(race.get("time") or "")
        rid = fld = None
        for r in con.execute(
                "SELECT race_id, course, off_time, field_size FROM races WHERE date=?",
                (date,)).fetchall():
            if (brief_builder._norm_course(r["course"]) == ck
                    and brief_builder._time24(r["off_time"]) == tk):
                rid, fld = r["race_id"], r["field_size"]
                break
        if not rid:
            return out
        out["field"] = fld
        rows = con.execute(
            "SELECT horse, horse_norm, finish_pos, beaten_lengths, starting_price,"
            " draw, official_rating FROM runners"
            " WHERE race_id=? AND finish_pos IS NOT NULL AND finish_pos != ''",
            (rid,)).fetchall()

        def _pos(r):
            try:
                return int(r["finish_pos"])
            except (TypeError, ValueError):
                return 99

        rows = sorted(rows, key=_pos)
        for r in rows[:3]:
            sp = f" SP {r['starting_price']:.1f}" if r["starting_price"] else ""
            dr = f" dr{r['draw']}" if r["draw"] is not None else ""
            orr = f" OR {r['official_rating']}" if r["official_rating"] else ""
            out["lines"].append(f"  {r['finish_pos']}. {r['horse']}{dr}{orr}{sp}")
        hn, wn = _norm_horse(horse), _norm_horse(winner)
        for r in rows:
            if r["horse_norm"] == hn:
                out["pick_pos"] = r["finish_pos"]
                out["pick_beaten"] = r["beaten_lengths"]
                out["pick_sp"] = r["starting_price"]
            if r["horse_norm"] == wn:
                out["winner_sp"] = r["starting_price"]
    except Exception:
        pass
    finally:
        try:
            con.close()
        except Exception:
            pass
    return out


FACTOR_WORDS = ("draw", "ground", "jockey", "trainer", "course", "distance",
                "class", "market", "fitness", "pace", "weight", "form")


def _build_brief(date: str, race: dict, horse: str, winner: str,
                 entry: dict | None, voice: str, prior: dict) -> str:
    won = _norm_horse(horse) == _norm_horse(winner)
    odds = _odds_for(race, horse)
    odds_s = f"{odds:.2f}" if odds else "?"
    ctx = _result_context(date, race, horse, winner)
    model_line = ""
    if entry:
        top3 = (entry.get("model_prediction") or {}).get("top_3") or []
        if top3:
            bits = [f"{t.get('name')} ({t.get('win_prob', '?')}%)" for t in top3[:3]]
            model_line = "Model top 3: " + ", ".join(bits)

    dist = race.get("distance_f")
    cond = " · ".join(str(b) for b in (
        f"{dist:.1f}".rstrip("0").rstrip(".") + "f" if dist else None,
        race.get("going"), race.get("race_class")) if b)

    if won:
        result = f"WON — {horse} beat the field"
        if ctx.get("winner_sp"):
            result += f" at SP {ctx['winner_sp']:.1f}"
        result += "."
    else:
        result = f"LOST — winner was {winner}"
        if ctx.get("winner_sp"):
            result += f" (SP {ctx['winner_sp']:.1f})"
        result += f". {horse}"
        if ctx.get("pick_pos"):
            result += f" finished {ctx['pick_pos']}/{ctx.get('field') or '?'}"
            if ctx.get("pick_beaten"):
                result += f", beaten {ctx['pick_beaten']:.1f}L"
        else:
            result += " did not finish in the recorded places"
        if ctx.get("pick_sp"):
            result += f" (SP {ctx['pick_sp']:.1f}, was {odds_s} on the morning board)"
        result += "."
    finish_block = ("First home:\n" + "\n".join(ctx["lines"]) + "\n") if ctx.get("lines") else ""

    # Prior verdict block — probability-aware, judges NO BET calls too
    wp = prior.get("win_prob")
    wp_s = f"{wp:.2f}" if isinstance(wp, (int, float)) else "?"
    if prior.get("no_bet"):
        prior_block = (f"Your prior VERDICT: NO BET (you gave the likeliest horse "
                       f"only {wp_s}). Judge that abstention against the result: a "
                       "well-backed easy winner means you were too cautious; a "
                       "chaotic result vindicates the pass.")
    elif prior.get("pick"):
        factors = ", ".join(str(f) for f in (prior.get("missing_factors") or [])[:5]) or "(none)"
        prior_block = (f"Your prior VERDICT: {prior['pick']} with win_prob {wp_s}.\n"
                       f"Your prior key_risk: {prior.get('key_risk') or '(none)'}\n"
                       f"Your prior missing_factors: {factors}\n"
                       f"Say whether {wp_s} was too high, too low, or fair in hindsight.")
    else:
        prior_block = "You gave no verdict on this race."

    # Extended tier: the winner's pre-race profile, so 'what we missed' is
    # grounded in facts that were knowable before the off.
    winner_line = ""
    if not won and brief_builder.voice_budget(voice)[0] >= 1200:
        try:
            con = bets_store.connect()
            con.row_factory = sqlite3.Row
            fs = brief_builder.FeatureStore(con, date)
            wn = _norm_horse(winner)
            wr = next((r for r in race.get("runners") or []
                       if _norm_horse(r.get("name") or "") == wn), None)
            if wr:
                rp = brief_builder.load_rp_card(date)
                ck = brief_builder._norm_course(race.get("course") or "")
                tk = brief_builder._time24(race.get("time") or "")
                line = brief_builder._runner_core_line(
                    wr, race, fs, rp["runners"].get((ck, tk, wn)), False)
                winner_line = f"Winner's pre-race profile (knowable before the off):\n{line}\n"
            con.close()
        except Exception:
            winner_line = ""

    return f"""Date: {date}
Race: {race.get('course')} {race.get('time')} — {race.get('name') or ''} ({cond})
Board pick (the horse on yesterday's card): {horse} @ {odds_s}
Result: {result}
{finish_block}{winner_line}{prior_block}
{model_line}

Focus on THIS horse's run (pace, trip, ground, mark, fitness) — not a full race tip.
Base your verdict on the result facts above, not speculation.
Reply with ONE JSON object only (no markdown), keys with double-quoted string values:
  "horse_verdict": 2-4 sentences on why {horse} won or lost,
  "what_we_missed": one line (use if lost) OR "what_we_got_right": one line (use if won),
  "improve_next_time": one actionable change — START it with the single factor word
   that matters most, chosen from: {", ".join(FACTOR_WORDS)}.
Example shape: {{"horse_verdict":"...","what_we_missed":"...","improve_next_time":"draw: ..."}}
"""


def _parse_pm_json(text: str) -> dict | None:
    if not text or text.startswith("ERROR") or text.startswith("["):
        return None
    raw = text.strip()
    # Strip markdown fences
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    obj = None
    try:
        i, j = raw.find("{"), raw.rfind("}")
        if i >= 0 and j > i:
            obj = json.loads(raw[i:j + 1])
    except Exception:
        obj = None

    def _field(name: str) -> str:
        if isinstance(obj, dict):
            val = (obj.get(name) or "").strip() if isinstance(obj.get(name), str) else ""
            if val:
                return val
        # Quoted value
        m = re.search(rf'"{name}"\s*:\s*"((?:\\.|[^"\\])*)"', raw, re.S)
        if m:
            return m.group(1).replace('\\"', '"').strip()
        # Unquoted value (NVIDIA often omits quotes on long prose)
        m = re.search(
            rf'"{name}"\s*:\s*(.+?)(?=\s*,\s*"|\s*\n\s*"|\s*}})',
            raw, re.S)
        if m:
            return m.group(1).strip().strip(",").strip().strip('"').strip()
        return ""

    verdict = _field("horse_verdict")
    if not verdict:
        return None
    out = {
        "horse_verdict": verdict[:800],
        "improve_next_time": _field("improve_next_time")[:300],
    }
    missed = _field("what_we_missed")
    got = _field("what_we_got_right")
    if missed:
        out["what_we_missed"] = missed[:300]
    if got:
        out["what_we_got_right"] = got[:300]
    return out


def _call_voice(voice: str, prompt: str, dry_run: bool = False) -> str:
    if dry_run:
        return json.dumps({
            "horse_verdict": f"[dry-run {voice}] Horse ran to form on the day.",
            "what_we_missed": "Dry-run placeholder.",
            "improve_next_time": "Weigh trip and ground more carefully.",
        })
    from ai_daily_analysis import RAW_CALLERS, _filtered_providers
    allowed = {k for k, _, _ in _filtered_providers()}
    if voice not in allowed and voice not in RAW_CALLERS:
        return f"ERROR: voice {voice} not available"
    fn = RAW_CALLERS.get(voice)
    if not fn:
        return f"ERROR: no raw caller for {voice}"
    # Soft token cap for short JSON replies
    prev = os.environ.get("GOING_AI_MAX_TOKENS")
    os.environ.setdefault("GOING_AI_MAX_TOKENS", "512")
    try:
        return fn(prompt) or ""
    finally:
        if prev is None and "GOING_AI_MAX_TOKENS" in os.environ and prev is None:
            # leave setdefault as-is for other voices in the same run
            pass


def _find_race_slot(pm: dict, race_name: str, horse: str,
                    course: str | None = None, time: str | None = None) -> dict | None:
    """Match PM race row: prefer race+horse, then race name, then course+time."""
    hn = _norm_horse(horse)
    rn = (race_name or "").strip().lower()
    rows = pm.get("races") or []
    for row in rows:
        if (row.get("race_name") or "").strip().lower() == rn and _norm_horse(row.get("horse") or "") == hn:
            return row
    if rn:
        for row in rows:
            if (row.get("race_name") or "").strip().lower() == rn:
                return row
    if course or time:
        ck = (course or "").strip().lower()
        tk = (time or "").strip().lower()
        for row in rows:
            if ((row.get("course") or "").strip().lower() == ck
                    and (row.get("time") or "").strip().lower() == tk):
                return row
    return None


def _connections_for(race: dict, horse: str) -> tuple[str, str]:
    """Return (jockey, trainer) for a horse on the card."""
    hn = _norm_horse(horse)
    for r in race.get("runners") or []:
        if _norm_horse(r.get("name") or "") == hn:
            return (
                (r.get("jockey") or "").strip(),
                (r.get("trainer") or "").strip(),
            )
    return "", ""


def _settled_targets(date: str, card: dict, report: dict, win_map: dict) -> list[dict]:
    """One target per card race with a known winner + board pick."""
    targets = []
    for race in card.get("races") or []:
        course = race.get("course") or ""
        time = race.get("time") or ""
        name = race.get("name") or ""
        winner = _lookup_winner(win_map, course, time)
        if not winner:
            continue
        entry = _race_entry(report, course, time, name)
        horse = _model_pick(entry, race)
        if not horse:
            continue
        won = _norm_horse(horse) == _norm_horse(winner)
        jockey, trainer = _connections_for(race, horse)
        targets.append({
            "race_name": name,
            "course": course,
            "time": time,
            "horse": horse,
            "winner": winner,
            "result": "won" if won else "lost",
            "odds": _odds_for(race, horse),
            "jockey": jockey,
            "trainer": trainer,
            "entry": entry,
            "race": race,
        })
    return targets


def run(date: str | None = None, card_path: str | Path | None = None,
        fresh: bool = False, dry_run: bool = False,
        voices: tuple[str, ...] | None = None,
        write_json: bool = True) -> dict:
    _load_env()
    date = date or (dt.date.today() - dt.timedelta(days=1)).isoformat()
    voices = voices or live_pm_voices()
    card = _load_card(Path(card_path) if card_path else None, date)
    report = _load_ai_report(date)
    path_s = str(card_path) if card_path else str(REPO / "yesterday.json")
    win_map = _winner_map(date, path_s if Path(path_s).exists() else None)

    pm = {"date": date, "generated": None, "races": []} if fresh else load_pm(date)
    if fresh:
        pm = {"date": date, "generated": None, "races": []}

    targets = _settled_targets(date, card, report, win_map)
    print(f"Horse post-mortem {date}: {len(targets)} settled picks, "
          f"voices={','.join(voices) or '(none)'}")

    called = skipped = errors = 0
    for t in targets:
        slot = _find_race_slot(pm, t["race_name"], t["horse"])
        if slot is None:
            slot = {
                "race_name": t["race_name"],
                "course": t["course"],
                "time": t["time"],
                "horse": t["horse"],
                "winner": t["winner"],
                "result": t["result"],
                "odds": t["odds"],
                "jockey": t.get("jockey") or "",
                "trainer": t.get("trainer") or "",
                "voices": {},
            }
            pm.setdefault("races", []).append(slot)
        else:
            # Keep winner/result fresh if corpus caught up
            slot["winner"] = t["winner"]
            slot["result"] = t["result"]
            if t.get("jockey"):
                slot["jockey"] = t["jockey"]
            if t.get("trainer"):
                slot["trainer"] = t["trainer"]
            slot.setdefault("voices", {})

        for voice in voices:
            existing = (slot.get("voices") or {}).get(voice)
            if (not fresh and isinstance(existing, dict)
                    and existing.get("horse_verdict")):
                skipped += 1
                continue
            prior = _voice_prior(t["entry"], voice)
            brief = _build_brief(
                date, t["race"], t["horse"], t["winner"], t["entry"], voice, prior)
            try:
                raw = _call_voice(voice, brief, dry_run=dry_run)
                parsed = _parse_pm_json(raw)
                if not parsed:
                    errors += 1
                    slot["voices"][voice] = {
                        "error": (raw or "")[:200],
                        "raw_ok": False,
                        "prompt": brief,
                    }
                    print(f"  ! {t['race_name'][:40]} / {voice}: parse fail")
                else:
                    parsed["prior_pick"] = prior.get("pick")
                    parsed["prior_confidence"] = prior.get("confidence")
                    parsed["prompt"] = brief
                    slot["voices"][voice] = parsed
                    called += 1
                    print(f"  ✓ {t['horse']} @ {t['course']} {t['time']} / {voice}")
            except Exception as e:
                errors += 1
                slot["voices"][voice] = {
                    "error": str(e)[:200],
                    "raw_ok": False,
                    "prompt": brief,
                }
                print(f"  ! {voice}: {e}")

        # Checkpoint after each race
        pm["generated"] = dt.datetime.now().isoformat(timespec="seconds")
        if not dry_run:
            save_pm(pm, write_json=write_json)

    if dry_run:
        pm["generated"] = dt.datetime.now().isoformat(timespec="seconds")
        pm["dry_run"] = True
    else:
        save_pm(pm, write_json=write_json)

    summary = {
        "date": date,
        "targets": len(targets),
        "called": called,
        "skipped": skipped,
        "errors": errors,
        "db": str(going_paths.db_path()),
        "path": str(pm_path(date)),
        "voices": list(voices),
    }
    print(f"Done: called={called} skipped={skipped} errors={errors} "
          f"→ db={summary['db']} json={summary['path']}")
    return summary


def pm_for_race(date: str, race_name: str, horse: str) -> dict | None:
    """Lookup helper for serve.py — entry for race+horse or None."""
    return _find_race_slot(load_pm(date), race_name, horse)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Yesterday per-horse AI post-mortems")
    ap.add_argument("--date", default=(dt.date.today() - dt.timedelta(days=1)).isoformat())
    ap.add_argument("--card", default=None, help="Path to yesterday.json")
    ap.add_argument("--fresh", action="store_true", help="Rebuild all voices")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--providers", default="",
                    help="Comma list override (default: live credit-game voices)")
    ap.add_argument("--migrate-json", action="store_true",
                    help="Backfill horse_postmortems from JSON cache (no API calls)")
    ap.add_argument("--no-json", action="store_true",
                    help="Write DB only (skip ai_reports/horse_pm_*.json mirror)")
    args = ap.parse_args()
    if args.migrate_json:
        # With only --migrate-json, ingest all JSON caches; with --date, that day only.
        migrate_date = args.date if "--date" in sys.argv else None
        info = migrate_json_to_db(migrate_date)
        print(f"Migrated {info['files']} files → {info['rows']} DB rows "
              f"({going_paths.db_path()})")
        return
    voices = None
    if args.providers.strip():
        voices = tuple(p.strip() for p in args.providers.split(",") if p.strip())
    run(date=args.date, card_path=args.card, fresh=args.fresh,
        dry_run=args.dry_run, voices=voices, write_json=not args.no_json)


if __name__ == "__main__":
    main()
