"""
ai_track_record.py — Score AI daily analyses against actual race results.

Runs nightly on Lucy (after results are ingested). For every ai_analysis_*.json
report it:
1. Parses each AI's VERDICT pick (falls back to fuzzy extraction for old reports)
2. Looks up the actual winner in the results DB
3. Accumulates per-AI hit rates, consensus hit rates, and — most importantly —
   disagreement outcomes: when an AI disagreed with the model, who was right?
4. Tallies the "missing_factors" the AIs keep flagging, so recurring themes can
   feed the feature-ideation pipeline.

Output: ai_reports/track_record.json + console summary.

Cron (Lucy):
    30 22 * * * cd /home/pj/going && python3 ai_track_record.py >> logs/cron.log 2>&1
    40 22 * * * cd /home/pj/going && python3 ai_learning.py --register >> logs/cron.log 2>&1
"""
from __future__ import annotations

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
DB_PATH = str(going_paths.db_path())

AIS = ai_consensus.AIS
# Hobby credit-game voices scored in track_record (Lucy/ollama when enabled)
SCORE_VOICES = ("cursor", "nvidia", "groq", "cerebras", "ollama")


def norm_horse(name: str) -> str:
    """Normalize a horse name for matching: lowercase, strip country suffix and punctuation."""
    name = re.sub(r"\s*\([A-Z]{2,3}\)\s*$", "", name or "")
    return re.sub(r"[^a-z0-9]", "", name.lower())


def to_24h(t: str) -> str:
    """Racecard times like '3:24' are PM; convert to 'HH:MM' 24h."""
    try:
        h, m = t.strip().split(":")
        h = int(h)
        if h < 11:
            h += 12
        return f"{h:02d}:{int(m):02d}"
    except Exception:
        return t


def _minutes(t: str) -> int | None:
    try:
        hh, mm = to_24h(t).split(":")
        return int(hh) * 60 + int(mm)
    except Exception:
        return None


def norm_course(course: str) -> str:
    import bets_store
    return bets_store.norm_name(course)


def _time_variants(time: str) -> list[str]:
    """Racecard times may be 2:25 in reports and 14:25 or 2:25 in the corpus."""
    t = (time or "").strip()
    if not t:
        return []
    out = []
    t24 = to_24h(t)
    for cand in (t24, t):
        if cand and cand not in out:
            out.append(cand)
    # 14:25 vs 2:25 without PM assumption
    if ":" in t24:
        h, m = t24.split(":", 1)
        try:
            hi = int(h)
            if hi >= 12:
                short = f"{hi - 12}:{m}" if hi > 12 else f"12:{m}"
                if short not in out:
                    out.append(short)
                # also zero-padded short like 04:25 from card quirks
                if hi > 12:
                    padded = f"{hi - 12:02d}:{m}"
                    if padded not in out:
                        out.append(padded)
        except ValueError:
            pass
    return out


def _course_match(card_course: str, corpus_course: str) -> bool:
    a, b = norm_course(card_course), norm_course(corpus_course)
    if not a or not b:
        return False
    if a == b:
        return True
    # Newmarket vs Newmarket (July) / Newmarket (Rowley)
    return a.startswith(b) or b.startswith(a) or a in b or b in a


def _race_name_score(want: str, have: str) -> float:
    """Crude overlap score for race titles (0..1)."""
    import re
    def toks(s):
        s = re.sub(r"\(.*?\)", " ", (s or "").lower())
        return {t for t in re.split(r"[^a-z0-9]+", s) if len(t) > 2}
    A, B = toks(want), toks(have)
    if not A or not B:
        return 0.0
    return len(A & B) / max(len(A), len(B))


def _corpus_finishers(date: str) -> int:
    con = sqlite3.connect(DB_PATH)
    try:
        row = con.execute(
            "SELECT COUNT(*) FROM races ra JOIN runners r ON r.race_id=ra.race_id "
            "WHERE ra.date=? AND r.finish_pos IS NOT NULL AND r.finish_pos<>''",
            (date,),
        ).fetchone()
        return int(row[0]) if row else 0
    finally:
        con.close()


def sync_results_for_date(date: str, allow_network: bool = True, force: bool = False) -> bool:
    """Ingest results into going.db if missing (Betfair SP / rpscrape)."""
    if _corpus_finishers(date) and not force:
        return True
    if not allow_network:
        return _corpus_finishers(date) > 0
    try:
        import yesterday_results
        # force=True on live days so the 15-min refresh can pick up new finishers
        # despite the polite cooldown in yesterday_results.ensure
        yesterday_results.ensure(date, allow_network=True, force=force)
    except Exception:
        pass
    if _corpus_finishers(date):
        return True
    try:
        import ingest_results
        slash = date.replace("-", "/")
        ingest_results.ingest(slash, slash, ["gb", "ire"])
    except Exception:
        pass
    return _corpus_finishers(date) > 0


def find_winner(con, date: str, course: str, time: str,
                race_name: str | None = None) -> str | None:
    """Return the winning horse for a race, or None if no result yet."""
    courses = []
    for c in (norm_course(course), (course or "").lower().strip()):
        if c and c not in courses:
            courses.append(c)
    times = _time_variants(time)
    for c in courses:
        for t in times:
            race_id = f"{date}|{c}|{t}"
            row = con.execute(
                "SELECT horse FROM runners WHERE race_id=? AND CAST(finish_pos AS INT)=1",
                (race_id,),
            ).fetchone()
            if row:
                return row[0]
    # Fallback: normalized course + time join
    for t in times:
        row = con.execute(
            "SELECT r.horse, rc.course FROM runners r JOIN races rc ON rc.race_id=r.race_id "
            "WHERE rc.date=? AND CAST(r.finish_pos AS INT)=1 AND rc.off_time=? ",
            (date, t),
        ).fetchall()
        for horse, ccourse in row:
            if _course_match(course, ccourse):
                return horse

    # Soft match: same meeting + near off-time, or race-title overlap.
    # Racing API vs RP often differ by a few minutes; Newmarket (July) etc.
    want_m = _minutes(time)
    try:
        rows = con.execute(
            "SELECT r.horse, rc.course, rc.off_time, "
            "COALESCE(rc.race_name, rc.name, '') "
            "FROM runners r JOIN races rc ON rc.race_id=r.race_id "
            "WHERE rc.date=? AND CAST(r.finish_pos AS INT)=1",
            (date,),
        ).fetchall()
    except sqlite3.OperationalError:
        rows = con.execute(
            "SELECT r.horse, rc.course, rc.off_time, '' "
            "FROM runners r JOIN races rc ON rc.race_id=r.race_id "
            "WHERE rc.date=? AND CAST(r.finish_pos AS INT)=1",
            (date,),
        ).fetchall()
    best = None  # (score, horse)
    for horse, ccourse, off, rname in rows:
        if not _course_match(course, ccourse):
            continue
        score = 0.0
        off_m = _minutes(off or "")
        if want_m is not None and off_m is not None:
            delta = abs(want_m - off_m)
            if delta <= 2:
                score += 3.0
            elif delta <= 5:
                score += 2.0
            elif delta <= 12:
                score += 1.0
            else:
                score -= 0.5
        if race_name:
            score += 2.5 * _race_name_score(race_name, rname)
        if score >= 2.0 and (best is None or score > best[0]):
            best = (score, horse)
    return best[1] if best else None


BANKROLL_FILE = REPORT_DIR / "bankrolls.json"
START_BALANCE = 1000
# Fallback voice list if live filter is unavailable
CREDIT_RESET_VOICES = ("model", "cursor", "nvidia", "groq", "cerebras", "claude", "openai", "ollama")


def reset_credit_game(since: str | None = None, voices=None) -> dict:
    """Wipe balances/history and start a new season from `since` (default: today).

    Default voices = live credit_game_keys() so Lucy/ollama is included when
    GOING_AI_OLLAMA=1 and parked sit-outs are omitted.
    """
    since = since or dt.date.today().isoformat()
    if voices is None:
        try:
            voices = ai_consensus.credit_game_keys()
        except Exception:
            voices = CREDIT_RESET_VOICES
    bk = {
        "start": START_BALANCE,
        "since": since,
        "ais": {
            ai: {"balance": START_BALANCE, "settled": [], "settled_legs": [], "history": []}
            for ai in voices
        },
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    BANKROLL_FILE.write_text(json.dumps(bk, indent=2))
    return bk


def _leg_id(date: str, entry: dict) -> str:
    return f"{date}|{entry.get('race') or ''}|{norm_horse(entry.get('horse') or '')}"


def settle_credit_game(con) -> dict:
    """Settle credit-game stakes as race results land.

    Past days: settle remaining legs, then mark the day complete (refund
    unresolvable legs once the day is 3+ days old).
    Today: settle each finished race immediately so bankrolls update during
    the afternoon — unfinished races stay pending.
    """
    bk = {"start": START_BALANCE, "ais": {}}
    if BANKROLL_FILE.exists():
        try:
            bk = json.loads(BANKROLL_FILE.read_text())
        except Exception:
            pass
    # Legacy: top-level "model" → ais.model
    if isinstance(bk.get("model"), dict) and "model" not in (bk.get("ais") or {}):
        bk.setdefault("ais", {})["model"] = bk.pop("model")
    today = dt.date.today().isoformat()
    since = bk.get("since") or "1970-01-01"

    for path in sorted(REPORT_DIR.glob("ai_analysis_????-??-??.json")):
        try:
            report = json.loads(path.read_text())
        except Exception:
            continue
        date = report.get("date")
        allocations = report.get("allocations") or {}
        if not allocations or not date:
            continue
        if date < since:
            continue

        try:
            age = (dt.date.today() - dt.date.fromisoformat(date)).days
        except Exception:
            age = 0
        is_today = date == today

        for ai, alloc in allocations.items():
            rec = bk.setdefault("ais", {}).setdefault(
                ai, {"balance": START_BALANCE, "settled": [], "settled_legs": [], "history": []})
            if date in (rec.get("settled") or []):
                continue
            settled_legs = set(rec.get("settled_legs") or [])
            staked = returned = voided = wins = newly = 0
            pending = 0

            for e in alloc.get("entries") or []:
                lid = _leg_id(date, e)
                if lid in settled_legs:
                    continue
                credits = e.get("credits") or 0
                race = e.get("race") or ""
                course, _, time = race.rpartition(" ")
                winner = find_winner(con, date, course, time) if course else None
                odds = e.get("odds")
                if winner is None or not odds:
                    if (not is_today) and age >= 3:
                        voided += credits
                        settled_legs.add(lid)
                        newly += 1
                    else:
                        pending += 1
                    continue
                newly += 1
                staked += credits
                settled_legs.add(lid)
                if norm_horse(winner) == norm_horse(e.get("horse") or ""):
                    returned += credits * float(odds)
                    wins += 1

            if newly:
                rec["balance"] = round(float(rec.get("balance") or START_BALANCE)
                                       - staked + returned, 1)
                rec["settled_legs"] = sorted(settled_legs)
                hist = rec.setdefault("history", [])
                row = next((h for h in hist if h.get("date") == date), None)
                if row is None:
                    row = {"date": date, "staked": 0, "wins": 0, "returned": 0.0, "voided": 0}
                    hist.append(row)
                row["staked"] = int(row.get("staked") or 0) + int(staked)
                row["wins"] = int(row.get("wins") or 0) + int(wins)
                row["returned"] = round(float(row.get("returned") or 0) + returned, 1)
                row["voided"] = int(row.get("voided") or 0) + int(voided)

            if pending == 0 and not is_today:
                left = [
                    e for e in (alloc.get("entries") or [])
                    if _leg_id(date, e) not in settled_legs
                ]
                if not left and date not in (rec.get("settled") or []):
                    rec.setdefault("settled", []).append(date)

    BANKROLL_FILE.write_text(json.dumps(bk, indent=2))
    return bk


def refresh_outcomes(allow_network: bool = True) -> dict:
    """Pull today's (and yesterday's) results into the DB, then settle credit stakes.

    Used by the 15-min refresh and publish so WON/LOST pills and bankrolls
    stay current without waiting for the overnight job.
    """
    today = dt.date.today()
    yday = today - dt.timedelta(days=1)
    synced = {}
    for d in (yday.isoformat(), today.isoformat()):
        try:
            synced[d] = bool(sync_results_for_date(
                d, allow_network=allow_network, force=(d == today.isoformat())))
        except Exception as e:
            synced[d] = f"error:{e}"

    con = sqlite3.connect(DB_PATH)
    try:
        bk = settle_credit_game(con)
    finally:
        con.close()

    balances = {ai: rec.get("balance") for ai, rec in (bk.get("ais") or {}).items()}
    return {"synced": synced, "bankrolls": balances}


def main():
    con = sqlite3.connect(DB_PATH)
    today = dt.date.today().isoformat()

    # Pull recent results into corpus before scoring
    report_dates = set()
    for path in sorted(REPORT_DIR.glob("ai_analysis_*.json")):
        try:
            d = json.loads(path.read_text()).get("date")
            if d and d < today:
                report_dates.add(d)
        except Exception:
            pass
    for d in sorted(report_dates):
        if sync_results_for_date(d):
            print(f"  results OK for {d}")
        else:
            print(f"  results missing for {d} (will retry later)")

    stats = {ai: {"picks": 0, "wins": 0} for ai in SCORE_VOICES}
    stats["model"] = {"picks": 0, "wins": 0}
    stats["consensus_2plus"] = {"picks": 0, "wins": 0}
    stats["consensus_3"] = {"picks": 0, "wins": 0}
    disagreements = {ai: Counter() for ai in SCORE_VOICES}
    missing_factors = Counter()
    races_scored = 0
    races_unresolved = 0

    for path in sorted(REPORT_DIR.glob("ai_analysis_*.json")):
        try:
            report = json.loads(path.read_text())
        except Exception:
            continue
        date = report.get("date")
        for race in report.get("races", []):
            winner = find_winner(con, date, race.get("course") or "", race.get("time") or "")
            if not winner:
                races_unresolved += 1
                continue
            winner_n = norm_horse(winner)

            top3 = (race.get("model_prediction") or {}).get("top_3") or []
            model_pick = top3[0]["name"] if top3 else None

            picks = {}
            for ai in ai_consensus.analysis_voice_keys():
                text = race.get(f"{ai}_analysis") or ""
                verdict = ai_consensus.extract_verdict(text)
                if verdict:
                    picks[ai] = verdict["pick"]
                    for f in verdict.get("missing_factors") or []:
                        missing_factors[str(f).strip().lower()] += 1
                else:
                    cands = ai_consensus.extract_horse_picks(text)
                    picks[ai] = cands[0] if cands else None

            races_scored += 1
            if model_pick:
                stats["model"]["picks"] += 1
                if norm_horse(model_pick) == winner_n:
                    stats["model"]["wins"] += 1

            for ai in SCORE_VOICES:
                p = picks.get(ai)
                if not p:
                    continue
                hit = norm_horse(p) == winner_n
                stats[ai]["picks"] += 1
                if hit:
                    stats[ai]["wins"] += 1
                if model_pick and norm_horse(p) != norm_horse(model_pick):
                    model_hit = norm_horse(model_pick) == winner_n
                    if hit:
                        disagreements[ai]["ai_right"] += 1
                    elif model_hit:
                        disagreements[ai]["model_right"] += 1
                    else:
                        disagreements[ai]["both_wrong"] += 1

            # Consensus: 2+ / 3 AIs on the same horse
            tally = Counter(norm_horse(p) for p in picks.values() if p)
            if tally:
                top, n = tally.most_common(1)[0]
                if n >= 2:
                    stats["consensus_2plus"]["picks"] += 1
                    if top == winner_n:
                        stats["consensus_2plus"]["wins"] += 1
                if n == 3:
                    stats["consensus_3"]["picks"] += 1
                    if top == winner_n:
                        stats["consensus_3"]["wins"] += 1

    bankrolls = settle_credit_game(con)
    con.close()

    def rate(s):
        return round(s["wins"] / s["picks"], 3) if s["picks"] else None

    out = {
        "races_scored": races_scored,
        "races_unresolved": races_unresolved,
        "hit_rates": {k: {**v, "rate": rate(v)} for k, v in stats.items()},
        "disagreements_vs_model": {ai: dict(c) for ai, c in disagreements.items()},
        "missing_factors_top20": missing_factors.most_common(20),
        "bankrolls": {ai: rec.get("balance") for ai, rec in (bankrolls.get("ais") or {}).items()},
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "track_record.json").write_text(json.dumps(out, indent=2))

    print(f"Scored {races_scored} races ({races_unresolved} without results yet)")
    for k, v in out["hit_rates"].items():
        print(f"  {k:16s} picks={v['picks']:4d} wins={v['wins']:4d} rate={v['rate']}")
    for ai, d in out["disagreements_vs_model"].items():
        if d:
            print(f"  disagree[{ai}]: {d}")
    if missing_factors:
        print("  top missing factors:", missing_factors.most_common(5))
    if out["bankrolls"]:
        board = sorted(out["bankrolls"].items(), key=lambda kv: kv[1] or 0, reverse=True)
        print("  bankrolls: " + " · ".join(f"{ai} {bal}" for ai, bal in board))
    model = out["hit_rates"].get("model", {})
    if model.get("picks"):
        print(f"  model prediction rate: {model['wins']}/{model['picks']} "
              f"({model.get('rate', 0):.1%})")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Score AI track record + settle credit game")
    ap.add_argument("--reset-credits", action="store_true",
                    help="wipe bankrolls and start a new season from today (or --since)")
    ap.add_argument("--since", default=None,
                    help="YYYY-MM-DD season start when used with --reset-credits")
    ap.add_argument("--refresh-outcomes", action="store_true",
                    help="sync today/yesterday results + settle credit stakes, then exit")
    args = ap.parse_args()
    if args.reset_credits:
        bk = reset_credit_game(since=args.since)
        print(f"Credit game reset → start={bk['start']} since={bk['since']} "
              f"voices={','.join(bk['ais'])}")
    if args.refresh_outcomes:
        out = refresh_outcomes(allow_network=True)
        print(f"Outcomes: synced={out.get('synced')} bankrolls={out.get('bankrolls')}")
        if not args.reset_credits:
            raise SystemExit(0)
    main()
