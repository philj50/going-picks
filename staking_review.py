"""
staking_review.py — nightly post-mortem of each voice's betting DAY BOOK.

Race post-mortems judge predictions; this judges the staking decisions the
betting-freedom game introduced: concentration, spreading, and sitting out.
Rule-computed (no LLM calls). Results persist to ai_reports/staking_review.json
and feed a "YOUR STAKING RECORD" line back into the next day-book prompt.

    python3 staking_review.py                 # review yesterday + today
    python3 staking_review.py --date 2026-07-18
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sqlite3
from pathlib import Path

REPO = Path(__file__).parent
OUT_PATH = REPO / "ai_reports" / "staking_review.json"


def _load() -> dict:
    if not OUT_PATH.exists():
        return {}
    try:
        return json.loads(OUT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def review_day(date: str) -> dict:
    """Score every voice's day book for a settled date. Returns {voice: {...}}.

    Also captures which prompt variant each voice was assigned to that day
    (for A/B testing efficacy tracking)."""
    import ai_consensus
    import ai_track_record as tr
    import prompt_variants

    path = REPO / "ai_reports" / f"ai_analysis_{date}.json"
    if not path.exists():
        return {}
    rep = json.loads(path.read_text(encoding="utf-8"))
    allocs = rep.get("allocations") or {}
    if not allocs:
        return {}

    con = sqlite3.connect(tr.DB_PATH)
    win_cache: dict = {}

    def winner(course: str, time: str):
        key = (course, time)
        if key not in win_cache:
            try:
                win_cache[key] = tr.find_winner(con, date, course, time)
            except Exception:
                win_cache[key] = None
        return win_cache[key]

    out = {}
    try:
        for voice, alloc in allocs.items():
            entries = alloc.get("entries") or []
            n_bets = len(entries)
            staked = wins = unresolved = 0
            pnl = 0.0
            for e in entries:
                race = e.get("race") or ""
                course, _, time = race.rpartition(" ")
                w = winner(course, time) if course else None
                odds = e.get("odds")
                if not w or not odds:
                    unresolved += 1
                    continue
                credits = int(e.get("credits") or 0)
                staked += credits
                if tr.norm_horse(w) == tr.norm_horse(e.get("horse") or ""):
                    wins += 1
                    pnl += credits * (float(odds) - 1.0)
                else:
                    pnl -= credits

            # Unbacked picks: verdicts the voice chose NOT to stake
            staked_horses = {tr.norm_horse(e.get("horse") or "") for e in entries}
            ub_n = ub_wins = 0
            for race in rep.get("races") or []:
                v = ai_consensus.extract_verdict(race.get(f"{voice}_analysis") or "")
                if not v or v.get("no_bet"):
                    continue
                hn = tr.norm_horse(v.get("pick") or "")
                if not hn or hn in staked_horses:
                    continue
                w = winner(race.get("course") or "", race.get("time") or "")
                if not w:
                    continue
                ub_n += 1
                if tr.norm_horse(w) == hn:
                    ub_wins += 1

            if not n_bets:
                style = "sat out"
            elif n_bets == 1 and staked >= 60:
                style = "all-in"
            elif n_bets <= 3:
                style = "concentrated"
            else:
                style = "spread"

            if not n_bets:
                judgment = ("good pass — unbacked picks went "
                            f"{ub_wins}/{ub_n}" if ub_n and ub_wins / ub_n <= 0.2
                            else f"missed {ub_wins} winner(s) among {ub_n} unbacked picks"
                            if ub_n else "quiet day")
            else:
                judgment = (f"{'profit' if pnl > 0 else 'loss' if pnl < 0 else 'flat'} "
                            f"{pnl:+.0f}cr from {wins}/{n_bets - unresolved} staked")

            # Capture which A/B test variant was assigned to this voice that day
            variant = prompt_variants.select_variant_for_voice(
                voice, dt.date.fromisoformat(date))
            variant_name = prompt_variants.VARIANTS[variant]["name"]

            out[voice] = {
                "n_bets": n_bets, "staked": staked, "wins": wins,
                "unresolved": unresolved, "pnl": round(pnl, 1),
                "picks": alloc.get("picks"), "method": alloc.get("method"),
                "unbacked": ub_n, "unbacked_wins": ub_wins,
                "style": style, "judgment": judgment,
                "variant": variant, "variant_name": variant_name,
            }
    finally:
        con.close()

    if out:
        data = _load()
        data[date] = out
        OUT_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out


def staking_line(voice: str, days: int = 14) -> str:
    """One-line day-book feedback for the next allocation prompt."""
    data = _load()
    if not data:
        return ""
    cutoff = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    rows = [d[voice] for date, d in sorted(data.items())
            if date >= cutoff and voice in d]
    if not rows:
        return ""
    bet_days = [r for r in rows if r.get("n_bets")]
    sat_days = [r for r in rows if not r.get("n_bets")]
    bits = []
    if bet_days:
        w = sum(r.get("wins") or 0 for r in bet_days)
        n = sum((r.get("n_bets") or 0) - (r.get("unresolved") or 0) for r in bet_days)
        pnl = sum(r.get("pnl") or 0 for r in bet_days)
        avg = sum(r.get("n_bets") or 0 for r in bet_days) / len(bet_days)
        bits.append(f"{len(bet_days)} betting day(s): {w}/{n} staked picks won,"
                    f" P&L {pnl:+.0f}cr, avg {avg:.1f} bets/day")
    if sat_days:
        ub = sum(r.get("unbacked") or 0 for r in sat_days)
        ubw = sum(r.get("unbacked_wins") or 0 for r in sat_days)
        bits.append(f"{len(sat_days)} sat-out day(s) — unbacked picks went {ubw}/{ub}")
    styles = [r.get("style") for r in bet_days if r.get("style")]
    if styles:
        from collections import Counter
        bits.append("usual style: " + Counter(styles).most_common(1)[0][0])
    if not bits:
        return ""
    return f"\nYOUR STAKING RECORD (last {days} days): " + " · ".join(bits) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Nightly day-book staking review.")
    ap.add_argument("--date", default=None,
                    help="YYYY-MM-DD, or today/yesterday (default: yesterday AND today)")
    args = ap.parse_args()
    if args.date:
        d = args.date
        if d == "today":
            d = dt.date.today().isoformat()
        elif d == "yesterday":
            d = (dt.date.today() - dt.timedelta(days=1)).isoformat()
        dates = [d]
    else:
        dates = [(dt.date.today() - dt.timedelta(days=1)).isoformat(),
                 dt.date.today().isoformat()]
    for d in dates:
        out = review_day(d)
        print(f"{d}: reviewed {len(out)} voice day-books")
        for voice, r in sorted(out.items()):
            print(f"  {voice}: {r['style']} — {r['judgment']}")


if __name__ == "__main__":
    main()
