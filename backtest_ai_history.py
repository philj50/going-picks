"""
backtest_ai_history.py — Retrospective AI evaluation on historical races.

Samples completed races from a post-knowledge-cutoff window (default Feb-Jun
2026, so none of the AIs can have memorized the results), reconstructs the
same pre-race brief the daily job builds, runs the same three AIs, and scores
their picks against the actual winners.

Resumable: results append to ai_reports/backtest/results.jsonl keyed by
race_id; already-scored races are skipped on re-run.

Usage:
    python3 backtest_ai_history.py --n 50                 # pilot
    python3 backtest_ai_history.py --n 500                # full run
    python3 backtest_ai_history.py --n 500 --skip-ollama  # cloud AIs only
    python3 backtest_ai_history.py --summary              # just recompute summary
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sqlite3
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ai_consensus
import ai_config
import race_stats
from ai_daily_analysis import PROVIDERS, _filtered_providers

REPO = Path(__file__).parent
OUT_DIR = REPO / "ai_reports" / "backtest"
RESULTS = OUT_DIR / "results.jsonl"
SUMMARY_NAME = "summary.json"
DB_PATH = "/mnt/nvme/going/db/going.db"

D_FROM = "2026-02-01"   # after all AI knowledge cutoffs
D_TO = "2026-06-28"


def parse_odds(val) -> float | None:
    """Odds may be decimal ('4.5') or fractional ('7/2'). Return decimal odds."""
    if val is None:
        return None
    s = str(val).strip()
    try:
        f = float(s)
        return f if f > 1.0 else None
    except ValueError:
        pass
    m = re.match(r"^(\d+)/(\d+)$", s)
    if m:
        return 1.0 + int(m.group(1)) / int(m.group(2))
    return None


def norm_horse(name: str) -> str:
    name = re.sub(r"\s*\([A-Z]{2,3}\)\s*$", "", name or "")
    return re.sub(r"[^a-z0-9]", "", name.lower())


def recent_form(con, horse_norm: str, before_date: str, limit=4) -> list:
    rows = con.execute(
        "SELECT r.finish_pos FROM runners r JOIN races rc ON rc.race_id=r.race_id "
        "WHERE r.horse_norm=? AND rc.date<? AND r.finish_pos IS NOT NULL "
        "ORDER BY rc.date DESC LIMIT ?",
        (horse_norm, before_date, limit),
    ).fetchall()
    return [r[0] for r in rows]


def sample_races(con, n: int, done: set) -> list:
    """Random completed races in the window with a known winner and 5-16 runners."""
    rows = con.execute(
        "SELECT rc.race_id, rc.date, rc.course, rc.off_time, rc.race_type, "
        "rc.handicap_flag, rc.field_size "
        "FROM races rc "
        "WHERE rc.date BETWEEN ? AND ? "
        "AND rc.field_size BETWEEN 5 AND 16 "
        "AND EXISTS (SELECT 1 FROM runners r WHERE r.race_id=rc.race_id "
        "            AND CAST(r.finish_pos AS INT)=1) "
        "ORDER BY RANDOM() LIMIT ?",
        (D_FROM, D_TO, n * 2),
    ).fetchall()
    return [r for r in rows if r[0] not in done][:n]


def build_race(con, row) -> dict | None:
    race_id, date, course, off_time, race_type, handicap, field_size = row
    extra = con.execute(
        "SELECT distance_f, going, race_class FROM races WHERE race_id=?", (race_id,)
    ).fetchone() or (None, None, None)
    runners = []
    winner = None
    for r in con.execute(
        "SELECT horse, horse_norm, trainer, jockey, draw, weight_lbs, headgear, "
        "finish_pos, decision_price, starting_price, bsp_win, "
        "official_rating, rpr, top_speed "
        "FROM runners WHERE race_id=?", (race_id,)
    ):
        (horse, hn, trainer, jockey, draw, wt, headgear,
         finish_pos, dp, sp, bsp, or_, rpr, ts) = r
        odds = parse_odds(dp) or parse_odds(sp) or parse_odds(bsp)
        try:
            if int(finish_pos) == 1:
                winner = horse
        except (TypeError, ValueError):
            pass
        try:
            stats = race_stats.runner_stats(con, horse, trainer, jockey,
                                            course, extra[0], extra[1], date)
        except Exception:
            stats = {}
        runners.append({
            "name": horse,
            "trainer": trainer or "?",
            "jockey": jockey or "?",
            "draw": draw,
            "weight_lbs": wt,
            "headgear": headgear,
            "odds_decimal": odds or 0,
            "official_rating": or_,
            "rpr": rpr,
            "top_speed": ts,
            "last_positions": recent_form(con, hn, date),
            "stats_lines": race_stats.stats_lines(stats),
        })
    if not winner or len(runners) < 5:
        return None
    fav = min((x for x in runners if x["odds_decimal"]), key=lambda x: x["odds_decimal"], default=None)
    distance_f, going, race_class = extra
    return {
        "distance_f": distance_f,
        "going": going,
        "race_class": race_class,
        "race_id": race_id,
        "name": f"{course} {off_time} ({date})".replace(f" ({date})", ""),
        "course": course,
        "time": off_time,
        "date": date,
        "is_flat": (race_type or "").lower() == "flat",
        "is_handicap": bool(handicap),
        "runners": runners,
        "winner": winner,
        "favourite": fav["name"] if fav else None,
    }


def compile_brief(race: dict, light: bool | None = None) -> str:
    """Mirror of ai_daily_analysis.compile_race_brief, minus model prediction."""
    if light is None:
        light = ai_config.light_brief()
    runners = list(race["runners"])
    if light:
        runners = sorted(runners, key=lambda r: r.get("odds_decimal") or 999)[:8]
    dist = race.get("distance_f")
    brief = f"""
RACE ANALYSIS BRIEF
==================

Race: {race['course']} {race['time']}
Course: {race['course']} | Time: {race['time']}
Type: {'Flat' if race['is_flat'] else 'Jumps'} | Handicap: {race['is_handicap']}
Distance: {f"{dist}f" if dist else '?'} | Going: {race.get('going') or '?'} | Class: {race.get('race_class') or '?'}
Field: {len(race['runners'])} runners

MODEL'S PREDICTION
------------------
No model prediction is available for this race — analyze it independently
from the data below. (Treat "agrees_with_model" in your verdict as null.)

RUNNERS
-------
"""
    for r in runners:
        odds = r.get("odds_decimal") or 0
        odds_str = f" @ {odds:.1f}" if odds > 0 else ""
        if light:
            form_str = "".join(str(p) for p in (r.get("last_positions") or [])[:4]) or "?"
            brief += (f"- {r['name']}{odds_str} | {r.get('trainer', '?')}/{r.get('jockey', '?')} "
                      f"| form {form_str}\n")
            continue
        brief += f"\n{r['name']} {odds_str}\n"
        brief += f"  Trainer: {r['trainer']}\n"
        brief += f"  Jockey: {r['jockey']}\n"
        brief += f"  Weight: {r.get('weight_lbs') or '?'}lbs | Draw: {r.get('draw') or '?'}\n"
        ratings = [f"OR {r['official_rating']}" if r.get("official_rating") else None,
                   f"RPR {r['rpr']}" if r.get("rpr") else None,
                   f"TS {r['top_speed']}" if r.get("top_speed") else None]
        ratings = [x for x in ratings if x]
        if ratings:
            brief += f"  Ratings: {' | '.join(ratings)}\n"
        if r["last_positions"]:
            brief += f"  Recent form (latest first): {', '.join(str(p) for p in r['last_positions'])}\n"
        if r.get("headgear"):
            brief += f"  Headgear: {r['headgear']}\n"
        for line in r.get("stats_lines") or []:
            brief += line + "\n"
    return brief


def pick_from(text: str, runner_names: list) -> str | None:
    verdict = ai_consensus.extract_verdict(text)
    if verdict:
        cand = verdict["pick"]
    else:
        cands = ai_consensus.extract_horse_picks(text)
        cand = cands[0] if cands else None
    if not cand:
        return None
    cn = norm_horse(cand)
    for name in runner_names:
        if cn == norm_horse(name) or cn in norm_horse(name) or norm_horse(name) in cn:
            return name
    return cand


def summarize():
    if not RESULTS.exists():
        print("No results yet.")
        return
    recs = [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]
    ais = list(ai_consensus.AIS)
    stats = {a: Counter() for a in ais + ["favourite", "consensus_2plus", "consensus_3"]}
    disagree_fav = {a: Counter() for a in ais}
    conf_buckets = {a: Counter() for a in ais}
    for rec in recs:
        wn = norm_horse(rec["winner"])
        fav = rec.get("favourite")
        if fav:
            stats["favourite"]["picks"] += 1
            if norm_horse(fav) == wn:
                stats["favourite"]["wins"] += 1
        picks = {}
        for a in ais:
            p = rec.get(f"{a}_pick")
            if not p:
                continue
            picks[a] = p
            hit = norm_horse(p) == wn
            stats[a]["picks"] += 1
            if hit:
                stats[a]["wins"] += 1
            conf = rec.get(f"{a}_confidence")
            if conf is not None:
                b = f"conf_{int(conf)//25*25}"  # 0-24 / 25-49 / 50-74 / 75-100
                conf_buckets[a][b + "_picks"] += 1
                if hit:
                    conf_buckets[a][b + "_wins"] += 1
            if fav and norm_horse(p) != norm_horse(fav):
                fav_hit = norm_horse(fav) == wn
                disagree_fav[a]["ai_right" if hit else ("fav_right" if fav_hit else "both_wrong")] += 1
        tally = Counter(norm_horse(p) for p in picks.values())
        if tally:
            top, nvotes = tally.most_common(1)[0]
            if nvotes >= 2:
                stats["consensus_2plus"]["picks"] += 1
                if top == wn:
                    stats["consensus_2plus"]["wins"] += 1
            if nvotes == 3:
                stats["consensus_3"]["picks"] += 1
                if top == wn:
                    stats["consensus_3"]["wins"] += 1

    out = {"races": len(recs), "generated": dt.datetime.now().isoformat(timespec="seconds")}
    print(f"\n=== BACKTEST SUMMARY ({len(recs)} races) ===")
    for k, c in stats.items():
        rate = round(c["wins"] / c["picks"], 3) if c["picks"] else None
        out[k] = {"picks": c["picks"], "wins": c["wins"], "rate": rate}
        print(f"  {k:16s} picks={c['picks']:4d} wins={c['wins']:4d} rate={rate}")
    out["disagree_vs_favourite"] = {a: dict(c) for a, c in disagree_fav.items()}
    out["confidence_calibration"] = {a: dict(c) for a, c in conf_buckets.items()}
    for a, c in disagree_fav.items():
        if c:
            print(f"  vs favourite [{a}]: {dict(c)}")
    (OUT_DIR / SUMMARY_NAME).write_text(json.dumps(out, indent=2))
    print(f"Summary -> {OUT_DIR / SUMMARY_NAME}")


def repair(args):
    """Re-run only the errored AI calls on races already in results.jsonl,
    then rewrite the file in place. Configured providers only."""
    recs = [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]
    active = {k: fn for k, fn, key_env in _filtered_providers()
              if not (k == "ollama" and args.skip_ollama)}
    targets = [rec for rec in recs
               if any(rec.get(f"{a}_error") or rec.get(f"{a}_pick") is None for a in active)]
    print(f"[{dt.datetime.now():%H:%M:%S}] Repairing {len(targets)} races "
          f"across providers: {sorted(active)}", flush=True)
    if not targets:
        return

    con = sqlite3.connect(DB_PATH)
    races_by_id = {}
    for rec in targets:
        row = con.execute(
            "SELECT race_id, date, course, off_time, race_type, handicap_flag, field_size "
            "FROM races WHERE race_id=?", (rec["race_id"],)).fetchone()
        if row:
            race = build_race(con, row)
            if race:
                races_by_id[rec["race_id"]] = race
    con.close()

    lock = threading.Lock()
    progress = {"n": 0}

    def fix(rec):
        race = races_by_id.get(rec["race_id"])
        if not race:
            return
        brief = compile_brief(race)
        names = [r["name"] for r in race["runners"]]
        todo = [(a, fn) for a, fn in active.items()
                if rec.get(f"{a}_error") or rec.get(f"{a}_pick") is None]
        with ThreadPoolExecutor(max_workers=max(1, len(todo))) as inner:
            futures = {a: inner.submit(fn, brief) for a, fn in todo}
            for a, fut in futures.items():
                try:
                    text = fut.result()
                except Exception as e:
                    text = f"ERROR: {e}"
                verdict = ai_consensus.extract_verdict(text)
                rec[f"{a}_pick"] = pick_from(text, names)
                rec[f"{a}_confidence"] = (verdict or {}).get("confidence")
                rec[f"{a}_agrees"] = (verdict or {}).get("agrees_with_model")
                rec[f"{a}_error"] = text.startswith("ERROR")
        with lock:
            progress["n"] += 1
            print(f"[{dt.datetime.now():%H:%M:%S}] repaired {progress['n']}/{len(targets)} "
                  f"{rec['race_id']}", flush=True)

    with ThreadPoolExecutor(max_workers=args.race_workers) as outer:
        list(outer.map(fix, targets))

    tmp = RESULTS.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in recs))
    tmp.replace(RESULTS)
    summarize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--skip-ollama", action="store_true")
    ap.add_argument("--race-workers", type=int, default=5,
                    help="races processed concurrently (each fans out to all providers)")
    ap.add_argument("--summary", action="store_true", help="only recompute summary")
    ap.add_argument("--repair", action="store_true",
                    help="re-run errored AI calls on already-recorded races")
    ap.add_argument("--dev", action="store_true", help="dev preset (groq, light brief, cache)")
    ap.add_argument("--providers", metavar="LIST", help="comma list of voices")
    ap.add_argument("--full-brief", action="store_true", help="verbose briefs")
    ap.add_argument("--tag", metavar="NAME",
                    help="separate arm: write results_NAME.jsonl / summary_NAME.json")
    ap.add_argument("--races-from", metavar="FILE",
                    help="backtest exactly the race_ids in this results.jsonl (A/B arms)")
    args = ap.parse_args()

    if args.dev:
        os.environ["GOING_AI_DEV"] = "1"
    if args.providers:
        os.environ["GOING_AI_PROVIDERS"] = args.providers
    if args.full_brief:
        os.environ["GOING_AI_LIGHT_BRIEF"] = "0"

    if args.tag:
        # A/B arm isolation: own results + summary files, never the main run's
        global RESULTS, SUMMARY_NAME
        RESULTS = OUT_DIR / f"results_{args.tag}.jsonl"
        SUMMARY_NAME = f"summary_{args.tag}.json"

    if args.summary:
        summarize()
        return
    if args.repair:
        repair(args)
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = set()
    if RESULTS.exists():
        for line in RESULTS.read_text().splitlines():
            try:
                done.add(json.loads(line)["race_id"])
            except Exception:
                pass

    # SQLite isn't thread-safe — hydrate every race up front, then close the
    # connection before any parallel work starts.
    con = sqlite3.connect(DB_PATH)
    if args.races_from:
        want = []
        for line in Path(args.races_from).read_text().splitlines():
            try:
                rid = json.loads(line).get("race_id")
            except Exception:
                continue
            if rid and rid not in done and rid not in want:
                want.append(rid)
        rows = []
        for rid in want[:args.n]:
            r = con.execute(
                "SELECT race_id, date, course, off_time, race_type, "
                "handicap_flag, field_size FROM races WHERE race_id=?",
                (rid,)).fetchone()
            if r:
                rows.append(r)
    else:
        rows = sample_races(con, args.n, done)
    races = [race for row in rows if (race := build_race(con, row))]
    con.close()
    print(f"[{dt.datetime.now():%H:%M:%S}] Backtesting {len(races)} races "
          f"({len(done)} already done), {args.race_workers} races in flight", flush=True)

    active = [(k, fn) for k, fn, key_env in _filtered_providers()
              if not (k == "ollama" and args.skip_ollama)]
    write_lock = threading.Lock()
    progress = {"n": 0}

    def analyze(race):
        brief = compile_brief(race)
        names = [r["name"] for r in race["runners"]]
        rec = {
            "race_id": race["race_id"],
            "date": race["date"],
            "winner": race["winner"],
            "favourite": race["favourite"],
            "field": len(names),
        }
        with ThreadPoolExecutor(max_workers=max(1, len(active))) as inner:
            futures = {k: inner.submit(fn, brief) for k, fn in active}
            for ai_name, fut in futures.items():
                try:
                    text = fut.result()
                except Exception as e:
                    text = f"ERROR: {e}"
                verdict = ai_consensus.extract_verdict(text)
                rec[f"{ai_name}_pick"] = pick_from(text, names)
                rec[f"{ai_name}_confidence"] = (verdict or {}).get("confidence")
                rec[f"{ai_name}_win_prob"] = (verdict or {}).get("win_prob")
                rec[f"{ai_name}_no_bet"] = bool((verdict or {}).get("no_bet"))
                rec[f"{ai_name}_agrees"] = (verdict or {}).get("agrees_with_model")
                rec[f"{ai_name}_error"] = text.startswith("ERROR")
        with write_lock:
            with open(RESULTS, "a") as f:
                f.write(json.dumps(rec) + "\n")
            progress["n"] += 1
            print(f"[{dt.datetime.now():%H:%M:%S}] {progress['n']}/{len(races)} "
                  f"{race['race_id']}", flush=True)

    with ThreadPoolExecutor(max_workers=args.race_workers) as outer:
        list(outer.map(analyze, races))

    summarize()


if __name__ == "__main__":
    main()
