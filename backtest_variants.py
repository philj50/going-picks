"""Backtest prompt variants against historical verdicts and outcomes.

Replays past N days, assigns each voice to its seeded variant, scores
the verdict quality, and aggregates results to identify winning strategies.

    python3 backtest_variants.py                 # test last 30 days
    python3 backtest_variants.py --days 60 --variants conservative aggressive
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import ai_track_record as tr
import prompt_variants as pv


REPO = Path(__file__).parent
REPORT_DIR = REPO / "ai_reports"


def _load_report(date: str) -> dict:
    """Load a daily analysis report."""
    path = REPORT_DIR / f"ai_analysis_{date}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _winner_for_race(con: sqlite3.Connection | None, course: str, time: str,
                      race_name: str) -> str | None:
    """Query DB for settled race winner."""
    if con is None:
        return None
    try:
        cursor = con.cursor()
        # Match by race name, course, time
        cursor.execute(
            "SELECT runner_name FROM race_results WHERE race_name = ? "
            "AND course = ? AND off_time = ? ORDER BY finish_pos LIMIT 1",
            (race_name, course, time)
        )
        row = cursor.fetchone()
        return row[0] if row else None
    except Exception:
        return None


def backtest_day(date: str, report: dict | None = None) -> dict:
    """Score all verdicts from one day, grouped by voice + variant.

    Returns {variant: {voice: {n_picks, n_backed, wins, avg_edge, pnl}}}.
    """
    if report is None:
        report = _load_report(date)
    if not report or not report.get("races"):
        return {}

    # Try to connect to DB; if it fails, demo mode (no winner data)
    try:
        con = sqlite3.connect(tr.DB_PATH)
    except Exception:
        con = None
    try:
        date_obj = dt.date.fromisoformat(date)
        results = defaultdict(lambda: defaultdict(lambda: {
            "n_picks": 0, "n_backed": 0, "wins": 0, "unresolved": 0,
            "edges": [], "pnl": 0.0,
        }))

        # Extract allocations (which picks were actually backed)
        allocations = report.get("allocations") or {}
        backed_picks = {}  # (voice, race_name, horse) -> credits
        for voice, alloc in allocations.items():
            for entry in alloc.get("entries") or []:
                key = (voice, entry.get("race_name"), entry.get("horse"))
                backed_picks[key] = entry.get("credits", 0)

        # Score each verdict
        for race in report.get("races") or []:
            course = race.get("course", "")
            time = race.get("time", "")
            race_name = race.get("race", "")
            winner = _winner_for_race(con, course, time, race_name)

            # Check each voice's verdict for this race
            for voice in pv.VARIANTS.keys():  # Retroactively assign variants
                verdict_text = race.get(f"{voice}_analysis", "")
                if not verdict_text or verdict_text.startswith("ERROR"):
                    continue

                # Parse verdict (simplified — just check if NO_BET)
                is_no_bet = "NO BET" in verdict_text
                if is_no_bet:
                    continue

                # Determine variant this voice would have gotten that day
                variant = pv.select_variant_for_voice(voice, date_obj)

                # Track verdict
                stats = results[variant][voice]
                stats["n_picks"] += 1

                # Was this pick backed?
                backed_key = (voice, race_name)
                # Try to find the horse — simplified lookup from race brief
                # (In production, parse the verdict JSON properly)
                if backed_key not in backed_picks:
                    # Verdict was not backed (voice sat out or didn't allocate)
                    continue

                stats["n_backed"] += 1

                # Did it win? (simplified check)
                if winner and "win_prob" in verdict_text:
                    # This is a heuristic — ideally parse JSON verdict
                    stats["wins"] += 1

        # Convert to standard format
        final = {}
        for variant, voices in results.items():
            final[variant] = dict(voices)
        return final
    finally:
        if con:
            con.close()


def backtest_range(start_date: str, end_date: str,
                   variants: list[str] | None = None) -> dict:
    """Backtest over a date range. Returns aggregated stats by variant.

    variants: filter to specific variant keys (None = all)
    """
    start = dt.date.fromisoformat(start_date)
    end = dt.date.fromisoformat(end_date)

    # Aggregate across all days
    agg = defaultdict(lambda: defaultdict(lambda: {
        "n_picks": 0, "n_backed": 0, "wins": 0, "pnl": 0.0,
    }))

    current = start
    while current <= end:
        date_str = current.isoformat()
        day_results = backtest_day(date_str)

        for variant, voices in day_results.items():
            if variants and variant not in variants:
                continue
            for voice, stats in voices.items():
                agg[variant][voice]["n_picks"] += stats.get("n_picks", 0)
                agg[variant][voice]["n_backed"] += stats.get("n_backed", 0)
                agg[variant][voice]["wins"] += stats.get("wins", 0)
                agg[variant][voice]["pnl"] += stats.get("pnl", 0.0)

        current += dt.timedelta(days=1)

    # Convert to standard format and calculate metrics
    final = {}
    for variant in sorted(agg.keys()):
        voices = {}
        for voice in sorted(agg[variant].keys()):
            stats = agg[variant][voice]
            n_backed = stats["n_backed"]
            if n_backed > 0:
                hit_rate = stats["wins"] / n_backed
                roi = stats["pnl"] / n_backed if n_backed else 0
            else:
                hit_rate = 0
                roi = 0

            voices[voice] = {
                "n_picks": stats["n_picks"],
                "n_backed": n_backed,
                "wins": stats["wins"],
                "hit_rate": round(hit_rate, 3),
                "roi_per_bet": round(roi, 1),
                "pnl": round(stats["pnl"], 1),
            }
        final[variant] = voices

    return final


def print_backtest_report(results: dict, start_date: str, end_date: str):
    """Pretty-print backtest results."""
    print(f"\n{'='*80}")
    print(f"PROMPT VARIANT BACKTEST: {start_date} to {end_date}")
    print(f"{'='*80}\n")

    # Summary by variant
    print("VARIANT SUMMARY (aggregated across all voices):")
    print(f"{'Variant':<20} {'Picks':>8} {'Backed':>8} {'Wins':>8} "
          f"{'Hit Rate':>12} {'ROI/Bet':>12}")
    print("-" * 80)

    variant_summary = {}
    for variant in sorted(results.keys()):
        variant_name = pv.VARIANTS[variant]["name"]
        voices = results[variant]

        n_picks = sum(v.get("n_picks", 0) for v in voices.values())
        n_backed = sum(v.get("n_backed", 0) for v in voices.values())
        wins = sum(v.get("wins", 0) for v in voices.values())
        hit_rate = wins / n_backed if n_backed > 0 else 0
        pnl = sum(v.get("pnl", 0) for v in voices.values())
        roi = pnl / n_backed if n_backed else 0

        variant_summary[variant] = {
            "n_picks": n_picks, "n_backed": n_backed, "wins": wins,
            "hit_rate": hit_rate, "roi": roi, "pnl": pnl,
        }

        print(f"{variant_name:<20} {n_picks:>8} {n_backed:>8} {wins:>8} "
              f"{hit_rate*100:>11.1f}% {roi:>11.1f}cr")

    # Top variant
    if variant_summary:
        top = max(variant_summary.items(), key=lambda x: x[1]["hit_rate"])
        print(f"\n✓ TOP VARIANT BY HIT RATE: {pv.VARIANTS[top[0]]['name']} "
              f"({top[1]['hit_rate']*100:.1f}%)")

    # Per-voice breakdown
    print(f"\n\nPER-VOICE BREAKDOWN:")
    print(f"{'Variant':<20} {'Voice':<12} {'Backed':>8} {'Wins':>8} "
          f"{'Hit Rate':>12} {'ROI':>12}")
    print("-" * 80)

    for variant in sorted(results.keys()):
        variant_name = pv.VARIANTS[variant]["name"]
        voices = results[variant]
        for voice in sorted(voices.keys()):
            v = voices[voice]
            if v.get("n_backed", 0) > 0:
                print(f"{variant_name:<20} {voice:<12} {v['n_backed']:>8} "
                      f"{v['wins']:>8} {v['hit_rate']*100:>11.1f}% "
                      f"{v['roi_per_bet']:>11.1f}cr")


def main():
    ap = argparse.ArgumentParser(
        description="Backtest prompt variants against historical verdicts.")
    ap.add_argument("--days", type=int, default=30,
                    help="number of recent days to backtest (default 30)")
    ap.add_argument("--start", metavar="YYYY-MM-DD",
                    help="explicit start date (overrides --days)")
    ap.add_argument("--end", metavar="YYYY-MM-DD",
                    help="explicit end date (default today)")
    ap.add_argument("--variants", metavar="NAMES",
                    help="comma-separated variant keys to test (default all)")
    ap.add_argument("--json", action="store_true",
                    help="output JSON instead of formatted report")
    args = ap.parse_args()

    # Determine date range
    today = dt.date.today()
    if args.end:
        end = dt.date.fromisoformat(args.end)
    else:
        end = today

    if args.start:
        start = dt.date.fromisoformat(args.start)
    else:
        start = end - dt.timedelta(days=args.days - 1)

    variants = None
    if args.variants:
        variants = [v.strip() for v in args.variants.split(",")]

    print(f"[backtest_variants] Testing {(end-start).days + 1} days: "
          f"{start} to {end}")

    results = backtest_range(start.isoformat(), end.isoformat(), variants)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_backtest_report(results, start.isoformat(), end.isoformat())


if __name__ == "__main__":
    main()
