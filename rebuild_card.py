"""
rebuild_card.py — reconstruct a day card JSON for a PAST date from the corpus
DB (runner lists, conditions, starting prices) plus that day's AI report
(race display names). Lets ai_daily_analysis.py --card/--date re-run a
historical day with fresh voices.

    python3 rebuild_card.py --date 2026-07-15 --out cards_2026-07-15.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sqlite3
from pathlib import Path

REPO = Path(__file__).parent
DB_PATH = "/mnt/nvme/going/db/going.db"


def _time24(t: str) -> str:
    m = re.match(r"^(\d{1,2}):(\d{2})", str(t or "").strip())
    if not m:
        return str(t or "")
    h, mi = int(m.group(1)), m.group(2)
    if h <= 10:
        h += 12
    return f"{h:02d}:{mi}"


def _norm_course(c: str) -> str:
    return re.sub(r"[^a-z]", "", re.sub(r"\(.*?\)", "", (c or "").lower()))


def rebuild(date: str) -> dict:
    report_path = REPO / "ai_reports" / f"ai_analysis_{date}.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row

    db_races = con.execute("SELECT * FROM races WHERE date=?", (date,)).fetchall()
    by_ct = {}
    for r in db_races:
        by_ct[(_norm_course(r["course"]), _time24(r["off_time"]))] = r

    races_out, missing = [], []
    for entry in report.get("races") or []:
        course, tdisp = entry.get("course"), entry.get("time")
        row = by_ct.get((_norm_course(course), _time24(tdisp)))
        if row is None:
            missing.append(f"{course} {tdisp}")
            continue
        runners = []
        for rr in con.execute("SELECT * FROM runners WHERE race_id=?",
                              (row["race_id"],)).fetchall():
            odds = rr["decision_price"] or rr["starting_price"] or rr["bsp_win"]
            rd = {"name": rr["horse"], "trainer": rr["trainer"] or "",
                  "jockey": rr["jockey"] or ""}
            if odds and odds > 1:
                rd["odds_decimal"] = round(float(odds), 2)
            if rr["draw"] is not None:
                rd["draw"] = rr["draw"]
            if rr["weight_lbs"] is not None:
                rd["weight_lbs"] = rr["weight_lbs"]
            if rr["headgear"]:
                rd["headgear"] = rr["headgear"]
            runners.append(rd)
        if not runners:
            missing.append(f"{course} {tdisp} (no runners)")
            continue
        races_out.append({
            "name": entry.get("race"),
            "course": row["course"],
            "time": tdisp,
            "date": date,
            "is_flat": (row["race_type"] or "Flat") == "Flat",
            "is_handicap": bool(row["handicap_flag"]),
            "distance_f": row["distance_f"],
            "going": row["going"],
            "race_class": row["race_class"],
            "off_dt": f"{date}T{_time24(tdisp)}:00",
            "runners": runners,
        })
    con.close()
    if missing:
        print(f"WARNING: {len(missing)} report races not matched in DB: "
              + "; ".join(missing[:6]))
    return {"name": f"rebuilt racecards {date}", "date": date, "races": races_out}


def main():
    ap = argparse.ArgumentParser(description="Rebuild a past day's card from the DB.")
    ap.add_argument("--date", required=True, metavar="YYYY-MM-DD")
    ap.add_argument("--out", default=None, metavar="FILE")
    args = ap.parse_args()
    dt.date.fromisoformat(args.date)  # validate
    card = rebuild(args.date)
    out = args.out or f"cards_{args.date}.json"
    Path(out).write_text(json.dumps(card, indent=2), encoding="utf-8")
    print(f"Wrote {out}: {len(card['races'])} races, "
          f"{sum(len(r['runners']) for r in card['races'])} runners")


if __name__ == "__main__":
    main()
