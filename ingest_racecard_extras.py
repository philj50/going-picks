"""
ingest_racecard_extras.py — persist rpscrape PRE-RACE racecard fields into the
corpus DB so history accumulates for briefs/backtests.

Adds nullable columns to `runners` (no existing reader breaks) and upserts
values from tools/rpscrape/racecards/YYYY-MM-DD.json by matching
(race date+course+off_time, horse_norm). The *_pre columns are the morning's
published figures — unlike the same-row rpr/top_speed, they are safe to use
in leak-free pre-race features.

Run in the evening cron (after results sync creates the day's races/runners):
    python3 ingest_racecard_extras.py --date today
Safe to re-run; only fills the matched rows.
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
RP_DIR = REPO / "tools" / "rpscrape" / "racecards"

NEW_COLS = [
    ("age", "INTEGER"), ("sex_code", "TEXT"),
    ("sire", "TEXT"), ("dam", "TEXT"), ("damsire", "TEXT"),
    ("last_run_days", "INTEGER"), ("trainer_rtf", "REAL"),
    ("headgear_first", "INTEGER"),
    ("ofr_pre", "INTEGER"), ("rpr_pre", "INTEGER"), ("ts_pre", "INTEGER"),
]


def norm_horse(name: str) -> str:
    name = re.sub(r"\s*\([A-Z]{2,3}\)\s*$", "", name or "")
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _norm_course(c: str) -> str:
    return re.sub(r"[^a-z]", "", re.sub(r"\(.*?\)", "", (c or "").lower()))


def _time24(t: str) -> str:
    m = re.match(r"^(\d{1,2}):(\d{2})", str(t or "").strip())
    if not m:
        return str(t or "")
    h, mi = int(m.group(1)), m.group(2)
    if h <= 10:
        h += 12
    return f"{h:02d}:{mi}"


def ensure_columns(con) -> list[str]:
    have = {r[1] for r in con.execute("PRAGMA table_info(runners)")}
    added = []
    for col, typ in NEW_COLS:
        if col not in have:
            con.execute(f"ALTER TABLE runners ADD COLUMN {col} {typ}")
            added.append(col)
    con.commit()
    return added


def _to_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def ingest(date: str) -> tuple[int, int]:
    path = RP_DIR / f"{date}.json"
    if not path.exists():
        print(f"No rpscrape racecard for {date} ({path})")
        return (0, 0)
    data = json.loads(path.read_text(encoding="utf-8"))

    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    ensure_columns(con)

    db_races = con.execute("SELECT race_id, course, off_time FROM races WHERE date=?",
                           (date,)).fetchall()
    by_ct = {(_norm_course(r["course"]), _time24(r["off_time"])): r["race_id"]
             for r in db_races}

    matched = updated = 0
    for region in (data or {}).values():
        for course_races in region.values():
            for off, race in course_races.items():
                ck = _norm_course(race.get("course") or "")
                tk = _time24(race.get("off_time") or off)
                race_id = by_ct.get((ck, tk))
                if not race_id:
                    continue
                matched += 1
                for rn in race.get("runners") or []:
                    vals = {
                        "age": _to_int(rn.get("age")),
                        "sex_code": rn.get("sex_code") or None,
                        "sire": rn.get("sire") or None,
                        "dam": rn.get("dam") or None,
                        "damsire": rn.get("damsire") or None,
                        "last_run_days": _to_int(rn.get("last_run")),
                        "trainer_rtf": rn.get("trainer_rtf"),
                        "headgear_first": 1 if rn.get("headgear_first") else 0,
                        "ofr_pre": _to_int(rn.get("ofr")),
                        "rpr_pre": _to_int(rn.get("rpr")),
                        "ts_pre": _to_int(rn.get("ts")),
                    }
                    sets = ", ".join(f"{k}=?" for k in vals)
                    cur = con.execute(
                        f"UPDATE runners SET {sets} WHERE race_id=? AND horse_norm=?",
                        (*vals.values(), race_id, norm_horse(rn.get("name") or "")))
                    updated += cur.rowcount
    con.commit()
    con.close()
    return matched, updated


def main():
    ap = argparse.ArgumentParser(description="Ingest rpscrape racecard extras into runners.")
    ap.add_argument("--date", default="today",
                    help="YYYY-MM-DD, or today/yesterday (default today)")
    args = ap.parse_args()
    date = args.date
    if date == "today":
        date = dt.date.today().isoformat()
    elif date == "yesterday":
        date = (dt.date.today() - dt.timedelta(days=1)).isoformat()
    dt.date.fromisoformat(date)
    matched, updated = ingest(date)
    print(f"{date}: matched {matched} races, updated {updated} runner rows")


if __name__ == "__main__":
    main()
