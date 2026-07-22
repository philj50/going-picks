"""
race_stats.py — Per-runner stats pack computed from the historical DB.

Gives the AI voices the 750k-record database in digest form: course/distance/
going records for the horse, strike rates for trainer, jockey and the combo.

Everything is computed strictly BEFORE a cutoff date, so the same code is
leak-free for backtests (cutoff = race date) and current for daily runs
(cutoff = tomorrow).
"""
from __future__ import annotations

import re
import sqlite3

DB_PATH = "/mnt/nvme/going/db/going.db"


def norm_horse(name: str) -> str:
    """Match the ingest's horse_norm: strip country suffix, lowercase, alnum only."""
    name = re.sub(r"\s*\([A-Z]{2,3}\)\s*$", "", name or "")
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _rec(con, sql, params) -> tuple[int, int]:
    row = con.execute(sql, params).fetchone()
    return (row[0] or 0, row[1] or 0)


def _fmt(runs: int, wins: int) -> str | None:
    if not runs:
        return None
    return f"{wins}/{runs} ({round(100 * wins / runs)}%)"


def runner_stats(con, horse: str, trainer: str, jockey: str,
                 course: str, distance_f, going: str, before: str) -> dict:
    """Formatted stat strings for one runner, all computed pre-`before` date."""
    hn = norm_horse(horse)
    win = "SUM(CASE WHEN CAST(r.finish_pos AS INT)=1 THEN 1 ELSE 0 END)"
    out = {}

    runs, wins = _rec(con,
        f"SELECT COUNT(*), {win} FROM runners r JOIN races rc ON rc.race_id=r.race_id "
        "WHERE r.horse_norm=? AND rc.date<? AND LOWER(rc.course)=?",
        (hn, before, (course or "").lower()))
    out["course"] = _fmt(runs, wins)

    if distance_f:
        runs, wins = _rec(con,
            f"SELECT COUNT(*), {win} FROM runners r JOIN races rc ON rc.race_id=r.race_id "
            "WHERE r.horse_norm=? AND rc.date<? AND ABS(rc.distance_f - ?) < 0.75",
            (hn, before, float(distance_f)))
        out["distance"] = _fmt(runs, wins)

    if going:
        runs, wins = _rec(con,
            f"SELECT COUNT(*), {win} FROM runners r JOIN races rc ON rc.race_id=r.race_id "
            "WHERE r.horse_norm=? AND rc.date<? AND LOWER(rc.going)=?",
            (hn, before, going.lower()))
        out["going"] = _fmt(runs, wins)

    if trainer and trainer != "?":
        runs, wins = _rec(con,
            f"SELECT COUNT(*), {win} FROM runners r JOIN races rc ON rc.race_id=r.race_id "
            "WHERE r.trainer=? AND rc.date<? AND rc.date>=date(?, '-60 days')",
            (trainer, before, before))
        out["trainer_60d"] = _fmt(runs, wins)

    if jockey and jockey != "?":
        runs, wins = _rec(con,
            f"SELECT COUNT(*), {win} FROM runners r JOIN races rc ON rc.race_id=r.race_id "
            "WHERE r.jockey=? AND rc.date<? AND rc.date>=date(?, '-60 days')",
            (jockey, before, before))
        out["jockey_60d"] = _fmt(runs, wins)

    if trainer and jockey and trainer != "?" and jockey != "?":
        runs, wins = _rec(con,
            f"SELECT COUNT(*), {win} FROM runners r JOIN races rc ON rc.race_id=r.race_id "
            "WHERE r.trainer=? AND r.jockey=? AND rc.date<? AND rc.date>=date(?, '-365 days')",
            (trainer, jockey, before, before))
        out["combo_1y"] = _fmt(runs, wins)

    # Most recent known ratings for the horse
    row = con.execute(
        "SELECT r.official_rating, r.rpr, r.top_speed FROM runners r "
        "JOIN races rc ON rc.race_id=r.race_id "
        "WHERE r.horse_norm=? AND rc.date<? AND (r.official_rating IS NOT NULL OR r.rpr IS NOT NULL) "
        "ORDER BY rc.date DESC LIMIT 1", (hn, before)).fetchone()
    if row:
        parts = [f"OR {row[0]}" if row[0] else None,
                 f"RPR {row[1]}" if row[1] else None,
                 f"TS {row[2]}" if row[2] else None]
        out["ratings"] = " | ".join(p for p in parts if p) or None

    return out


def stats_lines(stats: dict) -> list[str]:
    """Render a stats dict as brief-ready indented lines."""
    lines = []
    horse_bits = [f"{k}: {stats[k]}" for k in ("course", "distance", "going") if stats.get(k)]
    if horse_bits:
        lines.append("  Record — " + ", ".join(horse_bits))
    conn_bits = []
    if stats.get("trainer_60d"):
        conn_bits.append(f"trainer 60d: {stats['trainer_60d']}")
    if stats.get("jockey_60d"):
        conn_bits.append(f"jockey 60d: {stats['jockey_60d']}")
    if stats.get("combo_1y"):
        conn_bits.append(f"combo 1y: {stats['combo_1y']}")
    if conn_bits:
        lines.append("  Connections — " + ", ".join(conn_bits))
    if stats.get("ratings"):
        lines.append(f"  Last ratings: {stats['ratings']}")
    return lines
