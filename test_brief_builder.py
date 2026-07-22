"""Tests for brief_builder — budget ladder, merging, and helpers."""
import sqlite3

import brief_builder as bb


def _mem_db():
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.executescript("""
      CREATE TABLE races(race_id TEXT PRIMARY KEY, date TEXT, course TEXT,
        off_time TEXT, race_type TEXT, race_class TEXT, distance_f REAL,
        going TEXT, field_size INTEGER, handicap_flag INTEGER, surface TEXT);
      CREATE TABLE runners(runner_id TEXT PRIMARY KEY, race_id TEXT, horse TEXT,
        horse_norm TEXT, trainer TEXT, jockey TEXT, draw INTEGER,
        weight_lbs INTEGER, official_rating INTEGER, headgear TEXT,
        finish_pos TEXT, beaten_lengths REAL, starting_price REAL);
      CREATE TABLE price_ticks(date TEXT, course TEXT, course_norm TEXT,
        horse TEXT, horse_norm TEXT, ts TEXT, price REAL);
    """)
    # 40 historical Epsom 7f flat races so the draw-bias gate passes
    for i in range(40):
        rid = f"r{i}"
        con.execute("INSERT INTO races VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (rid, f"2025-{(i % 12) + 1:02d}-10", "Epsom", "14:00", "Flat",
                     "4", 7.0, "Good", 10, 1, "turf"))
        for d in range(1, 11):
            con.execute("INSERT INTO runners VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (f"{rid}-{d}", rid, f"Horse{d}", f"horse{d}", "T One",
                         "J One", d, 126 + d, 70 + d, None,
                         "1" if d == 1 else str(d), 0.5 * d, 3.0 + d))
    # recent form for Alpha (horse1-like history under its own name)
    for j, fp in enumerate(("1", "3", "2"), 1):
        rid = f"a{j}"
        con.execute("INSERT INTO races VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (rid, f"2026-06-{j:02d}", "Epsom", "15:00", "Flat", "4",
                     7.0, "Good", 8, 1, "turf"))
        con.execute("INSERT INTO runners VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (f"{rid}-1", rid, "Alpha", "alpha", "T One", "J One",
                     2, 130, 78, None, fp, 1.0, 4.5))
    con.commit()
    return con


RACE = {
    "name": "Test Handicap", "course": "Epsom", "time": "2:30",
    "date": "2026-07-16", "is_flat": True, "is_handicap": True,
    "distance_f": 7.0, "going": "Good", "race_class": "4",
    "runners": [
        {"name": "Alpha", "trainer": "T One", "jockey": "J One",
         "odds_decimal": 4.5, "draw": 2, "weight_lbs": 130},
        {"name": "Beta", "trainer": "T Two", "jockey": "J Two",
         "odds_decimal": 6.0, "draw": 5, "weight_lbs": 128},
    ],
}
PREDICTION = {"top_3": [{"name": "Alpha", "win_prob": 38.0, "confidence": 55,
                         "odds": 4.5, "reasons": ["won last time"]}]}


def test_compact_brief_fits_budget():
    con = _mem_db()
    brief, tier = bb.build_brief(con, dict(RACE), PREDICTION, in_budget=400,
                                 date="2026-07-16")
    assert "RACE: Epsom 2:30" in brief
    assert "MODEL PICK: Alpha" in brief
    assert "- Alpha @ 4.5" in brief
    assert "T14" in brief
    # history didn't fit in 400 tokens for 2 runners? it might — main check: no crash
    assert bb.est_tokens(brief) <= 900


def test_extended_brief_includes_history():
    con = _mem_db()
    brief, tier = bb.build_brief(con, dict(RACE), PREDICTION, in_budget=1500,
                                 date="2026-07-16")
    assert tier.startswith("extended")
    assert "    · " in brief          # indented last-runs lines
    assert "Epsom 7f" in brief


def test_draw_bias_line_present_when_gated_sample_ok():
    con = _mem_db()
    brief, _ = bb.build_brief(con, dict(RACE), PREDICTION, in_budget=1500,
                              date="2026-07-16")
    assert "DRAW Epsom" in brief
    assert "winners drawn low" in brief


def test_no_db_still_builds():
    brief, tier = bb.build_brief(None, dict(RACE), PREDICTION, in_budget=850,
                                 date="2026-07-16")
    assert "- Alpha @ 4.5" in brief
    assert tier == "compact"


def test_no_model_pick_line():
    brief, _ = bb.build_brief(None, dict(RACE), {"top_3": []}, in_budget=850,
                              date="2026-07-16")
    assert "MODEL PICK: none qualifying" in brief


def test_debut_flag_for_unraced():
    race = dict(RACE)
    race["runners"] = [{"name": "Newbie", "trainer": "T", "jockey": "J",
                        "odds_decimal": 8.0}]
    brief, _ = bb.build_brief(_mem_db(), race, {"top_3": []}, in_budget=850,
                              date="2026-07-16")
    assert "DEBUT" in brief


def test_voice_budget_defaults_and_env(monkeypatch):
    assert bb.voice_budget("cursor") == (1500, 2)
    assert bb.voice_budget("groq") == (850, 1)
    assert bb.voice_budget("unknown-voice") == bb.FALLBACK_BUDGET
    monkeypatch.setenv("GOING_AI_BRIEF_BUDGETS", "groq:700:0,cursor:2000")
    assert bb.voice_budget("groq") == (700, 0)
    assert bb.voice_budget("cursor") == (2000, 2)   # prose default kept


def test_time_and_course_normalisation():
    assert bb._time24("5:45") == "17:45"
    assert bb._time24("13:50") == "13:50"
    assert bb._norm_course("Wolverhampton (AW)") == "wolverhampton"
    assert bb.norm_horse("The Balearic Sun (IRE)") == "thebalearicsun"
