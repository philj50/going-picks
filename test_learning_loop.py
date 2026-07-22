"""Tests for the learning-loop upgrades: staking review, banded calibration,
contextual lessons, and the lessons A/B switch."""
import ai_daily_analysis as ada
import brief_builder as bb
import staking_review as sr


# --- staking_line aggregation -----------------------------------------------

def test_staking_line_aggregates_and_judges(monkeypatch):
    import datetime as dt
    today = dt.date.today()
    d1 = (today - dt.timedelta(days=1)).isoformat()
    d2 = (today - dt.timedelta(days=2)).isoformat()
    monkeypatch.setattr(sr, "_load", lambda: {
        d2: {"groq": {"n_bets": 3, "wins": 1, "unresolved": 0, "pnl": 12.0,
                      "picks": 20, "unbacked": 17, "unbacked_wins": 3,
                      "style": "concentrated"}},
        d1: {"groq": {"n_bets": 0, "wins": 0, "unresolved": 0, "pnl": 0,
                      "picks": 15, "unbacked": 15, "unbacked_wins": 1,
                      "style": "sat out"}},
    })
    line = sr.staking_line("groq")
    assert "1 betting day(s): 1/3 staked picks won" in line
    assert "+12cr" in line
    assert "1 sat-out day(s)" in line and "1/15" in line
    assert "usual style: concentrated" in line
    assert sr.staking_line("cursor") == ""      # no data → no line


# --- banded calibration ------------------------------------------------------

def test_bands_text_gates_on_n_and_buckets():
    recs = ([(0.5, True)] * 4 + [(0.45, False)] * 8      # ≥0.40: 4/12
            + [(0.3, True)] * 3 + [(0.28, False)] * 9    # 0.25–0.40: 3/12
            + [(0.1, False)] * 5)                        # <0.25: n=5 → gated out
    txt = ada._bands_text(recs)
    assert "≥0.40 won 33% (n=12)" in txt
    assert "0.25–0.40 won 25% (n=12)" in txt
    assert "<0.25" not in txt


def test_bands_empty_when_thin():
    assert ada._bands_text([(0.5, True)] * 5) == ""


# --- contextual lessons ------------------------------------------------------

THEMES = [
    {"theme": "draw position", "cites": 100, "maps_to_feature": "draw_bias"},
    {"theme": "ground", "cites": 90, "maps_to_feature": "going_win_rate"},
    {"theme": "well handicapped", "cites": 80, "maps_to_feature": "below_winning_mark"},
    {"theme": "jockey form", "cites": 70, "maps_to_feature": "jockey_in_form"},
    {"theme": "long layoff", "cites": 60, "maps_to_feature": "layoff_with_top_jockey"},
]


def _race(**kw):
    base = {"is_flat": True, "is_handicap": True, "going": "Soft"}
    base.update(kw)
    return base


def test_contextual_lessons_filters_by_race(monkeypatch):
    monkeypatch.setattr(bb, "_load_themes", lambda: THEMES)
    flat = bb.contextual_lessons(_race(), max_lr=30)
    assert "draw position" in flat and "ground" in flat
    assert "long layoff" not in flat            # nobody fresh

    jumps = bb.contextual_lessons(_race(is_flat=False), max_lr=30)
    assert "draw position" not in jumps         # draw noise over jumps
    assert "jockey form" in jumps

    non_hcp_good = bb.contextual_lessons(
        _race(is_handicap=False, going="Good"), max_lr=200)
    assert "well handicapped" not in non_hcp_good
    assert "ground" not in non_hcp_good         # plain Good going
    assert "long layoff" in non_hcp_good        # a 200-day absentee runs


def test_no_lessons_env_switch(monkeypatch):
    monkeypatch.setattr(bb, "_load_themes", lambda: THEMES)
    monkeypatch.setenv("GOING_AI_NO_LESSONS", "1")
    assert bb.contextual_lessons(_race()) == ""
    assert ada._prompt_lessons_block() == ""


def test_build_prompt_dedupes_lessons(monkeypatch):
    monkeypatch.setattr(ada, "_prompt_lessons_block",
                        lambda: "\n\nRecent lessons from settled post-mortems\n- x\n")
    monkeypatch.setattr(ada, "_calibration_line", lambda voice: "")
    with_marker = ada.build_prompt(
        "BRIEF\n\nLessons from settled post-mortems relevant to THIS race:\n- y", "groq")
    assert with_marker.count("essons from settled post-mortems") == 1
    without_marker = ada.build_prompt("BRIEF ONLY", "groq")
    assert "Recent lessons from settled post-mortems" in without_marker
