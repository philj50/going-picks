"""Tests for the betting-freedom day book: Kelly staking + LLM allocation."""
import json
import pytest

import ai_daily_analysis as ada


@pytest.fixture(autouse=True)
def mock_all_providers(monkeypatch):
    """Mock _filtered_providers to return all test voices."""
    def mock_filtered():
        # Default to a test voice; individual tests can override
        return [("groq", lambda _: '{}', None), ("test", lambda _: '{}', None)]
    monkeypatch.setattr(ada, "_filtered_providers", mock_filtered)


def _card(*runners):
    return {"races": [{"name": "Test Race", "runners":
                       [{"name": n, "odds_decimal": o} for n, o in runners]}]}


def _analyses(voice, verdicts):
    """verdicts: list of (race_name, course, time, verdict_json_or_None)."""
    races = []
    for name, course, time, vd in verdicts:
        row = {"race": name, "course": course, "time": time}
        if vd is not None:
            row[f"{voice}_analysis"] = "VERDICT: " + json.dumps(vd)
        races.append(row)
    return {"races": races}


# --- _kelly_entries ---------------------------------------------------------

def _w(horse, odds, p):
    return {"race": "Epsom 2:30", "race_name": "Test Race",
            "horse": horse, "odds": odds, "win_prob": p}


def test_kelly_no_edge_means_no_bets():
    assert ada._kelly_entries([_w("A", 3.0, 0.30), _w("B", 2.0, 0.45)]) == []


def test_kelly_concentrates_on_towering_edge():
    entries = ada._kelly_entries([_w("A", 4.0, 0.50), _w("B", 3.0, 0.35)])
    by = {e["horse"]: e["credits"] for e in entries}
    assert by["A"] > by["B"] * 2          # 33% Kelly vs 2.5%
    assert sum(by.values()) <= ada.DAILY_CREDITS


def test_kelly_caps_total_at_budget():
    entries = ada._kelly_entries([_w("A", 6.0, 0.60), _w("B", 6.0, 0.60),
                                  _w("C", 6.0, 0.60)])
    assert sum(e["credits"] for e in entries) <= ada.DAILY_CREDITS
    assert all(e["edge"] > 0 for e in entries)


def test_kelly_ignores_missing_prob_or_odds():
    assert ada._kelly_entries([_w("A", 0, 0.5), _w("B", 3.0, None)]) == []


# --- allocate_credits_rule --------------------------------------------------

def test_rule_kelly_stakes_edge_and_records_sitting_out():
    card = _card(("Alpha", 4.0), ("Beta", 2.0))
    analyses = _analyses("groq", [
        ("Test Race", "Epsom", "2:30",
         {"pick": "Alpha", "win_prob": 0.50}),      # edge +1.0 → bet
    ])
    out = ada.allocate_credits_rule(analyses, card)
    assert out["groq"]["entries"][0]["horse"] == "Alpha"
    assert out["groq"]["picks"] == 1
    assert out["groq"]["method"] == "kelly"

    # Same pick, but market already shorter than the voice's probability
    analyses2 = _analyses("groq", [
        ("Test Race", "Epsom", "2:30",
         {"pick": "Alpha", "win_prob": 0.20}),      # edge −0.2 → sit out
    ])
    out2 = ada.allocate_credits_rule(analyses2, card)
    assert out2["groq"]["entries"] == []
    assert out2["groq"]["total_staked"] == 0
    assert out2["groq"]["picks"] == 1


def test_rule_no_bet_verdicts_are_not_picks():
    card = _card(("Alpha", 4.0))
    analyses = _analyses("groq", [
        ("Test Race", "Epsom", "2:30", {"pick": "NO BET", "win_prob": 0.15}),
    ])
    out = ada.allocate_credits_rule(analyses, card)
    assert "groq" not in out          # no stakeable picks at all → not playing


# --- allocate_credits (LLM day book) ---------------------------------------

def _with_canned_reply(monkeypatch, voice, reply):
    monkeypatch.setitem(ada.RAW_CALLERS, voice, lambda prompt: reply)
    # Also mock _filtered_providers to include the test voice
    def mock_filtered():
        return [(voice, lambda _: reply, None)]
    monkeypatch.setattr(ada, "_filtered_providers", mock_filtered)


def test_llm_all_in_on_one_horse(monkeypatch):
    card = _card(("Alpha", 4.0), ("Beta", 2.0))
    analyses = _analyses("groq", [
        ("Test Race", "Epsom", "2:30", {"pick": "Alpha", "win_prob": 0.50}),
    ])
    _with_canned_reply(monkeypatch, "groq",
                       '{"allocations": [{"race": "Epsom 2:30", "horse": "Alpha", "credits": 100}]}')
    out = ada.allocate_credits(analyses, card)
    assert out["groq"]["total_staked"] == 100
    assert len(out["groq"]["entries"]) == 1
    assert out["groq"]["method"] == "self"
    assert "as MANY or as FEW" in out["groq"]["prompt"]


def test_llm_sit_out_is_recorded_not_dropped(monkeypatch):
    card = _card(("Alpha", 4.0))
    analyses = _analyses("groq", [
        ("Test Race", "Epsom", "2:30", {"pick": "Alpha", "win_prob": 0.50}),
    ])
    _with_canned_reply(monkeypatch, "groq", '{"allocations": []}')
    out = ada.allocate_credits(analyses, card)
    assert out["groq"]["entries"] == []
    assert out["groq"]["total_staked"] == 0
    assert out["groq"]["picks"] == 1


def test_llm_garbage_reply_falls_back_to_kelly(monkeypatch):
    card = _card(("Alpha", 4.0))
    analyses = _analyses("groq", [
        ("Test Race", "Epsom", "2:30", {"pick": "Alpha", "win_prob": 0.50}),
    ])
    _with_canned_reply(monkeypatch, "groq", "I would bet everything on Alpha!")
    out = ada.allocate_credits(analyses, card)
    assert out["groq"]["method"] == "kelly-fallback"
    assert out["groq"]["entries"][0]["horse"] == "Alpha"


def test_llm_hallucinated_horse_is_ignored(monkeypatch):
    card = _card(("Alpha", 4.0))
    analyses = _analyses("groq", [
        ("Test Race", "Epsom", "2:30", {"pick": "Alpha", "win_prob": 0.50}),
    ])
    _with_canned_reply(monkeypatch, "groq",
                       '{"allocations": [{"race": "Epsom 2:30", "horse": "Imaginary", "credits": 50}]}')
    out = ada.allocate_credits(analyses, card)
    assert out["groq"]["entries"] == []
