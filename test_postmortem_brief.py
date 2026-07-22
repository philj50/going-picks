"""Tests for the enriched post-mortem brief."""
import ai_horse_postmortem as pm


RACE = {"course": "Bath", "time": "2:31", "name": "Test Hcp",
        "distance_f": 5.5, "going": "Firm", "race_class": "Class 5",
        "runners": [{"name": "Safari Dream", "odds_decimal": 3.8},
                    {"name": "Grey Horizon", "odds_decimal": 5.0}]}


def _no_ctx(*a, **k):
    return {"lines": [], "pick_pos": None, "pick_beaten": None,
            "pick_sp": None, "winner_sp": None, "field": None}


def test_no_bet_prior_is_judged(monkeypatch):
    monkeypatch.setattr(pm, "_result_context", _no_ctx)
    prior = {"pick": "NO BET", "no_bet": True, "win_prob": 0.22,
             "key_risk": "", "missing_factors": []}
    brief = pm._build_brief("2026-07-15", RACE, "Safari Dream", "Grey Horizon",
                            None, "groq", prior)
    assert "NO BET" in brief
    assert "abstention" in brief
    assert "0.22" in brief


def test_pick_prior_asks_calibration(monkeypatch):
    monkeypatch.setattr(pm, "_result_context", _no_ctx)
    prior = {"pick": "Safari Dream", "no_bet": False, "win_prob": 0.45,
             "key_risk": "draw", "missing_factors": ["going"]}
    brief = pm._build_brief("2026-07-15", RACE, "Safari Dream", "Grey Horizon",
                            None, "groq", prior)
    assert "win_prob 0.45" in brief
    assert "too high, too low, or fair" in brief
    assert "LOST — winner was Grey Horizon" in brief


def test_factor_word_instruction_present(monkeypatch):
    monkeypatch.setattr(pm, "_result_context", _no_ctx)
    brief = pm._build_brief("2026-07-15", RACE, "Safari Dream", "Grey Horizon",
                            None, "groq", {})
    for w in ("draw", "ground", "market", "pace"):
        assert w in brief
    assert "improve_next_time" in brief


def test_result_facts_rendered(monkeypatch):
    def ctx(*a, **k):
        return {"lines": ["  1. Grey Horizon dr2 SP 5.0"], "pick_pos": "6",
                "pick_beaten": 4.5, "pick_sp": 4.2, "winner_sp": 5.0, "field": 10}
    monkeypatch.setattr(pm, "_result_context", ctx)
    brief = pm._build_brief("2026-07-15", RACE, "Safari Dream", "Grey Horizon",
                            None, "groq", {"pick": "Safari Dream", "win_prob": 0.3})
    assert "finished 6/10" in brief
    assert "beaten 4.5L" in brief
    assert "First home:" in brief
    assert "SP 5.0" in brief
