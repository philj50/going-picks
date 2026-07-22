"""Tests for quips.py rule firing, formatting, and day-stable selection."""
import quips


def _sig(**kw):
    base = {"horse": "Alpha", "odds": 5.0, "jockey": "J Doe", "trainer": "T Smith",
            "age": 5, "lr": 20, "form3": "231", "draw": 3, "course": "Epsom",
            "decision_prob": None, "market_prob": 0.2,
            "runs": 10, "wins": 2, "cd": (0, 1), "c": (1, 3), "g": (0, 2),
            "t14": (2, 12), "j14": (1, 10)}
    base.update(kw)
    return base


def test_on_fire_beats_everything():
    q = quips.quip_for_pick(_sig(form3="111", odds=1.5), "2026-07-17")
    assert "smoke alarm" in q or "on fire" in q


def test_cold_jockey_fires_and_formats():
    q = quips.quip_for_pick(_sig(j14=(0, 27)), "2026-07-17")
    assert "J Doe" in q and "27" in q


def test_hot_and_cold_yard():
    assert "T Smith" in quips.quip_for_pick(_sig(t14=(5, 16)), "2026-07-17") or \
        "red-hot" in quips.quip_for_pick(_sig(t14=(5, 16)), "2026-07-17")
    assert "0 from 18" in quips.quip_for_pick(_sig(t14=(0, 18)), "2026-07-17") or \
        "unplug" in quips.quip_for_pick(_sig(t14=(0, 18)), "2026-07-17")


def test_debut_and_maiden_and_veteran():
    assert quips.quip_for_pick(_sig(runs=0), "2026-07-17") is not None
    assert "Bless" in quips.quip_for_pick(_sig(runs=22, wins=0), "2026-07-17") or \
        "zero wins" in quips.quip_for_pick(_sig(runs=22, wins=0), "2026-07-17")
    assert "12" in quips.quip_for_pick(_sig(age=12), "2026-07-17")


def test_no_rule_no_quip():
    assert quips.quip_for_pick(_sig(), "2026-07-17") is None


def test_stable_within_day_varies_across_days():
    s = _sig(form3="111")
    a1 = quips.quip_for_pick(s, "2026-07-17")
    a2 = quips.quip_for_pick(s, "2026-07-17")
    assert a1 == a2
    days = {quips.quip_for_pick(s, f"2026-07-{d:02d}") for d in range(1, 20)}
    assert len(days) > 1


def test_intel_cd_and_gap_and_memory():
    s = _sig(cd=(2, 3), decision_prob=0.31, market_prob=0.18,
             pm={"date": "2026-07-12", "result": "lost", "note": "overrated the speed figure"})
    lines = quips.intel_for_pick(s, limit=4)
    joined = " ".join(lines)
    assert "2 of 3" in joined
    assert "31%" in joined and "18%" in joined
    assert "overrated the speed figure" in joined
