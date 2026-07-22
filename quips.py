"""
quips.py — unique intel lines + cheeky one-liners for race cards.

Everything is rule-based off our own corpus, so the joke is also information:
a quip only fires when the underlying stat is true. Selection is seeded by
(date, horse) so cards are stable within a day but vary day to day.

    intel_for_pick(con, race, runner_name, date_iso, ...)  -> list[str]
    quip_for_pick(signals)                                 -> str | None
    signals_for_pick(con, race, runner_name, date_iso, rp) -> dict

Edit the QUIPS table to tune the humour — each entry is
(rule_key, [template, ...]) and templates format against the signals dict.
"""
from __future__ import annotations

import hashlib
import re


# --------------------------------------------------------------------------
# signal gathering (cheap, indexed queries; everything optional)

def _norm_horse(name: str) -> str:
    name = re.sub(r"\s*\([A-Z]{2,3}\)\s*$", "", name or "")
    return re.sub(r"[^a-z0-9]", "", name.lower())


def signals_for_pick(con, race, runner, date_iso: str, rp_pick: dict | None = None,
                     decision_prob: float | None = None) -> dict:
    """Everything the intel/quip rules key off, from card + rp + corpus."""
    rp_pick = rp_pick or {}
    name = getattr(runner, "name", None) or (runner.get("name") if isinstance(runner, dict) else "?")
    odds = float(getattr(runner, "odds_decimal", None)
                 or (runner.get("odds_decimal") if isinstance(runner, dict) else 0) or 0)
    jockey = ((getattr(runner, "jockey", "") or "").strip()
              or (rp_pick.get("jockey") or "").strip())
    trainer = ((getattr(runner, "trainer", "") or "").strip()
               or (rp_pick.get("trainer") or "").strip())
    s = {
        "horse": name, "odds": odds, "jockey": jockey, "trainer": trainer,
        "age": rp_pick.get("age"), "lr": rp_pick.get("last_run"),
        "form3": str(rp_pick.get("form") or "")[-3:],
        "draw": getattr(runner, "draw", None),
        "course": getattr(race, "course", "") or "",
        "decision_prob": decision_prob,
        "market_prob": (1.0 / odds) if odds > 1 else None,
    }
    if con is None:
        return s
    try:
        import brief_builder
        fs = brief_builder.FeatureStore(con, date_iso)
        hn = _norm_horse(name)
        rec = fs.records(hn, s["course"], getattr(race, "distance_f", None),
                         getattr(race, "going", "") or "")
        s["runs"] = len(fs.hist(hn))
        s["wins"] = sum(1 for r in fs.hist(hn) if str(r["fp"]) == "1")
        s["cd"] = rec.get("CD", (0, 0))
        s["c"] = rec.get("C", (0, 0))
        s["g"] = rec.get("G", (0, 0))
        s["t14"] = fs.trainer14(trainer) if trainer else (0, 0)
        s["j14"] = fs.jockey14(jockey) if jockey else (0, 0)
        if s["lr"] is None and fs.hist(hn):
            try:
                import datetime as dt
                s["lr"] = (dt.date.fromisoformat(date_iso)
                           - dt.date.fromisoformat(fs.hist(hn)[0]["d"])).days
            except Exception:
                pass
        s["draw_line"] = fs.draw_bias_line(
            s["course"], getattr(race, "distance_f", None),
            bool(getattr(race, "is_handicap", False))) if getattr(race, "is_flat", True) else None
        if not s["form3"] and fs.hist(hn):
            def _fp(x):
                x = str(x)
                if x.isdigit():
                    return x if int(x) <= 9 else "0"
                return (x[:1] or "-").upper()
            s["form3"] = "".join(_fp(r["fp"]) for r in reversed(fs.hist(hn)[:3]))
    except Exception:
        pass
    # Post-mortem memory: were we burned on this horse or jockey recently?
    try:
        row = con.execute(
            """SELECT date, result, what_we_missed, improve_next_time
               FROM horse_postmortems
               WHERE (horse_norm = ? OR (jockey = ? AND jockey != ''))
                 AND date < ? AND date >= date(?, '-28 days')
                 AND COALESCE(what_we_missed, improve_next_time) IS NOT NULL
               ORDER BY date DESC LIMIT 1""",
            (_norm_horse(name), jockey, date_iso, date_iso)).fetchone()
        if row:
            s["pm"] = {"date": row[0], "result": row[1],
                       "note": (row[2] or row[3] or "").strip()}
    except Exception:
        pass
    return s


# --------------------------------------------------------------------------
# render-friendly cached lookup: base signals per (date, race, horse) are
# stable all day, so repeat page loads cost no queries. Odds/model prob are
# refreshed per call since they move intraday.

_SIG_CACHE: dict = {}


def cached_signals(race, runner, date_iso: str, rp_pick: dict | None = None,
                   decision_prob: float | None = None) -> dict:
    key = (date_iso, getattr(race, "course", "") or "",
           getattr(race, "time", "") or "", getattr(runner, "name", "") or "")
    base = _SIG_CACHE.get(key)
    if base is None:
        con = None
        try:
            import sqlite3
            import bets_store
            con = bets_store.connect()
            con.row_factory = sqlite3.Row
        except Exception:
            con = None
        base = signals_for_pick(con, race, runner, date_iso, rp_pick,
                                decision_prob=decision_prob)
        try:
            if con is not None:
                con.close()
        except Exception:
            pass
        if len(_SIG_CACHE) > 4000:
            _SIG_CACHE.clear()
        _SIG_CACHE[key] = dict(base)
    s = dict(base)
    odds = float(getattr(runner, "odds_decimal", None) or 0)
    s["odds"] = odds
    s["market_prob"] = (1.0 / odds) if odds > 1 else None
    s["decision_prob"] = decision_prob
    return s


# --------------------------------------------------------------------------
# intel: short factual lines nobody else has

def intel_for_pick(s: dict, limit: int = 2) -> list[str]:
    out = []
    cd_w, cd_n = s.get("cd") or (0, 0)
    c_w, c_n = s.get("c") or (0, 0)
    if cd_w:
        out.append(f"Won {cd_w} of {cd_n} at this course & distance.")
    elif c_w:
        out.append(f"Course winner — {c_w} from {c_n} round {s.get('course')}.")
    t_w, t_n = s.get("t14") or (0, 0)
    j_w, j_n = s.get("j14") or (0, 0)
    if t_n >= 8 and j_n >= 8:
        out.append(f"Fortnight form — yard {t_w}/{t_n}, jockey {j_w}/{j_n}.")
    dp, mp = s.get("decision_prob"), s.get("market_prob")
    if dp and mp and dp - mp >= 0.08:
        out.append(f"Model {dp * 100:.0f}% vs market {mp * 100:.0f}% — "
                   "our biggest-conviction gap type.")
    pm = s.get("pm")
    if pm and pm.get("note"):
        note = pm["note"][:90]
        out.append(f"Memory ({pm['date'][5:]}, {pm.get('result') or '?'}): "
                   f"“{note}”")
    if s.get("draw_line"):
        out.append(s["draw_line"])
    return out[:limit]


# --------------------------------------------------------------------------
# quips: (priority-ordered rule, templates). First firing rule wins;
# template chosen by a stable per-day hash. {x} fields come from signals.

QUIPS: list[tuple[str, list[str]]] = [
    ("on_fire", [        # last three runs all won
        "\U0001F525 Three on the bounce — someone check the smoke alarm.",
        "\U0001F525 Hat-trick seeker. The horse is, technically, on fire.",
    ]),
    ("cold_jockey", [    # 0 wins from 20+ rides in a fortnight
        "{jockey} is 0 from {j_n} this fortnight — due, or just crap?",
        "{jockey} couldn't win a raffle lately (0/{j_n}). Today's the day?",
    ]),
    ("hot_yard", [       # trainer 25%+ strike from 8+ runs
        "The {trainer} yard is red-hot — {t_w} winners from {t_n} in a fortnight.",
        "{trainer}'s toasters are all firing: {t_w}/{t_n} this fortnight.",
        "Whatever {trainer} is feeding them, it's working ({t_w}/{t_n} lately).",
        "{trainer} can't stop winning — {t_w} of the last {t_n}.",
    ]),
    ("cold_yard", [      # 0 from 15+ in a fortnight
        "The yard is colder than the going report — 0 from {t_n} this fortnight.",
        "{trainer}: 0 from {t_n} lately. Somebody unplug and replug the stable.",
    ]),
    ("long_layoff", [    # 200+ days off
        "First run in {lr} days — hope he remembers which way the track goes.",
        "{lr} days off. That's not a break, that's a gap year.",
    ]),
    ("career_maiden", [  # 0 wins from 15+ starts
        "Still a maiden after {runs} tries. Bless.",
        "{runs} races, zero wins — consistency of a sort.",
    ]),
    ("debut", [
        "First ever start — literally anyone's guess, including ours.",
        "Debutant. The form book is a blank page and so are we.",
    ]),
    ("veteran", [        # 10yo+
        "A {age}-year-old warrior — racing since before our model existed.",
        "{age} years old and still at it. Respect. Concern, but respect.",
    ]),
    ("odds_on", [        # notably short — not every mild favourite
        "Odds-on jolly — win a fiver, buy half a Freddo.",
        "Short enough to trip over. The bookies aren't scared.",
        "The market's already spent the winnings on this one.",
        "Practically a penalty kick, says the market. Famous last words.",
        "More chalk than a snooker hall.",
    ]),
    ("outsider", [       # 40+
        "{odds:.0f}/1-ish outsider — if he wins we're framing the printout.",
        "The market gives him no chance. The market is occasionally wrong.",
    ]),
]


def _fires(key: str, s: dict) -> bool:
    t_w, t_n = s.get("t14") or (0, 0)
    j_w, j_n = s.get("j14") or (0, 0)
    if key == "on_fire":
        return s.get("form3") == "111"
    if key == "cold_jockey":
        return bool(s.get("jockey")) and j_n >= 20 and j_w == 0
    if key == "hot_yard":
        return t_n >= 8 and t_w / t_n >= 0.25
    if key == "cold_yard":
        return bool(s.get("trainer")) and t_n >= 15 and t_w == 0
    if key == "long_layoff":
        return (s.get("lr") or 0) >= 200
    if key == "career_maiden":
        return (s.get("runs") or 0) >= 15 and "wins" in s and s["wins"] == 0
    if key == "debut":
        return s.get("runs") == 0 and "runs" in s
    if key == "veteran":
        return (s.get("age") or 0) >= 10
    if key == "odds_on":
        return 1.0 < (s.get("odds") or 0) < 1.65
    if key == "outsider":
        return (s.get("odds") or 0) >= 40
    return False


def quip_for_pick(s: dict, date_iso: str = "") -> str | None:
    t_w, t_n = s.get("t14") or (0, 0)
    j_w, j_n = s.get("j14") or (0, 0)
    fmt = dict(s, t_w=t_w, t_n=t_n, j_w=j_w, j_n=j_n)
    for key, templates in QUIPS:
        if not _fires(key, s):
            continue
        seed = hashlib.sha256(f"{date_iso}|{s.get('horse')}|{key}".encode()).digest()[0]
        tpl = templates[seed % len(templates)]
        try:
            return tpl.format(**fmt)
        except Exception:
            continue
    return None
