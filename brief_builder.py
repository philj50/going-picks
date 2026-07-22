"""
brief_builder.py — budget-driven race briefs for the AI voices.

Replaces the flat light/full brief with a per-voice token budget: every voice
gets the same *kinds* of facts (so the scoreboard stays comparable), richer
voices get more depth. Sections are added in priority order until the input
budget is spent:

  1. race header + model pick            (always)
  2. legend + per-runner core lines      (always)
  3. draw-bias line                      (flat races, sample-gated)
  4. last-3-runs history per runner      (drops to last-1, then none)

Per-runner data merges three sources, best first:
  - rpscrape racecard JSON (tools/rpscrape/racecards/YYYY-MM-DD.json) when
    present: OR/RPR/TS, full form string, age/sex, sire/damsire, headgear+first,
    days since run — all pre-race published figures, no leakage.
  - the 6-year corpus DB: going/distance/course/C&D records, trainer/jockey
    14-day form, last-3-runs detail, draw bias, last-known ratings.
  - the day card (The Racing API + Betfair): odds, draw, weight, connections.

The model's own scoring pipeline never reads any of this — briefs are for the
voices only, so the model stays stable (feature gate: monotonic lift only).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
from pathlib import Path

REPO = Path(__file__).parent
RP_RACECARD_DIR = REPO / "tools" / "rpscrape" / "racecards"

# ---------------------------------------------------------------------------
# Per-voice budgets: (input_tokens, prose_level)
# prose_level: 0 = VERDICT JSON only, 1 = one sentence, 2 = two short paragraphs
# Rationale: Cursor is paid (strongest reader, full prose); Lucy/ollama is paid
# in latency (full brief, JSON-only output); NVIDIA's free tier has generous
# token limits (full brief, short output); Groq/Cerebras/Mistral free tiers are
# tight (compact brief).
DEFAULT_BUDGETS: dict[str, tuple[int, int]] = {
    "cursor": (1500, 2),
    "claude": (1500, 2),
    "openai": (1500, 2),
    "ollama": (1500, 0),
    "nvidia": (1500, 1),
    "groq": (850, 1),
    "cerebras": (850, 1),
    "mistral": (850, 1),
    "gemini": (850, 1),
}
FALLBACK_BUDGET = (850, 1)


def _env_budgets() -> dict[str, tuple[int, int]]:
    """Optional override: GOING_AI_BRIEF_BUDGETS="cursor:2000:2,groq:700:1"."""
    raw = os.getenv("GOING_AI_BRIEF_BUDGETS", "").strip()
    out: dict[str, tuple[int, int]] = {}
    if not raw:
        return out
    for part in raw.split(","):
        bits = part.strip().split(":")
        if len(bits) >= 2:
            try:
                tokens = max(300, int(bits[1]))
                prose = max(0, min(2, int(bits[2]))) if len(bits) >= 3 else None
            except ValueError:
                continue
            default_prose = DEFAULT_BUDGETS.get(bits[0].lower(), FALLBACK_BUDGET)[1]
            out[bits[0].lower()] = (tokens, prose if prose is not None else default_prose)
    return out


def voice_budget(voice: str | None) -> tuple[int, int]:
    """(input_token_budget, prose_level) for a voice."""
    v = (voice or "").lower()
    env = _env_budgets()
    if v in env:
        return env[v]
    return DEFAULT_BUDGETS.get(v, FALLBACK_BUDGET)


def prose_level(voice: str | None) -> int:
    return voice_budget(voice)[1]


def est_tokens(s: str) -> int:
    return int(len(s) / 3.8)


# ---------------------------------------------------------------------------
# normalisation helpers

def norm_horse(name: str) -> str:
    name = re.sub(r"\s*\([A-Z]{2,3}\)\s*$", "", name or "")
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _norm_course(course: str) -> str:
    c = re.sub(r"\(.*?\)", "", course or "").strip().lower()
    return re.sub(r"[^a-z]", "", c)


def _time24(t: str) -> str:
    """'5:45' -> '17:45'; '13:50' stays. GB/IRE racing runs 11:00-21:30."""
    m = re.match(r"^(\d{1,2}):(\d{2})", str(t or "").strip())
    if not m:
        return str(t or "")
    h, mi = int(m.group(1)), m.group(2)
    if h <= 10:
        h += 12
    return f"{h:02d}:{mi}"


GOING_ABBR = {
    "firm": "F", "good to firm": "GF", "good": "G", "good to soft": "GS",
    "soft": "S", "soft to heavy": "SH", "heavy": "Hy", "yielding": "Y",
    "good to yielding": "GY", "yielding to soft": "YS",
    "standard": "St", "standard to slow": "StS", "standard to fast": "StF",
    "slow": "Sl", "fast": "Fst",
}


def _gabbr(going: str) -> str:
    return GOING_ABBR.get((going or "").strip().lower(), (going or "?")[:3])


def _fp_char(fp) -> str:
    if fp is None:
        return "-"
    s = str(fp)
    if s.isdigit():
        n = int(s)
        return str(n) if n <= 9 else "0"
    return s[:1].upper()  # PU->P F->F UR->U


def _wt(lbs) -> str:
    try:
        lbs = int(round(float(lbs)))
        return f"{lbs // 14}-{lbs % 14}"
    except (TypeError, ValueError):
        return "?"


# ---------------------------------------------------------------------------
# rpscrape racecard enrichment (pre-race Racing Post fields)

_RP_CACHE: dict[str, dict] = {}


def load_rp_card(date: str) -> dict:
    """rpscrape racecard for a date, keyed for merge.
    Returns {"races": {(course_norm, time24): race_dict},
             "runners": {(course_norm, time24, horse_norm): runner_dict}}."""
    if date in _RP_CACHE:
        return _RP_CACHE[date]
    out = {"races": {}, "runners": {}}
    path = RP_RACECARD_DIR / f"{date}.json"
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for region in (data or {}).values():
                for course_races in region.values():
                    for off, race in course_races.items():
                        ck = _norm_course(race.get("course") or "")
                        tk = _time24(race.get("off_time") or off)
                        out["races"][(ck, tk)] = race
                        for rn in race.get("runners") or []:
                            out["runners"][(ck, tk, norm_horse(rn.get("name") or ""))] = rn
        except Exception:
            pass
    _RP_CACHE[date] = out
    return out


# ---------------------------------------------------------------------------
# corpus feature store (leak-free: everything strictly before `before` date)

class FeatureStore:
    def __init__(self, con, before: str):
        self.con = con
        self.before = before  # race date — history is date < before + same-day earlier not counted
        self._hist: dict[str, list] = {}
        self._t14: dict[str, tuple[int, int]] = {}
        self._j14: dict[str, tuple[int, int]] = {}
        self._draw: dict[tuple, str | None] = {}
        self.mv_seen = False   # any market-move line rendered this race

    def hist(self, horse_norm: str) -> list:
        if horse_norm in self._hist:
            return self._hist[horse_norm]
        rows = []
        if self.con is not None:
            try:
                rows = self.con.execute(
                    "SELECT rr.finish_pos fp, rr.beaten_lengths bl, rr.starting_price sp,"
                    " rr.official_rating oro, ra.date d, ra.course co, ra.distance_f df,"
                    " ra.going g, ra.race_class cl, ra.field_size fsz"
                    " FROM runners rr JOIN races ra ON ra.race_id = rr.race_id"
                    " WHERE rr.horse_norm=? AND ra.date<?"
                    " AND rr.finish_pos IS NOT NULL AND rr.finish_pos != ''"
                    " ORDER BY ra.date DESC LIMIT 30",
                    (horse_norm, self.before)).fetchall()
            except Exception:
                rows = []
        self._hist[horse_norm] = rows
        return rows

    @staticmethod
    def _rec(rows, cond) -> tuple[int, int]:
        sel = [r for r in rows if cond(r)]
        return sum(1 for r in sel if str(r["fp"]) == "1"), len(sel)

    def records(self, horse_norm: str, course: str, distance_f, going: str) -> dict:
        rows = self.hist(horse_norm)
        cn = (course or "").strip().lower()
        gn = (going or "").strip().lower()
        out = {}
        out["G"] = self._rec(rows, lambda r: (r["g"] or "").lower() == gn) if gn else (0, 0)
        if distance_f:
            out["D"] = self._rec(rows, lambda r: r["df"] and abs(r["df"] - float(distance_f)) < 0.75)
        else:
            out["D"] = (0, 0)
        out["C"] = self._rec(rows, lambda r: (r["co"] or "").lower() == cn)
        out["CD"] = self._rec(rows, lambda r: (r["co"] or "").lower() == cn
                              and distance_f and r["df"]
                              and abs(r["df"] - float(distance_f)) < 0.75)
        return out

    def _form14(self, cache: dict, col: str, name: str) -> tuple[int, int]:
        if not name or name == "?":
            return (0, 0)
        if name in cache:
            return cache[name]
        w = n = 0
        if self.con is not None:
            try:
                d0 = (dt.date.fromisoformat(self.before) - dt.timedelta(days=14)).isoformat()
                row = self.con.execute(
                    f"SELECT SUM(CASE WHEN rr.finish_pos='1' THEN 1 ELSE 0 END), COUNT(*)"
                    f" FROM runners rr JOIN races ra ON ra.race_id = rr.race_id"
                    f" WHERE rr.{col}=? AND ra.date>=? AND ra.date<?",
                    (name, d0, self.before)).fetchone()
                w, n = (row[0] or 0), (row[1] or 0)
            except Exception:
                pass
        cache[name] = (w, n)
        return (w, n)

    def trainer14(self, name: str) -> tuple[int, int]:
        return self._form14(self._t14, "trainer", name)

    def jockey14(self, name: str) -> tuple[int, int]:
        return self._form14(self._j14, "jockey", name)

    def draw_bias_line(self, course: str, distance_f, is_handicap: bool) -> str | None:
        """Winners-by-draw-third at this course/distance, sample-gated (>=30 races)."""
        key = ((course or "").lower(), round(float(distance_f or 0), 1))
        if key in self._draw:
            return self._draw[key]
        line = None
        if self.con is not None and distance_f:
            try:
                rows = self.con.execute(
                    "SELECT rr.draw, ra.field_size fsz FROM runners rr"
                    " JOIN races ra ON ra.race_id = rr.race_id"
                    " WHERE LOWER(ra.course)=? AND ABS(ra.distance_f-?)<0.5"
                    " AND ra.race_type='Flat' AND rr.draw IS NOT NULL"
                    " AND rr.finish_pos='1' AND ra.field_size>=6 AND ra.date<?",
                    ((course or "").lower(), float(distance_f), self.before)).fetchall()
                n_races = len(rows)
                if n_races >= 30:
                    thirds = {"low": 0, "mid": 0, "high": 0}
                    for r in rows:
                        t = ("low" if r["draw"] <= r["fsz"] / 3
                             else "mid" if r["draw"] <= 2 * r["fsz"] / 3 else "high")
                        thirds[t] += 1
                    pct = {k: round(100 * v / n_races) for k, v in thirds.items()}
                    line = (f"DRAW {course} ~{float(distance_f):.0f}f (last 6yr, n={n_races}"
                            f" flat races): winners drawn low {pct['low']}% ·"
                            f" mid {pct['mid']}% · high {pct['high']}%")
            except Exception:
                line = None
        self._draw[key] = line
        return line

    def market_move(self, horse_norm: str) -> str | None:
        """Betfair drift/steam today: morning vs latest tick. Early market-open
        placeholder ticks (1.01-and-drifting) are excluded."""
        if self.con is None:
            return None
        try:
            rows = self.con.execute(
                "SELECT price FROM price_ticks WHERE date=? AND horse_norm=?"
                " AND ts >= ? AND price >= 1.2 ORDER BY ts",
                (self.before, horse_norm, f"{self.before}T08:00")).fetchall()
        except Exception:
            return None
        if len(rows) < 5:
            return None
        first, last = rows[0][0], rows[-1][0]
        if not first or not last or last < 1.2:
            return None
        if abs(first - last) / first < 0.15:
            return None
        self.mv_seen = True
        return f"mv {first:.1f}→{last:.1f}"


# ---------------------------------------------------------------------------
# race-contextual lessons: post-mortem themes included only when this race
# can actually use them (draw advice is noise in a jumps race, etc.)

_THEMES_CACHE = {"mtime": None, "themes": []}
_ALWAYS_FEATURES = frozenset({"cd_winner", "jockey_in_form", "yard_in_form",
                              "combo_in_form", "or_proxy"})


def _load_themes() -> list:
    path = REPO / "ai_reports" / "ai_learning.json"
    try:
        m = path.stat().st_mtime
        if _THEMES_CACHE["mtime"] != m:
            data = json.loads(path.read_text(encoding="utf-8"))
            _THEMES_CACHE.update(mtime=m, themes=list(data.get("themes") or []))
    except Exception:
        return []
    return _THEMES_CACHE["themes"]


def contextual_lessons(race: dict, *, max_lr: int | None = None,
                       has_market: bool = False, limit: int = 5) -> str:
    """Lesson block filtered to what THIS race can act on."""
    if os.getenv("GOING_AI_NO_LESSONS", "").strip().lower() in ("1", "true", "yes"):
        return ""
    themes = _load_themes()
    if not themes:
        return ""
    going = (race.get("going") or "").strip().lower()

    def applies(feat: str) -> bool:
        if feat in _ALWAYS_FEATURES:
            return True
        if feat == "draw_bias":
            return bool(race.get("is_flat", True))
        if feat == "going_win_rate":
            return going not in ("", "good", "standard")
        if feat == "below_winning_mark":
            return bool(race.get("is_handicap"))
        if feat == "layoff_with_top_jockey":
            return (max_lr or 0) >= 100
        if feat in ("drift_pct", "steamer_magnitude"):
            return has_market
        return True

    ranked = sorted(themes, key=lambda t: (1 if t.get("maps_to_feature") else 0,
                                           t.get("cites") or 0), reverse=True)
    lines = []
    unmapped_used = 0
    for t in ranked:
        theme = (t.get("theme") or "").strip()
        if not theme or len(theme) < 4:
            continue
        feat = t.get("maps_to_feature")
        if feat and not applies(feat):
            continue
        if not feat:
            if unmapped_used >= 2:
                continue
            unmapped_used += 1
        cites = t.get("cites")
        line = f"- {theme}"
        if feat:
            line += f" (→ {feat}" + (f", n={cites})" if cites else ")")
        elif cites:
            line += f" (n={cites})"
        lines.append(line[:120])
        if len(lines) >= limit:
            break
    if not lines:
        return ""
    return ("Lessons from settled post-mortems relevant to THIS race "
            "(apply if the data above supports them):\n" + "\n".join(lines))


# ---------------------------------------------------------------------------
# rendering

LEGEND_CORE = ("LEGEND: dr=draw · OR=official rating · wt=weight st-lb · form latest RIGHT"
               " ('0'=10th+, P/F/U=not completed) · LR=days since run · runs=career starts ·"
               " G/D/C/CD=wins-runs on today's going/distance/course/course+distance ·"
               " T14/J14=trainer/jockey wins-runs last 14 days · mv=today's price move")
LEGEND_HIST = " · indented = last runs (date course dist going class: pos/field, beaten, SP)"


def _form_string(rp_runner: dict | None, hist_rows: list, card_runner: dict) -> str:
    if rp_runner and rp_runner.get("form"):
        return str(rp_runner["form"])[-8:]
    if hist_rows:
        return "".join(_fp_char(r["fp"]) for r in reversed(hist_rows[:6]))
    lp = card_runner.get("last_positions") or []
    if lp:  # card convention: most recent FIRST — flip to racing convention
        return "".join(_fp_char(p) for p in reversed(lp))
    return "?"


def _runner_core_line(r: dict, race: dict, fs: FeatureStore, rp_runner: dict | None,
                      maiden_ctx: bool) -> str:
    hn = norm_horse(r.get("name") or "")
    hist = fs.hist(hn)
    odds = r.get("odds_decimal") or 0
    odds_s = f" @ {odds:.1f}" if odds and odds > 1.05 else ""
    draw = r.get("draw") if r.get("draw") is not None else (rp_runner or {}).get("draw")
    dr_s = f"dr{draw}" if draw is not None else "dr?"

    ofr = (rp_runner or {}).get("ofr")
    if ofr is None:
        ofr = next((row["oro"] for row in hist if row["oro"]), None)
    or_s = f"OR {ofr}" if ofr else "OR ?"
    wt_s = _wt(r.get("weight_lbs") if r.get("weight_lbs") is not None
               else (rp_runner or {}).get("lbs"))

    form = _form_string(rp_runner, hist, r)
    lr = (rp_runner or {}).get("last_run")
    if lr is None:
        lr = r.get("days_since_run")
    if lr is None and hist:
        try:
            lr = (dt.date.fromisoformat(fs.before) - dt.date.fromisoformat(hist[0]["d"])).days
        except Exception:
            lr = None
    runs = len(hist)
    form_bit = f"{form} LR{lr}" if lr is not None else form
    if runs == 0:
        form_bit = "DEBUT"

    rec = fs.records(hn, race.get("course"), race.get("distance_f"), race.get("going"))
    rec_s = " ".join(f"{k} {w}-{n}" for k, (w, n) in rec.items())

    t14 = fs.trainer14(r.get("trainer") or "")
    j14 = fs.jockey14(r.get("jockey") or "")
    conn_s = f"T14 {t14[0]}-{t14[1]} J14 {j14[0]}-{j14[1]}"

    extras = []
    age = (rp_runner or {}).get("age")
    sex = (rp_runner or {}).get("sex_code")
    if age:
        extras.append(f"{age}yo{(sex or '').lower()}")
    hg = r.get("headgear") or (rp_runner or {}).get("headgear")
    if hg:
        first = (rp_runner or {}).get("headgear_first") or r.get("headgear_first_time")
        extras.append(f"{str(hg).strip()}{'1' if first else ''}")
    rtf = (rp_runner or {}).get("trainer_rtf")
    if rtf is not None:
        extras.append(f"rtf {round(float(rtf))}%")
    mv = fs.market_move(hn)
    if mv:
        extras.append(mv)
    if maiden_ctx and runs <= 2 and rp_runner and rp_runner.get("sire"):
        sire_bit = f"by {rp_runner['sire']}"
        if rp_runner.get("damsire"):
            sire_bit += f" × {rp_runner['damsire']}"
        extras.append(sire_bit)
    extra_s = (" | " + " ".join(extras)) if extras else ""

    return (f"- {r.get('name')}{odds_s} | {dr_s} | {or_s} {wt_s} | {form_bit}"
            f" runs {runs} | {rec_s} | {conn_s}{extra_s}")


def _history_lines(hn: str, fs: FeatureStore, n: int) -> list[str]:
    out = []
    for x in fs.hist(hn)[:n]:
        bl = f" btn {x['bl']:.1f}L" if x["bl"] not in (None, 0) else ""
        sp = f" @ {x['sp']:.1f}" if x["sp"] else ""
        cl = str(x["cl"] or "").replace("Class ", "C") or "?"
        try:
            d = x["d"][5:]
        except Exception:
            d = "?"
        df = f"{x['df']:.0f}f" if x["df"] else "?f"
        out.append(f"    · {d} {x['co']} {df} {_gabbr(x['g'])} {cl}:"
                   f" {x['fp']}/{x['fsz'] or '?'}{bl}{sp}")
    return out


def _model_block(prediction: dict) -> str:
    top3 = (prediction or {}).get("top_3") or []
    if not top3:
        return "MODEL PICK: none qualifying in this race."
    p = top3[0]
    wp = p.get("win_prob")
    odds = p.get("odds") or 0
    line = f"MODEL PICK: {p.get('name')}"
    if wp is not None:
        line += f" — win prob {wp:.0f}%"
    if odds:
        line += f" (odds {odds:.1f})"
    if p.get("reasons"):
        line += f". Reasoning: {'; '.join(p['reasons'][:3])}"
    return line


def build_brief(con, race: dict, prediction: dict, in_budget: int = 850,
                date: str | None = None) -> tuple[str, str]:
    """Assemble a race brief within ~in_budget tokens. Returns (brief, tier)."""
    race_day = date or race.get("date") or dt.date.today().isoformat()
    fs = FeatureStore(con, race_day)
    rp = load_rp_card(race_day)
    ck = _norm_course(race.get("course") or "")
    tk = _time24(race.get("time") or "")
    rp_race = rp["races"].get((ck, tk))

    runners = sorted(race.get("runners") or [],
                     key=lambda r: r.get("odds_decimal") or 999)
    is_hcp = bool(race.get("is_handicap"))
    maiden_ctx = not is_hcp
    dist = race.get("distance_f")
    dist_s = f"{dist:.1f}".rstrip("0").rstrip(".") + "f" if dist else "?"
    rclass = race.get("race_class") or (rp_race or {}).get("race_class") or ""
    cls_s = f"Class {rclass}" if str(rclass).strip().isdigit() else (str(rclass) or "class ?")

    hdr = (f"RACE: {race.get('course')} {race.get('time')} · {dist_s}"
           f" {race.get('going') or '?'} · {cls_s}"
           f"{' handicap' if is_hcp else ''} · {len(runners)} run"
           f" · {'Flat' if race.get('is_flat', True) else 'Jumps'}")
    if rp_race:
        bits = []
        if rp_race.get("rating_band"):
            bits.append(str(rp_race["rating_band"]))
        if rp_race.get("age_band"):
            bits.append(str(rp_race["age_band"]))
        if rp_race.get("prize_winner"):
            bits.append(f"1st {rp_race['prize_winner']}")
        if bits:
            hdr += " · " + " · ".join(bits)

    core_lines = [_runner_core_line(r, race, fs, rp["runners"].get(
        (ck, tk, norm_horse(r.get("name") or ""))), maiden_ctx) for r in runners]

    parts = [hdr, _model_block(prediction), "", LEGEND_CORE, "", "RUNNERS"]
    parts += core_lines

    if race.get("is_flat", True):
        db_line = fs.draw_bias_line(race.get("course"), dist, is_hcp)
        if db_line:
            parts += ["", db_line]

    # Post-mortem memory: what our own overnight reviews said about these
    # horses/jockeys recently — unique corpus knowledge, budget permitting.
    pm_notes = ""
    if in_budget >= 1200:
        try:
            import ai_horse_postmortem as hpm
            pm_notes = hpm.connection_notes_for_brief(
                race.get("runners") or [], before=race_day, days=21, limit=5)
        except Exception:
            pm_notes = ""

    core_text = "\n".join(parts)
    used = est_tokens(core_text)
    tier = "compact"

    # history ladder: last-3 → last-1 → none
    for n_hist in (3, 1):
        hist_map = {}
        cost = 0
        for r in runners:
            hl = _history_lines(norm_horse(r.get("name") or ""), fs, n_hist)
            hist_map[r.get("name")] = hl
            cost += sum(est_tokens(l) + 1 for l in hl)
        if any(hist_map.values()) and used + cost <= in_budget:
            new_lines = []
            for r, line in zip(runners, core_lines):
                new_lines.append(line)
                new_lines.extend(hist_map.get(r.get("name")) or [])
            parts = [hdr, _model_block(prediction), "", LEGEND_CORE + LEGEND_HIST,
                     "", "RUNNERS"] + new_lines
            if race.get("is_flat", True):
                db_line = fs.draw_bias_line(race.get("course"), dist, is_hcp)
                if db_line:
                    parts += ["", db_line]
            tier = "extended" if n_hist == 3 else "extended-1"
            core_text = "\n".join(parts)
            break

    if pm_notes and est_tokens(core_text) + est_tokens(pm_notes) + 2 <= in_budget:
        core_text += "\n\n" + pm_notes.rstrip()

    # Race-contextual lessons (small; allowed to nudge past budget slightly)
    max_lr = 0
    for r in runners:
        rp_r = rp["runners"].get((ck, tk, norm_horse(r.get("name") or ""))) or {}
        for cand in (r.get("days_since_run"), rp_r.get("last_run")):
            try:
                max_lr = max(max_lr, int(cand))
            except (TypeError, ValueError):
                pass
    lessons = contextual_lessons(race, max_lr=max_lr, has_market=fs.mv_seen)
    if lessons and est_tokens(core_text) + est_tokens(lessons) <= in_budget + 100:
        core_text += "\n\n" + lessons

    return core_text, tier
