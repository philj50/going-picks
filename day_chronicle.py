"""Daily chronicle — markdown log of what happened on a racing day.

Written after Today refresh (throttled) and a richer EOD pass after settle.
Injected into race briefs so models can learn from recent days.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

import going_paths

ROOT = going_paths.repo()
REPORT_DIR = ROOT / "ai_reports"
PUBLIC_DAYS = ROOT / "public" / "days"


def chronicle_path(day: str | dt.date) -> Path:
    if isinstance(day, dt.date):
        day = day.isoformat()
    return REPORT_DIR / f"day_log_{day}.json"


def markdown_path(day: str | dt.date) -> Path:
    if isinstance(day, dt.date):
        day = day.isoformat()
    return REPORT_DIR / f"day_log_{day}.md"


def public_markdown_path(day: str | dt.date) -> Path:
    if isinstance(day, dt.date):
        day = day.isoformat()
    return PUBLIC_DAYS / f"{day}.md"


def _load_card(day: str) -> dict:
    name = "today.json" if day == dt.date.today().isoformat() else None
    if not name:
        for p in (ROOT / "yesterday.json", ROOT / "tomorrow.json"):
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
                if raw.get("date") == day:
                    return raw
            except Exception:
                pass
    for fname in ("today.json", "yesterday.json", "tomorrow.json"):
        p = ROOT / fname
        if not p.exists():
            continue
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            if raw.get("date") == day:
                return raw
        except Exception:
            pass
    return {}


def _load_report(day: str) -> dict:
    p = REPORT_DIR / f"ai_analysis_{day}.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def build_chronicle(day: str | None = None, *, eod: bool = False) -> dict:
    """Assemble chronicle payload for day (default today)."""
    day_s = day or dt.date.today().isoformat()
    card = _load_card(day_s)
    report = _load_report(day_s)
    potd = report.get("pick_of_the_day") or {}
    lines = []
    n_races = len(card.get("races") or [])
    n_analysed = len(report.get("races") or [])
    lines.append(f"# Going — {day_s}")
    lines.append("")
    lines.append(f"- Card races: **{n_races}** · AI report races: **{n_analysed}**")
    if potd.get("horse"):
        lines.append(
            f"- Pick of the day: **{potd.get('horse')}** "
            f"({potd.get('course')} {potd.get('time')})")
    # Outstanding voices
    try:
        missing = []
        for race in report.get("races") or []:
            for k in ("cursor", "nvidia", "groq", "cerebras", "ollama"):
                txt = race.get(f"{k}_analysis") or ""
                if not txt or str(txt).startswith("ERROR") or "VERDICT:" not in str(txt):
                    missing.append((race.get("course"), race.get("time"), k))
        if missing:
            by_voice: dict[str, int] = {}
            for _c, _t, k in missing:
                by_voice[k] = by_voice.get(k, 0) + 1
            bits = ", ".join(f"{k}×{n}" for k, n in sorted(by_voice.items()))
            lines.append(f"- Outstanding voices: **{len(set((c,t) for c,t,_ in missing))} races** ({bits})")
    except Exception:
        pass
    lines.append("")

    try:
        import day_cards
        rows = day_cards.day_voice_scoreboard(day_s)
        if rows:
            lines.append("## Voice scoreboard (staked hit rate)")
            for r in rows[:8]:
                lines.append(
                    f"- **{r['label']}**: {r['hits']}/{r['n']} "
                    f"({r['rate']*100:.0f}%)")
            lines.append("")
    except Exception:
        pass

    # Credit P&L strip
    try:
        bk = json.loads((REPORT_DIR / "bankrolls.json").read_text(encoding="utf-8"))
        lines.append("## Credit bankrolls")
        for ai, rec in (bk.get("ais") or {}).items():
            bal = rec.get("balance")
            if bal is None:
                continue
            day_pnl = None
            for h in rec.get("history") or []:
                if isinstance(h, dict) and h.get("date") == day_s:
                    day_pnl = h.get("pnl")
            pnl_bit = f" · day {float(day_pnl):+.0f}cr" if day_pnl is not None else ""
            lines.append(f"- **{ai}**: {float(bal):.0f}cr{pnl_bit}")
        lines.append("")
    except Exception:
        pass

    # Me paper bets
    try:
        me_path = REPORT_DIR / "me_bets.json"
        if me_path.exists():
            me = json.loads(me_path.read_text(encoding="utf-8"))
            day_bets = [p for p in (me.get("pending") or []) + (me.get("history") or [])
                        if p.get("date") == day_s]
            if day_bets:
                lines.append("## Me bets (paper)")
                for p in day_bets:
                    lines.append(
                        f"- £{p.get('stake_gbp', 0):.0f} **{p.get('horse')}** "
                        f"({p.get('course')} {p.get('time')}) · {p.get('status')}")
                lines.append("")
    except Exception:
        pass

    if eod:
        try:
            import ai_learning
            log = ai_learning.load_overnight_log(limit=3)
            if log:
                lines.append("## Overnight learning")
                for entry in log[:2]:
                    lines.append(f"- {entry.get('ts', '')}: "
                                 f"{entry.get('summary', entry)}")
                lines.append("")
        except Exception:
            pass
        try:
            import ai_horse_postmortem as hpm
            pm = hpm.load_pm(day_s)
            n_pm = len((pm or {}).get("races") or [])
            if n_pm:
                lines.append(f"## Post-mortems\n- Horse post-mortems: **{n_pm}** races\n")
        except Exception:
            pass

    lines.append("## Races")
    for race in (report.get("races") or [])[:60]:
        rn = race.get("race") or "?"
        course = race.get("course") or ""
        time_s = race.get("time") or ""
        top = (race.get("model_prediction") or {}).get("top_3") or []
        pick = top[0].get("name") if top else "—"
        voices = []
        for k in ("cursor", "nvidia", "groq", "cerebras", "ollama"):
            txt = race.get(f"{k}_analysis") or ""
            if txt and not str(txt).startswith("ERROR") and "VERDICT:" in str(txt):
                voices.append(k)
            elif not txt or str(txt).startswith("ERROR"):
                voices.append(f"{k}?")
        # stakes summary
        stake_bits = []
        for ai, alloc in (report.get("allocations") or {}).items():
            for e in alloc.get("entries") or []:
                if (e.get("race_name") or "") != rn:
                    continue
                cr = int(e.get("credits") or 0)
                if cr > 0:
                    stake_bits.append(f"{ai} {cr}cr→{e.get('horse')}")
        stake_s = f" · stakes: {', '.join(stake_bits)}" if stake_bits else ""
        lines.append(f"- **{course} {time_s}** — {rn}: model **{pick}** "
                     f"({', '.join(voices) or 'no AI'}){stake_s}")

    md = "\n".join(lines) + "\n"
    payload = {
        "date": day_s,
        "updated": dt.datetime.now().isoformat(timespec="seconds"),
        "eod": eod,
        "markdown": md,
        "n_races": n_races,
        "n_analysed": n_analysed,
    }
    return payload


def write_chronicle(day: str | None = None, *, eod: bool = False) -> Path:
    payload = build_chronicle(day, eod=eod)
    day_s = payload["date"]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    chronicle_path(day_s).write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    markdown_path(day_s).write_text(payload["markdown"], encoding="utf-8")
    PUBLIC_DAYS.mkdir(parents=True, exist_ok=True)
    public_markdown_path(day_s).write_text(payload["markdown"], encoding="utf-8")
    return markdown_path(day_s)


def excerpt_for_prompt(day: str | None = None, max_chars: int = 1200) -> str:
    """Short excerpt for race brief injection."""
    day_s = day or dt.date.today().isoformat()
    p = markdown_path(day_s)
    if not p.exists():
        try:
            write_chronicle(day_s, eod=False)
        except Exception:
            return ""
        p = markdown_path(day_s)
    if not p.exists():
        return ""
    try:
        text = p.read_text(encoding="utf-8")
    except Exception:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20].rstrip() + "\n…(truncated)"


def should_refresh_throttle(day: str, min_secs: int = 900) -> bool:
    """True if we should skip a refresh-triggered rewrite (default 15 min)."""
    p = chronicle_path(day)
    if not p.exists():
        return False
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        updated = data.get("updated") or ""
        if not updated:
            return False
        ts = dt.datetime.fromisoformat(updated)
        return (dt.datetime.now() - ts).total_seconds() < min_secs
    except Exception:
        return False
