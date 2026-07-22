"""
Client for The Racing API (https://www.theracingapi.com) + a mapper that turns
its racecards into the card shape this project uses (see sample_card.json), so
`analyse.py` can run on live data instead of hand-built JSON.

Pure standard library — no `requests` needed.

AUTH — credentials come from a `.env` file (or real environment variables),
never hard-coded. Copy `.env.example` to `.env` and fill it in:

    THERACINGAPI_USERNAME=your_username
    THERACINGAPI_PASSWORD=your_password

`.env` is git-ignored so your key is never committed. Real environment variables
still work and take precedence:

    $env:THERACINGAPI_USERNAME = "your_username"   # PowerShell, this shell only
    setx THERACINGAPI_USERNAME "your_username"      # persist (reopen terminal)

CALIBRATE THE MAPPING TO YOUR PLAN
The exact fields in a racecard depend on your subscription tier — the free tier
omits odds and most stats; higher tiers add them. The `_map_*` helpers below use
the documented field names with safe fallbacks and accept several aliases, but
field names DO drift. Run once:

    python fetch_today.py --raw --out raw.json

to dump a real response, eyeball the keys, and adjust the aliases here if needed.
Anything missing falls back to the model's neutral defaults — EXCEPT odds, since
without a price there is no value to assess. Runners with no odds are dropped and
reported.
"""
from __future__ import annotations
import base64
import json
import os
import ssl
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path

BASE_URL = "https://api.theracingapi.com"


class RacingAPIError(RuntimeError):
    """Any failure talking to the API (auth, network, HTTP error)."""


# --- .env support (no dependency) ------------------------------------------

def load_dotenv(path: str | os.PathLike | None = None) -> None:
    """Load KEY=VALUE lines from a `.env` file into os.environ. Real environment
    variables already set are left untouched (they win). Lines starting with #
    and blank lines are ignored; surrounding quotes are stripped. Silently does
    nothing if the file is absent — so it's safe to call unconditionally."""
    env_path = Path(path) if path else Path(__file__).with_name(".env")
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


# --- HTTP ------------------------------------------------------------------

def _auth_header() -> str:
    load_dotenv()
    user = os.environ.get("THERACINGAPI_USERNAME")
    pw = os.environ.get("THERACINGAPI_PASSWORD")
    if not user or not pw:
        raise RacingAPIError(
            "Missing credentials. Copy .env.example to .env and fill in "
            "THERACINGAPI_USERNAME / THERACINGAPI_PASSWORD (or set them as "
            "environment variables). See the module docstring."
        )
    token = base64.b64encode(f"{user}:{pw}".encode()).decode()
    return f"Basic {token}"


def _ssl_context(verify: bool, ca_bundle: str | None):
    """Build the TLS context.

    verify=True  (default) — normal certificate verification. If `ca_bundle`
                 (or the SSL_CERT_FILE / THERACINGAPI_CA_BUNDLE env var) points
                 at a PEM file, that CA is trusted too — the right fix behind a
                 corporate TLS-intercepting proxy: export the proxy's root CA
                 and point here.
    verify=False — skip verification entirely. Pragmatic escape hatch when a
                 proxy mangles certs, but it means you can't cryptographically
                 trust the connection. Opt-in only.
    """
    if not verify:
        return ssl._create_unverified_context()
    ca = ca_bundle or os.environ.get("THERACINGAPI_CA_BUNDLE")
    if ca:
        return ssl.create_default_context(cafile=ca)
    return None                                    # urllib uses its default


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def get(path: str, params: dict | None = None, timeout: float = 30.0,
        verify: bool | None = None, ca_bundle: str | None = None) -> dict:
    """GET a JSON endpoint with Basic auth. Raises RacingAPIError on failure.

    verify : None -> obey the THERACINGAPI_INSECURE env var (default secure);
             True/False -> force it.
    """
    if verify is None:
        verify = not _env_truthy("THERACINGAPI_INSECURE")
    url = BASE_URL + path
    if params:
        clean = {k: v for k, v in params.items() if v is not None}
        if clean:
            url += "?" + urllib.parse.urlencode(clean)
    req = urllib.request.Request(url, headers={
        "Authorization": _auth_header(),
        "Accept": "application/json",
        # Cloudflare blocks the default "Python-urllib/x" UA (error 1010), so
        # present a normal one.
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "HorseRacingValueAnalyser/1.0",
    })
    try:
        ctx = _ssl_context(verify, ca_bundle)
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:300]
        hint = " (check your credentials / plan)" if e.code in (401, 403) else ""
        raise RacingAPIError(f"HTTP {e.code} from {path}{hint}: {body}") from e
    except urllib.error.URLError as e:
        reason = str(e.reason)
        if "CERTIFICATE_VERIFY_FAILED" in reason:
            reason += ("\n  -> TLS verification failed (often a corporate proxy). "
                       "Fix securely by setting THERACINGAPI_CA_BUNDLE to your "
                       "proxy's root CA .pem, or bypass with --insecure / "
                       "THERACINGAPI_INSECURE=1.")
        raise RacingAPIError(f"Network error calling {path}: {reason}") from e


def fetch_racecards(day: str = "today", region: str | None = "gb",
                    endpoint: str = "free", verify: bool | None = None,
                    ca_bundle: str | None = None) -> dict:
    """Fetch racecards for a day.

    endpoint : 'free', 'standard' or 'pro' — match your subscription. Higher
               tiers include odds and trainer/jockey stats.
    region   : country filter, e.g. 'gb', 'ire'. None for all.
    verify / ca_bundle : TLS controls, see get().
    """
    return get(f"/v1/racecards/{endpoint}",
               {"day": day, "region_codes": region},
               verify=verify, ca_bundle=ca_bundle)


# --- field helpers ---------------------------------------------------------

def _first(d: dict, *keys, default=None):
    """Return the first present, non-empty value among several possible keys."""
    for k in keys:
        if k in d and d[k] not in (None, "", []):
            return d[k]
    return default


def _to_float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_int(v) -> int | None:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def parse_form(form: str, n: int = 4) -> list[int]:
    """Turn a form string like '1-2P3' or '4/213' into last_positions, most
    recent FIRST (the model's convention). Digits are finishing positions
    ('0' = 10th or worse); P/F/U/R/B/S (pulled up, fell, unseated, refused,
    brought down, slipped) become 0 = 'no completion'; separators are ignored."""
    if not form:
        return []
    positions: list[int] = []
    for ch in str(form):
        if ch.isdigit():
            positions.append(10 if ch == "0" else int(ch))
        elif ch.upper() in "PFURBS":
            positions.append(0)
        # '-', '/', spaces and anything else are separators — skip
    positions.reverse()                    # form reads oldest->newest; flip it
    return positions[:n]


def _decimal_odds(runner: dict) -> float | None:
    """A representative decimal price for the runner. Takes the median across
    whatever bookmakers are quoted; falls back to a single decimal/SP field."""
    odds = runner.get("odds")
    if isinstance(odds, list) and odds:
        decs = sorted(d for o in odds
                      if (d := _to_float(o.get("decimal"))) is not None)
        if decs:
            return decs[len(decs) // 2]
    single = _first(runner, "decimal", "sp_dec", "odds_decimal")
    return _to_float(single)


def _strike_rate(stats) -> float | None:
    """Pull a 0-1 strike rate from a stats blob like {'percent': 18} or
    {'wins': 4, 'runs': 19}."""
    if not isinstance(stats, dict):
        return None
    pct = _to_float(stats.get("percent"))
    if pct is not None:
        return pct / 100.0
    wins, runs = _to_float(stats.get("wins")), _to_float(stats.get("runs"))
    if wins is not None and runs:
        return wins / runs
    return None


def _is_handicap(race: dict) -> bool:
    text = " ".join(str(_first(race, k, default="")) for k in
                    ("race_name", "race_class", "type", "race_type")).lower()
    return "handicap" in text or " hcap" in text


def _is_flat(race: dict) -> bool:
    text = str(_first(race, "type", "race_type", "discipline", default="")).lower()
    if any(j in text for j in ("hurdle", "chase", "national hunt", "nh")):
        return False
    return True            # default to flat (draw factor stays relevant)


# --- mapping into our schema ----------------------------------------------

def map_runner(rd: dict) -> dict:
    """One API runner -> one of our runner dicts. Odds are included when the
    plan provides them; without a price the horse can still be ranked by
    ability (see pick.py), just not assessed for value."""
    out: dict = {
        "name": _first(rd, "horse", "name", "horse_name", default="Unknown"),
        "trainer": _first(rd, "trainer", default=""),
        "jockey": _first(rd, "jockey", default=""),
        "last_positions": parse_form(_first(rd, "form", default="")),
    }
    odds = _decimal_odds(rd)
    if odds is not None:
        out["odds_decimal"] = odds
    days = _to_int(_first(rd, "last_run", "days_since_run", "dslr"))
    if days is not None:
        out["days_since_run"] = days

    tsr = _strike_rate(_first(rd, "trainer_14_days", "trainer_stats"))
    if tsr is not None:
        out["trainer_sr"] = round(tsr, 3)
    jsr = _strike_rate(_first(rd, "jockey_14_days", "jockey_stats"))
    if jsr is not None:
        out["jockey_sr"] = round(jsr, 3)

    # Official rating ('ofr') is a calibrated ability measure; the speed_figure
    # factor ranks it within the field, so it's a sound relative-strength
    # signal even though it isn't a true speed figure. Prefer an actual figure
    # (RPR/Topspeed) if the plan provides one.
    fig = _to_float(_first(rd, "rpr", "ts", "ofr", "official_rating"))
    if fig is not None:
        out["speed_figure"] = fig
    lbs = _to_float(_first(rd, "lbs", "weight_lbs", "weight"))
    if lbs is not None:
        out["weight_lbs"] = lbs

    # Raw stall draw — live_enrich turns it into a draw advantage via the
    # course/distance bias table (flat only). The free tier supplies it.
    draw = _to_int(_first(rd, "draw", "stall"))
    if draw is not None:
        out["draw"] = draw

    # Raw headgear code (e.g. "b","p","v","t","h"); carried so live_enrich can
    # decide first-time-headgear against the horse's corpus history.
    hg = _first(rd, "headgear", "head_gear", default="")
    if hg:
        out["headgear"] = str(hg).strip()
        # If the feed also flags it as first-time, honour that directly.
        if str(_first(rd, "headgear_first", "hg_first", "headgear_run",
                      default="")).lower() in ("1", "true", "yes"):
            out["headgear_first_time"] = True

    # NB: trainer/jockey strike rates and run style aren't in the free racecard
    # (corpus-derived in live_enrich); course/distance/going-proven, class move
    # and draw advantage ARE now derivable because we carry the race conditions
    # (see map_race) + draw/headgear above.
    return out


def map_race(rd: dict) -> dict:
    runners = [map_runner(r) for r in rd.get("runners", [])]
    race = {
        "course": _first(rd, "course", "course_name", default="Unknown"),
        "time": _first(rd, "off_time", "off", "time", default=""),
        "name": _first(rd, "race_name", "name", default=""),
        "is_handicap": _is_handicap(rd),
        "is_flat": _is_flat(rd),
        "runners": runners,
    }
    # Today's conditions — present on the free tier and needed for the horse's
    # record-vs-today form (distance/going/class-proven, class move). Previously
    # discarded; carry them so live_enrich can derive those signals.
    dist = _to_float(_first(rd, "distance_f", "dist_f"))
    if dist is not None:
        race["distance_f"] = dist
    going = _first(rd, "going", "going_description", default="")
    if going and str(going).strip().lower() not in ("abandoned", ""):
        race["going"] = str(going).strip()
    rclass = _first(rd, "race_class", "class", default="")
    if rclass:
        race["race_class"] = str(rclass).strip()
    # Full start datetime, used to match Betfair markets to this race. Ignored
    # by the model/loader; just carried along in the JSON.
    off_dt = _first(rd, "off_dt", "off_dt_utc", "start_time")
    if off_dt:
        race["off_dt"] = off_dt
    return race


def card_dict_from_response(data: dict, name: str = "Live racecards") -> dict:
    """The full API response -> our card dict ({"name", "races": [...]}).
    Races where no runner has a price are dropped."""
    racecards = data.get("racecards", data.get("data", []))
    races = [m for rc in racecards if (m := map_race(rc))["runners"]]
    return {"name": name, "races": races}
