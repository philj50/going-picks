"""
One-shot fetch: grab racecards from The Racing API and write a card JSON that
`analyse.py` can run — so you're not hand-building cards.

Typical use
    python fetch_today.py                      # today's GB cards -> today.json
    python fetch_today.py --day tomorrow       # tomorrow's cards
    python fetch_today.py --region ire         # Irish cards
    python fetch_today.py --endpoint pro       # if your plan includes odds/stats
    python fetch_today.py --out cards/sat.json # choose the output file

Then:
    python analyse.py today.json 100

Calibrating the field mapping (do this once)
    python fetch_today.py --raw --out raw.json     # dump the raw API response
    python fetch_today.py --from-raw raw.json      # map a saved response offline

`--from-raw` needs no network or credentials — handy for checking the mapping
or re-mapping after you tweak racing_api.py.

Credentials: set THERACINGAPI_USERNAME / THERACINGAPI_PASSWORD (see racing_api.py).
"""
from __future__ import annotations
import argparse
import json
import sys

import racing_api


def _write_json(obj: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _merge_betfair(card: dict, args) -> None:
    """Fetch Betfair WIN odds and merge them into the card in place."""
    import betfair_api
    try:
        markets = betfair_api.list_win_markets(
            country=args.region.upper(),
            verify=False if args.insecure else None,
            ca_bundle=args.ca_bundle)
    except betfair_api.BetfairError as e:
        print(f"⚠ Betfair odds skipped: {e}", file=sys.stderr)
        return
    betfair_api.merge_into_card(card, markets)
    m = card.pop("_merge", {})
    print(f"  Betfair: matched {m.get('races_matched', 0)} races, priced "
          f"{m.get('runners_priced', 0)}/{m.get('runners_total', 0)} runners "
          f"from {m.get('markets', 0)} markets")


def _summarise(card: dict, out_path: str) -> None:
    races = card.get("races", [])
    runners = sum(len(r["runners"]) for r in races)
    priced = sum(1 for r in races for rn in r["runners"] if rn.get("odds_decimal"))
    print(f"  {len(races)} races, {runners} runners ({priced} with a price)")
    if not races:
        print("  ⚠ No races returned. Try --raw to inspect the response.")
        return
    if priced == 0:
        print("  ⓘ No odds in this data (free tier omits them). You can still")
        print(f"    rank by ability:  python pick.py {out_path}")
        print("    Then get odds manually and add them to the JSON for value.")
    for r in races[:12]:
        print(f"    {r['course']:<14} {r['time']:<6} "
              f"{len(r['runners']):>2} runners  {r['name'][:34]}")
    if len(races) > 12:
        print(f"    … and {len(races) - 12} more")


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch racecards into a card JSON.")
    ap.add_argument("--day", default="today",
                    help="today or tomorrow (default: today)")
    ap.add_argument("--region", default="gb",
                    help="region code, e.g. gb, ire (default: gb)")
    ap.add_argument("--regions", default="",
                    help="comma-separated regions to merge, e.g. gb,ire (overrides --region)")
    ap.add_argument("--endpoint", default="free",
                    help="free | standard | pro — match your plan (default: free)")
    ap.add_argument("--out", default="today.json",
                    help="output file (default: today.json)")
    ap.add_argument("--raw", action="store_true",
                    help="save the raw API response instead of a mapped card")
    ap.add_argument("--from-raw", metavar="FILE",
                    help="map a previously saved raw response (no network)")
    ap.add_argument("--ca-bundle", metavar="PEM",
                    help="trust this extra root CA (secure fix for a TLS proxy)")
    ap.add_argument("--insecure", action="store_true",
                    help="skip TLS verification (last resort behind a broken proxy)")
    ap.add_argument("--with-betfair", action="store_true",
                    help="merge live Betfair odds into the card (needs Betfair creds)")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    card_name = f"{args.day} racecards ({args.region})"

    regions = [r.strip() for r in (args.regions or "").split(",") if r.strip()]
    if not regions:
        regions = [args.region]

    def _fetch_one(region: str):
        return racing_api.fetch_racecards(
            day=args.day, region=region, endpoint=args.endpoint,
            verify=False if args.insecure else None, ca_bundle=args.ca_bundle)

    # Offline path: map a saved raw response.
    if args.from_raw:
        with open(args.from_raw, encoding="utf-8") as f:
            data = json.load(f)
        card = racing_api.card_dict_from_response(data, name=card_name)
        if args.with_betfair:
            _merge_betfair(card, args)
        _write_json(card, args.out)
        print(f"Mapped {args.from_raw} -> {args.out}")
        _summarise(card, args.out)
        return 0

    if args.insecure:
        print("⚠ TLS verification disabled (--insecure). The connection is not "
              "cryptographically trusted.", file=sys.stderr)

    # Live path: hit the API (merge multiple regions when requested).
    try:
        if len(regions) == 1:
            data = _fetch_one(regions[0])
            card = racing_api.card_dict_from_response(
                data, name=f"{args.day} racecards ({regions[0]})")
        else:
            merged_races = []
            for reg in regions:
                data = _fetch_one(reg)
                part = racing_api.card_dict_from_response(
                    data, name=f"{args.day} racecards ({reg})")
                merged_races.extend(part.get("races") or [])
            card = {
                "name": f"{args.day} racecards ({','.join(regions)})",
                "date": merged_races[0].get("date") if merged_races else None,
                "races": merged_races,
            }
    except racing_api.RacingAPIError as e:
        print(f"Fetch failed: {e}", file=sys.stderr)
        return 1

    if args.raw:
        _write_json(data, args.out)
        n = len(data.get("racecards", data.get("data", [])))
        print(f"Saved raw response ({n} racecards) -> {args.out}")
        print("Inspect the field names, then adjust racing_api.py if needed.")
        return 0

    if args.with_betfair:
        _merge_betfair(card, args)
    _write_json(card, args.out)
    print(f"Wrote {args.out}")
    _summarise(card, args.out)
    print(f"\nNext:  python analyse.py {args.out} 100")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
