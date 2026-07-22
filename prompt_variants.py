"""A/B test prompt variants to optimize prediction accuracy and ROI.

Each voice is randomly assigned to a variant each day (seeded by date+voice for
stability within day). Variants modify the analysis prompt's instructions and
emphasis. Staking review tracks performance by variant to identify which strategies
work best per voice.
"""
import hashlib
import datetime as dt


VARIANTS = {
    "standard": {
        "name": "Standard",
        "instruction": "Balance confidence and edge. Back picks where you believe the win probability justifies the odds.",
    },
    "conservative": {
        "name": "Conservative",
        "instruction": "Only back horses where you are highly confident (≥65% win probability). Prefer picks with clear, obvious edge. Sit out races where you're unsure.",
    },
    "aggressive": {
        "name": "Aggressive",
        "instruction": "Back any pick with positive edge, even if confidence is moderate. Maximize the number of picks with edge >0. Concentration is fine.",
    },
    "high_confidence_only": {
        "name": "High Confidence Only",
        "instruction": "Only back top-tier, unambiguous picks. If you have fewer than 4 truly strong picks today, consider sitting out entirely.",
    },
    "value_hunter": {
        "name": "Value Hunter",
        "instruction": "Focus on finding mispriced horses—picks where your win probability exceeds the market-implied probability by 3+ points. Look for value in overlooked runners.",
    },
}


def select_variant_for_voice(voice: str, date: dt.date = None) -> str:
    """Deterministically select a variant for a voice on a given date.

    Same voice gets same variant all day; different day = different variant.
    Seeded by (date, voice) so it's reproducible.
    """
    if date is None:
        date = dt.date.today()

    seed_str = f"{date.isoformat()}:{voice}"
    hash_val = int(hashlib.md5(seed_str.encode()).hexdigest(), 16)
    variant_keys = sorted(VARIANTS.keys())
    return variant_keys[hash_val % len(variant_keys)]


def get_variant_instruction(variant_key: str) -> str:
    """Get the instruction text for a variant."""
    return VARIANTS.get(variant_key, VARIANTS["standard"])["instruction"]


def variant_modifier_for_prompt(voice: str, date: dt.date = None) -> str:
    """Return the variant instruction to inject into the analysis prompt."""
    variant = select_variant_for_voice(voice, date)
    instruction = get_variant_instruction(variant)
    return f"\n\nPROMPT VARIANT ({VARIANTS[variant]['name']}): {instruction}"
