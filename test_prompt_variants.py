"""Tests for A/B prompt variant selection and application."""
import datetime as dt
import prompt_variants as pv


def test_variant_selection_stable_within_day():
    """Same voice+date should always get the same variant."""
    date = dt.date(2026, 7, 22)
    for voice in ["claude", "groq", "openai"]:
        v1 = pv.select_variant_for_voice(voice, date)
        v2 = pv.select_variant_for_voice(voice, date)
        assert v1 == v2, f"{voice} got different variants on same day"


def test_variant_changes_across_days():
    """Same voice on different dates should (probably) get different variants."""
    voice = "claude"
    dates = [
        dt.date(2026, 7, 20),
        dt.date(2026, 7, 21),
        dt.date(2026, 7, 22),
    ]
    variants = [pv.select_variant_for_voice(voice, d) for d in dates]
    # With 5 variants and 3 dates, we should see variation (not guaranteed but likely)
    assert len(set(variants)) > 1, "Variants should vary across days"


def test_variant_selection_distributed():
    """Variants should be roughly distributed across voices and days."""
    voices = ["claude", "openai", "groq", "gemini", "cerebras"]
    dates = [dt.date(2026, 7, d) for d in range(1, 31)]

    counts = {}
    for voice in voices:
        for date in dates:
            var = pv.select_variant_for_voice(voice, date)
            counts[var] = counts.get(var, 0) + 1

    # All variants should appear at least once in 150 samples (5 voices × 30 days)
    assert len(counts) == len(pv.VARIANTS), f"Not all variants used: {counts}"


def test_variant_modifier_includes_instruction():
    """Variant modifier should include the instruction text."""
    date = dt.date(2026, 7, 22)
    voice = "claude"
    modifier = pv.variant_modifier_for_prompt(voice, date)

    # Should include variant name and instruction
    assert "PROMPT VARIANT" in modifier
    assert ":" in modifier  # Format is "PROMPT VARIANT (Name): instruction"

    # Should contain actual instruction text from variant
    variant = pv.select_variant_for_voice(voice, date)
    instruction = pv.get_variant_instruction(variant)
    assert instruction in modifier


def test_all_variants_defined():
    """All variants should have required fields."""
    required = ["name", "instruction"]
    for key, variant in pv.VARIANTS.items():
        for field in required:
            assert field in variant, f"Variant {key} missing {field}"
        assert variant["instruction"], f"Variant {key} has empty instruction"
        assert variant["name"], f"Variant {key} has empty name"


def test_variant_keys_stable():
    """Variant keys should match VARIANTS dict."""
    keys = sorted(pv.VARIANTS.keys())
    assert len(keys) >= 3, "Need at least 3 variants for good A/B distribution"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
