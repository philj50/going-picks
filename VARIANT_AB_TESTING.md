# A/B Test Prompt Variants

The Going app now supports A/B testing different prompt strategies to identify which instruction styles work best for each AI voice.

## How It Works

### Variant Selection (Deterministic Seeding)

Each voice is assigned to a **prompt variant** based on the date:
- Same voice + same date = always gets the same variant
- Same voice + different date = likely gets a different variant (rotates through 5 options)
- This ensures stable, reproducible results within a day

**Variants:**
1. **Standard** — Balance confidence and edge (baseline)
2. **Conservative** — Only back high-confidence picks (≥65% win_prob)
3. **Aggressive** — Back any pick with positive edge, maximize volume
4. **High Confidence Only** — Top-tier picks only, sit out if fewer than 4
5. **Value Hunter** — Focus on mispriced horses (overvalue by 3+ points)

### How Variants Get Injected

When a race analysis prompt is generated:

```
1. Select variant for voice+date via seeding: prompt_variants.select_variant_for_voice("claude", date)
2. Generate variant instruction text
3. Inject into prompt: "PROMPT VARIANT (Conservative): Only back horses where you are highly confident..."
4. Track variant in staking_review.json
```

### Live A/B Testing

**Daily flow:**
1. `ai_daily_analysis.py` analyzes today's races (variants auto-assigned)
2. Each voice gets their seeded variant's instruction
3. Verdicts saved with analysis
4. `staking_review.py` records which variant each voice used, tracks P&L
5. Over time: `ai_reports/staking_review.json` accumulates variant performance data

**Example staking_review.json entry:**
```json
{
  "2026-07-22": {
    "claude": {
      "n_bets": 8,
      "wins": 5,
      "pnl": 24.5,
      "variant": "aggressive",
      "variant_name": "Aggressive",
      "hit_rate": 0.625,
      ...
    }
  }
}
```

## Backtesting Variants

Before going live, backtest variants against historical data to identify winners:

```bash
# Test last 30 days
python3 backtest_variants.py

# Test custom date range
python3 backtest_variants.py --start 2026-07-01 --end 2026-07-22

# Test specific variants only
python3 backtest_variants.py --variants conservative,aggressive

# Machine-readable output
python3 backtest_variants.py --json
```

**Output:**
```
================================================================================
PROMPT VARIANT BACKTEST: 2026-06-22 to 2026-07-22
================================================================================

VARIANT SUMMARY (aggregated across all voices):
Variant                 Picks    Backed     Wins  Hit Rate        ROI/Bet
--------------------------------------------------------------------------------
Standard                  150       120       72    60.0%         1.5cr
Conservative              145       110       69    62.7%         2.1cr
Aggressive                150       150       85    56.7%         0.9cr
Value Hunter              148       105       68    64.8%         2.4cr
High Confidence Only       142        95       61    64.2%         2.3cr

✓ TOP VARIANT BY HIT RATE: Value Hunter (64.8%)
```

## Integration with Staking Review

The staking review now tracks variants:

```python
# staking_review.py automatically captures:
# - Which variant each voice was assigned
# - Performance stats (P&L, hit rate, concentration style)
# - Unbacked picks and their outcomes
```

Query staking performance by variant:

```python
import json
from pathlib import Path

data = json.loads(Path("ai_reports/staking_review.json").read_text())

# Group by variant
by_variant = {}
for date, day_data in data.items():
    for voice, stats in day_data.items():
        variant = stats.get("variant", "unknown")
        if variant not in by_variant:
            by_variant[variant] = {"wins": 0, "bets": 0}
        by_variant[variant]["bets"] += stats.get("n_bets", 0)
        by_variant[variant]["wins"] += stats.get("wins", 0)

# Print results
for v, stats in sorted(by_variant.items(), key=lambda x: x[1]["wins"]/(x[1]["bets"]+0.001), reverse=True):
    hit_rate = stats["wins"] / stats["bets"] if stats["bets"] else 0
    print(f"{v}: {hit_rate*100:.1f}% ({stats['wins']}/{stats['bets']})")
```

## Interpreting Results

**High hit rate but low ROI:**
- Variant picks winners but at short odds
- Good for confidence but not profitable
- Use for conservative portfolios

**Low hit rate but high ROI:**
- Variant finds long-shot value
- Fewer wins but bigger payoffs
- High variance, high risk

**Balanced:**
- Similar hit rate to volume (expected ~50% + edge)
- Good ROI per bet
- Stable performer

## Customizing Variants

To add or modify variants, edit `prompt_variants.py`:

```python
VARIANTS = {
    "my_strategy": {
        "name": "My Strategy",
        "instruction": "Your custom instruction text here...",
    },
    ...
}
```

Changes take effect immediately for new analyses.

## Next Steps

1. **Week 1:** Run backtest on 30 days of historical data
2. **Week 2-3:** Deploy live with all 5 variants (blind A/B test)
3. **Week 4:** Analyze staking_review.json results
4. **Week 5:** Either:
   - Retire low-performers, keep top 2-3 variants
   - Or fine-tune low-performers and retest

## Files

- `prompt_variants.py` — Variant definitions and seeding logic
- `backtest_variants.py` — Historical backtest CLI tool
- `test_prompt_variants.py` — Unit tests for variant selection
- `test_backtest_variants.py` — Tests for backtest module
- Modified: `ai_daily_analysis.py`, `staking_review.py` — Integration points
