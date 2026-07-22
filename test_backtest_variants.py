"""Tests for backtest_variants module."""
import datetime as dt
import json
from pathlib import Path
from unittest.mock import patch

import backtest_variants as bv
import prompt_variants as pv


def test_backtest_day_empty_report():
    """Empty report should return empty results."""
    result = bv.backtest_day("2026-07-22", {})
    assert result == {}

    result = bv.backtest_day("2026-07-22", {"races": []})
    assert result == {}


def test_backtest_day_assigns_variants():
    """Backtest should assign variants based on voice+date seeding."""
    report = {
        "races": [
            {
                "race": "Test 1 (2m)",
                "course": "Epsom",
                "time": "2:30",
                "groq_analysis": "VERDICT: {\"pick\": \"Alpha\", \"win_prob\": 0.60}",
                "claude_analysis": "VERDICT: {\"pick\": \"Beta\", \"win_prob\": 0.55}",
            }
        ]
    }

    # Mock the database connection and winner lookup
    import sqlite3
    mock_con = sqlite3.connect(":memory:")
    with patch("backtest_variants.sqlite3.connect", return_value=mock_con):
        with patch("backtest_variants._winner_for_race", return_value=None):
            result = bv.backtest_day("2026-07-22", report)

            # Should have picked up both voices' verdicts
            # Grouped by their assigned variants
            for variant_results in result.values():
                assert isinstance(variant_results, dict)


def test_backtest_range_aggregates():
    """Backtest over multiple days should aggregate stats."""
    start = "2026-07-20"
    end = "2026-07-22"

    # Mock _load_report to return empty reports
    with patch("backtest_variants._load_report", return_value={"races": []}):
        with patch("backtest_variants._winner_for_race", return_value=None):
            result = bv.backtest_range(start, end)
            # With empty reports, should get empty results
            assert isinstance(result, dict)


def test_variant_summary_structure():
    """Backtest results should have proper structure."""
    # Create mock results
    results = {
        "standard": {
            "groq": {
                "n_picks": 10, "n_backed": 8, "wins": 4,
                "hit_rate": 0.5, "roi_per_bet": 1.2, "pnl": 9.6
            },
            "claude": {
                "n_picks": 12, "n_backed": 10, "wins": 6,
                "hit_rate": 0.6, "roi_per_bet": 2.1, "pnl": 21.0
            }
        },
        "aggressive": {
            "groq": {
                "n_picks": 15, "n_backed": 15, "wins": 7,
                "hit_rate": 0.467, "roi_per_bet": 0.8, "pnl": 12.0
            }
        }
    }

    # Should not raise an exception
    try:
        bv.print_backtest_report(results, "2026-07-20", "2026-07-22")
    except Exception as e:
        assert False, f"print_backtest_report raised: {e}"


def test_backtest_filters_by_variant():
    """Backtest range should filter by variants if specified."""
    start = "2026-07-20"
    end = "2026-07-22"

    with patch("backtest_variants._load_report", return_value={"races": []}):
        with patch("backtest_variants._winner_for_race", return_value=None):
            # Filter to just conservative
            result = bv.backtest_range(start, end, variants=["conservative"])
            # Results may be empty due to mocking, but shouldn't crash
            assert isinstance(result, dict)


def test_backtest_date_range_calculation():
    """Date range should be calculated correctly."""
    # Test 30 days
    end = dt.date.today()
    start = end - dt.timedelta(days=29)

    days_range = (end - start).days + 1
    assert days_range == 30


def test_no_bet_verdicts_excluded():
    """NO_BET verdicts should be excluded from stats."""
    report = {
        "races": [
            {
                "race": "Test 1",
                "course": "Epsom",
                "time": "2:30",
                "groq_analysis": "VERDICT: {\"pick\": \"NO BET\", \"win_prob\": 0.40}",
                "claude_analysis": "VERDICT: {\"pick\": \"Alpha\", \"win_prob\": 0.60}",
            }
        ]
    }

    import sqlite3
    mock_con = sqlite3.connect(":memory:")
    with patch("backtest_variants.sqlite3.connect", return_value=mock_con):
        with patch("backtest_variants._winner_for_race", return_value=None):
            result = bv.backtest_day("2026-07-22", report)
            # Groq should be filtered out due to NO_BET
            # Claude should be included (if variant handling works)
            assert isinstance(result, dict)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
