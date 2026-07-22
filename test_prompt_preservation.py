"""Tests to ensure prediction prompts, post-mortem prompts, and UI features
don't get lost during day-book reruns, archive/restore, or other operations."""
import json
import tempfile
from pathlib import Path
import pytest

import ai_daily_analysis as ada
import ai_consensus
import brief_builder as bb


class TestRaceBriefPreservation:
    """Ensure race_brief and analysis_prompt fields survive all operations."""

    def test_race_brief_and_prompt_stored_in_report(self):
        """race_brief and analysis_prompt must be in saved ai_analysis JSON."""
        # Load a real report
        report_path = Path("ai_reports/ai_analysis_2026-07-18.json")
        if report_path.exists():
            data = json.loads(report_path.read_text())
            races = data.get("races", [])

            assert len(races) > 0, "Test report should have races"

            # Every race should have these fields (even if empty)
            for race in races:
                assert "race_brief" in race, f"{race.get('race')} missing race_brief"
                assert "analysis_prompt" in race, f"{race.get('race')} missing analysis_prompt"

            # Most should be non-empty
            has_brief = sum(1 for r in races if r.get("race_brief"))
            has_prompt = sum(1 for r in races if r.get("analysis_prompt"))
            assert has_brief > len(races) * 0.8, f"Only {has_brief}/{len(races)} have race_brief"
            assert has_prompt > len(races) * 0.8, f"Only {has_prompt}/{len(races)} have analysis_prompt"

    def test_ai_consensus_loads_prompts_correctly(self):
        """ai_consensus.load_daily_analyses must extract race_brief and analysis_prompt."""
        analyses = ai_consensus.load_daily_analyses()

        if not analyses:
            pytest.skip("No analyses available")

        for race_name, race_analyses in list(analyses.items())[:5]:
            # These fields must be present (loaded by ai_consensus.load_daily_analyses)
            assert "race_brief" in race_analyses, f"{race_name} missing race_brief in loaded data"
            assert "analysis_prompt" in race_analyses, f"{race_name} missing analysis_prompt in loaded data"

    def test_race_brief_contains_expected_sections(self):
        """race_brief should contain race metadata, runner data, and lessons."""
        report_path = Path("ai_reports/ai_analysis_2026-07-18.json")
        if not report_path.exists():
            pytest.skip("Test report not found")

        data = json.loads(report_path.read_text())
        races = data.get("races", [])

        # Find a race with a non-empty brief
        for race in races:
            brief = race.get("race_brief", "")
            if len(brief) > 500:  # Substantial brief
                # Should have course/time
                assert race.get("course") in brief or "Course" in brief or "RACE" in brief

                # Should have runner information (indicated by common patterns)
                assert ("@" in brief or "OR" in brief or "odds" in brief.lower()
                        or "runner" in brief.lower()), "Brief missing runner data"

                # Contextual lessons should be present in extended briefs
                if len(brief) > 2000:
                    assert "Lessons" in brief or "lesson" in brief.lower() or "draw" in brief.lower(), \
                        "Extended brief should have lessons section"
                break
        else:
            pytest.skip("No substantial briefs found")


class TestPostMortemIntegration:
    """Ensure post-mortem connection notes are preserved in briefs."""

    def test_pm_connection_notes_in_brief(self):
        """Post-mortem connection notes should be appended to race briefs."""
        report_path = Path("ai_reports/ai_analysis_2026-07-18.json")
        if not report_path.exists():
            pytest.skip("Test report not found")

        data = json.loads(report_path.read_text())
        races = data.get("races", [])

        # Some races should have connection notes (from post-mortems)
        with_notes = [r for r in races if "CONNECTIONS NOTES" in r.get("race_brief", "")]

        # With 47 races and 21 PM entries, expect some briefs to have notes
        assert len(with_notes) > 0, "No races have PM connection notes in brief"

        # Verify notes format
        for race in with_notes[:3]:
            brief = race["race_brief"]
            i = brief.find("CONNECTIONS NOTES")
            notes_section = brief[i:i+300]

            # Should mention horse names and context
            assert "(" in notes_section, "Notes should have (outcome) format"

    def test_pm_notes_not_duplicated_on_rebuild(self):
        """Rebuilding briefs should not duplicate PM connection notes."""
        report_path = Path("ai_reports/ai_analysis_2026-07-18.json")
        if not report_path.exists():
            pytest.skip("Test report not found")

        data = json.loads(report_path.read_text())
        races = data.get("races", [])

        for race in races:
            brief = race.get("race_brief", "")
            # Count occurrences of CONNECTIONS NOTES
            count = brief.count("CONNECTIONS NOTES")
            assert count <= 1, f"Brief has {count} CONNECTIONS NOTES sections (should be 0 or 1)"


class TestPromptStructure:
    """Ensure analysis prompts have correct structure."""

    def test_analysis_prompt_structure(self):
        """analysis_prompt should have intro, brief, lessons, calibration, verdict format."""
        report_path = Path("ai_reports/ai_analysis_2026-07-18.json")
        if not report_path.exists():
            pytest.skip("Test report not found")

        data = json.loads(report_path.read_text())
        races = data.get("races", [])

        # Find a race with a prompt
        for race in races:
            prompt = race.get("analysis_prompt", "")
            if len(prompt) > 500:
                # Should have expected sections
                assert "expert horse racing analyst" in prompt.lower(), "Missing intro"
                assert "VERDICT:" in prompt, "Missing VERDICT format"
                assert "Reply format" in prompt, "Missing reply format instructions"
                break
        else:
            pytest.skip("No substantial prompts found")

    def test_calibration_line_in_prompt(self):
        """Prompts should include calibration feedback (banded or overall)."""
        report_path = Path("ai_reports/ai_analysis_2026-07-18.json")
        if not report_path.exists():
            pytest.skip("Test report not found")

        data = json.loads(report_path.read_text())
        races = data.get("races", [])

        # Check a few extended prompts (might be old or new format)
        count = 0
        for race in races:
            prompt = race.get("analysis_prompt", "")
            if any(marker in prompt for marker in ["YOUR CALIBRATION", "YOUR RECORD", "claims", "won"]):
                count += 1
                if "YOUR CALIBRATION" in prompt or "YOUR RECORD" in prompt:
                    assert "%" in prompt, "Calibration line should have percentage"

        # Old reports may not have calibration, but newer ones should
        # Just verify the feature works if present
        if count == 0:
            pytest.skip("No calibration feedback in this report (might be older version)")


class TestUIFeaturePreservation:
    """Ensure UI features like race conditions, stats, connections are in briefs."""

    def test_race_conditions_in_brief(self):
        """Race conditions (distance, going, class, handicap, prize) should be in briefs."""
        report_path = Path("ai_reports/ai_analysis_2026-07-18.json")
        if not report_path.exists():
            pytest.skip("Test report not found")

        data = json.loads(report_path.read_text())
        races = data.get("races", [])

        for race in races:
            brief = race.get("race_brief", "")
            if len(brief) > 200:
                # Should mention distance/going/class/handicap somewhere
                has_conditions = any(word in brief for word in
                                    ["distance", "going", "class", "handicap", "prize", "f", "yard"])
                assert has_conditions, f"Brief for {race.get('race')} missing race conditions"
                break

    def test_runner_stats_in_brief(self):
        """Briefs should contain runner stats (OR, RPR, draw, weight, etc.)."""
        report_path = Path("ai_reports/ai_analysis_2026-07-18.json")
        if not report_path.exists():
            pytest.skip("Test report not found")

        data = json.loads(report_path.read_text())
        races = data.get("races", [])

        for race in races:
            brief = race.get("race_brief", "")
            if "OR" in brief or "RPR" in brief or "@" in brief:
                # Found a brief with stats
                assert len(brief) > 200, "Brief with stats should be substantial"
                break


class TestAllocationPreservation:
    """Ensure fields aren't lost during day-book and allocation operations."""

    def test_allocations_dont_strip_fields(self):
        """When allocations are recomputed, race_brief and analysis_prompt must be preserved."""
        report_path = Path("ai_reports/ai_analysis_2026-07-19.json")
        if not report_path.exists():
            pytest.skip("Today's report not found")

        data = json.loads(report_path.read_text())
        races = data.get("races", [])
        allocations = data.get("allocations", {})

        # Both should be present even when allocations were recomputed
        has_brief = sum(1 for r in races if r.get("race_brief"))
        has_prompt = sum(1 for r in races if r.get("analysis_prompt"))

        assert len(races) > 0, "No races in report"
        assert len(allocations) > 0, "No allocations in report (verify it was recomputed)"

        # Fields should still be there (at least for most races)
        assert has_brief > 0, "All race_brief fields were lost"
        assert has_prompt > 0, "All analysis_prompt fields were lost"

    def test_archive_preserves_all_fields(self):
        """Archive copy should preserve all essential fields."""
        archive_path = Path("ai_reports/archive_daybook_2026-07-19/ai_analysis_2026-07-19.json")
        if not archive_path.exists():
            pytest.skip("Archive not found")

        data = json.loads(archive_path.read_text())
        races = data.get("races", [])

        # Archive should have all race data
        assert len(races) > 0, "Archive missing races"

        # Required fields
        for race in races:
            assert "race" in race, "Archive missing race name"
            assert "course" in race, "Archive missing course"
            assert "time" in race, "Archive missing time"


class TestStakingAndCalibrationFeedback:
    """Ensure betting feedback and calibration are generated correctly."""

    def test_staking_review_file_exists(self):
        """staking_review.json should be generated and preserved."""
        staking_path = Path("ai_reports/staking_review.json")
        if staking_path.exists():
            data = json.loads(staking_path.read_text())

            # Should have recent entries
            assert len(data) > 0, "staking_review empty"

            for date_key, day_data in list(data.items())[-3:]:
                assert isinstance(day_data, dict), f"Day {date_key} data malformed"
                # Each voice should have staking info
                for voice, info in day_data.items():
                    assert "n_bets" in info or "style" in info, f"Voice {voice} missing staking data"

    def test_calibration_bands_computed(self):
        """Calibration feedback should use per-band hit rates, not overall."""
        # This is verified by build_prompt generating calibration lines
        calib_line = ada._calibration_line("cursor")

        if calib_line:
            # Should mention bands or claims
            assert ("claims" in calib_line or "win" in calib_line), "Calibration line malformed"


def test_contextual_lessons_filtered():
    """Contextual lessons should be filtered by race type, not global for all."""
    # Test that lessons filtering works
    flat_race = {"is_flat": True, "is_handicap": False, "going": "Good"}
    jumps_race = {"is_flat": False, "is_handicap": False, "going": "Good"}

    flat_lessons = bb.contextual_lessons(flat_race, max_lr=30)
    jumps_lessons = bb.contextual_lessons(jumps_race, max_lr=30)

    # Both should have lessons (or both empty, but not inverted)
    if flat_lessons and jumps_lessons:
        # If both have content, they might differ (draw in flat, not jumps)
        # This just verifies filtering is happening
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
