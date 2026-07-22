"""Tests for the win_prob/NO BET verdict schema in ai_consensus.extract_verdict."""
import ai_consensus as ac


def test_new_schema_win_prob():
    text = ('VERDICT: {"pick": "Alpha", "win_prob": 0.42, "second": "Beta", '
            '"agrees_with_model": true, "key_risk": "soft ground", '
            '"missing_factors": []}\nOne sentence.')
    v = ac.extract_verdict(text)
    assert v["pick"] == "Alpha"
    assert abs(v["win_prob"] - 0.42) < 1e-9
    assert v["confidence"] == 42          # derived for old consumers
    assert v["no_bet"] is False
    assert v["agrees_with_model"] is True


def test_old_schema_confidence_still_works():
    text = 'VERDICT: {"pick": "Alpha", "confidence": 70, "agrees_with_model": false}'
    v = ac.extract_verdict(text)
    assert v["confidence"] == 70
    assert abs(v["win_prob"] - 0.70) < 1e-9
    assert v["no_bet"] is False


def test_win_prob_given_as_percent_is_normalised():
    text = 'VERDICT: {"pick": "Alpha", "win_prob": 62}'
    v = ac.extract_verdict(text)
    assert abs(v["win_prob"] - 0.62) < 1e-9
    assert v["confidence"] == 62


def test_no_bet_detected():
    for pick in ("NO BET", "no bet", "No_Bet", "PASS", "none"):
        v = ac.extract_verdict(
            f'VERDICT: {{"pick": "{pick}", "win_prob": 0.1}}')
        assert v is not None and v["no_bet"] is True, pick
    assert ac.is_no_bet({"pick": "NO BET"}) is True
    assert ac.is_no_bet({"pick": "Alpha"}) is False


def test_agrees_with_model_null_passes_through():
    v = ac.extract_verdict(
        'VERDICT: {"pick": "Alpha", "win_prob": 0.3, "agrees_with_model": null}')
    assert v["agrees_with_model"] is None


def test_truncated_fallback_with_win_prob():
    text = 'VERDICT: {"pick": "Alpha", "win_prob": 0.35, "second": "Be'
    v = ac.extract_verdict(text)
    assert v["pick"] == "Alpha"
    assert abs(v["win_prob"] - 0.35) < 1e-9
    assert v["confidence"] == 35


def test_consensus_skips_no_bet_votes():
    analyses = {k: "" for k in ac.analysis_voice_keys()}
    analyses["groq"] = 'VERDICT: {"pick": "Alpha", "win_prob": 0.4}'
    analyses["cerebras"] = 'VERDICT: {"pick": "NO BET", "win_prob": 0.1}'
    out = ac.compute_consensus("race", analyses, ["Alpha", "Beta"])
    assert out["picks"]["groq"] == "Alpha"
    assert out["picks"]["cerebras"] is None
    assert out["ai_votes"].get("Alpha") == 1
    assert "NO BET" not in out["ai_votes"]
