"""Tests for the guardrails: each check must be able to FAIL, and must pass on real engine output."""

import math
from dataclasses import replace

import pytest

from src.guardrails import (
    MAX_QUERY_CHARS, MAX_RESULTS, errors, find_contradictions, validate_count, validate_overrides,
    validate_request, validate_results, validate_sources,
)
from src.intent import Constraint, catalog_vocabulary
from src.main import PROFILES, SONGS_CSV
from src.recommender import (
    DiversityConfig, Recommendation, RANKING_MODES, ScoringWeights, load_songs, rank_songs,
)

SONGS = load_songs(str(SONGS_CSV))
GENRES, MOODS = catalog_vocabulary(SONGS)
PREFS = {"genre": "lofi", "mood": "chill", "energy": 0.4, "likes_acoustic": True}
WEIGHTS = ScoringWeights()


def good(k=5, prefs=PREFS, diversity=None, songs=SONGS):
    return rank_songs(prefs, songs, k, WEIGHTS, diversity)


def check(recs, k=5, prefs=PREFS, diversity=None, constraints=(), candidates=SONGS, variety=None):
    return validate_results(recs, catalog=SONGS, candidates=candidates, prefs=prefs, weights=WEIGHTS,
                            diversity=diversity, constraints=constraints, k=k, variety_requested=variety)


def codes(issues, severity=None):
    return {i.code for i in issues if severity is None or i.severity == severity}


# --- request checks -----------------------------------------------------------------------

def test_a_normal_request_is_cleaned_and_accepted():
    clean, issues = validate_request("  calm\tmusic \n for   studying  ")
    assert clean == "calm music for studying" and issues == []


@pytest.mark.parametrize("query, code", [
    (None, "INVALID_QUERY"), (42, "INVALID_QUERY"), ("", "EMPTY_QUERY"), ("   \n\t ", "EMPTY_QUERY"),
    ("\x00\x01", "EMPTY_QUERY"), ("x" * (MAX_QUERY_CHARS + 1), "QUERY_TOO_LONG"),
])
def test_bad_requests_are_rejected_with_a_specific_reason(query, code):
    _, issues = validate_request(query)
    assert codes(errors(issues)) == {code}


def test_control_characters_are_replaced_not_passed_through():
    clean, issues = validate_request("jazz\x00music\x07now")
    assert clean == "jazz music now" and issues == []


@pytest.mark.parametrize("count", [0, -1, MAX_RESULTS + 1, 3.5, "3", True])
def test_invalid_counts_are_rejected(count):
    assert codes(validate_count(count)) == {"INVALID_COUNT"}


@pytest.mark.parametrize("count", [None, 1, 5, MAX_RESULTS])
def test_valid_counts_are_accepted(count):
    assert validate_count(count) == []


def test_override_validation():
    assert validate_overrides({}, GENRES, MOODS) == []
    assert validate_overrides({"energy": 0.9, "genre": "jazz", "likes_acoustic": True}, GENRES, MOODS) == []
    for bad in ({"energy": 7}, {"loudness": 1}, {"genre": "reggaeton"}, {"mood": "sleepy"}, {"likes_acoustic": "yes"}):
        assert codes(validate_overrides(bad, GENRES, MOODS)) == {"INVALID_PREFERENCE"}, bad


def test_contradiction_detection():
    ban = Constraint("exclude", "genre", "!=", "pop", "t")
    low = Constraint("numeric", "energy", "<=", 0.3, "t")
    high = Constraint("numeric", "energy", ">=", 0.8, "t")
    assert find_contradictions({"genre": "pop"}, [ban]) == ["genre 'pop' is both requested and excluded"]
    assert "cannot both be true" in find_contradictions({}, [low, high])[0]
    assert "outside the hard limit" in find_contradictions({"energy": 0.9}, [low])[0]
    assert find_contradictions({"energy": 0.2}, [low]) == []
    assert find_contradictions({"genre": "jazz"}, [ban]) == []


# --- result checks on real engine output ------------------------------------------------------

@pytest.mark.parametrize("mode", list(RANKING_MODES))
def test_real_engine_output_passes_for_every_profile_and_mode(mode):
    weights = RANKING_MODES[mode].weights
    for _, prefs in PROFILES:
        recs = rank_songs(prefs, SONGS, 5, weights)
        issues = validate_results(recs, catalog=SONGS, candidates=SONGS, prefs=prefs, weights=weights,
                                  diversity=None, constraints=(), k=5)
        assert errors(issues) == [], (mode, prefs)


def test_real_engine_output_with_variety_penalties_passes():
    diversity = DiversityConfig(artist_penalty=0.75, genre_penalty=0.5)
    for _, prefs in PROFILES:
        recs = rank_songs(prefs, SONGS, 8, WEIGHTS, diversity)
        issues = validate_results(recs, catalog=SONGS, candidates=SONGS, prefs=prefs, weights=WEIGHTS,
                                  diversity=diversity, constraints=(), k=8)
        assert errors(issues) == []


def test_display_rounding_is_tolerated_but_real_inconsistency_is_not():
    # Reasons are printed with 2 decimals, so the listed points rarely add to the score exactly.
    recs = good(30)
    assert any(abs(sum(float(r.split("(")[-1].rstrip(")")) for r in rec.reasons) - rec.score) > 1e-12 for rec in recs)
    assert errors(check(recs, k=30)) == []
    off_by_a_lot = [replace(recs[0], reasons=tuple(r.replace("(+", "(+9") for r in recs[0].reasons))] + recs[1:5]
    assert "EXPLANATION_MISMATCH" in codes(check(off_by_a_lot[:5]))


# --- fault injection: every one of these must be caught -----------------------------------------

def test_duplicate_song_is_caught():
    recs = good()
    assert "DUPLICATE_SONG" in codes(check([recs[0], recs[0]] + recs[1:4]))


def test_nonexistent_song_is_caught():
    recs = good()
    ghost = dict(recs[0].song, id=999, title="Imaginary Anthem")
    assert "UNKNOWN_SONG" in codes(check([replace(recs[0], song=ghost)] + recs[1:]))


def test_a_catalog_song_with_altered_fields_is_caught():
    recs = good()
    altered = dict(recs[1].song, energy=0.99)
    assert "UNKNOWN_SONG" in codes(check([recs[0], replace(recs[1], song=altered)] + recs[2:]))


def test_hard_constraint_violation_is_caught():
    recs = good()
    constraint = Constraint("exclude", "genre", "!=", recs[0].song["genre"], "t")
    assert "CONSTRAINT_VIOLATION" in codes(check(recs, constraints=[constraint]))


def test_a_wrong_score_is_caught():
    recs = good()
    assert "SCORE_MISMATCH" in codes(check([replace(recs[0], score=recs[0].score + 1.0)] + recs[1:]))


def test_non_finite_scores_are_caught():
    recs = good()
    assert "SCORE_INVALID" in codes(check([replace(recs[0], score=math.nan)] + recs[1:]))


def test_an_explanation_belonging_to_another_song_is_caught():
    recs = good()
    swapped = [replace(recs[0], reasons=recs[1].reasons)] + recs[1:]
    assert "EXPLANATION_MISMATCH" in codes(check(swapped))


def test_wrong_order_too_many_and_shortfall_are_caught():
    recs = good()
    assert "ORDER_VIOLATION" in codes(check(list(reversed(recs))))
    assert "TOO_MANY_RESULTS" in codes(check(good(5), k=3))
    assert "RESULT_SHORTFALL" in codes(check(recs[:2]))


def test_fewer_results_are_fine_when_candidates_really_ran_out():
    few = SONGS[:2]
    issues = check(good(5, songs=few), k=5, candidates=few)
    assert errors(issues) == [] and "FEWER_THAN_REQUESTED" in codes(issues)


def test_missed_variety_request_is_a_warning_only_when_variety_was_achievable():
    prefs = {"genre": "pop", "mood": "happy", "energy": 0.85, "likes_acoustic": False}
    plain = rank_songs(prefs, SONGS, 6, WEIGHTS)
    assert len({r.song["artist"] for r in plain}) < 6
    issues = check(plain, k=6, prefs=prefs, variety={"artist": True})
    assert codes(issues, "warning") == {"DIVERSITY_GOAL_MISSED"} and errors(issues) == []
    varied = rank_songs(prefs, SONGS, 6, WEIGHTS, DiversityConfig(artist_penalty=3.0))
    assert "DIVERSITY_GOAL_MISSED" not in codes(check(varied, k=6, prefs=prefs, diversity=DiversityConfig(artist_penalty=3.0),
                                                      variety={"artist": True}))
    one_artist = [s for s in SONGS if s["artist"] == "Neon Echo"]
    forced = rank_songs(prefs, one_artist, 3, WEIGHTS)
    assert "DIVERSITY_GOAL_MISSED" not in codes(check(forced, k=3, prefs=prefs, candidates=one_artist, variety={"artist": True}))


def test_a_poor_best_match_is_flagged_as_weak():
    prefs = {"genre": "metal", "mood": "angry", "energy": 0.97}
    recs = rank_songs(prefs, [s for s in SONGS if s["genre"] == "classical"], 2, WEIGHTS)
    issues = check(recs, k=2, prefs=prefs, candidates=[s for s in SONGS if s["genre"] == "classical"])
    assert "WEAK_MATCH" in codes(issues, "warning")


def test_source_references_must_exist():
    valid, problems = validate_sources(["KB-A", "KB-MADE-UP", "KB-A", "KB-B"], ["KB-A", "KB-B"])
    assert valid == ["KB-A", "KB-B"]
    assert [p.code for p in problems] == ["UNSUPPORTED_SOURCE"] and "KB-MADE-UP" in problems[0].detail


def test_a_tampered_variety_penalty_line_is_caught_by_the_points_check_alone():
    """The base reasons are untouched, so only the 'points listed vs score' check can see this."""
    diversity = DiversityConfig(artist_penalty=0.75)
    prefs = {"genre": "pop", "mood": "happy", "energy": 0.85, "likes_acoustic": False}
    recs = rank_songs(prefs, SONGS, 8, WEIGHTS, diversity)
    index = next(i for i, r in enumerate(recs) if any(x.startswith("variety penalty") for x in r.reasons))
    tampered = tuple(x.replace("(-0.75)", "(-0.05)") if x.startswith("variety penalty") else x for x in recs[index].reasons)
    assert tampered != recs[index].reasons
    bad = recs[:index] + [replace(recs[index], reasons=tampered)] + recs[index + 1:]
    issues = check(bad, k=8, prefs=prefs, diversity=diversity)
    assert codes(errors(issues)) == {"EXPLANATION_MISMATCH"} and "display rounding" in errors(issues)[0].detail
    assert errors(check(recs, k=8, prefs=prefs, diversity=diversity)) == []
