"""Tests for the intent classifier, the keyword baseline, the datasets and the request parser."""

import math

import pytest

from src.intent import (
    DEV_FILE, EVAL_LABELS, HELDOUT_FILE, TRAIN_FILE, TRAIN_LABELS, UNCERTAIN, UNSUPPORTED, Constraint,
    NaiveBayesIntentClassifier, accuracy, catalog_vocabulary, compare_models, default_classifier,
    keyword_baseline, load_intent_examples, overlapping_examples, parse_request, parse_require,
    summarize_comparison, tune_on_dev,
)
from src.main import SONGS_CSV
from src.recommender import load_songs

SONGS = load_songs(str(SONGS_CSV))
GENRES, MOODS = catalog_vocabulary(SONGS)


def parse(text):
    return parse_request(text, GENRES, MOODS)


@pytest.fixture(scope="module")
def train():
    return load_intent_examples(TRAIN_FILE, TRAIN_LABELS)


@pytest.fixture(scope="module")
def dev():
    return load_intent_examples(DEV_FILE)


@pytest.fixture(scope="module")
def heldout():
    return load_intent_examples(HELDOUT_FILE)


# --- datasets ---------------------------------------------------------------------------

def test_splits_have_the_documented_shape(train, dev, heldout):
    assert (len(train), len(dev), len(heldout)) == (144, 34, 66)
    assert {label for _, label in train} == set(TRAIN_LABELS)
    assert all(sum(1 for _, other in train if other == label) == 24 for label in TRAIN_LABELS)
    assert {label for _, label in heldout} == set(EVAL_LABELS)
    assert UNCERTAIN not in {label for _, label in train}     # abstention is never a trained class


def test_no_sentence_appears_in_more_than_one_split(train, dev, heldout):
    assert overlapping_examples(train, dev, heldout) == []


def test_overlap_check_actually_detects_a_leak(train):
    leaked = [(train[0][0].upper() + "!", "study")]
    assert overlapping_examples(train, leaked) == [leaked[0][0]]


def test_dataset_loader_rejects_bad_files(tmp_path):
    def write(text):
        path = tmp_path / "x.csv"
        path.write_text(text, encoding="utf-8")
        return path
    with pytest.raises(ValueError, match="not in"):
        load_intent_examples(write("text,label\nhello,dancing\n"))
    with pytest.raises(ValueError, match="empty text"):
        load_intent_examples(write("text,label\n  ,study\n"))
    with pytest.raises(ValueError, match="duplicate"):
        load_intent_examples(write("text,label\nStudy now,study\nstudy NOW!,study\n"))


# --- specialized classifier ----------------------------------------------------------------

@pytest.mark.parametrize("text, label", [
    ("quiet music so I can study for my exam", "study"),
    ("pump me up for the gym", "workout"),
    ("songs for my birthday party", "party"),
    ("help me relax and unwind", "relax"),
    ("show me something new and different", "discovery"),
    ("what is the weather today", UNSUPPORTED),
])
def test_classifier_recognises_clear_requests(text, label):
    assert default_classifier().predict(text).label == label


def test_scores_are_a_relative_distribution_over_trained_labels(train):
    prediction = default_classifier().predict("fast songs for the gym")
    assert set(prediction.scores) == set(TRAIN_LABELS)
    assert math.isclose(sum(prediction.scores.values()), 1.0, abs_tol=1e-4)
    assert prediction.best_label == max(prediction.scores, key=prediction.scores.get)
    assert prediction.score == prediction.scores[prediction.best_label]


def test_abstains_when_no_word_was_seen_in_training():
    prediction = default_classifier().predict("zzz qqq xyzzy")
    assert prediction.label == UNCERTAIN
    assert prediction.known == 0 and "seen in training" in prediction.reason
    assert default_classifier().predict("").label == UNCERTAIN


def test_abstains_when_the_best_score_is_below_the_threshold(train):
    strict = NaiveBayesIntentClassifier(min_score=1.0).fit(train)
    prediction = strict.predict("songs for the gym")
    assert prediction.label == UNCERTAIN and prediction.best_label == "workout"
    assert "below the abstention threshold" in prediction.reason


def test_training_is_deterministic(train):
    a, b = NaiveBayesIntentClassifier().fit(train), NaiveBayesIntentClassifier().fit(list(reversed(train)))
    for text in ("songs for the gym", "calm study music", "what time is it", "music"):
        assert a.predict(text).scores == b.predict(text).scores


def test_training_and_use_reject_bad_input(train):
    with pytest.raises(ValueError, match="zero examples"):
        NaiveBayesIntentClassifier().fit([])
    with pytest.raises(ValueError, match="unknown training label"):
        NaiveBayesIntentClassifier().fit([("hi", "dancing")])
    with pytest.raises(RuntimeError, match="not been trained"):
        NaiveBayesIntentClassifier().predict("hi")
    for kwargs in ({"alpha": 0}, {"alpha": float("nan")}, {"min_score": 1.5}):
        with pytest.raises(ValueError):
            NaiveBayesIntentClassifier(**kwargs)


def test_shipped_settings_are_the_best_on_dev_not_held_out(train, dev):
    grid = tune_on_dev(train, dev)
    assert len(grid) == 30 and grid[0][3] >= grid[-1][3]
    shipped = default_classifier()
    assert accuracy(lambda t: shipped.predict(t).label, dev) == grid[0][3]


# --- baseline and held-out comparison ----------------------------------------------------------

def test_keyword_baseline_counts_hits_and_abstains_on_ties_or_misses():
    assert keyword_baseline("music for studying and homework").label == "study"
    assert keyword_baseline("gym").hits["workout"] == 1
    assert keyword_baseline("study at the gym").label == UNCERTAIN       # tie between two labels
    assert keyword_baseline("what is the weather").label == UNCERTAIN    # no hits
    assert keyword_baseline(None).label == UNCERTAIN


def test_baseline_can_never_answer_unsupported(train, dev, heldout):
    labels = {keyword_baseline(text).label for text, _ in train + dev + heldout}
    assert UNSUPPORTED not in labels


def test_comparison_summary_is_internally_consistent(heldout):
    rows = compare_models(heldout)
    summary = summarize_comparison(rows)
    assert summary["overall"]["n"] == len(heldout)
    assert sum(v["n"] for v in summary["per_label"].values()) == len(heldout)
    assert sum(v["specialized_correct"] for v in summary["per_label"].values()) == summary["overall"]["specialized_correct"]
    accounted = (summary["overall"]["specialized_correct"] + len(summary["baseline_only_correct"])
                 + len(summary["both_wrong"]))
    assert accounted == len(heldout)       # every example is: specialized right, or baseline-only right, or both wrong


def test_reported_heldout_numbers_are_reproducible(heldout):
    """These are the numbers quoted in model_card.md; if the model or data changes, update the docs."""
    summary = summarize_comparison(compare_models(heldout))
    assert (summary["overall"]["specialized_correct"], summary["overall"]["baseline_correct"]) == (45, 18)
    assert (summary["goals_only"]["specialized_correct"], summary["goals_only"]["baseline_correct"]) == (36, 12)
    assert len(summary["both_wrong"]) == 19


# --- explicit request parser --------------------------------------------------------------------

def test_a_plain_request_has_no_explicit_content():
    parsed = parse("Find calm instrumental music for studying")
    assert (parsed.count, parsed.genres, parsed.moods, parsed.decade) == (None, [], [], None)
    assert parsed.constraints == [] and parsed.not_applied == [] and not parsed.platform_cue


@pytest.mark.parametrize("text, count", [
    ("top 3 songs", 3), ("give me five tracks", 5), ("find 7 songs for me", 7), ("suggest two", 2),
    ("give me 0 songs", 0), ("recommend 100 songs", 100), ("show me -3 tracks", -3), ("TOP 10", 10),
])
def test_count_is_read_from_several_phrasings(text, count):
    assert parse(text).count == count


def test_a_decade_is_not_mistaken_for_a_count():
    assert parse("give me 90s songs").count is None


def test_genres_and_moods_use_the_catalog_vocabulary():
    parsed = parse("some chill lofi with no metal")
    assert parsed.genres == ["lofi"] and parsed.moods == ["chill"]
    assert [c.describe() for c in parsed.constraints] == ["genre != metal"]


def test_multiword_and_hyphenated_genres_match_whole_and_do_not_leak_parts():
    assert parse("indie pop please").genres == ["indie pop"]
    assert parse("some hip-hop").genres == ["hip hop"]
    assert parse("r&b for tonight").genres == ["r&b"]


def test_several_genres_are_all_reported():
    assert parse("jazz or blues").genres == ["jazz", "blues"]


def test_requesting_and_excluding_the_same_thing_is_a_contradiction():
    parsed = parse("pop music but no pop")
    assert "genre 'pop' is both requested and excluded" in parsed.contradictions
    assert parsed.genres == []


@pytest.mark.parametrize("text, expected", [
    ("instrumental only please", "instrumentalness >= 0.50"),
    ("songs with no vocals", "instrumentalness >= 0.50"),
    ("something without too much energy", "energy <= 0.60"),
    ("under 100 bpm", "tempo_bpm <= 100"),
    ("faster than 130 bpm", "tempo_bpm >= 130"),
])
def test_hard_limit_phrases_become_constraints(text, expected):
    assert [c.describe() for c in parse(text).constraints] == [expected]


def test_out_of_range_tempo_is_invalid_not_silently_applied():
    parsed = parse("under 900 bpm")
    assert parsed.constraints == [] and "900" in parsed.invalid[0]


def test_decades():
    assert parse("music from the 90s").decade == 1990
    assert parse("1980s hits").decade == 1980
    assert parse("00s pop").decade == 2000
    gone = parse("songs from the 60s")
    assert gone.decade is None and "1960s" in gone.not_applied[0]


@pytest.mark.parametrize("text, fragment", [
    ("music in spanish", "language"),
    ("clean songs only", "content rating"),
    ("the latest releases", "recency"),
    ("songs about heartbreak", "lyric"),
    ("play it on spotify", "spotify"),
])
def test_unsupported_requirements_are_reported_not_applied(text, fragment):
    parsed = parse(text)
    assert parsed.constraints == []
    assert any(fragment in note for note in parsed.not_applied)


def test_constraint_semantics():
    song = {"genre": "Metal", "energy": 0.5, "tempo_bpm": 100, "popularity": None}
    assert not Constraint("exclude", "genre", "!=", "metal", "t").allows(song)
    assert Constraint("exclude", "genre", "!=", "pop", "t").allows(song)
    assert Constraint("numeric", "energy", "<=", 0.5, "t").allows(song)
    assert not Constraint("numeric", "energy", ">=", 0.6, "t").allows(song)
    assert not Constraint("numeric", "popularity", ">=", 10, "t").allows(song)   # unreadable value cannot satisfy


def test_cli_requirement_parsing():
    assert parse_require("energy<=0.5").describe() == "energy <= 0.50"
    assert parse_require("tempo >= 120").describe() == "tempo_bpm >= 120"
    for bad in ("energy=0.5", "loudness<=1", "energy<=7", "", "energy<=abc"):
        with pytest.raises(ValueError):
            parse_require(bad)
