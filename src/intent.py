"""
Intent interpretation for VibeFinder AI (standard library only).

Three separate pieces live here:

1. NaiveBayesIntentClassifier: a small multinomial Naive Bayes text classifier trained on the
   versioned synthetic dataset in data/intent_*_v1.csv. It is a statistical classifier, NOT a
   language model. Its `score` is a normalized posterior used only to compare labels and to
   abstain; it is not a calibrated probability.
2. keyword_baseline: a plain keyword matcher used as the comparison point for (1).
3. parse_request: a rule-based parser for what the user states *explicitly* (result count,
   a genre or mood from the catalog vocabulary, a decade, exclusions, a few hard-limit
   phrases) plus requirements the catalog cannot satisfy. It is deliberately a small,
   documented grammar, not general language understanding.
"""

import csv
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.recommender import FEATURE_RANGES
from src.retrieval import GOALS
from src.textutil import content_tokens, normalize

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATASET_VERSION = "v1"
TRAIN_FILE = DATA_DIR / f"intent_train_{DATASET_VERSION}.csv"
DEV_FILE = DATA_DIR / f"intent_dev_{DATASET_VERSION}.csv"
HELDOUT_FILE = DATA_DIR / f"intent_heldout_{DATASET_VERSION}.csv"

UNSUPPORTED = "unsupported"
UNCERTAIN = "uncertain"
TRAIN_LABELS: Tuple[str, ...] = tuple(GOALS) + (UNSUPPORTED,)
EVAL_LABELS: Tuple[str, ...] = TRAIN_LABELS + (UNCERTAIN,)

# Chosen by `tune_on_dev` (grid search over the DEV file only; the held-out file was not used).
DEFAULT_ALPHA = 0.1
DEFAULT_MIN_SCORE = 0.5
DEFAULT_BIGRAMS = False

Example = Tuple[str, str]


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def _example_key(text: str) -> str:
    """Lowercased words only, so punctuation and case never hide a duplicate."""
    return " ".join(re.findall(r"[a-z0-9&]+", normalize(text)))


def load_intent_examples(path: Path, allowed: Sequence[str] = EVAL_LABELS) -> List[Example]:
    """Read a `text,label` CSV, rejecting blank text, unknown labels and duplicates within the file."""
    examples: List[Example] = []
    seen = set()
    with open(path, newline="", encoding="utf-8") as handle:
        for line_no, row in enumerate(csv.DictReader(handle), start=2):
            text, label = (row.get("text") or "").strip(), (row.get("label") or "").strip()
            if not text:
                raise ValueError(f"{path.name} line {line_no}: empty text")
            if label not in allowed:
                raise ValueError(f"{path.name} line {line_no}: label {label!r} not in {', '.join(allowed)}")
            key = _example_key(text)
            if key in seen:
                raise ValueError(f"{path.name} line {line_no}: duplicate example {text!r}")
            seen.add(key)
            examples.append((text, label))
    return examples


def overlapping_examples(*splits: Sequence[Example]) -> List[str]:
    """Sentences that appear in more than one split (compared after lowercasing/punctuation)."""
    seen: Dict[str, int] = {}
    overlaps = []
    for index, split in enumerate(splits):
        for text, _ in split:
            key = _example_key(text)
            if key in seen and seen[key] != index:
                overlaps.append(text)
            seen.setdefault(key, index)
    return overlaps


# ---------------------------------------------------------------------------
# Specialized model: multinomial Naive Bayes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Prediction:
    """
    label       the answer: a trained label, or "uncertain" when the model abstains
    best_label  the highest-scoring trained label even when it abstained
    score       relative score of best_label among the trained labels (normalized posterior,
                NOT a calibrated probability)
    margin      score of best_label minus the runner-up
    known       how many of the request's features were seen in training
    reason      why the model abstained ("" when it did not)
    """
    label: str
    best_label: str
    score: float
    margin: float
    known: int
    reason: str = ""
    scores: Mapping[str, float] = field(default_factory=dict)


class NaiveBayesIntentClassifier:
    """Multinomial Naive Bayes over stemmed words (optionally also word pairs) with additive (Lidstone) smoothing."""

    def __init__(self, alpha: float = DEFAULT_ALPHA, min_score: float = DEFAULT_MIN_SCORE,
                 bigrams: bool = DEFAULT_BIGRAMS):
        if not (math.isfinite(alpha) and alpha > 0):
            raise ValueError(f"alpha must be a finite number > 0, got {alpha!r}")
        if not (0.0 <= min_score <= 1.0):
            raise ValueError(f"min_score must be between 0 and 1, got {min_score!r}")
        self.alpha, self.min_score, self.bigrams = alpha, min_score, bigrams
        self.labels: Tuple[str, ...] = ()
        self._log_prior: Dict[str, float] = {}
        self._log_like: Dict[str, Dict[str, float]] = {}
        self._log_unseen: Dict[str, float] = {}
        self._vocab: frozenset = frozenset()

    def _features(self, text: str) -> List[str]:
        words = content_tokens(text)
        if not self.bigrams:
            return words
        return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]

    def fit(self, examples: Sequence[Example]) -> "NaiveBayesIntentClassifier":
        if not examples:
            raise ValueError("cannot train on zero examples")
        counts: Dict[str, Dict[str, int]] = {}
        docs: Dict[str, int] = {}
        for text, label in examples:
            if label not in TRAIN_LABELS:
                raise ValueError(f"unknown training label {label!r}; allowed: {', '.join(TRAIN_LABELS)}")
            docs[label] = docs.get(label, 0) + 1
            bucket = counts.setdefault(label, {})
            for feature in self._features(text):
                bucket[feature] = bucket.get(feature, 0) + 1
        self.labels = tuple(sorted(docs))
        self._vocab = frozenset(f for bucket in counts.values() for f in bucket)
        size, total_docs = len(self._vocab), sum(docs.values())
        for label in self.labels:
            self._log_prior[label] = math.log(docs[label] / total_docs)
            denominator = sum(counts[label].values()) + self.alpha * size
            self._log_like[label] = {f: math.log((counts[label].get(f, 0) + self.alpha) / denominator)
                                     for f in self._vocab}
            self._log_unseen[label] = math.log(self.alpha / denominator)
        return self

    def predict(self, text: str) -> Prediction:
        if not self.labels:
            raise RuntimeError("classifier has not been trained")
        known = [f for f in self._features(text if isinstance(text, str) else "") if f in self._vocab]
        if not known:
            return Prediction(UNCERTAIN, self.labels[0], 0.0, 0.0, 0, "none of the request's words were seen in training")
        joint = {label: self._log_prior[label] + sum(self._log_like[label][f] for f in known)
                 for label in self.labels}
        top = max(joint.values())
        weights = {label: math.exp(value - top) for label, value in joint.items()}
        total = sum(weights.values())
        scores = {label: round(w / total, 6) for label, w in weights.items()}
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        best, best_score = ranked[0]
        margin = best_score - ranked[1][1] if len(ranked) > 1 else best_score
        if best_score < self.min_score:
            return Prediction(UNCERTAIN, best, best_score, round(margin, 6), len(known),
                              f"best relative score {best_score:.2f} is below the abstention threshold {self.min_score:.2f}",
                              scores)
        return Prediction(best, best, best_score, round(margin, 6), len(known), "", scores)


# ---------------------------------------------------------------------------
# Baseline: keyword matching
# ---------------------------------------------------------------------------

BASELINE_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "study": ("study", "studying", "homework", "exam", "focus", "concentrate", "reading", "revision"),
    "workout": ("workout", "gym", "exercise", "run", "running", "cardio", "training", "lifting"),
    "party": ("party", "dance", "dancing", "club", "celebration", "celebrate"),
    "relax": ("relax", "relaxing", "calm", "chill", "unwind", "peaceful", "sleep"),
    "discovery": ("discover", "new", "explore", "surprise", "different", "fresh"),
}
_BASELINE_STEMS = {label: frozenset(content_tokens(" ".join(words))) for label, words in BASELINE_KEYWORDS.items()}


@dataclass(frozen=True)
class BaselinePrediction:
    label: str
    hits: Mapping[str, int]


def keyword_baseline(text: str) -> BaselinePrediction:
    """Pick the label with the most distinct keyword hits; no hits or a tie gives "uncertain".
    It has no concept of out-of-domain requests, so it can never answer "unsupported"."""
    words = set(content_tokens(text if isinstance(text, str) else ""))
    hits = {label: len(words & stems) for label, stems in _BASELINE_STEMS.items()}
    best = max(hits.values())
    winners = [label for label, count in hits.items() if count == best]
    return BaselinePrediction(winners[0] if best > 0 and len(winners) == 1 else UNCERTAIN, hits)


# ---------------------------------------------------------------------------
# Training, tuning and comparison
# ---------------------------------------------------------------------------

def accuracy(classify, examples: Sequence[Example]) -> float:
    return sum(classify(text) == label for text, label in examples) / len(examples) if examples else 0.0


def tune_on_dev(train: Sequence[Example], dev: Sequence[Example]) -> List[Tuple[float, float, bool, float]]:
    """Grid-search the three model settings on the DEV split only. Returns rows
    (alpha, min_score, bigrams, dev_accuracy), best first; ties favour the earlier grid entry."""
    rows = []
    for alpha, min_score, bigrams in product((0.1, 0.5, 1.0), (0.3, 0.4, 0.5, 0.6, 0.7), (False, True)):
        model = NaiveBayesIntentClassifier(alpha, min_score, bigrams).fit(train)
        rows.append((alpha, min_score, bigrams, accuracy(lambda t: model.predict(t).label, dev)))
    return sorted(rows, key=lambda row: -row[3])


@lru_cache(maxsize=1)
def default_classifier() -> NaiveBayesIntentClassifier:
    """The shipped model: default settings trained on the full training file."""
    return NaiveBayesIntentClassifier().fit(load_intent_examples(TRAIN_FILE, TRAIN_LABELS))


@dataclass(frozen=True)
class ComparisonRow:
    text: str
    expected: str
    specialized: str
    specialized_score: float
    baseline: str


def compare_models(examples: Sequence[Example],
                   classifier: Optional[NaiveBayesIntentClassifier] = None) -> List[ComparisonRow]:
    """Run the specialized model and the keyword baseline on the same examples."""
    model = classifier or default_classifier()
    rows = []
    for text, expected in examples:
        prediction = model.predict(text)
        rows.append(ComparisonRow(text, expected, prediction.label, prediction.score,
                                  keyword_baseline(text).label))
    return rows


def summarize_comparison(rows: Sequence[ComparisonRow]) -> Dict[str, Any]:
    """Counts and accuracies for both models, overall and on the five listening goals only."""
    def stats(subset: Sequence[ComparisonRow]) -> Dict[str, Any]:
        n = len(subset)
        spec = sum(r.specialized == r.expected for r in subset)
        base = sum(r.baseline == r.expected for r in subset)
        return {"n": n, "specialized_correct": spec, "baseline_correct": base,
                "specialized_accuracy": spec / n if n else 0.0, "baseline_accuracy": base / n if n else 0.0}

    per_label = {label: stats([r for r in rows if r.expected == label]) for label in EVAL_LABELS
                 if any(r.expected == label for r in rows)}
    return {
        "overall": stats(rows),
        "goals_only": stats([r for r in rows if r.expected in GOALS]),
        "per_label": per_label,
        "specialized_only_correct": [r for r in rows if r.specialized == r.expected != r.baseline],
        "baseline_only_correct": [r for r in rows if r.baseline == r.expected != r.specialized],
        "both_wrong": [r for r in rows if r.specialized != r.expected and r.baseline != r.expected],
    }


# ---------------------------------------------------------------------------
# Explicit request parsing (a small documented grammar)
# ---------------------------------------------------------------------------

FEATURE_ALIASES = {"tempo": "tempo_bpm", "bpm": "tempo_bpm", "decade": "release_decade"}
ENERGY_CAP = 0.60          # "without too much energy" is read as energy <= 0.60 (and said so in the response)
INSTRUMENTAL_FLOOR = 0.50  # "instrumental only" / "no vocals" is read as instrumentalness >= 0.50
SUPPORTED_DECADES = (1970, 1980, 1990, 2000, 2010, 2020)

_NUMBER_WORDS = {w: i for i, w in enumerate("one two three four five six seven eight nine ten".split(), start=1)}
_NUM = r"(-?\d+|" + "|".join(_NUMBER_WORDS) + r")"
_COUNT_PATTERNS = (
    re.compile(rf"\btop\s+{_NUM}\b"),
    re.compile(rf"(?<![\w.-]){_NUM}\s+(?:songs?|tracks?|tunes?|recommendations?|suggestions?)\b"),
    re.compile(rf"\b(?:give|show|find|recommend|suggest|list)\s+(?:me\s+)?{_NUM}\b"),
)
_NEGATOR = re.compile(r"\b(?:no|not|without|except|exclude|excluding|avoid|avoiding|skip|minus)\s+"
                      r"(?:(?:any|the|too much|much|more)\s+)?$")
_INSTRUMENTAL_ONLY = re.compile(r"\b(?:instrumentals? only|only instrumentals?|no vocals|without vocals|"
                                r"no lyrics|without lyrics|no singing|without singing)\b")
_ENERGY_CAP = re.compile(r"\b(?:without too much energy|not too much energy|not too energetic|"
                         r"low energy only|not too intense)\b")
_TEMPO_MAX = re.compile(r"\b(?:under|below|less than|slower than|at most)\s+(\d{1,4})\s*bpm\b")
_TEMPO_MIN = re.compile(r"\b(?:over|above|more than|faster than|at least)\s+(\d{1,4})\s*bpm\b")
_DECADE = re.compile(r"\b(?:(19|20)(\d)0|(\d)0)s\b")

_LANGUAGES = ("spanish|french|german|korean|japanese|portuguese|italian|hindi|yoruba|swahili|arabic|"
              "mandarin|chinese|russian")
_UNSUPPORTED_REQUIREMENTS = (
    (re.compile(rf"\b(?:in|sung in)\s+({_LANGUAGES})\b"),
     "a language ({0}): the catalog has no language field"),
    (re.compile(r"\b(explicit|clean|family friendly|kid friendly|radio edit)\b"),
     "a content rating ('{0}'): the catalog has no content-rating field"),
    (re.compile(r"\b(new releases?|latest|newest|brand new|released in (?:19|20)\d\d|this year|last year|just released)\b"),
     "recency ('{0}'): the catalog stores only a release decade"),
    (re.compile(r"\b(lyrics about|songs about)\b"),
     "lyric topics ('{0}'): the catalog has no lyric data"),
    (re.compile(r"\b(spotify|apple music|youtube|soundcloud|pandora|deezer|tidal|amazon music)\b"),
     "playing or linking on '{0}': this assistant only lists songs from its fictional catalog"),
)


@dataclass(frozen=True)
class Constraint:
    """
    A HARD constraint: a song that fails it is removed before ranking.
    kind "exclude": genre/mood must not equal `value`.   kind "numeric": feature op value.
    """
    kind: str
    feature: str
    op: str
    value: Any
    origin: str
    source_text: str = ""

    def allows(self, song: Mapping[str, Any]) -> bool:
        raw = song.get(self.feature)
        if self.kind == "exclude":
            return not (isinstance(raw, str) and raw.strip().casefold() == self.value)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw):
            return False                      # a value we cannot read cannot satisfy a limit
        return raw >= self.value if self.op == ">=" else raw <= self.value

    def describe(self) -> str:
        if self.kind == "exclude":
            return f"{self.feature} != {self.value}"
        shown = f"{self.value:g}" if self.feature in ("tempo_bpm", "popularity", "release_decade") \
            else f"{self.value:.2f}"
        return f"{self.feature} {self.op} {shown}"


@dataclass
class ParsedRequest:
    """What the user explicitly asked for, in a form the planner can use."""
    count: Optional[int] = None
    genres: List[str] = field(default_factory=list)
    moods: List[str] = field(default_factory=list)
    decade: Optional[int] = None
    constraints: List[Constraint] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    invalid: List[str] = field(default_factory=list)
    not_applied: List[str] = field(default_factory=list)
    platform_cue: bool = False


def catalog_vocabulary(songs: Iterable[Mapping[str, Any]]) -> Tuple[List[str], List[str]]:
    """(genres, moods) present in the catalog, casefolded and sorted."""
    songs = list(songs)
    return (sorted({str(s["genre"]).strip().casefold() for s in songs}),
            sorted({str(s["mood"]).strip().casefold() for s in songs}))


def _find_terms(text: str, terms: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Positive and negated occurrences of vocabulary terms (longest term first, spans masked)."""
    masked, positive, negated = text, [], []
    for term in sorted(terms, key=lambda t: (-len(t), t)):
        pattern = re.compile(rf"(?<![a-z0-9&]){re.escape(term)}(?![a-z0-9&])")
        for match in list(pattern.finditer(masked)):
            if _NEGATOR.search(masked[:match.start()]):
                negated.append(term)
            else:
                positive.append(term)
            masked = masked[:match.start()] + " " * (match.end() - match.start()) + masked[match.end():]
    order = {t: text.find(t) for t in set(positive + negated)}
    return (sorted(dict.fromkeys(positive), key=order.get), sorted(dict.fromkeys(negated), key=order.get))


def parse_request(text: str, genres: Sequence[str], moods: Sequence[str]) -> ParsedRequest:
    """Extract explicit, checkable statements from the request. Everything else is left to the
    intent model and the knowledge base."""
    parsed = ParsedRequest()
    raw = text.lower()
    norm = " ".join(normalize(text).split())

    for pattern in _COUNT_PATTERNS:
        match = pattern.search(raw)
        if match:
            word = match.group(1)
            parsed.count = _NUMBER_WORDS[word] if word in _NUMBER_WORDS else int(word)
            break

    for kind, vocabulary in (("genre", genres), ("mood", moods)):
        positive, negated = _find_terms(norm, vocabulary)
        both = [t for t in positive if t in negated]
        if both:
            parsed.contradictions.append(f"{kind} '{both[0]}' is both requested and excluded")
        (parsed.genres if kind == "genre" else parsed.moods).extend(t for t in positive if t not in both)
        for term in negated:
            parsed.constraints.append(Constraint("exclude", kind, "!=", term, "request", f"no {term}"))

    if _INSTRUMENTAL_ONLY.search(norm):
        parsed.constraints.append(Constraint("numeric", "instrumentalness", ">=", INSTRUMENTAL_FLOOR, "request",
                                             _INSTRUMENTAL_ONLY.search(norm).group(0)))
    if _ENERGY_CAP.search(norm):
        parsed.constraints.append(Constraint("numeric", "energy", "<=", ENERGY_CAP, "request",
                                             _ENERGY_CAP.search(norm).group(0)))
    for pattern, op in ((_TEMPO_MAX, "<="), (_TEMPO_MIN, ">=")):
        for match in pattern.finditer(norm):
            value = float(match.group(1))
            low, high = FEATURE_RANGES["tempo_bpm"]
            if low <= value <= high:
                parsed.constraints.append(Constraint("numeric", "tempo_bpm", op, value, "request", match.group(0)))
            else:
                parsed.invalid.append(f"tempo limit {match.group(1)} BPM is outside {low:g}-{high:g}")

    for match in _DECADE.finditer(norm):
        if match.group(3) is not None:
            digit = int(match.group(3))
            decade = 2000 + 10 * digit if digit <= 2 else 1900 + 10 * digit
        else:
            decade = int(match.group(1) + match.group(2) + "0")
        if decade in SUPPORTED_DECADES:
            parsed.decade = decade
        else:
            parsed.not_applied.append(f"the {decade}s: the catalog covers {SUPPORTED_DECADES[0]}s-{SUPPORTED_DECADES[-1]}s only")

    for pattern, template in _UNSUPPORTED_REQUIREMENTS:
        match = pattern.search(norm)
        if match:
            parsed.not_applied.append(template.format(match.group(1)))
            if template.startswith("playing"):
                parsed.platform_cue = True
    return parsed


def parse_require(spec: str, origin: str = "cli") -> Constraint:
    """Parse a CLI hard limit such as 'energy<=0.5' or 'tempo>=120'. Raises ValueError if invalid."""
    match = re.fullmatch(r"\s*([a-z_]+)\s*(>=|<=)\s*(-?\d+(?:\.\d+)?)\s*", spec or "")
    if not match:
        raise ValueError(f"could not read {spec!r}; use FEATURE>=NUMBER or FEATURE<=NUMBER, e.g. energy<=0.5")
    feature = FEATURE_ALIASES.get(match.group(1), match.group(1))
    if feature not in FEATURE_RANGES:
        raise ValueError(f"unknown feature {match.group(1)!r}; choose from: {', '.join(FEATURE_RANGES)}")
    value, (low, high) = float(match.group(3)), FEATURE_RANGES[feature]
    if not low <= value <= high:
        raise ValueError(f"{feature} limit must be between {low:g} and {high:g}, got {match.group(3)}")
    return Constraint("numeric", feature, match.group(2), value, origin, spec.strip())
