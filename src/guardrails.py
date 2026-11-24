"""
Guardrails for VibeFinder AI: functional checks that run inside the agent workflow.

Every check can fail, and a failure changes what the system does (an error is returned, a
correction is attempted, or the output is withheld). Three layers:

1. Request checks    validate_request, validate_overrides, find_contradictions
2. Result checks     validate_results re-derives each score and explanation with the Module 3
                     scorer and compares them with what the engine returned; it also checks
                     hard constraints, duplicates, unknown songs, ordering and result count
3. Source checks     validate_sources makes sure every cited knowledge ID really exists

Rounding: the Module 3 recommender shows each reason with 2 decimals but keeps full precision
in the score. A gap between the points shown and the score smaller than half a display unit
per reason is display rounding, not an error; anything larger is a real inconsistency.
"""

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.intent import Constraint
from src.recommender import (
    DiversityConfig, Recommendation, ScoringWeights, max_score, score_breakdown, validate_prefs,
)

MAX_QUERY_CHARS = 500
MAX_RESULTS = 15
DEFAULT_RESULTS = 5
EXACT_TOLERANCE = 1e-9
WEAK_MATCH_RATIO = 0.40       # warn when the best score is below 40% of a perfect score
DISPLAY_HALF_UNIT = 0.005     # reasons are printed with 2 decimals

ERROR, WARNING, INFO = "error", "warning", "info"

# Result problems that re-ranking from the trusted candidate list can fix.
INTEGRITY_CODES = frozenset({
    "DUPLICATE_SONG", "UNKNOWN_SONG", "CONSTRAINT_VIOLATION", "TOO_MANY_RESULTS", "SCORE_INVALID",
    "SCORE_MISMATCH", "EXPLANATION_MISMATCH", "ORDER_VIOLATION", "RESULT_SHORTFALL",
})


@dataclass(frozen=True)
class Issue:
    """One guardrail finding."""
    code: str
    severity: str
    detail: str
    song_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {"code": self.code, "severity": self.severity, "detail": self.detail}
        if self.song_id is not None:
            data["song_id"] = self.song_id
        return data


def errors(issues: Sequence[Issue]) -> List[Issue]:
    return [i for i in issues if i.severity == ERROR]


# ---------------------------------------------------------------------------
# 1. Request checks
# ---------------------------------------------------------------------------

def validate_request(query: Any, count: Any = None) -> Tuple[str, List[Issue]]:
    """Return (cleaned query, issues). Control characters become spaces; whitespace is collapsed."""
    issues: List[Issue] = []
    if not isinstance(query, str):
        return "", [Issue("INVALID_QUERY", ERROR, f"the request must be text, got {type(query).__name__}")]
    clean = " ".join(re.sub(r"[\x00-\x1f\x7f]", " ", query).split())
    if not clean:
        issues.append(Issue("EMPTY_QUERY", ERROR, "the request is empty; describe what you want to listen to"))
    elif len(clean) > MAX_QUERY_CHARS:
        issues.append(Issue("QUERY_TOO_LONG", ERROR,
                            f"the request is {len(clean)} characters; the limit is {MAX_QUERY_CHARS}"))
    issues.extend(validate_count(count))
    return clean, issues


def validate_count(count: Any) -> List[Issue]:
    """None means 'use the default'. Anything else must be a whole number from 1 to MAX_RESULTS."""
    if count is None:
        return []
    if isinstance(count, bool) or not isinstance(count, int):
        return [Issue("INVALID_COUNT", ERROR, f"the number of songs must be a whole number, got {count!r}")]
    if not 1 <= count <= MAX_RESULTS:
        return [Issue("INVALID_COUNT", ERROR,
                      f"the number of songs must be between 1 and {MAX_RESULTS}, got {count}")]
    return []


def validate_overrides(prefer: Mapping[str, Any], genres: Sequence[str], moods: Sequence[str]) -> List[Issue]:
    """Check explicit preference overrides (--prefer): known keys, valid values, catalog vocabulary."""
    if not prefer:
        return []
    try:
        clean = validate_prefs(prefer)
    except ValueError as err:
        return [Issue("INVALID_PREFERENCE", ERROR, str(err))]
    issues = []
    for key, vocabulary in (("genre", genres), ("mood", moods)):
        if key in clean and clean[key] not in vocabulary:
            issues.append(Issue("INVALID_PREFERENCE", ERROR,
                                f"{key} {clean[key]!r} is not in the catalog; choose from: {', '.join(vocabulary)}"))
    return issues


def numeric_bounds(constraints: Sequence[Constraint]) -> Dict[str, List[Optional[float]]]:
    bounds: Dict[str, List[Optional[float]]] = {}
    for c in constraints:
        if c.kind != "numeric":
            continue
        low, high = bounds.setdefault(c.feature, [None, None])
        if c.op == ">=":
            bounds[c.feature][0] = c.value if low is None else max(low, c.value)
        else:
            bounds[c.feature][1] = c.value if high is None else min(high, c.value)
    return bounds


def find_contradictions(explicit_prefs: Mapping[str, Any], constraints: Sequence[Constraint]) -> List[str]:
    """
    Contradictions between things the user stated explicitly: a genre or mood both wanted and
    excluded, hard limits that cannot all be true, or a stated target outside a hard limit.
    (Inferred targets that fall outside a hard limit are adjusted, not treated as contradictions.)
    """
    problems: List[str] = []
    excluded = {(c.feature, c.value) for c in constraints if c.kind == "exclude"}
    for key in ("genre", "mood"):
        if key in explicit_prefs and (key, explicit_prefs[key]) in excluded:
            problems.append(f"{key} '{explicit_prefs[key]}' is both requested and excluded")
    for feature, (low, high) in numeric_bounds(constraints).items():
        if low is not None and high is not None and low > high:
            problems.append(f"{feature} >= {low:g} and {feature} <= {high:g} cannot both be true")
            continue
        target = explicit_prefs.get("tempo_bpm" if feature == "tempo_bpm" else
                                    "decade" if feature == "release_decade" else feature)
        if isinstance(target, (int, float)) and not isinstance(target, bool):
            if (low is not None and target < low) or (high is not None and target > high):
                limit = f">= {low:g}" if low is not None and target < low else f"<= {high:g}"
                problems.append(f"the requested {feature} target {target:g} lies outside the hard limit {feature} {limit}")
    return problems


# ---------------------------------------------------------------------------
# 2. Result checks
# ---------------------------------------------------------------------------

_POINTS = re.compile(r"\(([+-]\d+\.\d{2})\)$")


def _check_one(rec: Recommendation, catalog_by_id: Mapping[int, Mapping[str, Any]],
               prefs: Mapping[str, Any], weights: ScoringWeights,
               constraints: Sequence[Constraint], diversity_on: bool) -> List[Issue]:
    song = rec.song
    song_id = song.get("id") if isinstance(song, Mapping) else None
    if not isinstance(song, Mapping) or song_id not in catalog_by_id:
        return [Issue("UNKNOWN_SONG", ERROR, f"song id {song_id!r} is not in the catalog", song_id)]
    official = catalog_by_id[song_id]
    found: List[Issue] = []
    if dict(song) != dict(official):
        found.append(Issue("UNKNOWN_SONG", ERROR,
                           f"'{song.get('title')}' does not match the catalog entry for id {song_id}", song_id))
    for constraint in constraints:
        if not constraint.allows(official):
            found.append(Issue("CONSTRAINT_VIOLATION", ERROR,
                               f"'{official['title']}' violates the hard constraint {constraint.describe()}", song_id))
    if not isinstance(rec.score, (int, float)) or not math.isfinite(rec.score):
        found.append(Issue("SCORE_INVALID", ERROR, f"score {rec.score!r} is not a finite number", song_id))
        return found

    base = score_breakdown(prefs, official, weights)
    shown = [r for r in rec.reasons if not r.startswith("variety penalty")]
    if shown != list(base.reasons):
        found.append(Issue("EXPLANATION_MISMATCH", ERROR,
                           f"the explanation for '{official['title']}' differs from a fresh scoring of that song", song_id))
    if not diversity_on and abs(rec.score - base.total) > EXACT_TOLERANCE:
        found.append(Issue("SCORE_MISMATCH", ERROR,
                           f"'{official['title']}' scored {rec.score:.4f} but recomputing gives {base.total:.4f}", song_id))
    if diversity_on and rec.score > base.total + EXACT_TOLERANCE:
        found.append(Issue("SCORE_MISMATCH", ERROR,
                           f"'{official['title']}' scored {rec.score:.4f}, above its base score {base.total:.4f}, "
                           "but variety penalties can only lower a score", song_id))
    points = [_POINTS.search(r) for r in rec.reasons]
    if all(points):
        listed = sum(float(m.group(1)) for m in points)
        allowance = DISPLAY_HALF_UNIT * len(points) + EXACT_TOLERANCE      # display rounding only
        if abs(listed - rec.score) > allowance:
            found.append(Issue("EXPLANATION_MISMATCH", ERROR,
                               f"the points listed for '{official['title']}' add to {listed:.2f} but the score is "
                               f"{rec.score:.2f}; that is more than display rounding", song_id))
    else:
        found.append(Issue("EXPLANATION_MISMATCH", ERROR,
                           f"a reason for '{official['title']}' has no readable points", song_id))
    return found


def validate_results(recs: Sequence[Recommendation], *, catalog: Sequence[Mapping[str, Any]],
                     candidates: Sequence[Mapping[str, Any]], prefs: Mapping[str, Any],
                     weights: ScoringWeights, diversity: Optional[DiversityConfig],
                     constraints: Sequence[Constraint], k: int,
                     variety_requested: Optional[Mapping[str, bool]] = None) -> List[Issue]:
    """Check an engine result against the catalog, the plan and a fresh scoring. Returns all findings."""
    catalog_by_id = {s["id"]: s for s in catalog}
    diversity_on = bool(diversity and diversity.enabled)
    issues: List[Issue] = []

    seen = set()
    for rec in recs:
        song_id = rec.song.get("id") if isinstance(rec.song, Mapping) else None
        if song_id in seen:
            issues.append(Issue("DUPLICATE_SONG", ERROR, f"song id {song_id} appears more than once", song_id))
        seen.add(song_id)
        issues.extend(_check_one(rec, catalog_by_id, prefs, weights, constraints, diversity_on))

    if len(recs) > k:
        issues.append(Issue("TOO_MANY_RESULTS", ERROR, f"{len(recs)} songs returned but at most {k} were requested"))
    expected = min(k, len(candidates))
    if len(recs) < expected:
        issues.append(Issue("RESULT_SHORTFALL", ERROR,
                            f"{len(recs)} songs returned although {expected} valid candidates were available"))
    scores = [r.score for r in recs if isinstance(r.score, (int, float)) and math.isfinite(r.score)]
    if any(later > earlier + EXACT_TOLERANCE for earlier, later in zip(scores, scores[1:])):
        issues.append(Issue("ORDER_VIOLATION", ERROR, "scores are not in best-first order"))

    if len(recs) < k and len(recs) == expected:
        issues.append(Issue("FEWER_THAN_REQUESTED", INFO,
                            f"only {len(recs)} of {k} requested songs satisfy the constraints"))

    clean_recs = [r for r in recs if isinstance(r.song, Mapping)]
    for dimension in ("artist", "genre"):
        if variety_requested and variety_requested.get(dimension) and clean_recs:
            values = [str(r.song.get(dimension)).strip().casefold() for r in clean_recs]
            available = {str(c.get(dimension)).strip().casefold() for c in candidates}
            if len(set(values)) < len(values) and len(available) >= len(values):
                repeated = sorted({v for v in values if values.count(v) > 1})
                issues.append(Issue("DIVERSITY_GOAL_MISSED", WARNING,
                                    f"variety across {dimension}s was requested but {', '.join(repeated)} repeats "
                                    f"although {len(available)} different {dimension}s were available"))

    if prefs and scores:
        try:
            best = max_score(prefs, weights)
        except ValueError:
            best = 0.0
        if best > 0 and scores[0] < WEAK_MATCH_RATIO * best:
            issues.append(Issue("WEAK_MATCH", WARNING,
                                f"the best match scores {scores[0]:.2f} of a possible {best:.2f}; "
                                "nothing in the catalog is a close fit"))
    return issues


# ---------------------------------------------------------------------------
# 3. Source checks
# ---------------------------------------------------------------------------

def validate_sources(cited: Sequence[str], known_ids: Sequence[str]) -> Tuple[List[str], List[Issue]]:
    """Keep only cited knowledge IDs that exist. Returns (valid ids, issues for the removed ones)."""
    known = set(known_ids)
    valid = [c for c in dict.fromkeys(cited) if c in known]
    problems = [Issue("UNSUPPORTED_SOURCE", ERROR, f"cited source {c!r} does not exist in the knowledge collection")
                for c in dict.fromkeys(cited) if c not in known]
    return valid, problems
