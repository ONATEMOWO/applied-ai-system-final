"""
The VibeFinder AI agent: a bounded, linear tool-calling workflow around the Module 3 engine.

    validate input -> classify intent -> parse explicit request -> route
      -> retrieve knowledge -> build plan -> filter catalog (hard constraints)
      -> rank with the existing engine -> validate results
      -> at most MAX_CORRECTIONS corrective steps -> final status

It always terminates: there is no loop except the correction loop, which runs at most
MAX_CORRECTIONS times and never repeats the same strategy. Every step records a structured
event (see Trace). Events are observable facts (tool arguments, tool results, decisions),
not a transcript of reasoning.

How retrieved knowledge changes the output
------------------------------------------
Retrieved sections carry validated profile blocks. The plan is built FROM them:
  * the goal section supplies inferred soft targets (only if the request itself contains a
    keyword of that profile: the classifier alone is not enough),
  * matching modifier sections add or override soft targets,
  * matching strategy sections choose the ranking mode and variety penalties,
  * feature/policy sections become the notes in the response.
With retrieval disabled none of that exists, and only what the user stated explicitly is used.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from src.guardrails import (
    DEFAULT_RESULTS, ERROR, INTEGRITY_CODES, WARNING, Issue, errors, find_contradictions, numeric_bounds,
    validate_count, validate_overrides, validate_request, validate_results, validate_sources,
)
from src.intent import (
    UNCERTAIN, UNSUPPORTED, Constraint, NaiveBayesIntentClassifier, ParsedRequest, Prediction,
    catalog_vocabulary, default_classifier, parse_request,
)
from src.recommender import (
    DEFAULT_MODE, RANKING_MODES, DiversityConfig, Recommendation, get_mode, max_score, rank_songs,
)
from src.retrieval import GOALS, Hit, KnowledgeBase

MAX_CORRECTIONS = 2
STRONG_VARIETY_PENALTY = 3.0
DEFAULT_CONFLICT_SPREAD = 0.4
CONFLICT_SPREAD = {"tempo_bpm": 50.0, "popularity": 40.0, "decade": 30.0}
_FEATURE_OF_PREF = {"decade": "release_decade"}
STATUSES = ("ok", "needs_clarification", "unsupported", "invalid_input", "no_match", "failed_validation")

RecommendFn = Callable[..., List[Recommendation]]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AssistantRequest:
    """Everything a caller can control. Only `query` is required."""
    query: Any
    count: Optional[int] = None
    mode: Optional[str] = None
    goal: Optional[str] = None
    prefer: Mapping[str, Any] = field(default_factory=dict)
    require: Sequence[Constraint] = ()
    exclude_genres: Sequence[str] = ()
    artist_penalty: Optional[float] = None
    genre_penalty: Optional[float] = None
    use_retrieval: bool = True
    guardrails: bool = True

    def describe(self) -> Dict[str, Any]:
        options = {"count": self.count, "mode": self.mode, "goal": self.goal, "prefer": dict(self.prefer),
                   "require": [c.describe() for c in self.require], "exclude_genres": list(self.exclude_genres),
                   "artist_penalty": self.artist_penalty, "genre_penalty": self.genre_penalty,
                   "retrieval": self.use_retrieval, "guardrails": self.guardrails}
        return {k: v for k, v in options.items() if v is not None and v != {} and v != []}


@dataclass
class PrefEntry:
    """One soft scoring target and where it came from."""
    value: Any
    origin: str                      # "explicit" or "inferred"
    source: str                      # "request", "cli", or a KB-* id
    overridden: List[str] = field(default_factory=list)


@dataclass
class Plan:
    """What the agent decided to ask the recommendation engine, and why."""
    count: int = DEFAULT_RESULTS
    goal: Optional[str] = None
    goal_source: str = "none"                      # classifier | override | none
    goal_profile_applied: bool = False
    goal_note: str = ""
    soft: Dict[str, PrefEntry] = field(default_factory=dict)
    constraints: List[Constraint] = field(default_factory=list)
    mode: str = DEFAULT_MODE
    mode_source: str = "default"
    artist_penalty: float = 0.0
    genre_penalty: float = 0.0
    diversity_source: str = "default"
    variety_requested: Dict[str, bool] = field(default_factory=lambda: {"artist": False, "genre": False})
    applied_sources: List[str] = field(default_factory=list)
    adjustments: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    not_applied: List[str] = field(default_factory=list)

    @property
    def prefs(self) -> Dict[str, Any]:
        return {key: entry.value for key, entry in self.soft.items()}

    def diversity_config(self) -> Optional[DiversityConfig]:
        config = DiversityConfig(self.artist_penalty, self.genre_penalty)
        return config if config.enabled else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count, "goal": self.goal, "goal_source": self.goal_source,
            "goal_profile_applied": self.goal_profile_applied, "goal_note": self.goal_note,
            "soft_preferences": {k: {"value": e.value, "origin": e.origin, "source": e.source,
                                     "overridden": e.overridden} for k, e in self.soft.items()},
            "hard_constraints": [{"rule": c.describe(), "origin": c.origin} for c in self.constraints],
            "mode": self.mode, "mode_source": self.mode_source,
            "diversity": {"artist_penalty": self.artist_penalty, "genre_penalty": self.genre_penalty,
                          "source": self.diversity_source},
            "applied_sources": list(self.applied_sources), "adjustments": list(self.adjustments),
            "conflicts": list(self.conflicts), "not_applied": list(self.not_applied),
        }


class Trace:
    """An ordered list of structured runtime events. Contains no timestamps, so runs are comparable."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def add(self, step: str, event: str, **data: Any) -> None:
        self.events.append({"seq": len(self.events) + 1, "step": step, "event": event, **_plain(data)})


def _plain(value: Any) -> Any:
    """JSON-friendly copy: tuples become lists, floats are rounded."""
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(v) for v in value]
    return value


@dataclass
class AgentResult:
    status: str
    message: str = ""
    query: str = ""
    prediction: Optional[Prediction] = None
    plan: Optional[Plan] = None
    recommendations: List[Recommendation] = field(default_factory=list)
    hits: List[Hit] = field(default_factory=list)
    candidates: int = 0
    catalog_size: int = 0
    perfect_score: Optional[float] = None
    warnings: List[Issue] = field(default_factory=list)
    infos: List[Issue] = field(default_factory=list)
    corrections: List[Dict[str, Any]] = field(default_factory=list)
    failures: List[Issue] = field(default_factory=list)
    cited_sources: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_used: bool = True
    guardrails_used: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return _plain({
            "status": self.status, "message": self.message, "query": self.query,
            "intent": None if self.prediction is None else {
                "label": self.prediction.label, "best_label": self.prediction.best_label,
                "relative_score": self.prediction.score, "margin": self.prediction.margin,
                "reason": self.prediction.reason},
            "plan": None if self.plan is None else self.plan.to_dict(),
            "recommendations": [{"rank": i, "id": r.song["id"], "title": r.song["title"],
                                 "artist": r.song["artist"], "score": round(r.score, 4), "reasons": list(r.reasons)}
                                for i, r in enumerate(self.recommendations, start=1)],
            "candidates": self.candidates, "catalog_size": self.catalog_size, "perfect_score": self.perfect_score,
            "warnings": [i.to_dict() for i in self.warnings], "failures": [i.to_dict() for i in self.failures],
            "corrections": self.corrections, "cited_sources": self.cited_sources, "notes": self.notes,
            "retrieval_used": self.retrieval_used, "guardrails_used": self.guardrails_used,
            "events": self.events,
        })


def _short(text: Any, limit: int = 200) -> str:
    text = text if isinstance(text, str) else repr(text)
    return text if len(text) <= limit else text[:limit] + "..."


def _fmt(value: Any) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------

class Agent:
    """
    songs            the catalog (list of dicts from load_songs)
    kb               the knowledge collection
    classifier       the intent model (default: the shipped, trained one)
    recommend_fn     the ranking engine (default: the Module 3 rank_songs). Replaceable only so
                     tests can inject a faulty engine and prove the guardrails catch it.
    """

    def __init__(self, songs: Sequence[Mapping[str, Any]], kb: KnowledgeBase,
                 classifier: Optional[NaiveBayesIntentClassifier] = None,
                 recommend_fn: Optional[RecommendFn] = None):
        self.songs = list(songs)
        self.kb = kb
        self.classifier = classifier or default_classifier()
        self.recommend_fn = recommend_fn or rank_songs
        self.genres, self.moods = catalog_vocabulary(self.songs)

    # -- helpers -----------------------------------------------------------------------

    def _tool(self, trace: Trace, step: str, name: str, args: Mapping[str, Any],
              fn: Callable[[], Any], summarize: Callable[[Any], Mapping[str, Any]]) -> Any:
        trace.add(step, "tool_call", tool=name, args=args)
        result = fn()
        trace.add(step, "tool_result", tool=name, **summarize(result))
        return result

    def _finish(self, trace: Trace, req: AssistantRequest, status: str, message: str, **fields: Any) -> AgentResult:
        result = AgentResult(status=status, message=message, catalog_size=len(self.songs),
                             retrieval_used=req.use_retrieval, guardrails_used=req.guardrails, **fields)
        trace.add("final", "final_status", status=status, returned=[r.song["id"] for r in result.recommendations],
                  warnings=[w.code for w in result.warnings], corrections=len(result.corrections))
        result.events = trace.events
        return result

    # -- the workflow ------------------------------------------------------------------

    def run(self, req: AssistantRequest) -> AgentResult:
        trace = Trace()
        trace.add("input", "request_received", query=_short(req.query), options=req.describe())

        # 1. input validation
        clean, issues = validate_request(req.query, req.count)
        issues += validate_overrides(req.prefer, self.genres, self.moods)
        if req.mode is not None and req.mode not in RANKING_MODES:
            issues.append(Issue("INVALID_MODE", ERROR, f"unknown ranking mode {req.mode!r}; choose from: {', '.join(RANKING_MODES)}"))
        if req.goal is not None and req.goal not in GOALS:
            issues.append(Issue("INVALID_GOAL", ERROR, f"unknown goal {req.goal!r}; choose from: {', '.join(GOALS)}"))
        for name, value in (("artist_penalty", req.artist_penalty), ("genre_penalty", req.genre_penalty)):
            if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool)
                                      or not math.isfinite(value) or value < 0):
                issues.append(Issue("INVALID_PENALTY", ERROR, f"{name} must be a finite number >= 0, got {value!r}"))
        trace.add("input", "input_validated", ok=not errors(issues), issues=[i.to_dict() for i in issues])
        if errors(issues):
            return self._finish(trace, req, "invalid_input",
                                "Invalid input: " + "; ".join(i.detail for i in errors(issues)),
                                query=_short(req.query), failures=errors(issues))

        # 2. interpret: intent model + explicit-request parser
        prediction = self._tool(
            trace, "interpret", "classify_intent", {"override": req.goal},
            lambda: self._classify(clean, req.goal),
            lambda p: {"label": p.label, "best_label": p.best_label, "relative_score": p.score, "margin": p.margin,
                       "known_features": p.known, "source": "override" if req.goal else "classifier",
                       "reason": p.reason})
        parsed: ParsedRequest = self._tool(
            trace, "interpret", "parse_explicit_request", {},
            lambda: parse_request(clean, self.genres, self.moods),
            lambda p: {"count": p.count, "genres": p.genres, "moods": p.moods, "decade": p.decade,
                       "hard_constraints": [c.describe() for c in p.constraints],
                       "not_applied": p.not_applied, "contradictions": p.contradictions, "invalid": p.invalid})

        count = req.count if req.count is not None else parsed.count
        problems = errors(validate_count(count)) + [Issue("INVALID_REQUEST", ERROR, text) for text in parsed.invalid]
        if problems:
            trace.add("decision", "route", route="invalid_input", reason=problems[0].detail)
            return self._finish(trace, req, "invalid_input", "Invalid input: " + "; ".join(p.detail for p in problems),
                                query=clean, prediction=prediction, failures=problems)
        count = count if count is not None else DEFAULT_RESULTS

        explicit: Dict[str, Tuple[Any, str]] = {}
        for key, values, label in (("genre", parsed.genres, "genres"), ("mood", parsed.moods, "moods")):
            if len(values) > 1:
                message = (f"Your request names several {label} ({', '.join(values)}) but I can score one "
                           f"{key} at a time. Pick one, for example 'Find {values[0]} music', or use --prefer {key}=NAME.")
                trace.add("decision", "route", route="needs_clarification", reason=f"multiple {label}")
                return self._finish(trace, req, "needs_clarification", message, query=clean, prediction=prediction)
            if values:
                explicit[key] = (values[0], "request")
        if parsed.decade is not None:
            explicit["decade"] = (parsed.decade, "request")
        for key, value in req.prefer.items():
            explicit[key] = ({"genre": str(value).strip().casefold(), "mood": str(value).strip().casefold()}.get(key, value), "cli")
        constraints = list(parsed.constraints) + list(req.require) + [
            Constraint("exclude", "genre", "!=", g.strip().casefold(), "cli", f"--exclude-genre {g}") for g in req.exclude_genres]

        contradictions = parsed.contradictions + find_contradictions({k: v[0] for k, v in explicit.items()}, constraints)
        if contradictions:
            trace.add("decision", "route", route="needs_clarification", reason="contradictory explicit constraints",
                      contradictions=contradictions)
            return self._finish(
                trace, req, "needs_clarification",
                "Your request contradicts itself: " + "; ".join(contradictions) + ". Remove one of the conflicting parts and try again.",
                query=clean, prediction=prediction)

        if prediction.label == UNSUPPORTED:
            trace.add("decision", "route", route="unsupported", reason="the intent model classified this as out of scope")
            return self._finish(
                trace, req, "unsupported",
                "That looks outside what I can do. VibeFinder AI only recommends songs from a fictional 30-song catalog "
                "for listening goals such as studying, working out, partying, relaxing or discovering something new.",
                query=clean, prediction=prediction)
        trace.add("decision", "route", route="recommend",
                  reason=f"intent '{prediction.label}'" + (" with explicit preferences" if explicit else ""))

        # 3. retrieve knowledge
        goal = prediction.label if prediction.label in GOALS else None
        goal_note = ""
        hits: List[Hit] = []
        if req.use_retrieval:
            query_text = clean + (f" goal_{goal}" if goal else "")
            hits = self._tool(
                trace, "retrieve", "retrieve_knowledge", {"query": clean, "goal_token": f"goal_{goal}" if goal else None},
                lambda: self.kb.search(query_text),
                lambda hs: {"hits": [{"id": h.section.id, "kind": h.section.kind, "score": h.score,
                                      "matched": list(h.matched)} for h in hs]})
            goal, hits, goal_note = self._resolve_goal(trace, req, clean, prediction, goal, hits)
        else:
            trace.add("retrieve", "retrieval_disabled", note="no knowledge is used; only explicit preferences count")

        # 4. build the plan
        goal_source = "override" if req.goal else ("classifier" if goal == prediction.label else "retrieval")
        plan = self._build_plan(trace, req, goal, goal_source, parsed, hits, explicit, constraints, count)
        plan.goal_note = goal_note
        if plan.conflicts:
            return self._finish(
                trace, req, "needs_clarification",
                "Your request mixes goals that pull in opposite directions: " + "; ".join(plan.conflicts) +
                ". Say which one matters most, or state a preference explicitly (for example --prefer energy=0.9).",
                query=clean, prediction=prediction, plan=plan, hits=hits)
        if not plan.soft:
            reason = ("I could not tell what kind of music you want." if prediction.label == UNCERTAIN
                      else "I found no preferences I can score against.")
            if not req.use_retrieval:
                reason += " Knowledge retrieval is disabled, so only an explicit genre, mood, decade or --prefer value can be used."
            trace.add("decision", "clarify", reason=reason)
            return self._finish(
                trace, req, "needs_clarification",
                reason + " Try something like 'calm instrumental music for studying', 'energetic music for a workout', "
                "or name a genre or mood.", query=clean, prediction=prediction, plan=plan, hits=hits,
                notes=plan.adjustments)

        # 5. filter by hard constraints, then call the existing engine
        candidates, steps = self._tool(
            trace, "recommend", "filter_catalog", {"constraints": [c.describe() for c in plan.constraints]},
            lambda: self._filter(plan.constraints),
            lambda r: {"catalog": len(self.songs), "candidates": len(r[0]), "removed_by": r[1]})
        if not candidates:
            detail = ", ".join(f"{s['constraint']} (removes {s['removed']})" for s in steps) or "the constraints"
            trace.add("decision", "no_match", reason="no catalog song satisfies every hard constraint")
            return self._finish(
                trace, req, "no_match",
                f"No song in the {len(self.songs)}-song catalog satisfies every hard constraint: {detail}. "
                "I will not relax a hard constraint silently; loosen one and try again.",
                query=clean, prediction=prediction, plan=plan, hits=hits, candidates=0)

        weights = get_mode(plan.mode).weights
        try:
            recs = self._tool(
                trace, "recommend", "rank_songs",
                {"mode": plan.mode, "k": plan.count, "candidates": len(candidates),
                 "artist_penalty": plan.artist_penalty, "genre_penalty": plan.genre_penalty},
                lambda: self.recommend_fn(plan.prefs, candidates, plan.count, weights, plan.diversity_config()),
                lambda rs: {"returned": [r.song.get("id") if isinstance(r.song, Mapping) else None for r in rs],
                            "scores": [r.score for r in rs]})
        except ValueError as err:
            trace.add("recommend", "engine_error", error=str(err))
            return self._finish(trace, req, "failed_validation", f"The recommendation engine rejected the plan: {err}",
                                query=clean, prediction=prediction, plan=plan, hits=hits, candidates=len(candidates),
                                failures=[Issue("ENGINE_ERROR", ERROR, str(err))])

        # 6. validate, then at most MAX_CORRECTIONS corrective steps
        corrections: List[Dict[str, Any]] = []
        found: List[Issue] = []
        if req.guardrails:
            found = self._validate(trace, recs, candidates, plan, weights)
            tried = set()
            while len(corrections) < MAX_CORRECTIONS:
                codes = {i.code for i in found if i.severity in (ERROR, WARNING)}
                if codes & INTEGRITY_CODES and "rerank_with_reference_engine" not in tried:
                    strategy = "rerank_with_reference_engine"
                elif "DIVERSITY_GOAL_MISSED" in codes and "strengthen_variety" not in tried:
                    strategy = "strengthen_variety"
                else:
                    break
                tried.add(strategy)
                before = sorted(codes & (INTEGRITY_CODES | {"DIVERSITY_GOAL_MISSED"}))
                recs = self._correct(trace, strategy, before, len(corrections) + 1, recs, candidates, plan, weights)
                weights = get_mode(plan.mode).weights
                found = self._validate(trace, recs, candidates, plan, weights)
                corrections.append({"attempt": len(corrections) + 1, "strategy": strategy, "addressed": before,
                                    "remaining": [i.code for i in found if i.severity in (ERROR, WARNING)]})
        else:
            trace.add("validate", "validation_skipped", reason="guardrails disabled for this run")

        failures = errors(found)
        if failures:
            trace.add("decision", "withhold_results", reason="validation errors remain after bounded correction",
                      codes=[i.code for i in failures])
            return self._finish(
                trace, req, "failed_validation",
                "I withheld the results because they failed validation and could not be corrected: "
                + "; ".join(i.detail for i in failures[:3]) + ".",
                query=clean, prediction=prediction, plan=plan, hits=hits, candidates=len(candidates),
                corrections=corrections, failures=failures)

        # 7. cite sources and return
        cited, notes = self._collect_sources(plan, hits)
        cited, bad_sources = validate_sources(cited, self.kb.ids)
        trace.add("validate", "sources_checked", cited=cited, removed=[i.detail for i in bad_sources])
        perfect = max_score(plan.prefs, weights)
        return self._finish(
            trace, req, "ok", "", query=clean, prediction=prediction, plan=plan, hits=hits,
            recommendations=recs, candidates=len(candidates), perfect_score=perfect,
            warnings=[i for i in found if i.severity == WARNING] + bad_sources,
            infos=[i for i in found if i.severity not in (ERROR, WARNING)],
            corrections=corrections, cited_sources=cited, notes=notes)

    # -- tools -------------------------------------------------------------------------

    def _classify(self, text: str, override: Optional[str]) -> Prediction:
        if override:
            return Prediction(override, override, 1.0, 1.0, 0, "goal supplied explicitly", {})
        return self.classifier.predict(text)

    def _resolve_goal(self, trace: Trace, req: AssistantRequest, clean: str, prediction: Prediction,
                      goal: Optional[str], hits: List[Hit]) -> Tuple[Optional[str], List[Hit], str]:
        """
        Let literal goal words in the request correct the intent model. A goal has keyword
        evidence when a word of its knowledge profile appears in the request (the injected
        goal_<label> token does not count). If the model's goal has no evidence but exactly one
        other goal does, that goal is used instead (and retrieval is repeated for it). An
        explicit --goal is never changed.
        """
        evidence = {h.section.goal: [m for m in h.matched if not m.startswith("goal_")]
                    for h in hits if h.section.kind == "goal"}
        with_evidence = sorted(g for g, found in evidence.items() if found)
        final, note = goal, ""
        if req.goal is None and goal not in with_evidence and len(with_evidence) == 1:
            final = with_evidence[0]
            said = f"the intent model said '{prediction.label}'" + (
                f" (relative score {prediction.score:.2f})" if prediction.label in GOALS else " (it abstained)")
            note = (f"{said}, but the request contains the {final} word(s) "
                    f"{', '.join(repr(w) for w in evidence[final])}, so '{final}' was used")
            hits = self._tool(
                trace, "retrieve", "retrieve_knowledge", {"query": clean, "goal_token": f"goal_{final}", "repeat": True},
                lambda: self.kb.search(f"{clean} goal_{final}"),
                lambda hs: {"hits": [{"id": h.section.id, "kind": h.section.kind, "score": h.score,
                                      "matched": list(h.matched)} for h in hs]})
        trace.add("retrieve", "goal_resolved", model_goal=goal, goals_with_keyword_evidence=with_evidence,
                  final_goal=final, changed=final != goal)
        return final, hits, note

    def _filter(self, constraints: Sequence[Constraint]) -> Tuple[List[Mapping[str, Any]], List[Dict[str, Any]]]:
        remaining, steps = list(self.songs), []
        for constraint in constraints:
            kept = [s for s in remaining if constraint.allows(s)]
            steps.append({"constraint": constraint.describe(), "removed": len(remaining) - len(kept)})
            remaining = kept
        return remaining, steps

    def _validate(self, trace: Trace, recs: Sequence[Recommendation], candidates: Sequence[Mapping[str, Any]],
                  plan: Plan, weights: Any) -> List[Issue]:
        def run() -> List[Issue]:
            return validate_results(recs, catalog=self.songs, candidates=candidates, prefs=plan.prefs, weights=weights,
                                    diversity=plan.diversity_config(), constraints=plan.constraints, k=plan.count,
                                    variety_requested=plan.variety_requested)
        return self._tool(trace, "validate", "validate_results", {"returned": len(recs)}, run,
                          lambda issues: {"ok": not errors(issues), "issues": [i.to_dict() for i in issues]})

    def _correct(self, trace: Trace, strategy: str, addressed: List[str], attempt: int, recs: Sequence[Recommendation],
                 candidates: Sequence[Mapping[str, Any]], plan: Plan, weights: Any) -> List[Recommendation]:
        trace.add("correct", "correction_attempt", attempt=attempt, strategy=strategy, addressed=addressed)
        if strategy == "rerank_with_reference_engine":
            # Re-rank the de-duplicated, constraint-filtered catalog entries with the trusted Module 3 engine.
            unique = list({s["id"]: s for s in candidates}.values())
            return rank_songs(plan.prefs, unique, plan.count, weights, plan.diversity_config())
        # strengthen_variety: raise the penalty for the dimension(s) the user asked to diversify.
        before = (plan.artist_penalty, plan.genre_penalty)
        if plan.variety_requested["artist"]:
            plan.artist_penalty = max(plan.artist_penalty, STRONG_VARIETY_PENALTY)
        if plan.variety_requested["genre"]:
            plan.genre_penalty = max(plan.genre_penalty, STRONG_VARIETY_PENALTY)
        plan.adjustments.append(
            f"variety request was not met by the first ranking; penalties raised from artist {before[0]:g}/genre {before[1]:g} "
            f"to artist {plan.artist_penalty:g}/genre {plan.genre_penalty:g}")
        return self.recommend_fn(plan.prefs, list(candidates), plan.count, weights, plan.diversity_config())

    # -- planning ----------------------------------------------------------------------

    def _build_plan(self, trace: Trace, req: AssistantRequest, goal: Optional[str], goal_source: str,
                    parsed: ParsedRequest, hits: Sequence[Hit], explicit: Mapping[str, Tuple[Any, str]],
                    constraints: Sequence[Constraint], count: int) -> Plan:
        plan = Plan(count=count, goal=goal, goal_source=goal_source if goal else "none",
                    constraints=list(constraints), not_applied=list(parsed.not_applied))
        applied: List[str] = []
        conflicts: List[Tuple[str, str]] = []          # (preference key, description)

        def put(key: str, value: Any, origin: str, source: str) -> None:
            old = plan.soft.get(key)
            entry = PrefEntry(value, origin, source, list(old.overridden) if old else [])
            if old is not None and old.value != value:
                entry.overridden.append(f"{old.source}={_fmt(old.value)}")
            plan.soft[key] = entry

        # layer 1: the goal profile (needs keyword evidence in the request unless the goal was given explicitly)
        goal_hit = next((h for h in hits if h.section.kind == "goal" and h.section.goal == goal), None)
        if goal and goal_hit is not None:
            evidence = [m for m in goal_hit.matched if not m.startswith("goal_")]
            if evidence or goal_source == "override":
                plan.goal_profile_applied = True
                applied.append(goal_hit.section.id)
                for key, value in goal_hit.section.prefs.items():
                    put(key, value, "inferred", goal_hit.section.id)
            else:
                plan.adjustments.append(
                    f"the intent model suggested '{goal}', but no word in the request matches that goal's knowledge "
                    f"profile ({goal_hit.section.id}), so the profile was not applied")
        elif goal and req.use_retrieval:
            plan.adjustments.append(f"the intent model suggested '{goal}', but no matching knowledge profile was retrieved")

        # layer 2: phrase modifiers (only those whose keywords appear in the request)
        for hit in hits:
            if hit.section.kind != "modifier" or not hit.matched:
                continue
            used = False
            for key, value in hit.section.prefs.items():
                old = plan.soft.get(key)
                if old is None:
                    put(key, value, "inferred", hit.section.id)
                    used = True
                elif old.value == value:
                    used = True
                elif isinstance(value, str) or isinstance(old.value, str):
                    plan.adjustments.append(f"{hit.section.id} wanted {key}={value} but {old.source} already set {old.value}")
                else:
                    spread = float(value != old.value) if isinstance(value, bool) else abs(float(value) - float(old.value))
                    limit = 0.0 if isinstance(value, bool) else CONFLICT_SPREAD.get(key, DEFAULT_CONFLICT_SPREAD)
                    if round(spread, 6) > limit:
                        conflicts.append((key, f"{key}: {old.source} wants {_fmt(old.value)} but {hit.section.id} wants {_fmt(value)}"))
                    elif old.source.startswith("KB-GOAL"):
                        put(key, value, "inferred", hit.section.id)
                        used = True
                    else:
                        plan.adjustments.append(f"{hit.section.id} wanted {key}={_fmt(value)} but higher-ranked {old.source} already set {_fmt(old.value)}")
            if used:
                applied.append(hit.section.id)

        # layer 3: explicit preferences always win over inferred ones
        for key, (value, source) in explicit.items():
            put(key, value, "explicit", source)
        plan.conflicts = [text for key, text in conflicts if key not in explicit]

        # hard constraints adjust inferred targets rather than being violated or ignored
        bounds = numeric_bounds(plan.constraints)
        for key, entry in list(plan.soft.items()):
            low, high = bounds.get(_FEATURE_OF_PREF.get(key, key), (None, None))
            if entry.origin == "inferred" and isinstance(entry.value, float):
                clamped = entry.value if low is None else max(entry.value, low)
                clamped = clamped if high is None else min(clamped, high)
                if clamped != entry.value:
                    plan.adjustments.append(f"inferred {key} target {_fmt(entry.value)} ({entry.source}) moved to "
                                            f"{_fmt(clamped)} to respect your hard limit")
                    entry.value = clamped
        excluded = {(c.feature, c.value) for c in plan.constraints if c.kind == "exclude"}
        for key in ("genre", "mood"):
            entry = plan.soft.get(key)
            if entry is not None and entry.origin == "inferred" and (key, entry.value) in excluded:
                plan.adjustments.append(f"inferred {key} '{entry.value}' ({entry.source}) dropped because you excluded it")
                del plan.soft[key]

        # layer 4: ranking mode. explicit option > strategy section > goal profile > default
        strategy_hits = [h for h in hits if h.section.kind == "strategy" and h.matched]
        mode_hit = next((h for h in strategy_hits if h.section.mode), None)
        if req.mode:
            plan.mode, plan.mode_source = req.mode, "cli"
        elif mode_hit is not None:
            plan.mode, plan.mode_source = mode_hit.section.mode, mode_hit.section.id
            applied.append(mode_hit.section.id)
        elif plan.goal_profile_applied and goal_hit.section.mode:
            plan.mode, plan.mode_source = goal_hit.section.mode, goal_hit.section.id

        # layer 5: variety. explicit option > strategy sections > goal profile > off
        div_sources: List[str] = []
        for dimension, cli_value in (("artist", req.artist_penalty), ("genre", req.genre_penalty)):
            from_strategy = [(h.section.diversity[dimension], h.section.id) for h in strategy_hits if dimension in h.section.diversity]
            if cli_value is not None:
                value, source = float(cli_value), "cli"
                plan.variety_requested[dimension] = value > 0
            elif from_strategy:
                value, source = max(v for v, _ in from_strategy), "+".join(sorted({i for _, i in from_strategy}))
                plan.variety_requested[dimension] = value > 0
                applied.extend(i for _, i in from_strategy)
            elif plan.goal_profile_applied and dimension in goal_hit.section.diversity:
                value, source = goal_hit.section.diversity[dimension], goal_hit.section.id
            else:
                continue
            setattr(plan, f"{dimension}_penalty", value)
            div_sources.extend(source.split("+"))

        if div_sources:
            plan.diversity_source = " + ".join(dict.fromkeys(div_sources))
        plan.applied_sources = list(dict.fromkeys(applied))
        trace.add("plan", "plan_built", **plan.to_dict())
        return plan

    def _collect_sources(self, plan: Plan, hits: Sequence[Hit]) -> Tuple[List[str], List[str]]:
        """Source ids to cite, and the explanatory notes (the `summary` lines of cited sections)."""
        if not hits:
            return [], []
        used_features = set(plan.soft) | {_FEATURE_OF_PREF.get(k, k) for k in plan.soft}
        used_features |= {c.feature for c in plan.constraints}
        if "likes_acoustic" in plan.soft:
            used_features.add("acousticness")
        if {"genre", "mood"} & set(plan.soft):
            used_features.add("genre_mood")
        feature_ids = [h.section.id for h in hits if h.section.kind == "feature"
                       and (h.section.feature in used_features or
                            (h.section.feature == "decade" and "release_decade" in used_features))][:3]
        policy = ["KB-POLICY-SYNTHETIC"]
        if any(e.origin == "inferred" for e in plan.soft.values()):
            policy.append("KB-POLICY-INFERENCE")
        if not (plan.artist_penalty or plan.genre_penalty):
            policy.append("KB-POLICY-FILTER-BUBBLE")
        policy.append("KB-POLICY-COVERAGE")
        cited = plan.applied_sources + feature_ids + policy
        notes = [f"[{sid}] {self.kb.get(sid).summary}" for sid in cited if sid in self.kb]
        return cited, notes
