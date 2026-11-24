"""
Evaluation harness for VibeFinder AI.

    python3 -m src.evaluate                       # run everything, print PASS/FAIL and a summary
    python3 -m src.evaluate --write-traces traces # also save representative runtime traces as JSON

Four parts, all executed against the real system (nothing is mocked except the deliberately
faulty engines in part 4, which are labelled as injected faults):

  1. Scenario cases    predefined requests with an expected outcome; the actual outcome is checked.
  2. Intent model      keyword baseline vs the Naive Bayes model on the held-out file (reported,
                       not pass/fail), plus dataset integrity checks (pass/fail).
  3. Retrieval         the same request with and without knowledge retrieval.
  4. Guardrails        engine faults delivered with guardrails off vs corrected/withheld with them on.

A passing run shows the system behaves as specified on these cases. It is not evidence of
recommendation quality for real listeners. Exit status is 1 if any pass/fail check fails.
"""

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import src.agent as agent_module
from src.agent import Agent, AgentResult, AssistantRequest
from src.assistant import build_agent, compare_retrieval
from src.intent import (
    DEFAULT_ALPHA, DEFAULT_BIGRAMS, DEFAULT_MIN_SCORE, DEV_FILE, EVAL_LABELS, HELDOUT_FILE, TRAIN_FILE,
    TRAIN_LABELS, UNCERTAIN, UNSUPPORTED, compare_models, default_classifier, load_intent_examples,
    overlapping_examples, parse_require, summarize_comparison, tune_on_dev,
)
from src.recommender import Recommendation, rank_songs, score_breakdown

TRACE_DIR = Path(__file__).resolve().parent.parent / "traces"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def fmt_options(options: Optional[Dict[str, Any]]) -> str:
    """Options as readable text (constraints shown by their rule, not their repr)."""
    if not options:
        return ""
    parts = []
    for key, value in options.items():
        if key == "require":
            value = [c.describe() for c in value]
        elif key == "exclude_genres":
            value = list(value)
        parts.append(f"{key}={value}")
    return ", ".join(parts)


def _ids(result: AgentResult) -> List[int]:
    return [r.song["id"] for r in result.recommendations]


def _titles(result: AgentResult) -> str:
    return ", ".join(r.song["title"] for r in result.recommendations) or "(none)"


def describe(result: AgentResult) -> str:
    goal = result.plan.goal if result.plan and result.plan.goal else (result.prediction.label if result.prediction else "-")
    shown = f"top {len(result.recommendations)}: {_titles(result)}" if result.recommendations else "no recommendations"
    extra = f"; corrections={len(result.corrections)}" if result.corrections else ""
    return f"status={result.status}; goal={goal}; {shown}{extra}"


def _all(result: AgentResult, feature: str, test: Callable[[float], bool]) -> bool:
    return bool(result.recommendations) and all(test(r.song[feature]) for r in result.recommendations)


def _mean(result: AgentResult, feature: str) -> float:
    return sum(r.song[feature] for r in result.recommendations) / len(result.recommendations)


# ---------------------------------------------------------------------------
# 1. scenario cases
# ---------------------------------------------------------------------------

@dataclass
class Case:
    id: str
    category: str
    query: Any
    expect: str
    check: Callable[[AgentResult], Optional[str]]      # returns None on success, else why it failed
    options: Dict[str, Any] | None = None


def _status(expected: str) -> Callable[[AgentResult], Optional[str]]:
    return lambda r: None if r.status == expected and not r.recommendations else \
        f"expected status {expected} with no recommendations, got {r.status} with {len(r.recommendations)}"


def _goal_or_clarify(goal: str) -> Callable[[AgentResult], Optional[str]]:
    def check(r: AgentResult) -> Optional[str]:
        if r.status == "needs_clarification" and not r.recommendations:
            return None                                        # a safe abstention is acceptable
        if r.status == "ok" and r.plan.goal == goal:
            return None
        return f"expected goal '{goal}' or a clarification, got {describe(r)}"
    return check


def _need(cond: bool, why: str) -> Optional[str]:
    return None if cond else why


def _study(r: AgentResult) -> Optional[str]:
    if r.status != "ok" or r.plan.goal != "study":
        return f"expected an ok study plan, got {describe(r)}"
    return _need(_all(r, "instrumentalness", lambda v: v >= 0.5) and _all(r, "energy", lambda v: v <= 0.5),
                 "a result is not both instrumental (>= 0.5) and calm (energy <= 0.5)")


def _workout(r: AgentResult) -> Optional[str]:
    if r.status != "ok" or r.plan.goal != "workout":
        return f"expected an ok workout plan, got {describe(r)}"
    return _need(_all(r, "energy", lambda v: v >= 0.8), "a result has energy below 0.8")


def _party(r: AgentResult) -> Optional[str]:
    if r.status != "ok" or r.plan.goal != "party":
        return f"expected an ok party plan, got {describe(r)}"
    artists = [x.song["artist"] for x in r.recommendations]
    return _need(len(set(artists)) == len(artists) and _all(r, "valence", lambda v: v >= 0.6),
                 f"artists repeat or a song is not bright: {artists}")


def _relax(r: AgentResult) -> Optional[str]:
    if r.status != "ok" or r.plan.goal != "relax":
        return f"expected an ok relax plan, got {describe(r)}"
    return _need(_all(r, "energy", lambda v: v <= 0.60) and r.candidates < r.catalog_size,
                 "a result exceeds the energy cap, or the cap removed nothing")


def _fewer(r: AgentResult) -> Optional[str]:
    return _need(r.status == "ok" and 0 < len(r.recommendations) < 10 and len(r.recommendations) == r.candidates
                 and _all(r, "energy", lambda v: v <= 0.25),
                 f"expected fewer than 10 valid songs, got {describe(r)}")


def _distinct(feature: str, n: int) -> Callable[[AgentResult], Optional[str]]:
    def check(r: AgentResult) -> Optional[str]:
        values = [x.song[feature] for x in r.recommendations]
        return _need(r.status == "ok" and len(values) == n and len(set(values)) == len(values),
                     f"expected {n} different {feature}s, got {values}")
    return check


def _plan(test: Callable[[Any], bool], why: str) -> Callable[[AgentResult], Optional[str]]:
    return lambda r: _need(r.status == "ok" and test(r.plan), f"{why}; got {describe(r)}")


def _no_ban(r: AgentResult) -> Optional[str]:
    if r.status != "ok":
        return f"expected ok, got {describe(r)}"
    return _need(all(x.song["genre"] not in ("pop", "electronic") for x in r.recommendations) and r.candidates < r.catalog_size,
                 "a banned genre was returned or nothing was filtered")


def _slow(r: AgentResult) -> Optional[str]:
    return _need(r.status == "ok" and _all(r, "tempo_bpm", lambda v: v <= 80), f"expected only songs under 80 BPM, got {describe(r)}")


def _disclosed(r: AgentResult) -> Optional[str]:
    return _need(r.status == "ok" and any("language" in n for n in r.plan.not_applied),
                 "the unsupported language request was not disclosed as not applied")


def build_cases() -> List[Case]:
    return [
        Case("study-basic", "study", "Find calm instrumental music for studying",
             "ok, goal study; every song instrumental (>= 0.5) and calm (energy <= 0.5)", _study),
        Case("workout-basic", "workout", "Give me energetic music for a workout",
             "ok, goal workout; every song has energy >= 0.8", _workout),
        Case("party-diverse", "party", "Recommend upbeat party music with different artists",
             "ok, goal party; five different artists, every song valence >= 0.6", _party),
        Case("relax-capped", "relaxation", "I want something relaxing without too much energy",
             "ok, goal relax; hard cap energy <= 0.60 removes songs and every result obeys it", _relax),
        Case("unfamiliar-study", "unfamiliar wording", "background sounds while I finish my essay",
             "goal study, or a safe clarification (never a wrong confident goal)", _goal_or_clarify("study")),
        Case("unfamiliar-workout", "unfamiliar wording", "tunes to keep me moving on the treadmill",
             "goal workout, or a safe clarification", _goal_or_clarify("workout")),
        Case("unfamiliar-relax", "unfamiliar wording", "something mellow for the evening",
             "goal relax, or a safe clarification", _goal_or_clarify("relax")),
        Case("unfamiliar-party", "unfamiliar wording", "what should be playing when the guests arrive tonight",
             "goal party, or a safe clarification", _goal_or_clarify("party")),
        Case("ambiguous-bare", "ambiguous", "music please", "needs_clarification, no songs", _status("needs_clarification")),
        Case("ambiguous-vague", "ambiguous", "something good", "needs_clarification, no songs", _status("needs_clarification")),
        Case("ambiguous-unsure", "ambiguous", "I am not sure",
             "needs_clarification (the model alone says 'discovery'; no word supports it)", _status("needs_clarification")),
        Case("unsupported-weather", "unsupported", "What is the weather like today", "unsupported, no songs", _status("unsupported")),
        Case("unsupported-booking", "unsupported", "book me a hotel in paris",
             "no songs (unsupported or a clarification)",
             lambda r: _need(r.status in ("unsupported", "needs_clarification") and not r.recommendations, f"got {describe(r)}")),
        Case("unsupported-streaming", "unsupported", "play jazz on spotify", "no songs; it cannot play on a service", 
             lambda r: _need(r.status in ("unsupported", "needs_clarification") and not r.recommendations, f"got {describe(r)}")),
        Case("unsupported-language", "unsupported", "calm instrumental music for studying in spanish",
             "ok, and the language request is disclosed as not applied", _disclosed),
        Case("contradiction-goal", "contradictory", "Give me calm music for an intense workout",
             "needs_clarification naming the energy conflict", _status("needs_clarification")),
        Case("contradiction-genre", "contradictory", "pop music but no pop", "needs_clarification", _status("needs_clarification")),
        Case("contradiction-limits", "contradictory", "music for studying",
             "needs_clarification: energy >= 0.9 and energy <= 0.3 cannot both hold", _status("needs_clarification"),
             {"require": (parse_require("energy>=0.9"), parse_require("energy<=0.3"))}),
        Case("insufficient-fewer", "insufficient matches", "music for studying",
             "ok with fewer than 10 songs, all valid (energy <= 0.25), shortfall stated", _fewer,
             {"count": 10, "require": (parse_require("energy<=0.25"),)}),
        Case("insufficient-none", "insufficient matches", "music for studying",
             "no_match: no song has energy >= 0.99; constraint is not silently relaxed", 
             lambda r: _need(r.status == "no_match" and r.candidates == 0 and not r.recommendations, f"got {describe(r)}"),
             {"require": (parse_require("energy>=0.99"),)}),
        Case("diversity-artists", "diversity", "pop party music with different artists",
             "ok; six songs, six different artists", _distinct("artist", 6), {"count": 6}),
        Case("diversity-genres", "diversity", "party music with different genres",
             "ok; five songs, five different genres", _distinct("genre", 5)),
        Case("override-explicit", "explicit override", "Find calm instrumental music for studying",
             "the explicit energy 0.9 replaces the inferred 0.3 and is recorded as explicit",
             _plan(lambda p: p.soft["energy"].value == 0.9 and p.soft["energy"].origin == "explicit", "energy not overridden"),
             {"prefer": {"energy": 0.9}}),
        Case("hard-exclude", "hard constraint", "energetic music for a workout, no pop, no electronic",
             "ok; no pop or electronic song, and the exclusion removed candidates", _no_ban),
        Case("hard-tempo", "hard constraint", "relaxing music under 80 bpm",
             "ok; every song is 80 BPM or slower", _slow),
        Case("malformed-empty", "malformed", "", "invalid_input", _status("invalid_input")),
        Case("malformed-none", "malformed", None, "invalid_input", _status("invalid_input")),
        Case("malformed-count", "malformed", "give me 0 songs of jazz", "invalid_input (count must be 1-15)", _status("invalid_input")),
        Case("malformed-long", "malformed", "calm " * 200, "invalid_input (request too long)", _status("invalid_input")),
        Case("malformed-pref", "malformed", "music for studying", "invalid_input (energy must be 0-1)", _status("invalid_input"),
             {"prefer": {"energy": 5}}),
    ]


def run_cases(cases: Sequence[Case]) -> List[Tuple[Case, AgentResult, Optional[str]]]:
    agent = build_agent()
    rows = []
    for case in cases:
        result = agent.run(AssistantRequest(query=case.query, **(case.options or {})))
        rows.append((case, result, case.check(result)))
    return rows


# ---------------------------------------------------------------------------
# 3. retrieval ablation
# ---------------------------------------------------------------------------

@dataclass
class Ablation:
    id: str
    query: str
    expect: str
    options: Dict[str, Any]
    check: Callable[[AgentResult, AgentResult], Optional[str]]


def _distinct_artists(r: AgentResult) -> int:
    return len({x.song["artist"] for x in r.recommendations})


def build_ablations() -> List[Ablation]:
    def answerable(w: AgentResult, wo: AgentResult) -> Optional[str]:
        return _need(w.status == "ok" and wo.status == "needs_clarification",
                     f"expected ok with retrieval and a clarification without, got {w.status} / {wo.status}")

    def energy_up(w: AgentResult, wo: AgentResult) -> Optional[str]:
        if w.status != "ok" or wo.status != "ok":
            return f"both runs should answer, got {w.status} / {wo.status}"
        if _ids(w) == _ids(wo):
            return "retrieval did not change the ranking"
        return _need(_mean(w, "energy") > _mean(wo, "energy"),
                     f"mean energy {_mean(w, 'energy'):.2f} (with) is not above {_mean(wo, 'energy'):.2f} (without)")

    def variety(w: AgentResult, wo: AgentResult) -> Optional[str]:
        return _need(_distinct_artists(w) == 6 and _distinct_artists(wo) < 6,
                     f"distinct artists with/without: {_distinct_artists(w)}/{_distinct_artists(wo)}")

    def no_ranking_effect(w: AgentResult, wo: AgentResult) -> Optional[str]:
        same_order = _ids(w) == _ids(wo)
        scores_differ = [round(r.score, 4) for r in w.recommendations] != [round(r.score, 4) for r in wo.recommendations]
        return _need(w.status == wo.status == "ok" and same_order and scores_differ,
                     f"expected identical order but different scores; same_order={same_order} scores_differ={scores_differ}")

    def instrumental_up(w: AgentResult, wo: AgentResult) -> Optional[str]:
        if w.status != "ok" or wo.status != "ok":
            return f"both runs should answer, got {w.status} / {wo.status}"
        return _need(_mean(w, "instrumentalness") > _mean(wo, "instrumentalness"),
                     f"mean instrumentalness {_mean(w, 'instrumentalness'):.2f} (with) vs {_mean(wo, 'instrumentalness'):.2f} (without)")

    return [
        Ablation("answerable", "Find calm instrumental music for studying",
                 "with retrieval: a plan and songs; without: a clarification (nothing explicit to score)", {}, answerable),
        Ablation("workout-genre", "pop music for my workout",
                 "retrieval changes the ranking and raises mean energy of the results", {}, energy_up),
        Ablation("variety", "pop party music with different artists",
                 "retrieval reads 'different artists' and gives 6 distinct artists; without it artists repeat", {"count": 6}, variety),
        Ablation("study-genre", "jazz for studying",
                 "retrieval raises mean instrumentalness of the results", {}, instrumental_up),
        Ablation("no-effect", "chill lofi for studying",
                 "explicit genre+mood already decide the order: same order, different scores (retrieval has no ranking effect)", {},
                 no_ranking_effect),
    ]


# ---------------------------------------------------------------------------
# 4. guardrail effectiveness (injected engine faults)
# ---------------------------------------------------------------------------

@dataclass
class Fault:
    id: str
    description: str
    query: str
    options: Dict[str, Any]
    engine: Callable[..., List[Recommendation]]
    defective: Callable[[AgentResult], bool]       # independent, simple check that the delivered output is bad


def _wrap_engine(mutate: Callable[[List[Recommendation]], List[Recommendation]]) -> Callable[..., List[Recommendation]]:
    def engine(prefs, songs, k, weights=None, diversity=None):
        return mutate(rank_songs(prefs, songs, k, weights, diversity))
    return engine


def build_faults() -> List[Fault]:
    catalog_ids = {s["id"] for s in build_agent().songs}
    first_pop = next(s for s in build_agent().songs if s["genre"] == "pop")

    def stubborn(prefs, songs, k, weights=None, diversity=None):
        strong = diversity is not None and diversity.artist_penalty >= 3.0
        return rank_songs(prefs, songs, k, weights, diversity if strong else None)

    def bad_score(r: AgentResult) -> bool:
        return any(abs(x.score - score_breakdown(r.plan.prefs, x.song).total) > 1e-6 for x in r.recommendations)

    def bad_text(r: AgentResult) -> bool:
        return any(list(x.reasons) != score_breakdown(r.plan.prefs, x.song).reasons for x in r.recommendations)

    study = "Find calm instrumental music for studying"
    return [
        Fault("duplicate-song", "the engine returns the first song twice", study, {},
              _wrap_engine(lambda rs: [rs[0], rs[0]] + rs[1:4]),
              lambda r: len(_ids(r)) != len(set(_ids(r)))),
        Fault("nonexistent-song", "the engine invents a song (id 777)", study, {},
              _wrap_engine(lambda rs: [replace(rs[0], song=dict(rs[0].song, id=777, title="Imaginary Anthem"))] + rs[1:]),
              lambda r: any(i not in catalog_ids for i in _ids(r))),
        Fault("constraint-violation", "the engine ignores 'no pop' and puts a pop song first",
              "energetic music for a workout", {"exclude_genres": ("pop",)},
              _wrap_engine(lambda rs: [Recommendation(first_pop, rs[0].score, tuple(rs[0].reasons))] + rs[1:]),
              lambda r: any(x.song["genre"] == "pop" for x in r.recommendations)),
        Fault("wrong-score", "the engine reports a score 1.5 too high", study, {},
              _wrap_engine(lambda rs: [replace(rs[0], score=rs[0].score + 1.5)] + rs[1:]), bad_score),
        Fault("wrong-explanation", "the engine attaches song 2's explanation to song 1", study, {},
              _wrap_engine(lambda rs: [replace(rs[0], reasons=rs[1].reasons)] + rs[1:]), bad_text),
        Fault("missed-variety", "the engine ignores variety unless the penalty is at least 3.0",
              "pop party music with different artists", {"count": 6}, stubborn,
              lambda r: len({x.song["artist"] for x in r.recommendations}) < len(r.recommendations)),
    ]


def run_fault(fault: Fault, guardrails: bool) -> AgentResult:
    base = build_agent()
    agent = Agent(base.songs, base.kb, recommend_fn=fault.engine)
    return agent.run(AssistantRequest(query=fault.query, guardrails=guardrails, **fault.options))


# ---------------------------------------------------------------------------
# traces
# ---------------------------------------------------------------------------

def write_traces(directory: Path) -> List[Dict[str, str]]:
    """Run representative requests and save their runtime events as JSON, plus an index README."""
    directory.mkdir(parents=True, exist_ok=True)
    agent = build_agent()
    faults = {f.id: f for f in build_faults()}
    runs: List[Tuple[str, str, str, Optional[str], AgentResult]] = []

    def real(name: str, description: str, query: str, flags: str = "", **options: Any) -> None:
        result = agent.run(AssistantRequest(query=query, **options))
        runs.append((name, description, f'python3 -m src.assistant --query "{query}"{flags} --trace-file traces/{name}.json', None, result))

    def injected(name: str, description: str, fault_id: str) -> None:
        result = run_fault(faults[fault_id], guardrails=True)
        runs.append((name, description, "python3 -m src.evaluate --write-traces traces", faults[fault_id].description, result))

    real("study", "Study request: classifier, retrieval, plan from three sections, validation passes.",
         "Find calm instrumental music for studying")
    real("workout", "Workout request: explicit mood from the wording plus a retrieved goal profile.",
         "Give me energetic music for a workout")
    real("party_goal_correction", "The intent model says 'discovery'; the word 'party' corrects the goal; retrieval selects artist variety.",
         "Recommend upbeat party music with different artists")
    real("relax_hard_constraint", "A hard energy cap parsed from the wording filters the catalog before ranking.",
         "I want something relaxing without too much energy")
    real("clarification_conflict", "Contradictory goal and wording: the agent stops and asks instead of guessing.",
         "Give me calm music for an intense workout")
    real("unsupported", "Out-of-scope request: routed to 'unsupported' before retrieval.", "What is the weather like today")
    real("retrieval_disabled", "Same study request with retrieval disabled: nothing explicit to score, so it clarifies.",
         "Find calm instrumental music for studying", " --no-retrieval", use_retrieval=False)
    real("insufficient_matches", "Only a few songs satisfy the hard limit, so fewer than requested are returned.",
         "music for studying", " --count 10 --require 'energy<=0.25'", count=10, require=(parse_require("energy<=0.25"),))
    real("variety_correction_natural", "NOT injected: the first ranking missed 'different genres'; validation flagged it and one stronger re-rank fixed it.",
         "party music with different genres")
    injected("guardrail_duplicate_corrected", "INJECTED FAULT: the engine returns a duplicate; the validator catches it and one correction fixes it.",
             "duplicate-song")
    injected("guardrail_variety_corrected", "INJECTED FAULT: the engine misses a variety request; one stronger re-rank fixes it.",
             "missed-variety")

    index = ["# Runtime traces", "",
             "Structured events recorded by the application agent (tool calls, tool results, decisions, validation, corrections)",
             "for representative requests. They are observable execution records, not reasoning transcripts, and contain no timestamps,",
             "so regenerating them gives identical files (`tests/test_evaluate.py` checks this).", "",
             "Regenerate all of them with `python3 -m src.evaluate --write-traces traces`.", "",
             "| File | What it shows | Status | Command |", "|---|---|---|---|"]
    entries = []
    for name, description, command, fault, result in runs:
        payload = {"description": description, "command": command, "injected_fault": fault, "query": result.query,
                   "status": result.status, "events": result.events}
        (directory / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        index.append(f"| [{name}.json]({name}.json) | {description} | {result.status} | `{command}` |")
        entries.append({"file": f"{name}.json", "status": result.status})
    (directory / "README.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    return entries


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def _rule(title: str) -> str:
    return f"\n{'=' * 78}\n{title}\n{'=' * 78}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m src.evaluate", description=__doc__.split("\n\n")[0])
    parser.add_argument("--write-traces", metavar="DIR", help="also save representative runtime traces (JSON) into DIR")
    args = parser.parse_args(argv)
    failed = 0
    totals: Dict[str, Tuple[int, int]] = {}

    # --- 1 -------------------------------------------------------------------------------------
    print(_rule("1. SCENARIO CASES (real system, predefined expectations)"))
    rows = run_cases(build_cases())
    for case, result, problem in rows:
        shown = case.query if isinstance(case.query, str) and len(case.query) < 90 else repr(case.query)[:80] + "..."
        print(f"[{'PASS' if problem is None else 'FAIL'}] {case.id}  ({case.category})")
        print(f"       input   : {shown}")
        if case.options:
            print(f"       options : {fmt_options(case.options)}")
        print(f"       expected: {case.expect}")
        print(f"       actual  : {describe(result)}")
        if problem:
            print(f"       failure : {problem}")
    passed = sum(1 for _, _, p in rows if p is None)
    totals["scenario cases"] = (passed, len(rows))

    # --- 2 -------------------------------------------------------------------------------------
    print(_rule("2. INTENT MODEL: keyword baseline vs Naive Bayes (held-out file, reported not pass/fail)"))
    train = load_intent_examples(TRAIN_FILE, TRAIN_LABELS)
    dev = load_intent_examples(DEV_FILE)
    held = load_intent_examples(HELDOUT_FILE)
    integrity = [
        ("no sentence appears in more than one split", overlapping_examples(train, dev, held) == []),
        ("training labels are the six trained labels only", {l for _, l in train} == set(TRAIN_LABELS)),
        ("held-out file covers all seven evaluation labels", {l for _, l in held} == set(EVAL_LABELS)),
    ]
    for name, ok in integrity:
        print(f"[{'PASS' if ok else 'FAIL'}] dataset integrity: {name}")
    totals["dataset integrity"] = (sum(ok for _, ok in integrity), len(integrity))

    grid = tune_on_dev(train, dev)
    print(f"\nsettings chosen on the DEV file only: alpha={DEFAULT_ALPHA}, abstain below {DEFAULT_MIN_SCORE}, "
          f"bigrams={DEFAULT_BIGRAMS}  (dev accuracy {grid[0][3]:.3f}; the held-out file was not used to choose them)")
    print(f"sizes: train {len(train)}, dev {len(dev)}, held-out {len(held)}")
    comparison = compare_models(held)
    summary = summarize_comparison(comparison)

    def line(label: str, stats: Dict[str, Any]) -> str:
        return (f"  {label:<26} n={stats['n']:>2}   keyword baseline {stats['baseline_correct']:>2} "
                f"({stats['baseline_accuracy']:.1%})   Naive Bayes {stats['specialized_correct']:>2} ({stats['specialized_accuracy']:.1%})")
    print("\nheld-out accuracy (same examples, same labels):")
    print(line("all 66 examples", summary["overall"]))
    print(line("five listening goals only", summary["goals_only"]))
    for label, stats in summary["per_label"].items():
        print(line(f"label: {label}", stats))
    print(f"\n  specialized right, baseline wrong: {len(summary['specialized_only_correct'])}"
          f" | baseline right, specialized wrong: {len(summary['baseline_only_correct'])}"
          f" | both wrong: {len(summary['both_wrong'])}")
    print("\nNaive Bayes failures (expected -> predicted; 'uncertain' = the model abstained):")
    for r in comparison:
        if r.specialized != r.expected:
            kind = "abstained" if r.specialized == UNCERTAIN else "WRONG"
            print(f"  [{kind:<9}] {r.expected:>11} -> {r.specialized:<11} (relative score {r.specialized_score:.2f})  {r.text}")
    print("\nbaseline wins (baseline right, Naive Bayes wrong):")
    for r in summary["baseline_only_correct"] or []:
        print(f"  {r.expected:>11}: baseline {r.baseline}, Naive Bayes {r.specialized} ({r.specialized_score:.2f})  {r.text}")
    if not summary["baseline_only_correct"]:
        print("  none")

    # pipeline-level goal decision (post-hoc: the corroboration and correction rules were designed after seeing the model's held-out failures)
    agent = build_agent()
    def pipeline_label(text: str) -> str:
        result = agent.run(AssistantRequest(query=text))
        if result.status == "unsupported":
            return UNSUPPORTED
        return result.plan.goal if result.plan and result.plan.goal_profile_applied else UNCERTAIN
    pipeline = [pipeline_label(r.text) for r in comparison]
    pipe_ok = sum(p == r.expected for p, r in zip(pipeline, comparison))
    goal_rows = [(p, r) for p, r in zip(pipeline, comparison) if r.expected in ("study", "workout", "party", "relax", "discovery")]
    wrong_model = sum(1 for r in comparison if r.specialized not in (r.expected, UNCERTAIN))
    wrong_pipe = sum(1 for p, r in zip(pipeline, comparison) if p not in (r.expected, UNCERTAIN))
    print(f"\npipeline goal decision with retrieval evidence rules (POST-HOC, not an independent measurement):")
    print(f"  correct {pipe_ok}/{len(comparison)} ({pipe_ok / len(comparison):.1%}) vs Naive Bayes alone "
          f"{summary['overall']['specialized_correct']}/{len(comparison)}")
    print(f"  confident wrong answers (neither correct nor an abstention): {wrong_pipe} in the pipeline vs {wrong_model} for the model alone")
    print(f"  five goals only: {sum(p == r.expected for p, r in goal_rows)}/{len(goal_rows)}")

    # --- 3 -------------------------------------------------------------------------------------
    print(_rule("3. RETRIEVAL ENABLED vs DISABLED (same input, same catalog)"))
    ok_count = 0
    ablations = build_ablations()
    for ab in ablations:
        with_kb, without, diff = compare_retrieval(AssistantRequest(query=ab.query, **ab.options))
        problem = ab.check(with_kb, without)
        ok_count += problem is None
        print(f"[{'PASS' if problem is None else 'FAIL'}] {ab.id}")
        print(f"       input   : {ab.query}" + (f"  [{fmt_options(ab.options)}]" if ab.options else ""))
        print(f"       expected: {ab.expect}")
        print(f"       with    : {describe(with_kb)}")
        print(f"       without : {describe(without)}")
        if with_kb.recommendations and without.recommendations:
            print(f"       measured: mean energy {_mean(with_kb, 'energy'):.2f} vs {_mean(without, 'energy'):.2f}; "
                  f"mean instrumentalness {_mean(with_kb, 'instrumentalness'):.2f} vs {_mean(without, 'instrumentalness'):.2f}; "
                  f"distinct artists {_distinct_artists(with_kb)} vs {_distinct_artists(without)}; "
                  f"shared songs {diff['shared_songs']}/{len(with_kb.recommendations)}")
        if problem:
            print(f"       failure : {problem}")
    totals["retrieval comparison"] = (ok_count, len(ablations))

    # --- 4 -------------------------------------------------------------------------------------
    print(_rule("4. GUARDRAIL EFFECTIVENESS (injected engine faults, guardrails on vs off)"))
    print("Each fault is injected into the recommendation engine on purpose. A case passes when the faulty output\n"
          "reaches the user with guardrails OFF (the fault is real) and does NOT with guardrails ON.\n")
    ok_count = 0
    faults = build_faults()
    for fault in faults:
        off, on = run_fault(fault, False), run_fault(fault, True)
        bad_off = fault.defective(off)
        bad_on = fault.defective(on) if on.recommendations else False
        problems = []
        if not bad_off:
            problems.append("with guardrails off the fault did not reach the user, so the test proves nothing")
        if bad_on or on.status not in ("ok", "failed_validation"):
            problems.append(f"with guardrails on the defect was still delivered (status {on.status})")
        ok_count += not problems
        print(f"[{'PASS' if not problems else 'FAIL'}] {fault.id}: {fault.description}")
        print(f"       input        : {fault.query}" + (f"  [{fmt_options(fault.options)}]" if fault.options else ""))
        print(f"       guardrails OFF: delivered defective output = {bad_off}  (status {off.status})")
        fixes = "; ".join(f"{c['strategy']} addressed {','.join(c['addressed'])}" for c in on.corrections) or "none"
        print(f"       guardrails ON : status {on.status}; defective output delivered = {bad_on}; corrections: {fixes}")
        for problem in problems:
            print(f"       failure      : {problem}")
    totals["guardrail faults"] = (ok_count, len(faults))

    # --- traces ---------------------------------------------------------------------------------
    if args.write_traces:
        entries = write_traces(Path(args.write_traces))
        print(_rule("TRACES"))
        print(f"wrote {len(entries)} traces and an index to {args.write_traces}/")

    # --- summary --------------------------------------------------------------------------------
    print(_rule("SUMMARY"))
    for name, (passed, total) in totals.items():
        failed += total - passed
        print(f"  {name:<22} {passed}/{total} passed")
    grand_passed = sum(p for p, _ in totals.values())
    grand_total = sum(t for _, t in totals.values())
    print(f"\n  TOTAL: {grand_passed}/{grand_total} checks passed, {failed} failed")
    print("  (Passing shows the system behaves as specified on these cases; it does not prove recommendation quality.)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
