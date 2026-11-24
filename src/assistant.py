"""
VibeFinder AI command line: ask for music in plain language.

    python3 -m src.assistant --query "Find calm instrumental music for studying"

Response generation is deterministic template filling (retrieval-augmented template generation).
It is not an LLM: every sentence is assembled from the plan, the validated results and the
`summary` lines of the knowledge sections the plan actually used.

Exit status: 0 when the request was handled (recommendations, a clarification, an
out-of-scope notice or an honest "no match"), 1 when results were withheld because they failed
validation, 2 for invalid input.
"""

import argparse
import json
import textwrap
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.agent import Agent, AgentResult, AssistantRequest
from src.guardrails import DEFAULT_RESULTS, MAX_RESULTS
from src.intent import FEATURE_ALIASES, Constraint, parse_require
from src.main import SONGS_CSV
from src.recommender import DEFAULT_MODE, RANKING_MODES, load_songs
from src.report import MIN_WIDTH, format_recommendation_list, format_recommendation_table
from src.retrieval import GOALS, load_knowledge

DEFAULT_WIDTH = 100
DEMO_QUERIES = (
    "Find calm instrumental music for studying",
    "Give me energetic music for a workout",
    "Recommend upbeat party music with different artists",
)
_PREFER_KEYS = ("genre", "mood", "likes_acoustic", "energy", "valence", "danceability", "tempo_bpm",
                "popularity", "decade", "instrumentalness", "speechiness", "liveness")


@lru_cache(maxsize=1)
def build_agent() -> Agent:
    """The shipped agent: the 30-song catalog, the knowledge collection and the trained intent model."""
    return Agent(load_songs(str(SONGS_CSV)), load_knowledge())


def run_assistant(query: Any, **options: Any) -> AgentResult:
    """Programmatic entry point: run_assistant("calm music for studying", count=3, use_retrieval=False)."""
    return build_agent().run(AssistantRequest(query=query, **options))


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _wrap(text: str, width: int, indent: str = "", subsequent: Optional[str] = None) -> str:
    return textwrap.fill(text, width=width, initial_indent=indent,
                         subsequent_indent=indent if subsequent is None else subsequent,
                         break_long_words=True, break_on_hyphens=False)


def _value(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    return f"{value:g}" if isinstance(value, float) else str(value)


def _describe_pref(key: str, entry: Any) -> str:
    where = {"request": "your request", "cli": "--prefer"}.get(entry.source, entry.source)
    note = f"; replaces {', '.join(entry.overridden)}" if entry.overridden else ""
    return f"{key} = {_value(entry.value)}   [{entry.origin}: {where}{note}]"


def format_response(result: AgentResult, width: int = DEFAULT_WIDTH, layout: str = "table") -> str:
    """Render an AgentResult as plain text."""
    out: List[str] = ["VibeFinder AI", "=============", _wrap(f"Request : {result.query}", width, "", "          "),
                      f"Status  : {result.status}"]
    plan = result.plan

    if result.prediction is not None and result.status != "invalid_input":
        p = result.prediction
        if p.reason == "goal supplied explicitly":
            out.append(f"Goal    : {p.label} (set with --goal)")
        elif plan is not None and plan.goal_note:
            out.append(_wrap(f"Goal    : {plan.goal} ({plan.goal_note})", width, "", "          "))
        elif p.label in GOALS or p.label == "unsupported":
            out.append(_wrap(f"Goal    : {p.label} (intent model; relative score {p.score:.2f}, not a calibrated probability)",
                             width, "", "          "))
        else:
            out.append(_wrap(f"Goal    : not determined ({p.reason})", width, "", "          "))
    out.append("")

    if result.status != "ok":
        out.append(_wrap(result.message, width))
        if plan is not None and plan.adjustments:
            out += ["", "What I did with your request:"] + [_wrap(a, width, "  - ", "    ") for a in plan.adjustments]
        if plan is not None and plan.not_applied:
            out += ["", "Not applied (the catalog cannot satisfy these):"] + [_wrap(a, width, "  - ", "    ") for a in plan.not_applied]
        if result.failures and result.status == "failed_validation":
            out += ["", "Validation findings:"] + [_wrap(f"{i.code}: {i.detail}", width, "  - ", "    ") for i in result.failures]
        return _fit("\n".join(out), width)

    out.append("Plan (what the ranking engine was asked to do)")
    explicit = [(k, e) for k, e in plan.soft.items() if e.origin == "explicit"]
    inferred = [(k, e) for k, e in plan.soft.items() if e.origin == "inferred"]
    for label, rows in (("explicit", explicit), ("inferred", inferred)):
        if rows:
            out.append(_wrap(f"  Soft {label} preferences (scoring targets, never filters):", width, "", "    "))
            out += [_wrap(_describe_pref(k, e), width, "    ", "      ") for k, e in rows]
    if not plan.soft:
        out.append("  Soft preferences: none")
    if plan.constraints:
        out.append("  Hard constraints (songs that fail are removed before ranking):")
        out += [_wrap(f"{c.describe()}   [{c.origin}]", width, "    ", "      ") for c in plan.constraints]
    else:
        out.append("  Hard constraints: none")
    out.append(_wrap(f"  Ranking mode: {plan.mode} ({_where(plan.mode_source)})", width, "", "    "))
    if plan.artist_penalty or plan.genre_penalty:
        origin = "set with an option" if plan.diversity_source == "cli" else f"from {plan.diversity_source}"
        out.append(_wrap(f"  Variety penalty: artist {plan.artist_penalty:g}, genre {plan.genre_penalty:g} ({origin})",
                         width, "", "    "))
    else:
        out.append("  Variety penalty: off")
    out.append(f"  Candidates: {result.candidates} of {result.catalog_size} songs passed the hard constraints")
    out.append("")

    shown = len(result.recommendations)
    asked = plan.count
    head = f"Top {shown}" + (f" (you asked for {asked})" if shown < asked else "")
    out.append(f"{head}; a perfect match would score {result.perfect_score:.2f}:")
    recs = result.recommendations
    out.append(format_recommendation_table(recs, width) if layout == "table" else format_recommendation_list(recs, width))
    out.append("")

    if result.notes:
        out.append(_wrap("Why this plan (project-authored guidance, not external research)", width, "", "  "))
        out += [_wrap(n, width, "  ", "      ") for n in result.notes]
        out.append("")

    caveats = list(plan.adjustments)
    caveats += [f"not applied: {text}" for text in plan.not_applied]
    caveats += [i.detail for i in result.infos]
    caveats += [f"warning: {w.detail}" for w in result.warnings]
    if not result.retrieval_used:
        caveats.append("knowledge retrieval was disabled: no inferred preferences, strategies or source notes were used")
        caveats.append("all songs and numbers are fictional; this is not a real streaming recommendation")
    if caveats:
        out.append("Assumptions, adjustments and limits")
        out += [_wrap(c, width, "  - ", "    ") for c in caveats]
        out.append("")

    if result.corrections:
        out.append(_wrap("Validation: " + "; ".join(
            f"correction {c['attempt']} ({c['strategy']}) addressed {', '.join(c['addressed'])}" for c in result.corrections),
            width, "", "            "))
    elif result.guardrails_used:
        out.append(_wrap("Validation: passed (scores and explanations re-derived, hard constraints, duplicates and sources checked)",
                         width, "", "            "))
    else:
        out.append("Validation: SKIPPED (guardrails disabled)")
    if result.cited_sources:
        scores = {h.section.id: h.score for h in result.hits}
        cited = [f"{sid} ({scores[sid]:.2f})" if sid in scores else f"{sid} (fixed note)" for sid in result.cited_sources]
        out.append(_wrap("Sources (retrieval score): " + ", ".join(cited), width, "", "                           "))
    return _fit("\n".join(out), width)


def _fit(text: str, width: int) -> str:
    """Final pass: wrap any line wider than `width`, keeping its indentation (table lines are already narrow enough)."""
    lines = []
    for line in text.split("\n"):
        if len(line) <= width:
            lines.append(line)
            continue
        indent = line[:len(line) - len(line.lstrip())]
        lines.append(textwrap.fill(line.strip(), width=width, initial_indent=indent, subsequent_indent=indent + "  ",
                                   break_long_words=True, break_on_hyphens=False))
    return "\n".join(lines)


def _where(source: str) -> str:
    return {"default": "default", "cli": "set with an option"}.get(source, f"from {source}")


def format_trace(events: Sequence[Dict[str, Any]], width: int = DEFAULT_WIDTH) -> str:
    """One compact line per runtime event."""
    lines = [_wrap("Runtime trace (structured events; the full JSON is available with --trace-file)", width)]
    for e in events:
        extras = {k: v for k, v in e.items() if k not in ("seq", "step", "event", "tool")}
        name = f"{e['event']}:{e['tool']}" if "tool" in e else e["event"]
        body = json.dumps(extras, separators=(",", ":"), sort_keys=False)
        lines.append(_wrap(f"{e['seq']:>2}. [{e['step']}] {name} {body}", width, "", "      "))
    return "\n".join(lines)


def compare_retrieval(request: AssistantRequest) -> Tuple[AgentResult, AgentResult, Dict[str, Any]]:
    """Run the same request with and without knowledge retrieval and summarise the difference."""
    from dataclasses import replace
    with_kb = build_agent().run(replace(request, use_retrieval=True))
    without = build_agent().run(replace(request, use_retrieval=False))
    def ids(r: AgentResult) -> List[int]:
        return [rec.song["id"] for rec in r.recommendations]

    a, b = ids(with_kb), ids(without)
    def prefs(r: AgentResult) -> Dict[str, Any]:
        return {} if r.plan is None else r.plan.prefs

    diff = {
        "with_status": with_kb.status, "without_status": without.status,
        "with_ids": a, "without_ids": b,
        "preferences_only_with_retrieval": sorted(set(prefs(with_kb)) - set(prefs(without))),
        "preferences_with_different_value": sorted(k for k in set(prefs(with_kb)) & set(prefs(without))
                                                   if prefs(with_kb)[k] != prefs(without)[k]),
        "mode": (with_kb.plan.mode if with_kb.plan else None, without.plan.mode if without.plan else None),
        "variety": ((with_kb.plan.artist_penalty, with_kb.plan.genre_penalty) if with_kb.plan else None,
                    (without.plan.artist_penalty, without.plan.genre_penalty) if without.plan else None),
        "shared_songs": len(set(a) & set(b)),
        "first_rank_that_differs": next((i for i, (x, y) in enumerate(zip(a, b), start=1) if x != y),
                                        None if a == b else min(len(a), len(b)) + 1),
    }
    return with_kb, without, diff


def format_comparison(query: str, with_kb: AgentResult, without: AgentResult, diff: Dict[str, Any],
                      width: int = DEFAULT_WIDTH) -> str:
    out = ["VibeFinder AI: retrieval comparison", "===================================",
           _wrap(f"Request : {query}", width, "", "          "),
           "Same input and same 30-song catalog; only knowledge retrieval differs.", ""]
    for title, result in (("WITH retrieval", with_kb), ("WITHOUT retrieval", without)):
        out.append(f"{title}  (status: {result.status})")
        if result.plan is not None and result.plan.soft:
            out += [_wrap(_describe_pref(k, e), width, "  ", "    ") for k, e in result.plan.soft.items()]
            out.append(f"  mode: {result.plan.mode}; variety: artist {result.plan.artist_penalty:g}, genre {result.plan.genre_penalty:g}")
        if result.recommendations:
            for i, rec in enumerate(result.recommendations, start=1):
                out.append(_wrap(f"{i}. {rec.song['title']} by {rec.song['artist']} (score {rec.score:.2f})", width, "  ", "     "))
        else:
            out.append(_wrap(result.message, width, "  ", "  "))
        out.append("")
    out.append("Difference")
    out.append(f"  preferences that exist only with retrieval: {', '.join(diff['preferences_only_with_retrieval']) or 'none'}")
    out.append(f"  preferences whose value differs: {', '.join(diff['preferences_with_different_value']) or 'none'}")
    out.append(f"  ranking mode (with, without): {diff['mode'][0]}, {diff['mode'][1]}")
    if diff["with_ids"] or diff["without_ids"]:
        out.append(f"  songs in both result lists: {diff['shared_songs']} of {max(len(diff['with_ids']), len(diff['without_ids']))}")
        out.append(f"  first rank where the lists differ: {diff['first_rank_that_differs'] or 'none (identical)'}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def _prefer(text: str) -> Tuple[str, Any]:
    key, sep, raw = text.partition("=")
    key = FEATURE_ALIASES.get(key.strip().lower(), key.strip().lower())
    if not sep or key not in _PREFER_KEYS:
        raise argparse.ArgumentTypeError(f"{text!r}: use KEY=VALUE with KEY one of {', '.join(_PREFER_KEYS)}")
    raw = raw.strip()
    if key in ("genre", "mood"):
        return key, raw
    if key == "likes_acoustic":
        if raw.lower() not in ("true", "false", "yes", "no"):
            raise argparse.ArgumentTypeError(f"{text!r}: likes_acoustic must be true or false")
        return key, raw.lower() in ("true", "yes")
    try:
        return key, float(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r}: {key} must be a number") from None


def _require(text: str) -> Constraint:
    try:
        return parse_require(text)
    except ValueError as err:
        raise argparse.ArgumentTypeError(str(err)) from None


def _width(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from None
    if value < MIN_WIDTH:
        raise argparse.ArgumentTypeError(f"width must be at least {MIN_WIDTH}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m src.assistant",
        description="VibeFinder AI: describe what you want to listen to and get recommendations from the "
                    "fictional 30-song catalog, with a visible plan, scores and validation.",
        epilog="examples:\n"
               "  python3 -m src.assistant --query \"Find calm instrumental music for studying\"\n"
               "  python3 -m src.assistant -q \"upbeat party music with different artists\" -n 3 --trace\n"
               "  python3 -m src.assistant -q \"chill lofi for studying\" --compare-retrieval\n"
               "  python3 -m src.assistant -q \"energetic music\" --prefer energy=0.95 --require tempo\\>=130\n"
               "  python3 -m src.assistant --demo\n\n"
               "exit status: 0 handled (including clarification, out-of-scope, no match); "
               "1 results withheld by validation; 2 invalid input",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-q", "--query", help="the request in plain language (required unless --demo)")
    parser.add_argument("-n", "--count", type=int, metavar="N",
                        help=f"how many songs (1-{MAX_RESULTS}; default {DEFAULT_RESULTS}; a phrase like 'top 3' also works)")
    parser.add_argument("--mode", choices=list(RANKING_MODES),
                        help=f"ranking strategy (default {DEFAULT_MODE}, unless the request or goal selects one)")
    parser.add_argument("--goal", choices=list(GOALS), help="set the listening goal yourself instead of using the intent model")
    parser.add_argument("--prefer", action="append", type=_prefer, default=[], metavar="KEY=VALUE",
                        help="explicit soft preference that overrides inferred ones, e.g. energy=0.9 or genre=jazz (repeatable)")
    parser.add_argument("--require", action="append", type=_require, default=[], metavar="FEATURE<=N",
                        help="hard limit such as energy<=0.5 or tempo>=120 (repeatable); songs that fail are removed")
    parser.add_argument("--exclude-genre", action="append", default=[], metavar="GENRE",
                        help="hard exclusion of a genre (repeatable)")
    parser.add_argument("--artist-penalty", type=float, metavar="P", help="variety: points subtracted per earlier pick by the same artist")
    parser.add_argument("--genre-penalty", type=float, metavar="P", help="variety: points subtracted per earlier pick in the same genre")
    parser.add_argument("--no-retrieval", action="store_true", help="do not use the knowledge collection (only explicit preferences count)")
    parser.add_argument("--compare-retrieval", action="store_true", help="run the request with and without retrieval and show the difference")
    parser.add_argument("--trace", action="store_true", help="print the structured runtime events after the response")
    parser.add_argument("--trace-file", metavar="PATH", help="write the runtime events as JSON (this stores the request text)")
    parser.add_argument("--json", action="store_true", help="print the full result as JSON instead of text")
    parser.add_argument("--format", choices=("table", "list"), default="table", help="recommendation layout (default table)")
    parser.add_argument("--width", type=_width, default=DEFAULT_WIDTH, metavar="N", help=f"maximum line width (default {DEFAULT_WIDTH})")
    parser.add_argument("--demo", action="store_true", help="run three example requests (study, workout, party)")
    return parser


def _exit_code(status: str) -> int:
    return {"failed_validation": 1, "invalid_input": 2}.get(status, 0)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.demo and args.query is None:
        parser.error("--query is required (or use --demo)")

    queries = list(DEMO_QUERIES) if args.demo else [args.query]
    code = 0
    for index, query in enumerate(queries):
        request = AssistantRequest(
            query=query, count=args.count, mode=args.mode, goal=args.goal, prefer=dict(args.prefer),
            require=tuple(args.require), exclude_genres=tuple(args.exclude_genre),
            artist_penalty=args.artist_penalty, genre_penalty=args.genre_penalty,
            use_retrieval=not args.no_retrieval)
        if index:
            print("\n" + "-" * min(args.width, 80) + "\n")
        if args.compare_retrieval:
            with_kb, without, diff = compare_retrieval(request)
            print(json.dumps({"with_retrieval": with_kb.to_dict(), "without_retrieval": without.to_dict(), "difference": diff},
                             indent=2) if args.json else format_comparison(query, with_kb, without, diff, args.width))
            result = with_kb
        else:
            result = build_agent().run(request)
            print(json.dumps(result.to_dict(), indent=2) if args.json else format_response(result, args.width, args.format))
            if args.trace and not args.json:
                print("\n" + format_trace(result.events, args.width))
        if args.trace_file:
            path = Path(args.trace_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"query": result.query, "status": result.status, "events": result.events},
                                       indent=2) + "\n", encoding="utf-8")
        code = max(code, _exit_code(result.status))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
