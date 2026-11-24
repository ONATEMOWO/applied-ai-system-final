"""Tests for the command line interface, the original entry point and the response text."""

import json

import pytest

from src.assistant import DEMO_QUERIES, build_parser, format_response, main, run_assistant
from src.main import main as original_main

STUDY = "Find calm instrumental music for studying"


def cli(capsys, *argv):
    code = main(list(argv))
    return code, capsys.readouterr().out


def test_help_documents_the_controls(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    text = capsys.readouterr().out
    for flag in ("--query", "--count", "--mode", "--prefer", "--require", "--exclude-genre", "--artist-penalty",
                 "--genre-penalty", "--no-retrieval", "--compare-retrieval", "--trace", "--trace-file", "--json", "--demo"):
        assert flag in text
    assert "exit status" in text


def test_a_query_is_required_unless_demo(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main([])
    assert exit_info.value.code == 2
    assert "--query is required" in capsys.readouterr().err


def test_a_normal_request_prints_a_grounded_response(capsys):
    code, out = cli(capsys, "--query", STUDY)
    assert code == 0
    for expected in ("Status  : ok", "Goal    : study", "Plan (what the ranking engine was asked to do)",
                     "Soft inferred preferences", "KB-GOAL-STUDY", "Focus Flow", "Why this plan",
                     "project-authored guidance", "Validation: passed", "Sources (retrieval score):", "KB-GOAL-STUDY (0.29)",
                     "KB-POLICY-SYNTHETIC (fixed note)"):
        assert expected in out


def test_every_listed_song_and_score_matches_the_result(capsys):
    result = run_assistant(STUDY)
    _, out = cli(capsys, "--query", STUDY)
    for rec in result.recommendations:
        assert rec.song["title"] in out and f"{rec.score:.2f}" in out


def test_non_answers_exit_zero_but_invalid_input_and_failures_do_not(capsys):
    assert cli(capsys, "-q", "what is the weather today")[0] == 0
    assert cli(capsys, "-q", "music please")[0] == 0
    assert cli(capsys, "-q", "music for studying", "--require", "energy>=0.99")[0] == 0
    assert cli(capsys, "-q", "")[0] == 2
    assert cli(capsys, "-q", STUDY, "-n", "0")[0] == 2


def test_controls_reach_the_plan(capsys):
    code, out = cli(capsys, "-q", STUDY, "-n", "2", "--mode", "mood-energy", "--prefer", "energy=0.9",
                    "--require", "tempo<=100", "--exclude-genre", "jazz", "--artist-penalty", "0.5", "--json")
    data = json.loads(out)
    assert code == 0 and len(data["recommendations"]) == 2
    plan = data["plan"]
    assert plan["mode"] == "mood-energy" and plan["mode_source"] == "cli"
    assert plan["soft_preferences"]["energy"] == {"value": 0.9, "origin": "explicit", "source": "cli",
                                                  "overridden": ["KB-GOAL-STUDY=0.4", "KB-MOD-CALM=0.3"]}
    rules = {c["rule"] for c in plan["hard_constraints"]}
    assert rules == {"tempo_bpm <= 100", "genre != jazz"}
    assert plan["diversity"]["artist_penalty"] == 0.5
    assert all(r["id"] and r["reasons"] for r in data["recommendations"])


def test_json_output_is_the_whole_result_and_only_json(capsys):
    _, out = cli(capsys, "-q", STUDY, "--json", "--trace")
    data = json.loads(out)                          # would raise if text were mixed in
    assert data["status"] == "ok" and data["events"][-1]["event"] == "final_status"
    assert data["cited_sources"] and data["intent"]["label"] == "study"
    assert data["retrieved"][0] == {"id": "KB-GOAL-STUDY", "kind": "goal", "score": 0.2916, "matched": ["study", "goal_study"]}
    assert [h["score"] for h in data["retrieved"]] == sorted((h["score"] for h in data["retrieved"]), reverse=True)


def test_trace_is_printed_on_request(capsys):
    _, plain = cli(capsys, "-q", STUDY)
    _, traced = cli(capsys, "-q", STUDY, "--trace")
    assert "Runtime trace" not in plain
    assert "Runtime trace" in traced and "tool_call:classify_intent" in traced and "plan_built" in traced
    assert "final_status" in traced


def test_trace_file_is_written_only_when_asked(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cli(capsys, "-q", STUDY)
    assert list(tmp_path.iterdir()) == []
    target = tmp_path / "out" / "trace.json"
    cli(capsys, "-q", STUDY, "--trace-file", str(target))
    saved = json.loads(target.read_text(encoding="utf-8"))
    assert saved["status"] == "ok" and saved["query"] == STUDY and saved["events"][0]["event"] == "request_received"


@pytest.mark.parametrize("bad", [
    ["--prefer", "energy"], ["--prefer", "loudness=1"], ["--prefer", "energy=loud"], ["--prefer", "likes_acoustic=maybe"],
    ["--require", "energy=0.5"], ["--require", "loudness<=1"], ["--mode", "turbo"], ["--goal", "sleep"],
    ["--width", "5"], ["--count", "many"],
])
def test_malformed_options_are_rejected_by_the_parser(capsys, bad):
    with pytest.raises(SystemExit) as exit_info:
        main(["-q", STUDY] + bad)
    assert exit_info.value.code == 2


def test_no_retrieval_flag_changes_the_outcome(capsys):
    _, out = cli(capsys, "-q", STUDY, "--no-retrieval")
    assert "Status  : needs_clarification" in out and "disabled" in out


def test_compare_retrieval_prints_both_runs_and_the_difference(capsys):
    code, out = cli(capsys, "-q", "pop music for my workout", "--compare-retrieval")
    assert code == 0
    assert "WITH retrieval" in out and "WITHOUT retrieval" in out and "Difference" in out
    assert "preferences that exist only with retrieval: " in out and "none" not in out.split("only with retrieval:")[1].split("\n")[0]


def test_demo_runs_three_distinct_goals(capsys):
    code, out = cli(capsys, "--demo")
    assert code == 0 and len(DEMO_QUERIES) == 3
    for goal in ("Goal    : study", "Goal    : workout", "Goal    : party"):
        assert goal in out


@pytest.mark.parametrize("width", [30, 45, 60, 100])
@pytest.mark.parametrize("layout", ["list", "table"])
def test_layout_and_width_are_honoured(capsys, layout, width):
    for query in (STUDY, "Recommend upbeat party music with different artists", "Give me calm music for an intense workout"):
        _, out = cli(capsys, "-q", query, "--format", layout, "--width", str(width), "--trace")
        assert max(len(line) for line in out.splitlines()) <= width, (query, layout, width)


def test_list_layout_numbers_the_songs(capsys):
    _, out = cli(capsys, "-q", STUDY, "--format", "list")
    assert "1. Focus Flow by LoRoom - score" in out


def test_clarification_responses_explain_and_recommend_nothing(capsys):
    _, out = cli(capsys, "-q", "Give me calm music for an intense workout")
    assert "needs_clarification" in out and "pull in opposite directions" in out
    assert "Top " not in out


def test_format_response_never_prints_validation_passed_for_withheld_results():
    from src.agent import Agent, AssistantRequest
    from src.assistant import build_agent
    from src.recommender import rank_songs
    base = build_agent()

    def bad(prefs, songs, k, weights=None, diversity=None):
        recs = rank_songs(prefs, songs, k, weights, diversity)
        return [recs[0], recs[0]] + recs[1:4]
    import src.agent as agent_module
    original = agent_module.rank_songs
    agent_module.rank_songs = bad
    try:
        result = Agent(base.songs, base.kb, recommend_fn=bad).run(AssistantRequest(STUDY))
    finally:
        agent_module.rank_songs = original
    text = format_response(result)
    assert result.status == "failed_validation" and "Validation: passed" not in text
    assert "DUPLICATE_SONG" in text and "withheld" in text


# --- the original Module 3 entry point is untouched ------------------------------------------------------

def test_original_entry_point_still_works(capsys):
    assert original_main([]) == 0
    out = capsys.readouterr().out
    assert "Loaded 30 songs" in out and "Happy high-energy pop listener" in out and "Edge case" in out


def test_parser_builds_without_side_effects():
    assert build_parser().parse_args(["-q", "x"]).count is None
