"""Tests for the agent workflow: routing, retrieval influence, constraints, validation and correction."""

import json
from dataclasses import replace

import pytest

import src.agent as agent_module
from src.agent import MAX_CORRECTIONS, Agent, AssistantRequest
from src.assistant import build_agent, compare_retrieval, run_assistant
from src.intent import parse_require
from src.recommender import RANKING_MODES, rank_songs

AGENT = build_agent()
SONG_IDS = {s["id"] for s in AGENT.songs}
STUDY = "Find calm instrumental music for studying"


def run(query=STUDY, **options):
    return AGENT.run(AssistantRequest(query=query, **options))


def tools(result):
    return [e["tool"] for e in result.events if e["event"] == "tool_call"]


def faulty_engine(mutate):
    """An engine that runs the real one and then corrupts the output."""
    def engine(prefs, songs, k, weights=None, diversity=None):
        return mutate(rank_songs(prefs, songs, k, weights, diversity))
    return engine


# --- workflow and trace ---------------------------------------------------------------------

def test_a_normal_request_runs_the_full_tool_sequence_in_order():
    result = run()
    assert result.status == "ok"
    assert tools(result) == ["classify_intent", "parse_explicit_request", "retrieve_knowledge", "filter_catalog",
                             "rank_songs", "validate_results"]
    steps = [e["step"] for e in result.events]
    assert steps.index("interpret") < steps.index("retrieve") < steps.index("plan") < steps.index("recommend") \
        < steps.index("validate") < steps.index("final")
    assert result.events[0]["event"] == "request_received" and result.events[-1]["event"] == "final_status"
    assert [e["seq"] for e in result.events] == list(range(1, len(result.events) + 1))


def test_the_trace_records_the_facts_a_reviewer_needs():
    result = run()
    by_tool = {e["tool"]: e for e in result.events if e["event"] == "tool_result"}
    assert by_tool["classify_intent"]["label"] == "study" and by_tool["classify_intent"]["source"] == "classifier"
    retrieved = {h["id"] for h in by_tool["retrieve_knowledge"]["hits"]}
    assert {"KB-GOAL-STUDY", "KB-MOD-CALM", "KB-MOD-INSTRUMENTAL"} <= retrieved
    plan = next(e for e in result.events if e["event"] == "plan_built")
    assert plan["mode"] == "balanced" and plan["soft_preferences"]["instrumentalness"]["source"] == "KB-GOAL-STUDY"
    assert by_tool["filter_catalog"]["candidates"] == 30
    assert by_tool["rank_songs"]["returned"] == [r.song["id"] for r in result.recommendations]
    assert next(e for e in result.events if e["event"] == "final_status")["status"] == "ok"
    json.dumps(result.events)                                  # must be plain JSON


def test_runs_are_repeatable_including_the_trace():
    assert run().to_dict() == run().to_dict()
    assert run("energetic music for a workout", count=3).to_dict() == run("energetic music for a workout", count=3).to_dict()


def test_nothing_is_written_to_disk(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run()
    assert list(tmp_path.iterdir()) == []


# --- retrieval changes the plan and the output -----------------------------------------------------

def test_retrieved_knowledge_supplies_the_preferences_and_each_one_is_traceable():
    result = run()
    assert result.plan.goal_profile_applied
    for key, entry in result.plan.soft.items():
        assert entry.origin == "inferred" and entry.source in AGENT.kb
        assert entry.source in result.plan.applied_sources
        assert AGENT.kb.get(entry.source).prefs[key] == entry.value      # the value really comes from that section


def test_without_retrieval_the_same_request_cannot_be_planned():
    result = run(use_retrieval=False)
    assert result.status == "needs_clarification" and result.hits == [] and result.plan.soft == {}
    assert "disabled" in result.message
    assert any(e["event"] == "retrieval_disabled" for e in result.events)
    assert "retrieve_knowledge" not in tools(result)


def test_retrieval_changes_the_ranking_when_the_user_only_names_a_genre():
    with_kb, without, diff = compare_retrieval(AssistantRequest("pop music for my workout"))
    assert with_kb.status == without.status == "ok"
    assert with_kb.plan.goal == "workout" and without.plan.goal == "workout"
    assert not without.plan.goal_profile_applied and with_kb.plan.goal_profile_applied
    assert set(without.plan.prefs) == {"genre"} and "energy" in with_kb.plan.prefs
    assert diff["first_rank_that_differs"] is not None
    assert [r.song["id"] for r in with_kb.recommendations] != [r.song["id"] for r in without.recommendations]
    def mean(r):
        return sum(x.song["energy"] for x in r.recommendations) / len(r.recommendations)

    assert mean(with_kb) > mean(without)


def test_retrieval_selects_variety_from_the_request_wording():
    with_kb = run("pop party music with different artists")
    plain = run("pop party music with different artists", use_retrieval=False)
    assert with_kb.plan.artist_penalty == 0.75 and "KB-STRAT-DIVERSITY-ARTIST" in with_kb.plan.applied_sources
    assert plain.plan.artist_penalty == 0.0


def test_retrieval_selects_the_ranking_mode_from_the_request_wording():
    assert run("match the mood of a rainy evening, relaxing").plan.mode == "mood-energy"
    assert run("relaxing music, stick to the genre jazz").plan.mode == "genre-first"
    assert run("relaxing music").plan.mode == "balanced"


def test_an_unapplied_goal_profile_is_reported_not_silently_used():
    result = run("I am not sure")
    assert result.status == "needs_clarification"
    assert any("not applied" in a for a in result.plan.adjustments)


def test_literal_goal_words_correct_a_model_label_that_has_no_evidence():
    result = run("Recommend upbeat party music with different artists")
    assert result.prediction.label == "discovery" and result.plan.goal == "party"
    assert result.plan.goal_source == "retrieval" and "KB-GOAL-PARTY" in result.plan.applied_sources
    assert any(e["event"] == "goal_resolved" and e["changed"] for e in result.events)
    assert tools(result).count("retrieve_knowledge") == 2


def test_an_explicit_goal_is_never_changed_by_retrieval():
    result = run("Recommend party music", goal="study")
    assert result.plan.goal == "study" and result.plan.goal_source == "override" and result.plan.goal_profile_applied


# --- explicit preferences override inferred ones ---------------------------------------------------

def test_explicit_wording_beats_the_goal_profile():
    plan = run("chill lofi for studying").plan
    assert plan.soft["mood"].origin == "explicit" and plan.soft["mood"].value == "chill"
    assert any("KB-GOAL-STUDY=focused" in o for o in plan.soft["mood"].overridden)
    assert plan.soft["genre"].origin == "explicit"
    assert plan.soft["instrumentalness"].origin == "inferred"


def test_cli_preferences_beat_both_wording_and_knowledge():
    plan = run(STUDY, prefer={"energy": 0.9}).plan
    assert plan.soft["energy"].value == 0.9 and plan.soft["energy"].origin == "explicit"
    assert plan.soft["energy"].source == "cli"


def test_an_explicit_value_resolves_what_would_otherwise_be_a_conflict():
    assert run("calm music for a workout").status == "needs_clarification"
    resolved = run("calm music for a workout", prefer={"energy": 0.5, "tempo_bpm": 100})
    assert resolved.status == "ok" and resolved.plan.soft["energy"].value == 0.5


def test_inferred_targets_move_inside_a_hard_limit_and_say_so():
    result = run("energetic music for a workout", require=[parse_require("energy<=0.6")])
    assert result.plan.soft["energy"].value <= 0.6
    assert any("moved to" in a for a in result.plan.adjustments)
    assert all(r.song["energy"] <= 0.6 for r in result.recommendations)


# --- hard constraints and catalog limits ---------------------------------------------------------------

def test_hard_constraints_remove_songs_before_ranking():
    result = run("energetic music for a workout", exclude_genres=["pop", "electronic"])
    assert result.status == "ok"
    assert all(r.song["genre"] not in ("pop", "electronic") for r in result.recommendations)
    removed = next(e for e in result.events if e.get("tool") == "filter_catalog" and e["event"] == "tool_result")
    assert removed["candidates"] < removed["catalog"] and removed["removed_by"][0]["removed"] > 0


def test_fewer_songs_are_returned_when_too_few_qualify_and_it_is_stated():
    result = run("music for studying", count=10, require=[parse_require("energy<=0.25")])
    assert result.status == "ok" and 0 < len(result.recommendations) < 10
    assert len(result.recommendations) == result.candidates
    assert any("only" in i.detail for i in result.infos)
    assert all(r.song["energy"] <= 0.25 for r in result.recommendations)


def test_no_candidates_gives_no_match_not_relaxed_constraints():
    result = run("music for studying", require=[parse_require("energy>=0.99")])
    assert result.status == "no_match" and result.recommendations == [] and result.candidates == 0
    assert "energy >= 0.99" in result.message


def test_never_invents_or_repeats_songs():
    for q in (STUDY, "energetic music for a workout", "party music", "surprise me with something new", "relaxing music"):
        result = run(q, count=15)
        ids = [r.song["id"] for r in result.recommendations]
        assert set(ids) <= SONG_IDS and len(ids) == len(set(ids)), q
        assert all(r.song == next(s for s in AGENT.songs if s["id"] == r.song["id"]) for r in result.recommendations)


def test_every_explanation_matches_a_fresh_scoring():
    from src.recommender import score_breakdown
    result = run(STUDY)
    for rec in result.recommendations:
        assert list(rec.reasons) == score_breakdown(result.plan.prefs, rec.song).reasons
        assert rec.score == pytest.approx(score_breakdown(result.plan.prefs, rec.song).total)


# --- routing ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("query, status", [
    ("", "invalid_input"), ("   ", "invalid_input"), (None, "invalid_input"), ("x" * 600, "invalid_input"),
    ("give me 0 songs", "invalid_input"), ("recommend 99 songs for the gym", "invalid_input"),
    ("what is the weather like today", "unsupported"), ("tell me a joke", "unsupported"),
    ("music please", "needs_clarification"), ("hmm", "needs_clarification"),
    ("jazz or blues", "needs_clarification"), ("pop music but no pop", "needs_clarification"),
    ("calm music for an intense workout", "needs_clarification"),
    ("workout music without too much energy and over 150 bpm and under 100 bpm", "needs_clarification"),
])
def test_requests_are_routed_to_the_right_outcome_with_no_recommendations(query, status):
    result = run(query)
    assert result.status == status and result.recommendations == []
    assert result.message


def test_invalid_options_are_rejected():
    assert run(STUDY, mode="turbo").status == "invalid_input"
    assert run(STUDY, goal="sleep").status == "invalid_input"
    assert run(STUDY, count=0).status == "invalid_input"
    assert run(STUDY, artist_penalty=-1).status == "invalid_input"
    assert run(STUDY, prefer={"energy": 5}).status == "invalid_input"
    assert run(STUDY, prefer={"genre": "reggaeton"}).status == "invalid_input"


def test_explicit_contradictions_are_found_across_query_and_options():
    assert run("metal music", exclude_genres=["metal"]).status == "needs_clarification"
    result = run("calm music", prefer={"energy": 0.9}, require=[parse_require("energy<=0.5")])
    assert result.status == "needs_clarification" and "outside the hard limit" in result.message


def test_unsupported_wording_is_disclosed_and_never_claimed_as_satisfied():
    result = run("calm instrumental music for studying in spanish, clean only")
    assert result.status == "ok"
    assert any("language" in n for n in result.plan.not_applied) and any("content rating" in n for n in result.plan.not_applied)
    assert all(c.feature not in ("language",) for c in result.plan.constraints)


def test_option_overrides_work():
    result = run(STUDY, count=3, mode="mood-energy", artist_penalty=1.0)
    assert len(result.recommendations) == 3 and result.plan.mode == "mood-energy"
    assert result.plan.artist_penalty == 1.0 and result.plan.diversity_source == "cli"


@pytest.mark.parametrize("mode", list(RANKING_MODES))
def test_every_ranking_mode_works_through_the_agent(mode):
    assert run(STUDY, mode=mode).status == "ok"


# --- validation, bounded correction, failure ----------------------------------------------------------------

def broken_agent(mutate):
    return Agent(AGENT.songs, AGENT.kb, recommend_fn=faulty_engine(mutate))


def with_duplicate(recs):
    return [recs[0], recs[0]] + recs[1:4]


def with_ghost(recs):
    return [replace(recs[0], song=dict(recs[0].song, id=777, title="Imaginary Anthem"))] + recs[1:]


def test_a_duplicate_from_the_engine_is_corrected_before_the_user_sees_it():
    result = broken_agent(with_duplicate).run(AssistantRequest(STUDY))
    ids = [r.song["id"] for r in result.recommendations]
    assert result.status == "ok" and len(ids) == len(set(ids)) == 5
    assert [c["strategy"] for c in result.corrections] == ["rerank_with_reference_engine"]
    assert "DUPLICATE_SONG" in result.corrections[0]["addressed"] and result.corrections[0]["remaining"] == []
    assert any(e["event"] == "correction_attempt" for e in result.events)


def test_a_nonexistent_song_from_the_engine_is_corrected():
    result = broken_agent(with_ghost).run(AssistantRequest(STUDY))
    assert result.status == "ok" and all(r.song["id"] in SONG_IDS for r in result.recommendations)
    assert result.corrections[0]["addressed"] == ["UNKNOWN_SONG"] or "UNKNOWN_SONG" in result.corrections[0]["addressed"]


def test_an_engine_that_ignores_hard_constraints_is_caught():
    forbidden = next(s for s in AGENT.songs if s["genre"] == "pop")

    def leak(recs):
        from src.recommender import Recommendation
        return [Recommendation(forbidden, recs[0].score, tuple(recs[0].reasons))] + recs[1:]
    result = broken_agent(leak).run(AssistantRequest("energetic music for a workout", exclude_genres=["pop"]))
    assert result.status == "ok" and all(r.song["genre"] != "pop" for r in result.recommendations)
    assert "CONSTRAINT_VIOLATION" in result.corrections[0]["addressed"]


def test_with_guardrails_off_the_same_bad_output_reaches_the_user():
    result = broken_agent(with_duplicate).run(AssistantRequest(STUDY, guardrails=False))
    ids = [r.song["id"] for r in result.recommendations]
    assert result.status == "ok" and len(ids) != len(set(ids)) and result.corrections == []
    assert any(e["event"] == "validation_skipped" for e in result.events)


def test_when_correction_cannot_help_the_results_are_withheld_and_the_run_ends(monkeypatch):
    monkeypatch.setattr(agent_module, "rank_songs", faulty_engine(with_duplicate))
    result = broken_agent(with_duplicate).run(AssistantRequest(STUDY))
    assert result.status == "failed_validation" and result.recommendations == []
    assert 1 <= len(result.corrections) <= MAX_CORRECTIONS
    assert result.failures and "withheld" in result.message
    assert result.events[-1]["event"] == "final_status"


def test_correction_never_repeats_a_strategy_and_is_bounded(monkeypatch):
    monkeypatch.setattr(agent_module, "rank_songs", faulty_engine(with_duplicate))
    result = broken_agent(with_duplicate).run(AssistantRequest(STUDY))
    strategies = [c["strategy"] for c in result.corrections]
    assert len(strategies) == len(set(strategies))


def test_an_engine_exception_becomes_a_failure_not_a_crash():
    def explode(*args, **kwargs):
        raise ValueError("engine is down")
    result = Agent(AGENT.songs, AGENT.kb, recommend_fn=explode).run(AssistantRequest(STUDY))
    assert result.status == "failed_validation" and "engine is down" in result.message


def test_a_missed_variety_request_triggers_one_stronger_rerank_that_works():
    def stubborn(prefs, songs, k, weights=None, diversity=None):
        strong = diversity is not None and diversity.artist_penalty >= 3.0
        return rank_songs(prefs, songs, k, weights, diversity if strong else None)
    result = Agent(AGENT.songs, AGENT.kb, recommend_fn=stubborn).run(
        AssistantRequest("pop party music with different artists", count=6))
    artists = [r.song["artist"] for r in result.recommendations]
    assert [c["strategy"] for c in result.corrections] == ["strengthen_variety"]
    assert len(set(artists)) == len(artists) and result.warnings == []
    assert result.plan.artist_penalty == 3.0 and any("raised" in a for a in result.plan.adjustments)


def test_when_stronger_variety_still_fails_it_is_reported_as_a_warning_and_the_run_still_ends():
    def ignore_variety(prefs, songs, k, weights=None, diversity=None):
        return rank_songs(prefs, songs, k, weights, None)

    result = Agent(AGENT.songs, AGENT.kb, recommend_fn=ignore_variety).run(
        AssistantRequest("pop party music with different artists", count=6))
    assert result.status == "ok" and len(result.corrections) == 1
    assert [w.code for w in result.warnings] == ["DIVERSITY_GOAL_MISSED"]


def test_cited_sources_all_exist_and_notes_quote_them():
    result = run()
    assert result.cited_sources and all(s in AGENT.kb for s in result.cited_sources)
    for sid in result.cited_sources:
        assert any(note.startswith(f"[{sid}]") for note in result.notes)
    assert "KB-POLICY-SYNTHETIC" in result.cited_sources


def test_run_assistant_convenience_function():
    assert run_assistant(STUDY, count=2).status == "ok"
    assert len(run_assistant(STUDY, count=2).recommendations) == 2


def test_the_correction_budget_is_enforced(monkeypatch):
    """A faulty engine that also ignores variety needs two different corrections; the budget caps them."""
    def duplicate_and_ignore_variety(prefs, songs, k, weights=None, diversity=None):
        return with_duplicate(rank_songs(prefs, songs, k, weights, None))
    request = AssistantRequest("pop party music", count=6, artist_penalty=0.05)   # too weak a penalty to separate the artists
    agent = Agent(AGENT.songs, AGENT.kb, recommend_fn=duplicate_and_ignore_variety)

    monkeypatch.setattr(agent_module, "MAX_CORRECTIONS", 1)
    capped = agent.run(request)
    assert len(capped.corrections) == 1

    monkeypatch.setattr(agent_module, "MAX_CORRECTIONS", 2)
    full = agent.run(request)
    assert [c["strategy"] for c in full.corrections] == ["rerank_with_reference_engine", "strengthen_variety"]
    assert full.status == "failed_validation"      # the second step re-calls the faulty engine, so it still fails
