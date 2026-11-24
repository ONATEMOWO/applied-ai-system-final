"""Tests for the evaluation harness and the committed runtime traces."""

import json
from pathlib import Path

import pytest

import src.evaluate as evaluate
from src.evaluate import (
    Case, build_ablations, build_cases, build_faults, compare_retrieval, main, run_cases, run_fault, write_traces,
)
from src.agent import AssistantRequest

REPO_TRACES = Path(__file__).resolve().parent.parent / "traces"


def test_the_harness_passes_on_the_shipped_system_and_prints_a_summary(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    for heading in ("1. SCENARIO CASES", "2. INTENT MODEL", "3. RETRIEVAL ENABLED vs DISABLED", "4. GUARDRAIL EFFECTIVENESS", "SUMMARY"):
        assert heading in out
    assert "TOTAL: 44/44 checks passed, 0 failed" in out
    assert "[FAIL]" not in out


def test_the_harness_reports_a_failed_expectation_and_exits_nonzero(capsys, monkeypatch):
    impossible = Case("impossible", "demo", "Find calm instrumental music for studying",
                      "an outcome that cannot happen", lambda r: "expected failure for the test")
    monkeypatch.setattr(evaluate, "build_cases", lambda: [impossible])
    assert main([]) == 1
    out = capsys.readouterr().out
    assert "[FAIL] impossible" in out and "failure : expected failure for the test" in out
    assert ", 1 failed" in out


def test_scenario_cases_cover_the_required_categories():
    categories = {c.category for c in build_cases()}
    assert {"study", "workout", "party", "relaxation", "unfamiliar wording", "ambiguous", "unsupported", "contradictory",
            "insufficient matches", "diversity", "malformed"} <= categories
    assert len({c.id for c in build_cases()}) == len(build_cases())


def test_every_scenario_case_passes_individually():
    for case, result, problem in run_cases(build_cases()):
        assert problem is None, (case.id, problem)


@pytest.mark.parametrize("ablation", build_ablations(), ids=lambda a: a.id)
def test_retrieval_ablations(ablation):
    with_kb, without, _ = compare_retrieval(AssistantRequest(query=ablation.query, **ablation.options))
    assert ablation.check(with_kb, without) is None


@pytest.mark.parametrize("fault", build_faults(), ids=lambda f: f.id)
def test_guardrails_change_what_the_user_receives(fault):
    off, on = run_fault(fault, guardrails=False), run_fault(fault, guardrails=True)
    assert fault.defective(off), "the injected fault must really reach the user when guardrails are off"
    assert not (on.recommendations and fault.defective(on))
    assert on.corrections and on.status == "ok"


# --- traces ---------------------------------------------------------------------------------

def test_traces_are_valid_json_with_the_required_events(tmp_path):
    entries = write_traces(tmp_path)
    assert len(entries) >= 8 and (tmp_path / "README.md").exists()
    for entry in entries:
        data = json.loads((tmp_path / entry["file"]).read_text(encoding="utf-8"))
        events = data["events"]
        assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
        assert events[0]["event"] == "request_received" and events[-1]["event"] == "final_status"
        assert events[-1]["status"] == data["status"] == entry["status"]
        assert not any("time" in key for e in events for key in e)           # no timestamps
        assert data["command"] and "description" in data


def test_the_trace_set_includes_each_kind_of_decision(tmp_path):
    write_traces(tmp_path)
    seen = set()
    for path in tmp_path.glob("*.json"):
        for e in json.loads(path.read_text(encoding="utf-8"))["events"]:
            seen.add(e.get("tool") or e["event"])
            if e["event"] == "final_status":
                seen.add("status:" + e["status"])
    for needed in ("classify_intent", "retrieve_knowledge", "plan_built", "filter_catalog", "rank_songs", "validate_results",
                   "correction_attempt", "goal_resolved", "retrieval_disabled", "status:ok", "status:needs_clarification",
                   "status:unsupported"):
        assert needed in seen, needed


def test_injected_faults_are_labelled_in_their_traces(tmp_path):
    write_traces(tmp_path)
    for path in tmp_path.glob("guardrail_*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["injected_fault"] and "INJECTED FAULT" in data["description"]
    natural = json.loads((tmp_path / "variety_correction_natural.json").read_text(encoding="utf-8"))
    assert natural["injected_fault"] is None
    assert any(e["event"] == "correction_attempt" for e in natural["events"])


def test_committed_traces_match_what_the_code_produces_now(tmp_path):
    """If this fails, behaviour changed: regenerate with `python3 -m src.evaluate --write-traces traces`."""
    write_traces(tmp_path)
    fresh = {p.name: p.read_text(encoding="utf-8") for p in tmp_path.iterdir()}
    committed = {p.name: p.read_text(encoding="utf-8") for p in REPO_TRACES.iterdir()}
    assert fresh == committed


def test_the_command_in_each_real_trace_reproduces_its_events(tmp_path):
    """Traces of real runs must be reproducible with the CLI command recorded inside them."""
    import shlex
    import src.assistant as assistant
    checked = 0
    for path in sorted(REPO_TRACES.glob("*.json")):
        committed = json.loads(path.read_text(encoding="utf-8"))
        if committed["injected_fault"] is not None:
            continue
        argv = shlex.split(committed["command"])
        assert argv[:3] == ["python3", "-m", "src.assistant"], committed["command"]
        argv = [str(tmp_path / "out.json") if part == "my_trace.json" else part for part in argv[3:]]
        assistant.main(argv)
        produced = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
        assert produced["events"] == committed["events"], path.name
        assert produced["status"] == committed["status"] and produced["query"] == committed["query"]
        checked += 1
    assert checked >= 8
