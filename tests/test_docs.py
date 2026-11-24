"""Documentation consistency: links, required sections, no placeholders, tested demo commands, repo hygiene."""

import io
import re
import shlex
import subprocess
from contextlib import redirect_stdout
from pathlib import Path

import pytest

import src.assistant as assistant
import src.main as original

ROOT = Path(__file__).resolve().parent.parent
DOCS = [ROOT / name for name in ("README.md", "model_card.md", "ai_interactions.md")] + [
    ROOT / "docs" / "presentation.md", ROOT / "docs" / "design.md", ROOT / "traces" / "README.md",
    ROOT / "data" / "intent_dataset.md", ROOT / "assets" / "README.md",
] + sorted((ROOT / "docs" / "module3").glob("*.md")) + sorted((ROOT / "knowledge").glob("*.md"))
LINK = re.compile(r"\]\(([^)\s]+)\)")


def slug(heading: str) -> str:
    text = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return text.replace(" ", "-")


def anchors(path: Path):
    return {slug(m.group(1)) for m in re.finditer(r"^#{1,6}\s+(.*?)\s*$", path.read_text(encoding="utf-8"), re.M)}


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_every_relative_link_and_anchor_resolves(doc):
    assert doc.exists(), f"missing document {doc}"
    text = doc.read_text(encoding="utf-8")
    for target in LINK.findall(text):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        path_part, _, anchor = target.partition("#")
        resolved = (doc.parent / path_part).resolve() if path_part else doc
        assert resolved.exists(), f"{doc.name}: broken link {target}"
        if anchor and resolved.suffix == ".md":
            assert anchor in anchors(resolved), f"{doc.name}: unknown anchor {target}"


def test_no_placeholders_or_unresolved_todos_in_the_deliverables():
    banned = re.compile(r"\b(TODO|TBD|FIXME|XXX|lorem ipsum|PLACEHOLDER|INSERT HERE|FILL IN)\b|@@\w+@@", re.I)
    for doc in DOCS[:5]:
        for number, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), start=1):
            assert not banned.search(line), f"{doc.name}:{number}: {line.strip()[:80]}"


def test_readme_has_the_required_sections():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    headings = {m.group(1).strip().lower() for m in re.finditer(r"^##\s+(.*)$", text, re.M)}
    for needed in ("original project", "new applied ai features", "architecture", "setup and commands",
                   "reproducible execution evidence", "retrieval", "agent workflow", "reliability and evaluation",
                   "design decisions", "limitations and future improvements", "brief reflection", "portfolio artifact"):
        assert needed in headings, needed
    assert "https://github.com/ONATEMOWO/applied-ai-system-final" in text
    assert "https://github.com/ONATEMOWO/ai110-module3show-musicrecommendersimulation-starter" in text
    assert "Music Recommender Simulation" in text and "Module 3" in text
    assert "diagrams/architecture.mmd" in text and "traces/" in text


def test_model_card_answers_the_four_assignment_questions_under_their_exact_wording():
    text = (ROOT / "model_card.md").read_text(encoding="utf-8")
    for question in (
        "What are the limitations or biases in your system?",
        "Could your AI be misused, and how would you prevent that?",
        "What surprised you while testing your AI's reliability?",
        "Describe your collaboration with AI during this project. Identify one suggestion you accepted and one you did not "
        "accept as written, and say why you rejected or changed it.",
    ):
        assert re.search(rf"^### Question \d: {re.escape(question)}$", text, re.M), question
    for needed in ("Intended use", "Data", "Synthetic dataset disclosure", "Baseline versus specialized model results",
                   "Guardrails and observed behavior", "Privacy considerations", "Future improvements"):
        assert needed in text, needed
    assert "not a calibrated probability" in text and "not conventional LLM-based RAG" in text


def test_ai_interactions_separates_the_two_kinds_of_agent_and_links_the_traces():
    text = (ROOT / "ai_interactions.md").read_text(encoding="utf-8")
    assert "coding agent" in text and "application agent" in text
    for trace in sorted((ROOT / "traces").glob("*.json")):
        assert f"traces/{trace.name}" in text, f"{trace.name} is not linked from ai_interactions.md"


def test_presentation_covers_every_required_section_within_seven_minutes():
    text = (ROOT / "docs" / "presentation.md").read_text(encoding="utf-8")
    for topic in ("Introduction and original project", "Motivation", "Architecture", "New AI features", "Live demonstration",
                  "Reliability and guardrails", "Evaluation results", "Ethical limitations", "Lessons learned and conclusion"):
        assert topic in text, topic
    spans = re.findall(r"\| (\d):(\d\d)–(\d):(\d\d) \|", text)
    minutes = [(int(a) * 60 + int(b), int(c) * 60 + int(d)) for a, b, c, d in spans]
    assert minutes[0][0] == 0 and 5 * 60 <= minutes[-1][1] <= 7 * 60
    assert all(prev[1] == nxt[0] for prev, nxt in zip(minutes, minutes[1:])), "timings must be contiguous"
    assert "has not been delivered" in text and "Backup if a live demo fails" in text


def test_every_demo_command_in_the_presentation_runs():
    text = (ROOT / "docs" / "presentation.md").read_text(encoding="utf-8")
    commands = [line.strip() for block in re.findall(r"```bash\n(.*?)```", text, re.S) for line in block.splitlines() if line.strip()]
    assert len(commands) >= 7
    for command in commands:
        argv = shlex.split(command)
        assert argv[:2] == ["python3", "-m"], command
        if argv[2] == "src.assistant":
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = assistant.main(argv[3:])
            assert code == 0 and "VibeFinder AI" in buffer.getvalue(), command
        else:
            assert argv[2] == "src.evaluate", command


def test_setup_instructions_match_the_repository():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    requirements = [line.strip() for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert requirements == ["pytest"]
    assert "pip install -r requirements.txt" in readme
    for command in ("python3 -m src.main", "python3 -m src.assistant --help", "python3 -m src.evaluate", "python3 -m pytest"):
        assert command in readme
    for module in ("main", "assistant", "evaluate", "retrieval"):
        assert (ROOT / "src" / f"{module}.py").exists()
    for needed in ("assets", "diagrams", "knowledge", "traces", "docs/module3"):
        assert (ROOT / needed).is_dir(), needed


def test_original_module3_files_are_unchanged_since_the_baseline_commit():
    """The Module 3 engine, CLI, report and data must be byte-identical to baseline commit 2e79736."""
    if subprocess.run(["git", "rev-parse", "--git-dir"], cwd=ROOT, capture_output=True).returncode != 0:
        pytest.skip("not a git checkout")
    if subprocess.run(["git", "cat-file", "-e", "2e79736"], cwd=ROOT, capture_output=True).returncode != 0:
        pytest.skip("baseline commit not available")
    diff = subprocess.run(
        ["git", "diff", "--stat", "2e79736", "--", "src/recommender.py", "src/main.py", "src/report.py", "src/experiments.py",
         "data/songs.csv", "tests/test_recommender.py", "tests/test_data.py", "tests/test_scoring.py", "tests/test_stretch.py",
         "tests/test_report.py"], cwd=ROOT, capture_output=True, text=True).stdout
    assert diff == "", diff


def test_no_caches_environments_or_secrets_are_tracked():
    if subprocess.run(["git", "rev-parse", "--git-dir"], cwd=ROOT, capture_output=True).returncode != 0:
        pytest.skip("not a git checkout")
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.splitlines()
    for path in tracked:
        assert not re.search(r"(__pycache__|\.pyc$|\.venv/|\.pytest_cache|\.DS_Store|\.env$|credentials|secret)", path), path
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in ("__pycache__/", ".venv/", ".pytest_cache/"):
        assert pattern in ignore


# --- saved evidence stays in step with the code -------------------------------------------------

def _captured(main, argv):
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        main(argv)
    return buffer.getvalue()


def test_saved_text_logs_match_what_the_commands_print_now():
    """Regenerate with the commands listed in assets/README.md if behaviour changes."""
    import src.evaluate as evaluate
    logs = ROOT / "assets" / "logs"
    def saved(name):
        return (logs / name).read_text(encoding="utf-8").rstrip("\n")      # logs are stored without trailing blank lines
    assert saved("src_main.txt") == _captured(original.main, ["--width", "100"]).rstrip("\n")
    assert saved("assistant_demo.txt") == _captured(assistant.main, ["--demo"]).rstrip("\n")
    assert saved("evaluate.txt") == _captured(evaluate.main, []).rstrip("\n")


def test_the_test_count_quoted_in_the_docs_is_the_real_count():
    collected = subprocess.run(["python3", "-m", "pytest", "--collect-only", "-q"], cwd=ROOT, capture_output=True, text=True).stdout
    count = re.search(r"(\d+) tests? collected", collected).group(1)
    log = (ROOT / "assets" / "logs" / "pytest.txt").read_text(encoding="utf-8")
    # The log is made with this very test deselected (it reads the log), so passed + deselected must equal the total.
    summary = re.search(r"(\d+) passed(?:, (\d+) deselected)?", log)
    assert summary and int(summary.group(1)) + int(summary.group(2) or 0) == int(count), \
        f"assets/logs/pytest.txt does not account for all {count} tests"
    for name in ("README.md", "ai_interactions.md"):
        assert re.search(rf"\b{count}\b", (ROOT / name).read_text(encoding="utf-8")), f"{name} should mention {count} tests"
