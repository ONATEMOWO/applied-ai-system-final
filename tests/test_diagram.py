"""Structural checks for diagrams/architecture.mmd (no Mermaid renderer is available offline)."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DIAGRAM = ROOT / "diagrams" / "architecture.mmd"
NODE = re.compile(r"^\s{4}([A-Z]+)(\(\[|\[\(|\[|\{)", re.M)
EDGE = re.compile(r"^\s{4}([A-Z]+)\s+(-->|-\.->)(?:\|\"[^\"|]+\"\|)?\s+([A-Z]+)\s*$", re.M)
CLOSERS = {"([": "])", "[(": ")]", "[": "]", "{": "}"}


@pytest.fixture(scope="module")
def text():
    return DIAGRAM.read_text(encoding="utf-8")


def test_the_file_is_a_mermaid_flowchart(text):
    assert text.startswith("flowchart TD\n")


def test_every_node_is_defined_once_with_matching_brackets_and_quotes(text):
    ids = []
    for line in text.splitlines():
        match = NODE.match(line)
        if not match:
            continue
        ids.append(match.group(1))
        opener = match.group(2)
        body = line[match.end():]
        assert body.endswith('"' + CLOSERS[opener]), line
        assert body.startswith('"') and body.count('"') == 2, line
    assert len(ids) == len(set(ids)) and len(ids) >= 20


def test_every_edge_connects_defined_nodes_and_every_node_is_connected(text):
    defined = {m.group(1) for m in NODE.finditer(text)}
    edges = EDGE.findall(text)
    assert len(edges) >= 30
    used = set()
    for source, _, target in edges:
        assert source in defined and target in defined, (source, target)
        used |= {source, target}
    assert defined == used, f"unconnected nodes: {sorted(defined - used)}"


def test_no_line_is_a_malformed_edge(text):
    for line in text.splitlines():
        if "-->" in line or "-.->" in line:
            assert EDGE.match(line), f"unparseable edge: {line!r}"


def test_class_assignments_use_defined_nodes_and_classes(text):
    defined = {m.group(1) for m in NODE.finditer(text)}
    classes = set(re.findall(r"^\s{4}classDef (\w+) ", text, re.M))
    for names, cls in re.findall(r"^\s{4}class ([A-Z,]+) (\w+);", text, re.M):
        assert cls in classes and set(names.split(",")) <= defined


def test_the_required_components_are_all_drawn(text):
    for label in ("User request", "Input validation", "Intent classifier", "Knowledge retrieval", "Knowledge collection",
                  "Recommendation plan", "Song catalog", "Existing Module 3 engine", "Output validation",
                  "Bounded correction", "Clarification", "Grounded response", "Evaluation harness", "Human review"):
        assert label in text, label


def test_every_file_named_in_the_diagram_exists_and_names_real_behaviour(text):
    for name in set(re.findall(r"\[((?:src/)?[\w./*]+\.(?:py|json|csv|md)|[\w/.*]+)\]", text)):
        if name.endswith(".py"):
            assert (ROOT / "src" / name).exists(), name
    assert (ROOT / "data" / "songs.csv").exists() and (ROOT / "knowledge").is_dir() and (ROOT / "traces").is_dir()
    from src import agent
    for strategy in ("rerank_with_reference_engine", "strengthen_variety"):
        assert strategy in text
        assert strategy in Path(agent.__file__).read_text(encoding="utf-8")
    kb_sections = sum(p.read_text(encoding="utf-8").count("\n## [KB-") for p in (ROOT / "knowledge").glob("*.md"))
    assert f"{kb_sections} sections" in text.replace("\n", " ")
    assert f"{len(list((ROOT / 'knowledge').glob('*.md')))} project-authored documents" in text.replace("<br/>", " ")


def test_readme_embeds_exactly_this_diagram():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    block = re.search(r"```mermaid\n(.*?)```", readme, re.S)
    assert block, "README.md must embed the Mermaid source"
    assert block.group(1) == DIAGRAM.read_text(encoding="utf-8")
