"""
Local knowledge collection and a small deterministic TF-IDF retriever (standard library only).

The knowledge lives in knowledge/*.md as project-authored Markdown. Every section has a
stable ID (`## [KB-GOAL-STUDY] Title`) and an optional fenced `profile` block:

    ```profile
    kind: goal | modifier | strategy | feature | policy
    keywords: comma, separated, phrases
    prefs: energy=0.40, likes_acoustic=true      (soft scoring targets)
    mode: balanced                               (a ranking mode)
    diversity: artist=0.75, genre=0.5            (variety penalties)
    summary: one sentence quoted in responses
    ```

Documents are DATA, not instructions. Only the whitelisted keys above are read, every value
is validated (preference values go through the recommender's own validate_prefs), and
free-text prose is only ever indexed and quoted, never interpreted. An unknown key or an
invalid value makes loading fail loudly instead of being ignored.

Retrieval is plain TF-IDF cosine similarity over (title + keywords + prose), with sublinear
term frequency and smoothed IDF. Results are ordered by (-score, source id), so a given
query always returns the same sections in the same order.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.recommender import RANKING_MODES, validate_prefs
from src.textutil import content_tokens, contains_phrase, tokens

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"
KINDS = ("goal", "modifier", "strategy", "feature", "policy")
GOALS = ("study", "workout", "party", "relax", "discovery")
PROFILE_KEYS = frozenset({"kind", "goal", "feature", "keywords", "prefs", "mode", "diversity", "summary"})
DIVERSITY_KEYS = ("artist", "genre")

_HEADER = re.compile(r"^##\s+\[(KB-[A-Z0-9]+(?:-[A-Z0-9]+)*)\]\s+(.+?)\s*$")


class KnowledgeError(ValueError):
    """A knowledge file is malformed (bad section, unknown key, invalid value)."""


@dataclass(frozen=True)
class Section:
    """One retrievable unit of knowledge with a stable ID."""
    id: str
    doc: str
    title: str
    kind: str
    text: str
    keywords: Tuple[str, ...]
    phrases: Tuple[Tuple[str, ...], ...]
    summary: str
    goal: Optional[str] = None
    feature: Optional[str] = None
    prefs: Mapping[str, Any] = field(default_factory=dict)
    mode: Optional[str] = None
    diversity: Mapping[str, float] = field(default_factory=dict)

    def matched_keywords(self, query_tokens: Sequence[str]) -> Tuple[str, ...]:
        """Keyword phrases that occur verbatim (after stemming) in the query."""
        seen, matched = set(), []
        for keyword, phrase in zip(self.keywords, self.phrases):
            if phrase not in seen and contains_phrase(list(query_tokens), phrase):
                seen.add(phrase)
                matched.append(keyword)
        return tuple(matched)


@dataclass(frozen=True)
class Hit:
    """A retrieved section, its cosine relevance score and the keyword phrases it matched."""
    section: Section
    score: float
    matched: Tuple[str, ...]


# ---------------------------------------------------------------------------
# Loading and validating knowledge files
# ---------------------------------------------------------------------------

def _parse_prefs(raw: str, where: str) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise KnowledgeError(f"{where}: preference {item!r} must look like key=value")
        key, value = (part.strip() for part in item.split("=", 1))
        if key in parsed:
            raise KnowledgeError(f"{where}: preference {key!r} listed twice")
        if key == "likes_acoustic":
            if value.lower() not in ("true", "false"):
                raise KnowledgeError(f"{where}: likes_acoustic must be true or false, got {value!r}")
            parsed[key] = value.lower() == "true"
        elif key in ("genre", "mood"):
            parsed[key] = value
        else:
            try:
                parsed[key] = float(value)
            except ValueError:
                raise KnowledgeError(f"{where}: {key} must be a number, got {value!r}") from None
    try:
        return validate_prefs(parsed)
    except ValueError as err:
        raise KnowledgeError(f"{where}: {err}") from None


def _parse_diversity(raw: str, where: str) -> Dict[str, float]:
    parsed: Dict[str, float] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        key, _, value = (part.strip() for part in item.partition("="))
        if key not in DIVERSITY_KEYS:
            raise KnowledgeError(f"{where}: diversity key must be one of {', '.join(DIVERSITY_KEYS)}, got {key!r}")
        try:
            number = float(value)
        except ValueError:
            raise KnowledgeError(f"{where}: diversity {key} must be a number, got {value!r}") from None
        if not math.isfinite(number) or number < 0:
            raise KnowledgeError(f"{where}: diversity {key} must be a finite number >= 0, got {value!r}")
        parsed[key] = number
    return parsed


def _parse_profile_block(lines: List[str], where: str) -> Dict[str, str]:
    block: Dict[str, str] = {}
    for line in lines:
        if not line.strip():
            continue
        key, sep, value = line.partition(":")
        key = key.strip()
        if not sep or key not in PROFILE_KEYS:
            raise KnowledgeError(f"{where}: unknown or malformed profile line {line.strip()!r}; "
                                 f"allowed keys: {', '.join(sorted(PROFILE_KEYS))}")
        if key in block:
            raise KnowledgeError(f"{where}: profile key {key!r} listed twice")
        block[key] = value.strip()
    return block


def _build_section(section_id: str, title: str, doc: str, body: List[str]) -> Section:
    where = f"{doc} [{section_id}]"
    prose: List[str] = []
    profile_lines: Optional[List[str]] = None
    inside = False
    for line in body:
        if not inside and line.strip() == "```profile":
            if profile_lines is not None:
                raise KnowledgeError(f"{where}: more than one profile block")
            inside, profile_lines = True, []
        elif inside and line.strip() == "```":
            inside = False
        elif inside:
            profile_lines.append(line)
        else:
            prose.append(line)
    if inside:
        raise KnowledgeError(f"{where}: unterminated profile block")
    if profile_lines is None:
        raise KnowledgeError(f"{where}: missing profile block")

    block = _parse_profile_block(profile_lines, where)
    kind = block.get("kind", "")
    if kind not in KINDS:
        raise KnowledgeError(f"{where}: kind must be one of {', '.join(KINDS)}, got {kind!r}")
    summary = block.get("summary", "")
    if not summary:
        raise KnowledgeError(f"{where}: summary is required")
    keywords = tuple(k.strip().lower() for k in block.get("keywords", "").split(",") if k.strip())
    if kind != "feature" and not keywords:
        raise KnowledgeError(f"{where}: keywords are required for kind {kind!r}")
    phrases = tuple(tuple(tokens(k)) for k in keywords)

    goal = block.get("goal")
    if kind == "goal" and goal not in GOALS:
        raise KnowledgeError(f"{where}: goal must be one of {', '.join(GOALS)}, got {goal!r}")
    if kind != "goal" and goal is not None:
        raise KnowledgeError(f"{where}: only goal sections may set 'goal'")
    prefs = _parse_prefs(block["prefs"], where) if "prefs" in block else {}
    if prefs and kind not in ("goal", "modifier"):
        raise KnowledgeError(f"{where}: only goal and modifier sections may set prefs")
    if kind == "modifier" and not prefs:
        raise KnowledgeError(f"{where}: a modifier needs prefs")
    mode = block.get("mode")
    if mode is not None and mode not in RANKING_MODES:
        raise KnowledgeError(f"{where}: unknown ranking mode {mode!r}; choose from {', '.join(RANKING_MODES)}")
    diversity = _parse_diversity(block["diversity"], where) if "diversity" in block else {}
    if kind == "strategy" and mode is None and not diversity:
        raise KnowledgeError(f"{where}: a strategy needs a mode or a diversity setting")
    return Section(section_id, doc, title, kind, " ".join(" ".join(prose).split()), keywords, phrases,
                   summary, goal=goal, feature=block.get("feature"), prefs=prefs, mode=mode,
                   diversity=diversity)


def parse_document(path: Path) -> List[Section]:
    """Split one Markdown file into sections. Text before the first section header is ignored."""
    sections: List[Section] = []
    current: Optional[Tuple[str, str]] = None
    body: List[str] = []

    def flush() -> None:
        if current is not None:
            sections.append(_build_section(current[0], current[1], path.name, body))

    for line in path.read_text(encoding="utf-8").splitlines():
        match = _HEADER.match(line)
        if match:
            flush()
            current, body = (match.group(1), match.group(2)), []
        elif line.startswith("## "):
            raise KnowledgeError(f"{path.name}: section header without a [KB-...] id: {line!r}")
        else:
            body.append(line)
    flush()
    return sections


# ---------------------------------------------------------------------------
# The index
# ---------------------------------------------------------------------------

class KnowledgeBase:
    """All sections plus a TF-IDF index over them."""

    def __init__(self, sections: Sequence[Section]):
        ids = [s.id for s in sections]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise KnowledgeError(f"duplicate section id(s): {', '.join(duplicates)}")
        self.sections: Tuple[Section, ...] = tuple(sorted(sections, key=lambda s: s.id))
        self._by_id = {s.id: s for s in self.sections}

        doc_tokens = {s.id: content_tokens(f"{s.title} {' '.join(s.keywords)} {s.text}") for s in self.sections}
        n_docs = len(self.sections)
        df = Counter(term for toks in doc_tokens.values() for term in set(toks))
        self._idf = {term: math.log((1 + n_docs) / (1 + count)) + 1.0 for term, count in df.items()}
        self._vectors = {sid: self._weigh(Counter(toks)) for sid, toks in doc_tokens.items()}

    def _weigh(self, counts: Mapping[str, int]) -> Dict[str, float]:
        raw = {t: (1.0 + math.log(c)) * self._idf[t] for t, c in counts.items() if t in self._idf and c > 0}
        norm = math.sqrt(sum(v * v for v in raw.values()))
        return {t: v / norm for t, v in raw.items()} if norm else {}

    @property
    def ids(self) -> Tuple[str, ...]:
        return tuple(s.id for s in self.sections)

    def __contains__(self, section_id: object) -> bool:
        return section_id in self._by_id

    def get(self, section_id: str) -> Section:
        """Look up a section by ID; an unknown ID raises KeyError (it is never invented)."""
        return self._by_id[section_id]

    def search(self, query: str, top_k: int = 8, min_score: float = 0.03) -> List[Hit]:
        """
        Return up to top_k sections with cosine score >= min_score, best first.
        Ties break on section id, so the order never depends on dict or file order.
        """
        if not isinstance(query, str) or top_k < 1:
            return []
        q_vec = self._weigh(Counter(content_tokens(query)))
        if not q_vec:
            return []
        q_tokens = tokens(query)
        hits = []
        for section in self.sections:
            vec = self._vectors[section.id]
            score = round(sum(weight * vec.get(term, 0.0) for term, weight in q_vec.items()), 4)
            if score >= min_score:
                hits.append(Hit(section, score, section.matched_keywords(q_tokens)))
        hits.sort(key=lambda h: (-h.score, h.section.id))
        return hits[:top_k]


def load_knowledge(directory: Optional[Path] = None) -> KnowledgeBase:
    """Load every *.md file in the knowledge directory (sorted by name) into one index."""
    directory = Path(directory) if directory is not None else KNOWLEDGE_DIR
    files = sorted(directory.glob("*.md"))
    if not files:
        raise KnowledgeError(f"no knowledge files found in {directory}")
    sections: List[Section] = []
    for path in files:
        sections.extend(parse_document(path))
    return KnowledgeBase(sections)


def check_catalog_vocabulary(kb: KnowledgeBase, songs: Sequence[Mapping[str, Any]]) -> List[str]:
    """Problems where a profile names a genre or mood that no catalog song has."""
    genres = {str(s["genre"]).strip().casefold() for s in songs}
    moods = {str(s["mood"]).strip().casefold() for s in songs}
    problems = []
    for section in kb.sections:
        if "genre" in section.prefs and section.prefs["genre"] not in genres:
            problems.append(f"{section.id}: genre {section.prefs['genre']!r} is not in the catalog")
        if "mood" in section.prefs and section.prefs["mood"] not in moods:
            problems.append(f"{section.id}: mood {section.prefs['mood']!r} is not in the catalog")
    return problems


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Explore the index: python3 -m src.retrieval "calm instrumental music for studying"."""
    import sys
    args = list(sys.argv[1:] if argv is None else argv)
    kb = load_knowledge()
    if not args:
        print(f"{len(kb.sections)} sections in {len({s.doc for s in kb.sections})} documents")
        for section in kb.sections:
            print(f"  {section.id:<30} {section.kind:<9} {section.doc}")
        return 0
    for hit in kb.search(" ".join(args)):
        print(f"{hit.score:.4f}  {hit.section.id:<30} {hit.section.kind:<9} matched: {', '.join(hit.matched) or '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
