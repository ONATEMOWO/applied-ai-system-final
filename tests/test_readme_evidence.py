"""Every command printed in README.md is re-run here and its output compared with the README.

A block that starts with '$ python3 -m MODULE ARGS' must match the program's real output:
exactly, unless the block contains '...' lines, in which case each run of lines between the
'...' markers must appear, contiguous and in order, in the real output.
"""

import io
import re
import shlex
from contextlib import redirect_stdout
from functools import lru_cache
from pathlib import Path

import pytest

import src.assistant as assistant
import src.evaluate as evaluate
import src.main as original

README = Path(__file__).resolve().parent.parent / "README.md"
BLOCK = re.compile(r"```text\n\$ python3 -m (src\.\w+)([^\n]*)\n(.*?)\n```", re.S)
RUNNERS = {"src.assistant": assistant.main, "src.evaluate": evaluate.main, "src.main": original.main}


@lru_cache(maxsize=None)
def real_output(module: str, args: str):
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        RUNNERS[module](shlex.split(args))
    return [line.rstrip() for line in buffer.getvalue().rstrip("\n").splitlines()]


def documented_blocks():
    return [(m.group(1), m.group(2).strip(), [line.rstrip() for line in m.group(3).splitlines()])
            for m in BLOCK.finditer(README.read_text(encoding="utf-8"))]


def contiguous_find(haystack, needle, start):
    for i in range(start, len(haystack) - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            return i
    return -1


def test_the_readme_documents_a_meaningful_number_of_runs():
    blocks = documented_blocks()
    modules = [b[0] for b in blocks]
    assert modules.count("src.assistant") >= 12 and "src.evaluate" in modules and "src.main" in modules
    goals = " ".join(args for module, args, _ in blocks if module == "src.assistant")
    for query in ("studying", "workout", "party"):
        assert query in goals


@pytest.mark.parametrize("module, args, documented", documented_blocks(),
                         ids=lambda v: v if isinstance(v, str) and len(v) < 60 else "")
def test_documented_output_matches_the_real_program(module, args, documented):
    actual = real_output(module, args)
    if "..." not in documented:
        assert documented == actual, f"README output for `{module} {args}` is out of date"
        return
    segments, current = [], []
    for line in documented:
        if line.strip() == "...":
            if current:
                segments.append(current)
            current = []
        else:
            current.append(line)
    if current:
        segments.append(current)
    position = 0
    for segment in segments:
        found = contiguous_find(actual, segment, position)
        assert found >= 0, f"README excerpt for `{module} {args}` no longer appears in the real output:\n" + "\n".join(segment[:4])
        position = found + len(segment)
