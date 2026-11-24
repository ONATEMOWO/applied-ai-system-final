"""
Tiny text helpers shared by the retriever and the intent classifier (standard library only).

Both components must see words the same way, so tokenizing and stemming live here.
The stemmer is deliberately crude (a handful of suffix rules, no dictionary): it only has
to make "studying", "studies" and "study" meet, and do so identically on both sides of a
comparison.
"""

import re
from typing import List, Tuple

# Function words only. Domain words such as "play", "recommend" or "workout" are NOT here:
# they carry signal for the intent classifier, and the retriever's IDF already
# down-weights words that appear in many sections.
STOPWORDS = frozenset("""
a an the and or of to for me my mine i im ive some something somethings anything any is are was were be
been being am with in on at it its this that these those please just really very so but if into from as by
we us you your can could would should will do does did get got have has had than then there their them
music song songs track tracks too much more less
""".split())


def normalize(text: str) -> str:
    """Lowercase, drop apostrophes, and turn hyphens into spaces ("hip-hop" -> "hip hop")."""
    return text.lower().replace("'", "").replace("’", "").replace("-", " ")


def stem(word: str) -> str:
    """A crude, consistent stemmer: parties->party, studying->study, relaxed->relax, dances->danc."""
    if len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith("ing") and len(word) > 5:
        word = word[:-3]
        if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "lsz":
            word = word[:-1]
    elif word.endswith("ed") and len(word) > 4:
        word = word[:-2]
        if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "lsz":
            word = word[:-1]
    elif word.endswith("es") and len(word) > 4:
        word = word[:-2]
    elif word.endswith("s") and len(word) > 3 and word[-2] not in "sui":
        word = word[:-1]
    if word.endswith("ly") and len(word) > 5:
        word = word[:-2]
    if word.endswith("e") and len(word) > 3:
        word = word[:-1]
    return word


def tokens(text: str) -> List[str]:
    """All stemmed word tokens in order, stopwords kept (used for exact phrase matching)."""
    return [stem(word) for word in re.findall(r"[a-z0-9_&]+", normalize(text))]


def content_tokens(text: str) -> List[str]:
    """Stemmed tokens with stopwords removed (used for TF-IDF and classification)."""
    return [stem(word) for word in re.findall(r"[a-z0-9_&]+", normalize(text)) if word not in STOPWORDS]


def contains_phrase(haystack: List[str], phrase: Tuple[str, ...]) -> bool:
    """True if `phrase` (a tuple of tokens) occurs as a contiguous run inside `haystack`."""
    n = len(phrase)
    return n > 0 and any(tuple(haystack[i:i + n]) == phrase for i in range(len(haystack) - n + 1))
