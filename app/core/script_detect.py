"""Script detection.

Which writing system is this string in? Decided by counting the characters
that fall inside each script's Unicode block, ignoring digits, punctuation and
whitespace. A string with a mix (common in real data: "જયેશ Patel") is
reported as the script of the majority of its letters, with the mix recorded
so the caller can see it happened.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from app.core.config import translit_config

LATIN = "latin"
UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ScriptProfile:
    """What scripts a string is written in."""

    script: str
    counts: dict[str, int]
    letter_count: int

    @property
    def is_mixed(self) -> bool:
        return len([c for c in self.counts.values() if c > 0]) > 1

    @property
    def confidence(self) -> float:
        if self.letter_count == 0:
            return 0.0
        return self.counts.get(self.script, 0) / self.letter_count


def profile_script(text: str) -> ScriptProfile:
    """Count letters per script and pick the dominant one."""
    config = translit_config()
    counts: dict[str, int] = dict.fromkeys(config.scripts, 0)
    counts[LATIN] = 0
    letters = 0

    for char in text:
        if not char.isalpha():
            continue
        letters += 1
        indic = config.script_for_codepoint(char)
        if indic is not None:
            counts[indic] += 1
        elif "LATIN" in unicodedata.name(char, ""):
            counts[LATIN] += 1

    if letters == 0:
        return ScriptProfile(script=UNKNOWN, counts=counts, letter_count=0)

    # Any Indic presence wins, even when Latin letters outnumber it.
    # "જયેશ Patel" is 3 Gujarati letters against 5 Latin, but calling it Latin
    # would mean skipping transliteration and leaving half the string
    # uncomparable. The reported script names the rules that were applied,
    # not the majority of the characters; `counts` keeps the full picture and
    # `is_mixed` says that it was mixed.
    indic = {name: n for name, n in counts.items() if name != LATIN and n > 0}
    if indic:
        best = max(indic.items(), key=lambda kv: kv[1])
        return ScriptProfile(script=best[0], counts=counts, letter_count=letters)
    return ScriptProfile(script=LATIN, counts=counts, letter_count=letters)


def detect_script(text: str) -> str:
    """The dominant script name: ``latin``, ``devanagari``, ``gujarati``, ``tamil``."""
    return profile_script(text).script


def is_indic(text: str) -> bool:
    return detect_script(text) not in {LATIN, UNKNOWN}
