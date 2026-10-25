"""Indic-tuned phonetic key.

Soundex and Metaphone encode English. On Indian names they preserve exactly
the distinctions that transliteration invents and merge nothing that Indic
speakers actually merge:

===============  ========  ==========  =========
name             Soundex   Metaphone   indic_key
===============  ========  ==========  =========
Patel            P340      PTL         PTL
Patil            P340      PTL         PTL
Jayesh           J200      JYX         JS
Jaishe           J200      JX          JS
Shah             S000      X           S
Chah             C000      X           S
Lakshmi          L252      LKXM        LKSM
Laxmi            L250      LKSM        LKSM
===============  ========  ==========  =========

Metaphone splits Jayesh from Jaishe, and neither Soundex nor Metaphone merges
Shah with Chah. ``indic_key`` gets both right, because the folding rules were
chosen for the confusions Indic transliteration actually produces rather than
for English orthography.

**This key is built for recall, and it collides on purpose.** It is a blocking
key: its job is to put every plausible variant of a name in the same bucket
cheaply, and pairwise scoring then throws out the ones that do not belong.
Dropping all non-initial vowels is what makes Jaysh and Jaishe meet, and the
same rule makes ``Jayesh`` and ``Joshi`` both ``JS``. That is a candidate pair,
not a match — ``name_string`` scores it at 0.4 and the pair is rejected. Judge
the key by whether true variants ever land in *different* buckets (a miss that
scoring can never recover) and not by whether unrelated names share one.

Five stages, all driven by ``config/phonetic.yaml``:

1. lowercase, strip to ``a-z``
2. apply the ordered digraph rewrites (aspirate and cluster folding)
3. map characters through the equivalence classes
4. collapse runs of the same class
5. drop vowel classes, keeping a marker for a word-initial vowel
"""

from __future__ import annotations

import re
from functools import lru_cache

from app.core.config import PhoneticConfig, phonetic_config

_NON_ALPHA = re.compile(r"[^a-z]+")


@lru_cache(maxsize=65536)
def indic_key(token: str) -> str:
    """Phonetic key for a single name token.

    Cached: a dataset of 5,000 records has only a few thousand distinct
    tokens, and the key is computed for every blocking strategy.
    """
    return _indic_key(token, phonetic_config())


def _indic_key(token: str, config: PhoneticConfig) -> str:
    cleaned = _NON_ALPHA.sub("", token.lower())
    if not cleaned:
        return ""

    folded = _apply_digraphs(cleaned, config)
    if not folded:
        return ""

    symbols: list[str] = []
    for position, char in enumerate(folded):
        symbols.append(_classify(char, position, config))

    if config.collapse_repeats:
        symbols = _collapse(symbols)

    key = "".join(s for s in symbols if s and s not in config.vowel_classes)
    if len(key) < config.min_key_length:
        # Everything folded away ("Aai", "Oo"): fall back to the first letter
        # so the token still blocks with its lookalikes.
        return cleaned[0].upper()
    return key


def _apply_digraphs(text: str, config: PhoneticConfig) -> str:
    """Single left-to-right pass, longest rule first at each position.

    A single pass matters: ``sh -> s`` followed by ``ch -> s`` would otherwise
    let a rule re-read its own output and turn "chh" into something else.
    """
    out: list[str] = []
    i, n = 0, len(text)
    rules_by_length: dict[int, list[tuple[str, str]]] = {}
    for pattern, replacement in config.digraphs:
        rules_by_length.setdefault(len(pattern), []).append((pattern, replacement))
    lengths = sorted(rules_by_length, reverse=True)

    while i < n:
        for length in lengths:
            frag = text[i : i + length]
            if len(frag) < length:
                continue
            for pattern, replacement in rules_by_length[length]:
                if frag == pattern:
                    out.append(replacement)
                    i += length
                    break
            else:
                continue
            break
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def _classify(char: str, position: int, config: PhoneticConfig) -> str:
    if position == 0:
        # `y` and `h` are real consonants word-initially (Yadav, Hemant) and
        # weak everywhere else.
        if char in config.semivowel_initial:
            return config.semivowel_initial[char]
        if char in config.consonant_initial:
            return config.consonant_initial[char]
        symbol = config.char_classes.get(char, char.upper())
        if symbol in config.vowel_classes:
            return config.initial_vowel_marker
        return symbol
    return config.char_classes.get(char, char.upper())


def _collapse(symbols: list[str]) -> list[str]:
    out: list[str] = []
    for symbol in symbols:
        if not out or out[-1] != symbol:
            out.append(symbol)
    return out


def phonetic_tokens(tokens: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Phonetic key per token, blanks dropped."""
    return tuple(key for key in (indic_key(t) for t in tokens) if key)


def phonetic_key(tokens: tuple[str, ...] | list[str]) -> str:
    """Order-insensitive key for a whole name.

    Sorted so "Jayesh Patel" and "Patel Jayesh" land in the same block.
    """
    return "-".join(sorted(phonetic_tokens(tokens)))


def token_phonetic_equal(left: str, right: str) -> bool:
    key_left, key_right = indic_key(left), indic_key(right)
    return bool(key_left) and key_left == key_right
