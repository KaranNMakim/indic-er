"""Indic -> Latin transliteration.

Abugida-aware, driven entirely by ``config/translit.yaml``. A consonant
carries an inherent vowel which a matra replaces and a virama suppresses, so
this cannot be done with a character-for-character table:

    राम    ->  r + (a) ... but the final 'a' is not pronounced  ->  "ram"
    लक्ष्मी  ->  l+a, k, sh, m+i                                  ->  "lakshmi"
    જયેશ   ->  j+a, y+e, sh                                      ->  "jayesh"

Mixed-script input ("જયેશ Patel") is handled per character, so each half is
transliterated by its own script's rules and Latin passes through untouched.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

from app.core.config import translit_config
from app.core.script_detect import detect_script

_WORD_RE = re.compile(r"[A-Za-z']+")


@dataclass(frozen=True, slots=True)
class ScriptTables:
    """Compiled lookup tables for one script."""

    name: str
    consonants: dict[str, str]
    independent: dict[str, str]
    matras: dict[str, str]
    signs: dict[str, str]
    virama: str
    inherent: str
    drop_final_inherent: bool
    max_key_length: int


@lru_cache(maxsize=8)
def _tables(script: str) -> ScriptTables | None:
    spec = translit_config().scripts.get(script)
    if spec is None:
        return None
    consonants = {str(k): str(v) for k, v in (spec.get("consonants") or {}).items()}
    independent = {str(k): str(v) for k, v in (spec.get("independent") or {}).items()}
    matras = {str(k): str(v) for k, v in (spec.get("matras") or {}).items()}
    signs = {str(k): str(v) for k, v in (spec.get("signs") or {}).items()}
    keys = list(consonants) + list(independent) + list(matras) + list(signs)
    return ScriptTables(
        name=script,
        consonants=consonants,
        independent=independent,
        matras=matras,
        signs=signs,
        virama=str(spec.get("virama", "")),
        inherent=str(spec.get("inherent", "a")),
        drop_final_inherent=bool(spec.get("drop_final_inherent", False)),
        max_key_length=max((len(k) for k in keys), default=1),
    )


def transliterate(text: str) -> str:
    """Transliterate any Indic runs in ``text`` to Latin.

    Latin characters, digits and punctuation pass through. Latin spelling
    variants from ``latin_variants`` are folded at the end, so "Amdavad" and
    "Ahmedabad" converge whichever script they arrived in.
    """
    if not text:
        return ""
    config = translit_config()
    text = unicodedata.normalize("NFC", text)
    out: list[str] = []
    i, n = 0, len(text)

    while i < n:
        char = text[i]
        script = config.script_for_codepoint(char)
        tables = _tables(script) if script else None
        if tables is None:
            out.append(char)
            i += 1
            continue

        match = _longest_match(text, i, tables)
        if match is None:
            # An in-range character the tables do not cover (a rare sign, a
            # digit from the script's own block): drop it rather than emit
            # a character the phonetic stage cannot use.
            i += 1
            continue

        kind, value, length = match
        i += length

        if kind != "consonant":
            out.append(value)
            continue

        out.append(value)
        # A consonant's vowel: a virama kills it, a matra replaces it,
        # otherwise the inherent vowel applies — unless this is a word-final
        # consonant in a script with schwa deletion.
        if i < n and text[i] == tables.virama:
            i += 1
            continue
        matra = _match_in(text, i, tables.matras, tables.max_key_length)
        if matra is not None:
            out.append(matra[0])
            i += matra[1]
            continue
        at_word_end = i >= n or config.script_for_codepoint(text[i]) is None
        if not (tables.drop_final_inherent and at_word_end):
            out.append(tables.inherent)

    return _fold_latin_variants("".join(out))


def _longest_match(text: str, index: int, tables: ScriptTables) -> tuple[str, str, int] | None:
    for length in range(min(tables.max_key_length, len(text) - index), 0, -1):
        frag = text[index : index + length]
        if frag in tables.consonants:
            return ("consonant", tables.consonants[frag], length)
        if frag in tables.independent:
            return ("vowel", tables.independent[frag], length)
        if frag in tables.matras:
            return ("matra", tables.matras[frag], length)
        if frag in tables.signs:
            return ("sign", tables.signs[frag], length)
        if tables.virama and frag == tables.virama:
            return ("virama", "", length)
    return None


def _match_in(text: str, index: int, table: dict[str, str], max_len: int) -> tuple[str, int] | None:
    for length in range(min(max_len, len(text) - index), 0, -1):
        frag = text[index : index + length]
        if frag in table:
            return (table[frag], length)
    return None


def _fold_latin_variants(text: str) -> str:
    variants = translit_config().latin_variants
    if not variants:
        return text
    lowered = text.lower()
    if not any(key in lowered for key in variants):
        return text

    def replace(match: re.Match[str]) -> str:
        word = match.group(0)
        return variants.get(word.lower(), word)

    return _WORD_RE.sub(replace, text)


def has_indic(text: str) -> bool:
    config = translit_config()
    return any(config.script_for_codepoint(ch) is not None for ch in text)


def to_latin(text: str) -> tuple[str, str]:
    """Return ``(dominant_script, latin_form)`` for ``text``.

    Transliteration runs whenever *any* Indic character is present, even when
    Latin is the dominant script: "જયેશ Patel" must come out fully Latin or
    the two halves can never be compared. Pure Latin input only gets variant
    folding, so this is safe to call on every record.
    """
    script = detect_script(text)
    if has_indic(text):
        return script, transliterate(text)
    return script, _fold_latin_variants(text)
