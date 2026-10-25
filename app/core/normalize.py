"""Normalization: raw field in, comparable form out.

The order of operations matters and is deliberate:

1. **transliterate first.** Honorific and suffix lists are written in Latin.
   Stripping "bhai" from "સુરેશભાઈ" only works once it reads "sureshabhai".
2. **strip business suffixes before tokenising.** They are multi-word
   ("pvt ltd", "& sons") and have to be matched against the string.
3. **honorifics as whole tokens, suffixes as token tails.** "Shri" is its own
   token; "bhai" is glued onto the end of one ("Jayeshbhai").
4. **phonetics last**, on the cleaned tokens, so honorifics never pollute a
   phonetic key — otherwise "Jayeshbhai" and "Jayesh" would block separately,
   which is the single most common failure in naive Indian-name matching.

Every step keeps what it removed. The web app shows it, and an auditor can ask
why a token disappeared.
"""

from __future__ import annotations

import re
from functools import lru_cache

from app.core.config import names_config
from app.core.models import InputRecord, NormalizedName, NormalizedRecord
from app.core.phonetic import phonetic_key, phonetic_tokens
from app.core.translit import to_latin

_PUNCT = re.compile(r"[^\w\s&]+", re.UNICODE)
_SPACES = re.compile(r"\s+")
_DIGITS = re.compile(r"\D+")


# --------------------------------------------------------------------------- #
# Names
# --------------------------------------------------------------------------- #


def normalize_name(raw: str, kind: str = "person") -> NormalizedName:
    """Full name pipeline. ``kind`` selects the business-suffix stage."""
    raw = (raw or "").strip()
    script, latin = to_latin(raw)
    flattened = _flatten(_apply_strip_patterns(latin))

    legal_suffix: str | None = None
    if kind == "business":
        flattened, legal_suffix = strip_business_suffixes(flattened)

    tokens = [t for t in flattened.split() if t]
    tokens, honorifics_removed = strip_honorifics(tokens)
    tokens, suffixes_removed = strip_affectionate_suffixes(tokens)

    if not tokens:
        # The whole name was honorifics and suffixes ("Shri Bhai"). Keep the
        # flattened form rather than emitting an empty record.
        tokens = [t for t in flattened.split() if t]

    final = tuple(tokens)
    return NormalizedName(
        raw=raw,
        script=script,
        transliterated=latin,
        tokens=final,
        phonetic_tokens=phonetic_tokens(final),
        phonetic_key=phonetic_key(final),
        honorifics_removed=tuple(honorifics_removed),
        suffixes_removed=tuple(suffixes_removed),
        legal_suffix=legal_suffix,
    )


@lru_cache(maxsize=1)
def _compiled_strip_patterns() -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(p, re.IGNORECASE) for p in names_config().strip_patterns)


def _apply_strip_patterns(text: str) -> str:
    """Remove honorifics that contain punctuation.

    "M/s Shah & Sons" would otherwise tokenise to ["m", "s", "shah"] and the
    stray initials would dilute every comparison.
    """
    for pattern in _compiled_strip_patterns():
        text = pattern.sub(" ", text)
    return text


def _flatten(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace."""
    text = text.lower().replace(".", " ")
    text = _PUNCT.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def strip_honorifics(tokens: list[str]) -> tuple[list[str], list[str]]:
    """Remove honorifics that stand as their own token."""
    config = names_config()
    kept: list[str] = []
    removed: list[str] = []
    for token in tokens:
        if token.rstrip(".") in config.honorifics:
            removed.append(token)
        else:
            kept.append(token)
    return kept, removed


def strip_affectionate_suffixes(tokens: list[str]) -> tuple[list[str], list[str]]:
    """Strip ``bhai`` / ``ben`` / ``ji`` style tails from each token.

    Guarded two ways, because this stage is where a careless implementation
    destroys real names: a stem shorter than ``min_stem_length`` is left
    alone, and tokens in ``suffix_exceptions`` (Juneja, Gowda, Panda) are
    never touched.
    """
    config = names_config()
    kept: list[str] = []
    removed: list[str] = []
    for token in tokens:
        if token in config.suffix_exceptions:
            kept.append(token)
            continue
        if token in config.affectionate_suffixes:
            # A bare "bhai"/"ben" token carries no identity.
            removed.append(token)
            continue
        stem, suffix = _strip_one_suffix(
            token, config.affectionate_suffixes, config.min_stem_length
        )
        if suffix:
            removed.append(suffix)
        kept.append(stem)
    return kept, removed


def _strip_one_suffix(
    token: str, suffixes: tuple[str, ...], min_stem: int
) -> tuple[str, str | None]:
    for suffix in suffixes:
        if len(token) <= len(suffix):
            continue
        if not token.endswith(suffix):
            continue
        stem = token[: -len(suffix)]
        if len(stem) < min_stem:
            continue
        return stem, suffix
    return token, None


def strip_business_suffixes(text: str) -> tuple[str, str | None]:
    """Pull legal and trade suffixes out of a business name.

    "Patel Trading Pvt Ltd" -> ("patel", "trading pvt ltd"). The suffix is
    kept, not discarded: it is weak evidence of identity but real evidence of
    what the record is, and the UI shows it.
    """
    config = names_config()
    working = f" {text} "
    found: list[str] = []
    for suffix in config.business_suffixes:
        padded = f" {suffix} "
        while padded in working:
            working = working.replace(padded, " ", 1)
            found.append(suffix)
    cleaned = _SPACES.sub(" ", working).strip()
    if not cleaned:
        # The name was nothing but suffixes ("Enterprises Pvt Ltd"): keep the
        # original rather than return an empty name.
        return text, " ".join(found) or None
    return cleaned, " ".join(dict.fromkeys(found)) or None


def is_common_surname(token: str) -> bool:
    return token.lower() in names_config().common_surnames


# --------------------------------------------------------------------------- #
# Phones
# --------------------------------------------------------------------------- #


def normalize_phone(raw: str | None) -> str | None:
    """Reduce an Indian phone number to its last 10 digits.

    Handles ``+91 98250 12345``, ``098250-12345``, ``0091 9825012345`` and
    ``9825012345``. Anything that cannot yield a plausible 10-digit mobile or
    landline number returns ``None`` — a junk phone must not become a matching
    signal.
    """
    if not raw:
        return None
    digits = _DIGITS.sub("", str(raw))
    if not digits:
        return None
    for prefix in ("0091", "91", "0"):
        if len(digits) > 10 and digits.startswith(prefix):
            digits = digits[len(prefix) :]
            break
    if len(digits) > 10:
        digits = digits[-10:]
    if len(digits) != 10 or digits[0] == "0":
        return None
    if len(set(digits)) == 1:
        # 0000000000 / 9999999999 are placeholders, not phone numbers.
        return None
    return digits


def phone_e164(last10: str | None) -> str | None:
    return f"+91{last10}" if last10 else None


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


def normalize_record(record: InputRecord) -> NormalizedRecord:
    """Normalize one input record end to end."""
    from app.core.address import parse_address  # local import: avoids a cycle

    return NormalizedRecord(
        record_id=record.record_id,
        kind=record.kind,
        source=record.source,
        raw_name=record.name,
        name=normalize_name(record.name, record.kind),
        address=parse_address(record.address) if record.address else None,
        phone_raw=record.phone,
        phone_last10=normalize_phone(record.phone),
        identifier_scheme=record.identifier_scheme,
        identifier_masked=record.identifier_masked,
    )


def normalize_records(records: list[InputRecord]) -> dict[str, NormalizedRecord]:
    """Normalize a batch, keyed by record id.

    Duplicate ids are rejected: silently dropping one would make the result
    wrong in a way nobody would notice.
    """
    out: dict[str, NormalizedRecord] = {}
    for record in records:
        if record.record_id in out:
            raise ValueError(f"duplicate record id in batch: {record.record_id!r}")
        out[record.record_id] = normalize_record(record)
    return out
