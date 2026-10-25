"""Indian address parsing.

Indian addresses do not have a field order. They have *landmarks*:

    "12, Shreeji Soc., Nr. Swaminarayan Mandir, Maninagar, Ahmedabad 380008"
    "Shreeji Society, opp Swaminarayan Temple, Maninagar, Amdavad, 380 008"
    "Flat 12 Shriji Soc, Maninagar, nr Swaminarayan temple, A'bad - 380008"

Those are the same place. A generic address parser trained on western formats
("number street, city, postcode") gets all three wrong. This parser instead:

* finds the pincode wherever it is, including ``380 008`` and ``380-008``
* expands abbreviations from a config list before doing anything else
  (``Soc.`` -> society, ``Nr.`` -> near, ``opp.`` -> opposite)
* captures the landmark phrase that follows a marker word, because
  "near Swaminarayan temple" is often the most distinctive thing in the string
* folds city aliases (Amdavad / A'bad -> ahmedabad)
* produces an order-insensitive ``core_tokens`` set for comparison, so token
  order and comma placement stop mattering

Everything it knows lives in ``config/address.yaml``.
"""

from __future__ import annotations

import re

from app.core.config import AddressConfig, address_config
from app.core.models import ParsedAddress
from app.core.translit import to_latin

_PUNCT = re.compile(r"[^\w\s,&/-]+")
_SPACES = re.compile(r"\s+")
_HOUSE = re.compile(r"^(?:[a-z]{0,2}[-/\s]?)?\d{1,5}(?:[-/][a-z0-9]{1,4})?$")
_HAS_DIGIT = re.compile(r"\d")


def parse_address(raw: str | None) -> ParsedAddress | None:
    """Parse an address string. ``None`` in, ``None`` out."""
    if raw is None or not str(raw).strip():
        return None
    return _parse(str(raw).strip(), address_config())


def _parse(raw: str, config: AddressConfig) -> ParsedAddress:
    _, latin = to_latin(raw)
    text = _flatten(latin)

    text, pincode = _extract_pincode(text, config)
    segments = [seg.strip() for seg in text.split(",")]
    segments = [_expand_segment(seg, config) for seg in segments]
    segments = [seg for seg in segments if seg]

    landmark, segments = _extract_landmark(segments, config)
    city, segments = _extract_named(segments, config.city_aliases)
    state, segments = _extract_named(segments, config.state_aliases)
    house, segments = _extract_house(segments)
    building = _find_marked(segments, config.building_markers)
    locality = _find_locality(segments, config, building)
    building = _clean_phrase(building, config)
    locality = _clean_phrase(locality, config)

    core = _core_tokens(segments, config, landmark)
    return ParsedAddress(
        raw=raw,
        house=house,
        building=building,
        landmark=landmark,
        locality=locality,
        city=city,
        state=state,
        pincode=pincode,
        core_tokens=core,
        expanded=", ".join(segments),
    )


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #


def _flatten(text: str) -> str:
    # Apostrophes are *removed*, not spaced: "A'bad" must stay one token so the
    # city alias list can fold it to ahmedabad. "/" is kept so a house number
    # like "15/A" survives as a single token.
    text = text.lower().replace("'", "").replace("’", "").replace(".", " ")
    text = _PUNCT.sub(" ", text)
    text = text.replace(" - ", " ")
    text = _SPACES.sub(" ", text)
    return re.sub(r"\s*,\s*", ", ", text).strip(" ,")


def _extract_pincode(text: str, config: AddressConfig) -> tuple[str, str | None]:
    """Pull out a 6-digit pincode, tolerating ``380 008`` and ``380-008``."""
    match = re.search(config.pincode_loose_pattern, text)
    if not match:
        return text, None
    candidate = f"{match.group(1)}{match.group(2)}"
    if not re.match(config.pincode_pattern, candidate):
        return text, None
    stripped = text[: match.start()] + " " + text[match.end() :]
    return _SPACES.sub(" ", stripped).strip(" ,-"), candidate


def _expand_segment(segment: str, config: AddressConfig) -> str:
    """Expand abbreviations token by token, then re-collapse whitespace."""
    out: list[str] = []
    for token in segment.split():
        bare = token.strip("-").rstrip(".")
        out.append(config.abbreviations.get(bare, bare))
    return _SPACES.sub(" ", " ".join(out)).strip(" -")


def _extract_landmark(segments: list[str], config: AddressConfig) -> tuple[str | None, list[str]]:
    """Capture the phrase introduced by a landmark marker.

    The marker's own segment is consumed, because "near Swaminarayan temple"
    belongs together — splitting it across fields loses the phrase.
    """
    for index, segment in enumerate(segments):
        tokens = segment.split()
        for position, token in enumerate(tokens):
            if token in config.landmark_markers:
                phrase = " ".join(tokens[position + 1 :]).strip()
                if not phrase:
                    continue
                remaining = segments[:index] + segments[index + 1 :]
                head = " ".join(tokens[:position]).strip()
                if head:
                    remaining.insert(index, head)
                return phrase, remaining
    return None, segments


def _extract_named(segments: list[str], aliases: dict[str, str]) -> tuple[str | None, list[str]]:
    """Find a city or state anywhere in the segments and fold it to canonical.

    Checked longest-first so "new delhi" wins over "delhi" and
    "tamil nadu" is not read as two unrelated tokens.
    """
    for index, segment in enumerate(segments):
        tokens = segment.split()
        for width in (3, 2, 1):
            for start in range(len(tokens) - width + 1):
                phrase = " ".join(tokens[start : start + width])
                canonical = aliases.get(phrase)
                if canonical is None:
                    continue
                leftover = " ".join(tokens[:start] + tokens[start + width :]).strip()
                remaining = segments[:index] + segments[index + 1 :]
                if leftover:
                    remaining.insert(index, leftover)
                return canonical, remaining
    return None, segments


def _extract_house(segments: list[str]) -> tuple[str | None, list[str]]:
    """Find the house/flat/plot number.

    Looks for a short numeric-ish token like ``12``, ``b-12``, ``402`` or
    ``a/7``. Only the token is consumed, never the rest of the segment, since
    "12 shreeji society" carries the building name too.
    """
    for index, segment in enumerate(segments):
        tokens = segment.split()
        for position, token in enumerate(tokens):
            if _HOUSE.match(token) and len(token) <= 8:
                leftover = " ".join(tokens[:position] + tokens[position + 1 :]).strip()
                remaining = segments[:index] + segments[index + 1 :]
                if leftover:
                    remaining.insert(index, leftover)
                return token, remaining
    return None, segments


def _find_marked(segments: list[str], markers: frozenset[str]) -> str | None:
    """Return the first segment that contains one of ``markers``."""
    for segment in segments:
        tokens = set(segment.split())
        if tokens & markers:
            return segment
    return None


def _find_locality(segments: list[str], config: AddressConfig, building: str | None) -> str | None:
    """Find the locality.

    First by marker word (nagar, colony, vihar, sector). Failing that, by
    position: the last remaining segment that is not the building and carries
    no digits is usually the locality in Indian addresses.
    """
    marked = _find_marked(segments, config.locality_markers)
    if marked and marked != building:
        return marked
    for segment in reversed(segments):
        if segment == building or _HAS_DIGIT.search(segment):
            continue
        if set(segment.split()) & config.building_markers:
            continue
        return segment
    return marked


def _clean_phrase(phrase: str | None, config: AddressConfig) -> str | None:
    """Drop stopwords from a display phrase.

    "flat shriji society" -> "shriji society": `flat` appears in half the
    addresses in any Indian dataset and identifies nothing.
    """
    if not phrase:
        return None
    kept = [t for t in phrase.split() if t not in config.stopwords]
    return " ".join(kept) or None


def _core_tokens(
    segments: list[str], config: AddressConfig, landmark: str | None
) -> tuple[str, ...]:
    """The identifying tokens, deduplicated and sorted.

    Sorted so comparison is order-insensitive: the same address written with
    its segments in a different order yields the same set. Stopwords and
    bare numbers are dropped — "flat", "floor", "block" and "2" appear in
    every address and distinguish nothing.
    """
    tokens: set[str] = set()
    for segment in segments:
        for token in segment.split():
            if len(token) < 3 or token in config.stopwords or token.isdigit():
                continue
            tokens.add(token)
    if landmark:
        for token in landmark.split():
            if len(token) >= 3 and token not in config.stopwords and not token.isdigit():
                tokens.add(token)
    return tuple(sorted(tokens))
