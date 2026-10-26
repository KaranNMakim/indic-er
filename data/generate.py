#!/usr/bin/env python
"""Synthetic data generator.

    py data/generate.py [--entities 1200] [--records 5000] [--seed 20251025]

SYNTHETIC ONLY. No real PII, no scraping. Every name, address and phone number
is assembled from the word lists in ``data/pools.yaml``; identifiers are
masked by construction (``XXXXX1234``) and a full identifier is never built at
any point.

The generator exists to be *hard in the right ways*. Random character noise is
easy to beat and proves nothing, so the noise here is the noise Indian data
actually has:

* the same name written in Gujarati, Devanagari or Tamil script
* curated Latin spelling variants (Jayesh/Jaysh/Jaish, Patel/Patil/Pattel)
* honorifics prefixed (Shri, Smt.) and fused (Jayeshbhai, Kokilaben)
* given names collapsed to initials, surnames moved to the front
* addresses that drift: abbreviations, a dropped landmark, a city alias, a
  missing pincode, segments reordered, the society spelled differently
* fields simply absent — a quarter of records have no phone, an eighth no
  address at all
* phone numbers in five different formats, and occasionally one wrong digit

Entities are also kept **distinguishable from their own data**: no two share
a name, locality and building, and house numbers are unique within a building.
Without that constraint the generator produced pairs like "Rekha Prajapati,
Harikrupa Society, Kalubha Road, Bhavnagar 364001" twice over as two different
people, which no system could separate and which therefore measures nothing.

and the two cases that break naive matching hardest:

* **same name, different person** — a twin entity sharing given name and
  surname, living somewhere else with a different phone
* **common-surname crowding** — Patel, Shah, Kumar and friends are
  over-represented on purpose, so blocks are dense and name similarity is
  close to worthless on its own

Every record carries slice tags in ``ground_truth.csv`` so ``make eval`` can
report accuracy per failure mode rather than one flattering average.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
POOLS = DATA_DIR / "pools.yaml"

DEFAULT_SEED = 20251025
DEFAULT_ENTITIES = 1200
DEFAULT_RECORDS = 5000

# How many records one entity appears as. Mean ~4.2, which turns 1,200
# entities into roughly 5,000 records.
CARDINALITY_WEIGHTS = {1: 8, 2: 12, 3: 18, 4: 20, 5: 17, 6: 12, 7: 8, 8: 5}

# Share of person entities that get a same-name twin living elsewhere.
TWIN_SHARE = 0.10
# Share of surname draws biased to the very common surnames.
COMMON_SURNAME_SHARE = 0.55

LANDMARK_SYNONYMS = {
    "Mandir": ["Temple", "Mandir", "Derasar"],
    "Temple": ["Mandir", "Temple"],
    "Masjid": ["Mosque", "Masjid"],
    "Bus Stand": ["Bus Stop", "Bus Stand", "Bus Depot"],
    "Railway Station": ["Rly Station", "Railway Station", "Station"],
}

ABBREVIATIONS = {
    "Society": ["Soc.", "Socy", "Society"],
    "Apartment": ["Apt.", "Apts", "Apartment"],
    "Flats": ["Flts", "Flats"],
    "Residency": ["Resi.", "Residency"],
    "Nivas": ["Nivas"],
    "Heights": ["Hts", "Heights"],
    "Road": ["Rd.", "Road"],
    "Nagar": ["Ngr", "Nagar"],
    "Sector": ["Sec.", "Sector"],
}

PHONE_FORMATS = [
    "{d}",
    "+91 {a} {b}",
    "0{d}",
    "+91-{a}-{b}",
    "{a} {b}",
    "+91{d}",
    "0{a}-{b}",
]


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Entity:
    """One real-world person or business, before any noise is applied."""

    entity_id: str
    kind: str
    region: str
    given: dict[str, Any]
    surname: dict[str, Any]
    business_word: str | None
    gender: str
    city: str
    city_spec: dict[str, Any]
    locality: dict[str, str]
    building: str
    house: str
    landmark: str
    landmark_marker: str
    phone_digits: str
    identifier_scheme: str
    identifier_last4: str
    is_twin: bool = False
    twin_of: str | None = None

    @property
    def true_name(self) -> str:
        if self.kind == "business":
            return f"{self.surname['latin']} {self.business_word}"
        return f"{self.given['latin']} {self.surname['latin']}"


@dataclass(slots=True)
class Record:
    record_id: str
    entity_id: str
    name: str
    kind: str
    address: str | None
    phone: str | None
    identifier_scheme: str | None
    identifier_masked: str | None
    source: str
    tags: list[str] = field(default_factory=list)

    def as_row(self) -> dict[str, str]:
        return {
            "id": self.record_id,
            "name": self.name,
            "kind": self.kind,
            "address": self.address or "",
            "phone": self.phone or "",
            "identifier_scheme": self.identifier_scheme or "",
            "identifier_masked": self.identifier_masked or "",
            "source": self.source,
        }


# --------------------------------------------------------------------------- #
# generator
# --------------------------------------------------------------------------- #


class Generator:
    def __init__(self, pools: dict[str, Any], seed: int) -> None:
        self.pools = pools
        self.rng = random.Random(seed)
        self.regions = list(pools["regions"])
        self.region_weights = [pools["regions"][r]["weight"] for r in self.regions]
        self._used_phones: set[str] = set()
        # Identity keys already taken, so no two entities become genuinely
        # indistinguishable. See the module docstring.
        self._used_identities: set[tuple[str, ...]] = set()
        self._used_houses: set[tuple[str, ...]] = set()

    # ------------------------------------------------------------- entities #

    def make_entities(self, count: int) -> list[Entity]:
        entities: list[Entity] = []
        for index in range(count):
            entities.append(self._make_entity(f"ent{index:05d}"))

        # Same name, different person. Built by copying the name and changing
        # everything else, which is exactly the pair that defeats a
        # name-similarity matcher.
        persons = [e for e in entities if e.kind == "person"]
        twin_count = int(len(persons) * TWIN_SHARE)
        for index, original in enumerate(self.rng.sample(persons, twin_count)):
            twin = self._make_entity(f"ent{count + index:05d}")
            twin.given = original.given
            twin.surname = original.surname
            twin.gender = original.gender
            twin.kind = "person"
            twin.business_word = None
            twin.is_twin = True
            twin.twin_of = original.entity_id
            # Force a different city so the pincode and address genuinely differ.
            while twin.city == original.city:
                twin_region = self._pick_region()
                twin.region = twin_region
                twin.city = self.rng.choice(self.pools["regions"][twin_region]["cities"])
            twin.city_spec = self.pools["cities"][twin.city]
            twin.locality, twin.building, twin.house = self._pick_place(
                twin.city, twin.city_spec, twin.given["latin"], twin.surname["latin"]
            )
            entities.append(twin)
        return entities

    def _pick_region(self) -> str:
        return self.rng.choices(self.regions, weights=self.region_weights, k=1)[0]

    def _make_entity(self, entity_id: str) -> Entity:
        region = self._pick_region()
        kind = "business" if self.rng.random() < 0.15 else "person"
        given_pool = self.pools["given_names"][region]
        given = self.rng.choice(given_pool)
        surname = self._pick_surname(region)
        gender = "female" if given["latin"][-1] in "aiy" else "male"

        city = self.rng.choice(self.pools["regions"][region]["cities"])
        city_spec = self.pools["cities"][city]
        locality, building, house = self._pick_place(
            city, city_spec, given["latin"], surname["latin"]
        )

        landmark = self.rng.choice(self.pools["landmarks"])
        return Entity(
            entity_id=entity_id,
            kind=kind,
            region=region,
            given=given,
            surname=surname,
            business_word=(
                self.rng.choice(self.pools["business_words"]) if kind == "business" else None
            ),
            gender=gender,
            city=city,
            city_spec=city_spec,
            locality=locality,
            building=building,
            house=house,
            landmark=landmark,
            landmark_marker=self.rng.choice(self.pools["landmark_markers"]),
            phone_digits=self._phone_digits(),
            identifier_scheme=self.rng.choice(self.pools["identifier_schemes"]),
            identifier_last4=f"{self.rng.randint(0, 9999):04d}",
        )

    def _pick_surname(self, region: str) -> dict[str, Any]:
        pool = self.pools["surnames"][region]
        common = [s for s in pool if s.get("common")]
        if common and self.rng.random() < COMMON_SURNAME_SHARE:
            return self.rng.choice(common)
        return self.rng.choice(pool)

    def _pick_place(
        self,
        city: str,
        city_spec: dict[str, Any],
        given: str,
        surname: str,
    ) -> tuple[dict[str, str], str, str]:
        """Choose a locality, building and house that keep this entity unique.

        Two people with the same name in the same building, separated only by
        a house number that a quarter of records omit, are not a hard case —
        they are an unanswerable one, and including them would measure the
        generator rather than the matcher. So a name may repeat within a
        locality only in a different building, and a house number is never
        reused inside one building.
        """
        localities = city_spec["localities"]
        buildings = self.pools["buildings"]
        for _ in range(60):
            locality = self.rng.choice(localities)
            building = self.rng.choice(buildings)
            identity = (given, surname, city, locality["name"], building)
            if identity in self._used_identities:
                continue
            for _ in range(40):
                house = self._house_number()
                place = (city, locality["name"], building, house)
                if place not in self._used_houses:
                    self._used_identities.add(identity)
                    self._used_houses.add(place)
                    return locality, building, house
        # Exhausted after many tries: fall back rather than loop forever. The
        # summary prints how often this happened so the pools can be widened.
        self._crowded = getattr(self, "_crowded", 0) + 1
        locality = self.rng.choice(localities)
        building = self.rng.choice(buildings)
        return locality, building, self._house_number()

    def _house_number(self) -> str:
        style = self.rng.random()
        if style < 0.45:
            return str(self.rng.randint(1, 299))
        if style < 0.70:
            return f"{self.rng.choice('ABCD')}-{self.rng.randint(1, 904)}"
        if style < 0.85:
            return f"{self.rng.randint(1, 60)}/{self.rng.choice('AB')}"
        return str(self.rng.randint(100, 999))

    def _phone_digits(self) -> str:
        """A unique 10-digit mobile-shaped number. Invented, not allocated."""
        while True:
            digits = f"{self.rng.choice('6789')}{self.rng.randint(0, 999999999):09d}"
            if digits not in self._used_phones:
                self._used_phones.add(digits)
                return digits

    # -------------------------------------------------------------- records #

    def make_records(self, entities: list[Entity], target: int) -> list[Record]:
        counts = self._allocate_cardinality(entities, target)
        records: list[Record] = []
        serial = 0
        for entity, count in zip(entities, counts, strict=True):
            for variant in range(count):
                serial += 1
                records.append(self._make_record(f"rec{serial:05d}", entity, variant))
        self.rng.shuffle(records)
        return records

    def _allocate_cardinality(self, entities: list[Entity], target: int) -> list[int]:
        values = list(CARDINALITY_WEIGHTS)
        weights = list(CARDINALITY_WEIGHTS.values())
        counts = [self.rng.choices(values, weights=weights, k=1)[0] for _ in entities]
        # Nudge toward the requested record count without distorting the shape.
        while sum(counts) > target:
            index = self.rng.randrange(len(counts))
            if counts[index] > 1:
                counts[index] -= 1
        while sum(counts) < target:
            index = self.rng.randrange(len(counts))
            if counts[index] < max(values):
                counts[index] += 1
        return counts

    def _make_record(self, record_id: str, entity: Entity, variant: int) -> Record:
        # Variant 0 is the clean rendering: in real life at least one system
        # usually holds tidy data, and a dataset where every copy is mangled
        # is easier, not harder.
        clean = variant == 0
        tags: list[str] = []
        if entity.is_twin:
            tags.append("same_name_twin")
        if entity.surname.get("common"):
            tags.append("common_surname")

        name = self._render_name(entity, clean, tags)
        address = self._render_address(entity, clean, tags)
        phone = self._render_phone(entity, clean, tags)

        if address is None and phone is None:
            tags.append("name_only")

        identifier_scheme = identifier_masked = None
        if not clean and self.rng.random() < 0.35:
            identifier_scheme = entity.identifier_scheme
            identifier_masked = f"XXXXX{entity.identifier_last4}"

        return Record(
            record_id=record_id,
            entity_id=entity.entity_id,
            name=name,
            kind=entity.kind,
            address=address,
            phone=phone,
            identifier_scheme=identifier_scheme,
            identifier_masked=identifier_masked,
            source=self.rng.choice(self.pools["sources"]),
            tags=tags,
        )

    # ----------------------------------------------------------------- name #

    def _render_name(self, entity: Entity, clean: bool, tags: list[str]) -> str:
        if entity.kind == "business":
            return self._render_business_name(entity, clean, tags)

        given, surname = entity.given, entity.surname
        use_script = (
            not clean
            and given.get("script")
            and surname.get("script")
            # Applied only when both name parts have a native form, which is
            # a minority of the pool, so the probability has to be high to
            # land ~12% of records in Indic script.
            and self.rng.random() < 0.45
        )
        if use_script:
            tags.append("script_variant")
            given_text = given["script"]
            surname_text = surname["script"]
            if self.rng.random() < 0.35:
                given_text = self._fuse_suffix_script(given_text, entity)
                tags.append("fused_suffix")
            return f"{given_text} {surname_text}"

        given_text = given["latin"]
        surname_text = surname["latin"]

        if not clean and given.get("variants") and self.rng.random() < 0.35:
            given_text = self.rng.choice(given["variants"])
            tags.append("spelling_variant")
        if not clean and surname.get("variants") and self.rng.random() < 0.35:
            surname_text = self.rng.choice(surname["variants"])
            tags.append("spelling_variant")

        if not clean and self.rng.random() < 0.22:
            given_text = self._fuse_suffix(given_text, entity)
            tags.append("fused_suffix")

        if not clean and self.rng.random() < 0.12:
            given_text = f"{given_text[0]}."
            tags.append("initials")

        parts = [given_text, surname_text]
        if not clean and self.rng.random() < 0.08:
            parts.reverse()
            tags.append("reordered")

        name = " ".join(parts)

        if not clean and self.rng.random() < 0.18:
            name = f"{self.rng.choice(self.pools['honorifics'])} {name}"
            tags.append("honorific")

        if not clean and self.rng.random() < 0.10:
            name = self._typo(name)
            tags.append("typo")

        return name

    def _render_business_name(self, entity: Entity, clean: bool, tags: list[str]) -> str:
        stem = entity.surname["latin"]
        if not clean and entity.surname.get("variants") and self.rng.random() < 0.30:
            stem = self.rng.choice(entity.surname["variants"])
            tags.append("spelling_variant")
        word = entity.business_word or "Trading"
        suffix = "" if clean else self.rng.choice(self.pools["business_suffixes"])
        if suffix:
            tags.append("business_suffix")
        name = " ".join(part for part in (stem, word, suffix) if part)
        if not clean and self.rng.random() < 0.10:
            name = f"M/s {name}"
            tags.append("honorific")
        if not clean and self.rng.random() < 0.08:
            name = self._typo(name)
            tags.append("typo")
        return name

    def _fuse_suffix(self, given: str, entity: Entity) -> str:
        key = "female_suffixes" if entity.gender == "female" else "male_suffixes"
        return f"{given}{self.rng.choice(self.pools[key])}"

    def _fuse_suffix_script(self, given_script: str, entity: Entity) -> str:
        # The native-script equivalents of bhai / ben.
        suffix = "ભાઈ" if entity.gender == "male" else "બેન"
        if entity.region != "gujarat":
            suffix = "जी" if entity.gender == "male" else "देवी"
        return f"{given_script}{suffix}"

    def _typo(self, text: str) -> str:
        """One realistic keyboard slip: a swap, a drop or a doubled letter."""
        letters = [index for index, char in enumerate(text) if char.isalpha()]
        if len(letters) < 4:
            return text
        position = self.rng.choice(letters[1:-1])
        style = self.rng.random()
        if style < 0.4 and position + 1 < len(text):
            return text[:position] + text[position + 1] + text[position] + text[position + 2 :]
        if style < 0.7:
            return text[:position] + text[position + 1 :]
        return text[:position] + text[position] + text[position:]

    # -------------------------------------------------------------- address #

    def _render_address(self, entity: Entity, clean: bool, tags: list[str]) -> str | None:
        if not clean and self.rng.random() < 0.12:
            tags.append("no_address")
            return None

        segments: list[str] = []

        house = entity.house
        if clean or self.rng.random() > 0.25:
            if not clean and self.rng.random() < 0.20:
                house = f"Flat {house}"
            segments.append(house)

        building = entity.building
        if not clean and self.rng.random() < 0.28:
            variants = self.pools["building_variants"].get(building)
            if variants:
                building = self.rng.choice(variants)
                tags.append("building_variant")
        if not clean and self.rng.random() < 0.40:
            building = self._abbreviate(building)
            tags.append("abbreviated")
        segments.append(building)

        if clean or self.rng.random() > 0.35:
            landmark = self._landmark_synonym(entity.landmark) if not clean else entity.landmark
            marker = (
                entity.landmark_marker if clean else self.rng.choice(self.pools["landmark_markers"])
            )
            segments.append(f"{marker} {landmark}")
        else:
            tags.append("no_landmark")

        locality = entity.locality["name"]
        if not clean and self.rng.random() < 0.30:
            locality = self._abbreviate(locality)
        segments.append(locality)

        city = entity.city.title()
        aliases = entity.city_spec.get("aliases") or []
        if not clean and aliases and self.rng.random() < 0.22:
            city = self.rng.choice(aliases)
            tags.append("city_alias")
        segments.append(city)

        if not clean and self.rng.random() < 0.20:
            segments.append(entity.city_spec["state"])

        if clean or self.rng.random() > 0.20:
            segments.append(self._format_pincode(entity.locality["pincode"], clean))
        else:
            tags.append("no_pincode")

        if not clean and self.rng.random() < 0.12:
            # Segment order is not stable across source systems.
            head = segments[:2]
            tail = segments[2:]
            self.rng.shuffle(tail)
            segments = head + tail
            tags.append("reordered_address")

        return ", ".join(segments)

    def _abbreviate(self, text: str) -> str:
        for full, options in ABBREVIATIONS.items():
            if full in text:
                return text.replace(full, self.rng.choice(options))
        return text

    def _landmark_synonym(self, landmark: str) -> str:
        for key, options in LANDMARK_SYNONYMS.items():
            if key in landmark:
                return landmark.replace(key, self.rng.choice(options))
        return landmark

    def _format_pincode(self, pincode: str, clean: bool) -> str:
        if clean:
            return pincode
        style = self.rng.random()
        if style < 0.70:
            return pincode
        if style < 0.85:
            return f"{pincode[:3]} {pincode[3:]}"
        return f"{pincode[:3]}-{pincode[3:]}"

    # ---------------------------------------------------------------- phone #

    def _render_phone(self, entity: Entity, clean: bool, tags: list[str]) -> str | None:
        if not clean and self.rng.random() < 0.25:
            tags.append("no_phone")
            return None
        digits = entity.phone_digits
        if not clean and self.rng.random() < 0.04:
            # One wrong digit: a transcription slip that makes the phone
            # actively misleading rather than merely absent.
            position = self.rng.randrange(1, 10)
            wrong = str((int(digits[position]) + self.rng.randint(1, 8)) % 10)
            digits = digits[:position] + wrong + digits[position + 1 :]
            tags.append("phone_typo")
        template = "{d}" if clean else self.rng.choice(PHONE_FORMATS)
        return template.format(d=digits, a=digits[:5], b=digits[5:])


# --------------------------------------------------------------------------- #
# showcase set
# --------------------------------------------------------------------------- #

SHOWCASE = [
    {
        "id": "s01",
        "name": "Jayeshbhai Patel",
        "kind": "person",
        "address": "12, Shreeji Soc., Nr. Swaminarayan Mandir, Maninagar, Ahmedabad 380008",
        "phone": "+91 98250 12345",
        "source": "kyc",
    },
    {
        "id": "s02",
        "name": "જયેશ પટેલ",
        "kind": "person",
        "address": "12 Shriji Society, opp Swaminarayan Temple, Maninagar, Ahmedabad",
        "phone": "9825012345",
        "source": "wallet",
    },
    {
        "id": "s03",
        "name": "J. Patil",
        "kind": "person",
        "address": "Shreeji Society, Maninagar, Amdavad, 380 008",
        "phone": "",
        "source": "logistics",
    },
    {
        "id": "s04",
        "name": "Shri Jayesh R. Pattel",
        "kind": "person",
        "address": "Flat 12, Shreji Soc, Maninagar, Ahmedabad, Gujarat, 380008",
        "phone": "098250-12345",
        "source": "ecommerce",
    },
    {
        "id": "s05",
        "name": "Jayesh Patel",
        "kind": "person",
        "address": "5, Sun Apt., Behind Bus Stand, Bopal, Ahmedabad 380058",
        "phone": "9811122233",
        "source": "kyc",
    },
    {
        "id": "s06",
        "name": "Jayesh Patel",
        "kind": "person",
        "address": "",
        "phone": "",
        "source": "crm",
    },
    {
        "id": "s07",
        "name": "Kokilaben Shah",
        "kind": "person",
        "address": "B-402, Gokul Apartment, Nr. Jain Derasar, Satellite, Ahmedabad 380015",
        "phone": "+91-98790-11223",
        "source": "kyc",
    },
    {
        "id": "s08",
        "name": "કોકિલા શાહ",
        "kind": "person",
        "address": "B 402 Gokul Apt, Jain Temple ni pase, Satellite, Ahmedabad 380015",
        "phone": "9879011223",
        "source": "wallet",
    },
    {
        "id": "s09",
        "name": "Smt. Kokila Sah",
        "kind": "person",
        "address": "B-402 Gokul Apartment, Satellite, A'bad",
        "phone": "",
        "source": "logistics",
    },
    {
        "id": "s10",
        "name": "Patel Trading Pvt. Ltd.",
        "kind": "business",
        "address": "7, Market Yard, Nr. Railway Station, Katargam, Surat 395004",
        "phone": "+91 261 2345678",
        "source": "crm",
    },
    {
        "id": "s11",
        "name": "M/s Patel Trading & Sons",
        "kind": "business",
        "address": "7 Market Yard, Katargam, Surath, 395 004",
        "phone": "02612345678",
        "source": "ecommerce",
    },
    {
        "id": "s12",
        "name": "Laxmi Subramaniam",
        "kind": "person",
        "address": "14, Anna Nagar, Opp. Hanuman Temple, T Nagar, Chennai 600017",
        "phone": "9840011223",
        "source": "kyc",
    },
    {
        "id": "s13",
        "name": "லக்ஷ்மி சுப்ரமணியன்",
        "kind": "person",
        "address": "14 Anna Ngr, T Nagar, Madras, 600017",
        "phone": "+91 98400 11223",
        "source": "wallet",
    },
    {
        "id": "s14",
        "name": "Nilesh Patel",
        "kind": "person",
        "address": "88, Silver Oak Apt, Sector 21, Gandhinagar 382021",
        "phone": "9900011122",
        "source": "kyc",
    },
]

SHOWCASE_TRUTH = {
    "s01": "show_jayesh",
    "s02": "show_jayesh",
    "s03": "show_jayesh",
    "s04": "show_jayesh",
    "s05": "show_jayesh_other",
    "s06": "show_jayesh_unknown",
    "s07": "show_kokila",
    "s08": "show_kokila",
    "s09": "show_kokila",
    "s10": "show_patel_trading",
    "s11": "show_patel_trading",
    "s12": "show_lakshmi",
    "s13": "show_lakshmi",
    "s14": "show_nilesh",
}


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def load_pools() -> dict[str, Any]:
    with POOLS.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_outputs(
    records: list[Record], entities: list[Entity], crowded: int = 0
) -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with (DATA_DIR / "records.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "name",
                "kind",
                "address",
                "phone",
                "identifier_scheme",
                "identifier_masked",
                "source",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(record.as_row())

    with (DATA_DIR / "ground_truth.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["record_id", "entity_id", "tags"])
        for record in records:
            writer.writerow(
                [record.record_id, record.entity_id, "|".join(sorted(set(record.tags)))]
            )

    with (DATA_DIR / "records.json").open("w", encoding="utf-8") as handle:
        json.dump([record.as_row() for record in records], handle, ensure_ascii=False, indent=1)

    with (DATA_DIR / "showcase.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"records": SHOWCASE, "ground_truth": SHOWCASE_TRUTH},
            handle,
            ensure_ascii=False,
            indent=1,
        )

    tag_counts: dict[str, int] = {}
    for record in records:
        for tag in set(record.tags):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    sizes: dict[str, int] = {}
    for record in records:
        sizes[record.entity_id] = sizes.get(record.entity_id, 0) + 1

    return {
        "records": len(records),
        "entities": len(entities),
        "entities_with_records": len(sizes),
        "mean_records_per_entity": round(len(records) / max(len(sizes), 1), 2),
        "largest_entity": max(sizes.values(), default=0),
        "singleton_entities": sum(1 for n in sizes.values() if n == 1),
        "twin_entities": sum(1 for e in entities if e.is_twin),
        "businesses": sum(1 for r in records if r.kind == "business"),
        "crowded_fallbacks": crowded,
        "tag_counts": dict(sorted(tag_counts.items())),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entities", type=int, default=DEFAULT_ENTITIES)
    parser.add_argument("--records", type=int, default=DEFAULT_RECORDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    pools = load_pools()
    generator = Generator(pools, args.seed)
    entities = generator.make_entities(args.entities)
    records = generator.make_records(entities, args.records)
    summary = write_outputs(records, entities, getattr(generator, "_crowded", 0))

    print(f"\nsynthetic dataset written to {DATA_DIR}")
    print(f"  seed                    {args.seed}")
    print(f"  records                 {summary['records']}")
    print(f"  true entities           {summary['entities_with_records']}")
    print(f"  mean records/entity     {summary['mean_records_per_entity']}")
    print(f"  largest entity          {summary['largest_entity']}")
    print(f"  singleton entities      {summary['singleton_entities']}")
    print(f"  same-name twin entities {summary['twin_entities']}")
    print(f"  business records        {summary['businesses']}")
    print(f"  crowded fallbacks       {summary['crowded_fallbacks']}")
    print("\n  slice tags (records carrying each):")
    for tag, count in summary["tag_counts"].items():
        print(f"    {tag:20} {count:6}  {count / summary['records']:6.1%}")
    print("\n  files: records.csv, ground_truth.csv, records.json, showcase.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
