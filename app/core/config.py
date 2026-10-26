"""Configuration loading.

All domain knowledge lives in ``config/*.yaml``; this module loads it, caches
it, and exposes typed accessors. Nothing in ``app/core`` hard-codes a
honorific, an abbreviation or a weight.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(os.environ.get("INDIC_ER_CONFIG_DIR", ROOT / "config"))
DATA_DIR = Path(os.environ.get("INDIC_ER_DATA_DIR", ROOT / "data"))
DOCS_DIR = ROOT / "docs"
WEB_DIR = ROOT / "web"


class ConfigError(RuntimeError):
    """A config file is missing, unparseable or fails validation."""


def load_yaml(name: str) -> dict[str, Any]:
    """Load ``config/<name>.yaml``."""
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    try:
        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:  # pragma: no cover - malformed file
        raise ConfigError(f"{path.name} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path.name} must contain a YAML mapping at the top level")
    return data


# --------------------------------------------------------------------------- #
# Typed views over the YAML
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class NamesConfig:
    honorifics: frozenset[str]
    affectionate_suffixes: tuple[str, ...]
    min_stem_length: int
    suffix_exceptions: frozenset[str]
    business_suffixes: tuple[str, ...]
    common_surnames: frozenset[str]
    ambiguous_tokens: frozenset[str]
    strip_patterns: tuple[str, ...] = ()

    @classmethod
    def from_yaml(cls, raw: dict[str, Any]) -> NamesConfig:
        suffixes = [s.lower() for s in raw.get("affectionate_suffixes", [])]
        # Longest first so "bhaii" is tried before "bhai" and "behn" before "ben".
        suffixes.sort(key=len, reverse=True)
        business = [s.lower() for s in raw.get("business_suffixes", [])]
        business.sort(key=len, reverse=True)
        return cls(
            honorifics=frozenset(h.lower().rstrip(".") for h in raw.get("honorifics", [])),
            affectionate_suffixes=tuple(suffixes),
            min_stem_length=int(raw.get("min_stem_length", 3)),
            suffix_exceptions=frozenset(s.lower() for s in raw.get("suffix_exceptions", [])),
            business_suffixes=tuple(business),
            common_surnames=frozenset(s.lower() for s in raw.get("common_surnames", [])),
            ambiguous_tokens=frozenset(s.lower() for s in raw.get("ambiguous_tokens", [])),
            strip_patterns=tuple(str(p) for p in raw.get("strip_patterns", [])),
        )


@dataclass(frozen=True)
class PhoneticConfig:
    digraphs: tuple[tuple[str, str], ...]
    char_classes: dict[str, str]
    vowel_classes: frozenset[str]
    semivowel_initial: dict[str, str]
    consonant_initial: dict[str, str]
    initial_vowel_marker: str
    collapse_repeats: bool
    min_key_length: int

    @classmethod
    def from_yaml(cls, raw: dict[str, Any]) -> PhoneticConfig:
        digraphs = tuple((str(a), str(b)) for a, b in raw.get("digraphs", []))
        return cls(
            digraphs=digraphs,
            char_classes={str(k): str(v) for k, v in raw.get("char_classes", {}).items()},
            vowel_classes=frozenset(raw.get("vowel_classes", ["V"])),
            semivowel_initial={
                str(k): str(v) for k, v in (raw.get("semivowel_initial") or {}).items()
            },
            consonant_initial={
                str(k): str(v) for k, v in (raw.get("consonant_initial") or {}).items()
            },
            initial_vowel_marker=str(raw.get("initial_vowel_marker", "A")),
            collapse_repeats=bool(raw.get("collapse_repeats", True)),
            min_key_length=int(raw.get("min_key_length", 1)),
        )


@dataclass(frozen=True)
class AddressConfig:
    pincode_pattern: str
    pincode_loose_pattern: str
    district_prefix_length: int
    abbreviations: dict[str, str]
    landmark_markers: frozenset[str]
    stopwords: frozenset[str]
    building_markers: frozenset[str]
    locality_markers: frozenset[str]
    city_aliases: dict[str, str]
    cities: frozenset[str]
    state_aliases: dict[str, str]
    states: frozenset[str]

    @classmethod
    def from_yaml(cls, raw: dict[str, Any]) -> AddressConfig:
        pin = raw.get("pincode", {})
        city_aliases: dict[str, str] = {}
        cities: set[str] = set()
        for canonical, aliases in (raw.get("cities") or {}).items():
            canonical = str(canonical).lower()
            cities.add(canonical)
            city_aliases[canonical] = canonical
            for alias in aliases or []:
                city_aliases[str(alias).lower()] = canonical
        state_aliases: dict[str, str] = {}
        states: set[str] = set()
        for canonical, aliases in (raw.get("states") or {}).items():
            canonical = str(canonical).lower()
            states.add(canonical)
            state_aliases[canonical] = canonical
            for alias in aliases or []:
                state_aliases[str(alias).lower()] = canonical
        return cls(
            pincode_pattern=str(pin.get("pattern", r"^[1-9][0-9]{5}$")),
            pincode_loose_pattern=str(
                pin.get("loose_pattern", r"(?<![0-9])([1-9][0-9]{2})[\s-]?([0-9]{3})(?![0-9])")
            ),
            district_prefix_length=int(pin.get("district_prefix_length", 3)),
            abbreviations={
                str(k).lower().rstrip("."): str(v).lower()
                for k, v in (raw.get("abbreviations") or {}).items()
            },
            landmark_markers=frozenset(str(m).lower() for m in raw.get("landmark_markers", [])),
            stopwords=frozenset(str(m).lower() for m in raw.get("stopwords", [])),
            building_markers=frozenset(str(m).lower() for m in raw.get("building_markers", [])),
            locality_markers=frozenset(str(m).lower() for m in raw.get("locality_markers", [])),
            city_aliases=city_aliases,
            cities=frozenset(cities),
            state_aliases=state_aliases,
            states=frozenset(states),
        )


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    weight: float
    description: str


@dataclass(frozen=True)
class ConflictRule:
    name: str
    description: str
    requires: tuple[str, ...]


@dataclass(frozen=True)
class WeightsConfig:
    features: dict[str, FeatureSpec]
    auto_merge: float
    review_low: float
    gate_enabled: bool
    min_evidence_weight: float
    require_non_name_support: bool
    non_name_support_min: float
    gate_strong_identifiers: tuple[str, ...]
    blocking_strategies: tuple[str, ...]
    max_block_size: int
    max_candidate_pairs: int
    conflicts_enabled: bool
    conflict_rules: tuple[ConflictRule, ...]
    conflict_override_score: float
    address_weak_below: float
    baseline_fuzzy_threshold: float
    baseline_description: str
    canonical_name_strategy: str
    canonical_prefer_latin: bool
    review_page_size: int
    review_order: str

    @classmethod
    def from_yaml(cls, raw: dict[str, Any]) -> WeightsConfig:
        features: dict[str, FeatureSpec] = {}
        for name, spec in (raw.get("features") or {}).items():
            features[str(name)] = FeatureSpec(
                name=str(name),
                weight=float(spec.get("weight", 0.0)),
                description=str(spec.get("description", "")).strip(),
            )
        if not features:
            raise ConfigError("weights.yaml defines no features")
        total = sum(f.weight for f in features.values())
        if total <= 0:
            raise ConfigError("weights.yaml feature weights sum to zero")

        thresholds = raw.get("thresholds") or {}
        auto_merge = float(thresholds.get("auto_merge", 0.72))
        review_low = float(thresholds.get("review_low", 0.55))
        if not 0.0 <= review_low <= auto_merge <= 1.0:
            raise ConfigError(
                f"thresholds must satisfy 0 <= review_low ({review_low}) "
                f"<= auto_merge ({auto_merge}) <= 1"
            )

        gate = raw.get("evidence_gate") or {}
        blocking = raw.get("blocking") or {}
        strategies = tuple(
            str(s["name"]) for s in blocking.get("strategies", []) if s.get("enabled", True)
        )
        conflicts = raw.get("conflicts") or {}
        rules = tuple(
            ConflictRule(
                name=str(r["name"]),
                description=str(r.get("description", "")).strip(),
                requires=tuple(str(x) for x in r.get("requires", [])),
            )
            for r in conflicts.get("rules", [])
        )
        baseline = raw.get("baseline") or {}
        canonical = raw.get("canonical") or {}
        review = raw.get("review") or {}
        return cls(
            features=features,
            auto_merge=auto_merge,
            review_low=review_low,
            gate_enabled=bool(gate.get("enabled", True)),
            min_evidence_weight=float(gate.get("min_evidence_weight", 0.0)),
            require_non_name_support=bool(gate.get("require_non_name_support", False)),
            non_name_support_min=float(gate.get("non_name_support_min", 0.5)),
            gate_strong_identifiers=tuple(
                str(name) for name in gate.get("strong_identifiers", ["phone"])
            ),
            blocking_strategies=strategies,
            max_block_size=int(blocking.get("max_block_size", 120)),
            max_candidate_pairs=int(blocking.get("max_candidate_pairs", 400_000)),
            conflicts_enabled=bool(conflicts.get("enabled", True)),
            conflict_rules=rules,
            conflict_override_score=float(conflicts.get("override_score", 0.92)),
            address_weak_below=float(conflicts.get("address_weak_below", 0.45)),
            baseline_fuzzy_threshold=float(baseline.get("fuzzy_threshold", 0.88)),
            baseline_description=str(baseline.get("description", "")).strip(),
            canonical_name_strategy=str(canonical.get("name_strategy", "most_complete")),
            canonical_prefer_latin=bool(canonical.get("prefer_latin", True)),
            review_page_size=int(review.get("page_size", 25)),
            review_order=str(review.get("order", "uncertainty")),
        )

    def with_overrides(
        self,
        *,
        feature_weights: dict[str, float] | None = None,
        auto_merge: float | None = None,
        review_low: float | None = None,
        conflicts_enabled: bool | None = None,
        gate_enabled: bool | None = None,
    ) -> WeightsConfig:
        """Return a copy with selected values replaced (used by ablations)."""
        features = dict(self.features)
        if feature_weights:
            for name, weight in feature_weights.items():
                if name not in features:
                    raise ConfigError(f"unknown feature in override: {name}")
                features[name] = FeatureSpec(
                    name=name, weight=weight, description=features[name].description
                )
        if not any(f.weight > 0 for f in features.values()):
            raise ConfigError("override leaves every feature weight at zero")
        new_auto = self.auto_merge if auto_merge is None else auto_merge
        new_low = self.review_low if review_low is None else review_low
        return WeightsConfig(
            features=features,
            auto_merge=new_auto,
            review_low=min(new_low, new_auto),
            gate_enabled=(self.gate_enabled if gate_enabled is None else gate_enabled),
            min_evidence_weight=self.min_evidence_weight,
            require_non_name_support=self.require_non_name_support,
            non_name_support_min=self.non_name_support_min,
            gate_strong_identifiers=self.gate_strong_identifiers,
            blocking_strategies=self.blocking_strategies,
            max_block_size=self.max_block_size,
            max_candidate_pairs=self.max_candidate_pairs,
            conflicts_enabled=(
                self.conflicts_enabled if conflicts_enabled is None else conflicts_enabled
            ),
            conflict_rules=self.conflict_rules,
            conflict_override_score=self.conflict_override_score,
            address_weak_below=self.address_weak_below,
            baseline_fuzzy_threshold=self.baseline_fuzzy_threshold,
            baseline_description=self.baseline_description,
            canonical_name_strategy=self.canonical_name_strategy,
            canonical_prefer_latin=self.canonical_prefer_latin,
            review_page_size=self.review_page_size,
            review_order=self.review_order,
        )


@dataclass(frozen=True)
class TranslitConfig:
    scripts: dict[str, dict[str, Any]]
    latin_variants: dict[str, str]
    _ranges: tuple[tuple[str, int, int], ...] = field(default=(), repr=False)

    @classmethod
    def from_yaml(cls, raw: dict[str, Any]) -> TranslitConfig:
        scripts = raw.get("scripts") or {}
        ranges = []
        for name, spec in scripts.items():
            lo, hi = spec.get("range", ["\u0000", "\u0000"])
            ranges.append((str(name), ord(lo), ord(hi)))
        return cls(
            scripts={str(k): v for k, v in scripts.items()},
            latin_variants={
                str(k).lower(): str(v).lower() for k, v in (raw.get("latin_variants") or {}).items()
            },
            _ranges=tuple(ranges),
        )

    def script_for_codepoint(self, char: str) -> str | None:
        cp = ord(char)
        for name, lo, hi in self._ranges:
            if lo <= cp <= hi:
                return name
        return None


@dataclass(frozen=True)
class Settings:
    """Runtime settings, from the environment."""

    llm_mode: str = "mock"
    anthropic_model: str = "claude-sonnet-5-5"
    anthropic_api_key: str | None = None
    db_path: Path = DATA_DIR / "indic_er.db"
    log_level: str = "INFO"
    log_json: bool = False

    @classmethod
    def from_env(cls) -> Settings:
        mode = os.environ.get("LLM_MODE", "mock").strip().lower()
        if mode not in {"mock", "live"}:
            raise ConfigError(f"LLM_MODE must be 'mock' or 'live', got {mode!r}")
        return cls(
            llm_mode=mode,
            anthropic_model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            db_path=Path(os.environ.get("INDIC_ER_DB", str(DATA_DIR / "indic_er.db"))),
            log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
            log_json=os.environ.get("LOG_JSON", "0") not in {"0", "", "false", "False"},
        )


# --------------------------------------------------------------------------- #
# Cached accessors
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def names_config() -> NamesConfig:
    return NamesConfig.from_yaml(load_yaml("names"))


@lru_cache(maxsize=1)
def phonetic_config() -> PhoneticConfig:
    return PhoneticConfig.from_yaml(load_yaml("phonetic"))


@lru_cache(maxsize=1)
def address_config() -> AddressConfig:
    return AddressConfig.from_yaml(load_yaml("address"))


@lru_cache(maxsize=1)
def weights_config() -> WeightsConfig:
    return WeightsConfig.from_yaml(load_yaml("weights"))


@lru_cache(maxsize=1)
def translit_config() -> TranslitConfig:
    return TranslitConfig.from_yaml(load_yaml("translit"))


@lru_cache(maxsize=1)
def settings() -> Settings:
    return Settings.from_env()


def reset_caches() -> None:
    """Drop every cached config. Used by tests that edit YAML on the fly."""
    for fn in (
        names_config,
        phonetic_config,
        address_config,
        weights_config,
        translit_config,
        settings,
    ):
        fn.cache_clear()
