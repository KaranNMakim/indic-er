# Changelog

All notable changes to indic-er. Phases follow the build order in
[PLAN.md](PLAN.md).

## Phase 2 — Matching pipeline

- **Pairwise features** (`app/core/features.py`): name spelling, name sound,
  name distinctiveness, address core, pincode, landmark and phone. Two
  governing rules — a missing field returns `None` and is dropped from both
  sides of the weighted mean, so absence is neutral; and names are compared by
  greedy token alignment, so reordering and dropped middle names cost nothing
  while an initial against a full token scores 0.90.
- **Weights rebalanced** so the name features sum to 0.46, below the 0.72
  merge threshold: two different people called Jayesh Patel cannot be merged
  on their name alone. The name-only corollary is documented and measured.
- **Blocking** (`app/core/blocking.py`): five keys chosen so a true pair stays
  reachable when several fields are wrong. Oversized blocks are skipped rather
  than truncated, because truncation silently strands specific records.
- **Clustering** (`app/core/cluster.py`): connected components plus a conflict
  check. When two cluster members disagree on hard evidence, the weakest
  `same_as` edge on the path between them is cut — unless it scores above the
  override threshold, or a reviewer confirmed it, in which case the conflict is
  reported rather than acted on.
- **Canonical records** with merge provenance per member pair, every name
  variant, every source script, and the most complete address in the cluster.
- **Baseline** (`app/core/baseline.py`): exact match plus Jaro-Winkler on raw
  strings, blocked on two characters. Implemented properly rather than
  strawmanned — it catches a trailing suffix and a plain typo, and fails on a
  leading honorific, a script change and Patel/Patil.
- **Review queue** (`app/core/review.py`): SQLite-backed, ordered by
  uncertainty, applied as hard overrides before clustering on the next run. A
  decided pair never returns to the queue. `threshold_shift()` reports where
  the accumulated decisions say the threshold should be, and says so honestly
  when the decisions overlap and no threshold can reproduce them.
- **Explanations** (`app/core/explain.py`): deterministic, rule-based,
  per-feature contributions plus a sentence that names the strongest and
  weakest evidence, says what could not be compared, and ranks an exact phone
  or pincode match above three middling name features because that is how a
  reviewer weighs it.
- **Pipeline** (`app/core/pipeline.py`): `resolve()` end to end with stage
  timings and a duplicate rate; `match_two()` scores any two records directly,
  including pairs blocking would never propose.
- 324 tests, 97% coverage on `app/core`.

## Phase 1 — Normalization

- **Script detection** (`app/core/script_detect.py`) for Latin, Devanagari,
  Gujarati and Tamil, by Unicode block. Any Indic presence wins over a Latin
  majority, so `"જયેશ Patel"` is still transliterated instead of being passed
  through half-unreadable.
- **Transliteration** (`app/core/translit.py`): abugida-aware, table-driven
  from `config/translit.yaml`. Handles the inherent vowel, matras, virama
  clusters and Hindi/Gujarati word-final schwa deletion, so राम → `ram` and
  लक्ष्मी → `lakshmi`. Latin city variants (Amdavad, Bombay, Bangalore) fold
  to one spelling.
- **Indic phonetic key** (`app/core/phonetic.py`): five documented stages
  driven by `config/phonetic.yaml`. Patel/Patil/Pattel → `PTL`,
  Jayesh/Jaishe/Jaysh → `JS`, Shah/Sah/Chah → `S`, Lakshmi/Laxmi → `LKSM`.
  Recall-first by design; collisions are resolved downstream by scoring.
- **Name normalization** (`app/core/normalize.py`): honorifics as whole
  tokens, affectionate suffixes as token tails (`Jayeshbhai` → `jayesh`),
  business legal suffixes captured rather than discarded. Everything removed
  is kept for the audit trail. Guards against destroying real names
  (`Juneja`, `Gowda`) and against eating a stem vowel (`Kokilaben` → `kokila`,
  not `kokil`).
- **Phone normalization** to the last 10 digits, rejecting placeholders so a
  junk phone never becomes a matching signal.
- **Address parsing** (`app/core/address.py`): pincode anywhere in the string
  including `380 008`, abbreviation expansion, landmark phrase capture
  (`Nr. Swaminarayan Mandir` → `swaminarayan temple`), city/state alias
  folding, and an order-insensitive `core_tokens` set.

## Phase 0 — Scaffold and knowledge-graph layer

- Project scaffold, `pyproject.toml`, ruff and pytest configuration.
- `tasks.py` task runner with a `Makefile` and `make.bat` delegating to it, so
  `make demo`, `make test` and `make eval` work with or without GNU make.
- All domain knowledge externalised to `config/*.yaml`: ontology, honorifics
  and suffixes, phonetic folding rules, address vocabulary, scoring weights.
- **Ontology** (`app/kg/ontology.py`): YAML in, validated Pydantic model out,
  JSON-LD and Turtle export via rdflib. A relation pointing at an undeclared
  class, an unknown property type, or a dropped provenance field fails the
  load.
- **Provenance** (`app/kg/provenance.py`): every fact carries source,
  timestamp, confidence and method (`rule` | `model` | `human`).
- **GraphStore** (`app/kg/graph_store.py`): interface plus a NetworkX +
  SQLite implementation, with ontology validation on every write and a
  Cytoscape export. Swappable for Neo4j/Kuzu without touching `app/core`.
- **LLMClient** (`app/kg/llm.py`): interface, deterministic offline mock, and
  a live Anthropic client selected by `LLM_MODE`. The matching core does not
  import it — enforced by a test.
- `/health` and `/ontology` endpoints; structured logging with a per-request
  correlation id.
