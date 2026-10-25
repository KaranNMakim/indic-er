# Changelog

All notable changes to indic-er. Phases follow the build order in
[PLAN.md](PLAN.md).

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
