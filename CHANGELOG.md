# Changelog

All notable changes to indic-er. Phases follow the build order in
[PLAN.md](PLAN.md).

## Fixes found while preparing the demo screenshot

- **A merge edge could veto its own merge.** `transitivity.veto_below` and
  `auto_merge` are independent settings, so any threshold below the veto floor
  put merge edges into the rejection index, where each one blocked itself.
  Every sweep point at or under 0.70 then returned identical numbers — the
  threshold sweep was a flat line that looked plausible and meant nothing. The
  shipped configuration was never affected (the floor sits below `auto_merge`),
  so the headline figures are unchanged, but the published sweep was wrong.
- **The sweep chart scaled from zero**, so a range of 0.80-0.96 rendered as a
  wall of identical bars. It now scales between its own minimum and maximum,
  and the caption describes the plateau computed from the data instead of
  asserting a shape that was true of an earlier configuration.
- **Stale assets and stale metrics.** Static files are served with
  `Cache-Control: no-cache` and API responses with `no-store`. Starlette sends
  an ETag but no `Cache-Control`, so browsers applied heuristic freshness and
  served an old build — and an old `/metrics` payload, which rendered numbers
  from a previous evaluation run.

376 tests.

## Transitivity check

A pairwise score carries no transitivity guarantee: A~B and B~C can both clear
the merge threshold while A~C was compared directly and **rejected**.
Agglomeration alone still put all three in one cluster, and in the evaluation
that was the entire residual false-positive class — every sampled
false-positive *pair* was decided `review` or `reject`, yet the records shared
an entity through a chain of other edges.

A merge is now refused when the cluster it would create would contain a member
pair that was scored and fell below `review_low`. Configurable under
`transitivity` in `config/weights.yaml`, with the same override semantics as
the other conflict rules (a reviewer's accept always wins; a pair at or above
0.95 merges anyway and is recorded as an unresolved conflict).

**The veto fires only on pairs that were actually scored.** A pair blocking
never proposed is unknown, not disagreement — letting absence veto a merge
would turn blocking's 99.6% recall ceiling into a precision ceiling too, and
would tear apart large true clusters where not every internal pair happens to
be a candidate.

| | precision | recall | pairwise F1 | B³ F1 | false-positive pairs |
|---|---|---|---|---|---|
| before | 0.688 | 0.941 | 0.795 | 0.921 | ~4,000 |
| **after** | **0.981** | 0.933 | **0.957** | **0.972** | **168** |

The veto floor is its own setting (`transitivity.veto_below`), not
`review_low`. They started as one knob, and widening the review band from 0.70
to 0.62 silently weakened the veto — F1 fell from 0.957 to 0.944. "Which pairs
does a human look at" and "which pairs are confident enough to block a merge"
are different questions and now have different numbers.

With precision no longer carried by the threshold, the sweep optimum moved from
0.84 to a flat 0.68–0.78 plateau; `auto_merge` is now **0.74**, which recovers
1.4 points of recall for one thousandth of precision.

Two further findings, reported because they are what the measurement says:

- **The field-based conflict rules now earn almost nothing** (ΔF1 −0.003). The
  transitivity check subsumes most of what they caught. Retained because they
  fire on pairs that were never scored, which transitivity by design cannot
  see.
- **Removing the phonetic scoring feature slightly improves F1** (0.961 vs
  0.957). Kept at a non-zero weight because the margin is within noise and the
  signal holds recall up on sparser data — but its indispensable job is
  blocking, where removing it costs 18 points of recall ceiling.

374 tests, 97% coverage on `app/core`.

## Phase 5 — Data, evaluation and docs

- **Synthetic data generator** (`data/generate.py`, pools in `data/pools.yaml`):
  5,000 records from 1,302 true entities, seeded and reproducible. Noise is the
  noise Indian data has — three scripts, curated Latin spelling variants,
  honorifics prefixed and fused, initials, address drift, five phone formats,
  absent fields — plus same-name twins and deliberate common-surname crowding.
  Entities are kept distinguishable from their own data (no two share a name,
  locality and building), because an unanswerable pair measures the generator
  rather than the matcher.
- **Evaluation** (`app/evaluation`): pairwise P/R/F1 and B-cubed, blocking
  recall, a 23-point threshold sweep, six scoring ablations, four blocking
  ablations, and per-slice recall by failure mode. Features are computed once
  and re-combined, so the sweep and ablations cost seconds rather than an hour.
- **`docs/eval-report.md` is generated**, not written: every number comes from
  the run that produced it.
- **README** with a mermaid architecture diagram, what is real vs mocked, and
  productisation notes (data acquisition, scale, DPDP compliance, pricing).
- **`docs/demo-script.md`**: three flows that work end to end.

### Measured results

| | pairwise F1 | B³ F1 | precision | recall |
|---|---|---|---|---|
| indic-er | **0.922** | **0.957** | 0.923 | 0.922 |
| baseline (exact + fuzzy) | 0.074 | 0.324 | 0.040 | 0.452 |

Blocking recall 0.9956. 364 tests, 97% coverage on `app/core`.
(Superseded by the transitivity check above: F1 0.922 → 0.950.)

## Phase 4 — Web app

- Playground (CSV/JSON paste or file drop, threshold sliders, entity cards with
  the full normalization trace, and a "Why?" panel showing per-feature
  contributions and highlighted matching tokens).
- Review queue with keyboard shortcuts (`A`/`R`/`S`/`U`), ordered by
  uncertainty, showing live threshold suggestions from accumulated decisions.
- Cytoscape cluster graph with per-node and per-edge provenance.
- Metrics page comparing the pipeline to the baseline, with the sweep,
  ablations and per-slice recall — and the live seeded numbers kept visibly
  separate from the ground-truth evaluation.
- No build step: ES modules, vendored Cytoscape, hand-written CSS, light/dark,
  responsive. Entity disclosures are keyboard-operable and expose
  `aria-expanded`.

## Phase 3 — API

- `POST /resolve`, `POST /match`, `GET /entities/{id}`, `GET /review-queue`,
  `POST /review/{pair_id}`, `GET /metrics`, plus `/health`, `/ontology`,
  `/showcase`, `/seed` and `/graph/{id}`.
- Response sizes capped deliberately: a 5,000-record resolve produces ~330k
  scored pairs, so merged and review-band pairs are returned and rejected ones
  counted.
- Structured logging with a per-request correlation id; the web app is served
  from the same origin, so the demo needs no CORS and no second process.
- `app/api/seed.py` pre-resolves the demo dataset so first paint is instant.

### Fixed while building

- **Graph writes committed per row**, making a 4,700-fact projection take 41
  seconds. Added a `bulk()` transaction context; it is now instant.

## Phase 2 — Matching pipeline

- **Pairwise features**: name spelling, name sound, name distinctiveness,
  address core, house number, pincode, landmark, phone. A missing field returns
  `None` and is dropped from both sides of the weighted mean, so absence is
  neutral. Names are compared by greedy token alignment, so reordering and
  dropped middle names cost nothing.
- **Blocking**: five keys chosen so a true pair stays reachable when several
  fields are wrong. Oversized blocks are skipped rather than truncated.
- **Clustering**: conflict-aware agglomeration — merge edges applied
  strongest-first, each refused if the cluster it would create trips a conflict
  rule.
- **Review queue**: SQLite-backed, ordered by uncertainty, applied as hard
  overrides before clustering. `threshold_shift()` reports where the decisions
  say the threshold should be, and says so plainly when they overlap and no
  threshold can reproduce them.
- **Baseline**: exact plus Jaro-Winkler on raw strings — implemented properly,
  not strawmanned.
- **Explanations**: deterministic, rule-based, with per-feature contributions.

### Fixed while building (all found by measurement, not review)

- **Transitive-closure blowup.** Connected components collapsed 4,702 of 5,000
  records into one "entity". Replaced with conflict-aware agglomeration that
  checks before the union instead of splitting after; largest cluster fell to
  19 and the run went from 43s to 17s.
- **Name-only bridges.** Because missing features renormalise the rest, a pair
  whose only comparable fields were the name scored 1.00 on an identical common
  name and chained unrelated clusters together. Added the **evidence gate**:
  enough of the weight must be computable *and* something outside the name must
  support the pair, else the pair is held at review, never rejected. Worth
  −0.335 F1 if removed — the single biggest contributor.
- **The address parser threw away the house number.** `core_tokens` drops short
  and numeric tokens by design, which also dropped "111" and "46/A", so two
  different households in one society scored 1.00 on address. Added a
  `house_number` feature.
- **Generic address tokens swamped the distinctive one.** "society", "dadar"
  and "west" agreed three times over and outvoted the building name, scoring
  0.93 for two different Nilesh Pawars. `core_tokens` now excludes type words,
  the locality and the city — the area signal is already carried by `pincode`.
- **The gate blocked pairs whose phone matched exactly.** An exact match on a
  hard identifier now waives the weight quorum.
- **Blocking missed initials.** "G. Pillai" and "Ganesan Pillai" share no full
  phonetic token. Added the `phonetic_initial_sorted` key; blocking recall rose
  from 0.951 to 0.996.
- **Thresholds were guesses.** The sweep showed 0.72 was far too low
  (P=0.525). Moved to 0.84, the joint optimum.
- **Canonical name was the longest string**, so a cluster published the one
  spelling in it that was a typo. Now the consensus spelling.
- **Trade words were stripped as legal suffixes**, merging "Patel Trading" with
  "Patel Textiles" and reducing a business's canonical name to "Patel". Only
  the legal wrapper is stripped now.

## Phase 1 — Normalization

- **Script detection** for Latin, Devanagari, Gujarati and Tamil. Any Indic
  presence wins over a Latin majority, so `"જયેશ Patel"` is still
  transliterated instead of passed through half-unreadable.
- **Transliteration**: abugida-aware and table-driven. Handles the inherent
  vowel, matras, virama clusters and word-final schwa deletion, so राम → `ram`
  and लक्ष्मी → `lakshmi`.
- **Indic phonetic key**: Patel/Patil/Pattel → `PTL`, Jayesh/Jaishe/Jaysh →
  `JS`, Shah/Sah/Chah → `S`, Lakshmi/Laxmi → `LKSM`. Recall-first by design;
  collisions are resolved downstream by scoring.
- **Name normalization**: honorifics as whole tokens, affectionate suffixes as
  token tails, business legal suffixes captured rather than discarded.
  Everything removed is kept for the audit trail, with guards against
  destroying real names (`Juneja`, `Gowda`) and against eating a stem vowel
  (`Kokilaben` → `kokila`, not `kokil`).
- **Phone normalization** to the last 10 digits, rejecting placeholders.
- **Address parsing**: pincode anywhere including `380 008`, abbreviation
  expansion, landmark phrase capture, city/state alias folding, and an
  order-insensitive token set.

## Phase 0 — Scaffold and knowledge-graph layer

- Project scaffold, `pyproject.toml`, ruff and pytest configuration.
- `tasks.py` task runner with a `Makefile` and `make.bat` delegating to it, so
  `make demo`, `make test` and `make eval` work with or without GNU make.
- All domain knowledge externalised to `config/*.yaml`.
- **Ontology**: YAML in, validated Pydantic model out, JSON-LD and Turtle
  export. A relation pointing at an undeclared class, an unknown property type
  or a dropped provenance field fails the load.
- **Provenance**: every fact carries source, timestamp, confidence and method.
- **GraphStore**: interface plus a NetworkX + SQLite implementation with
  ontology validation on every write.
- **LLMClient**: interface, deterministic offline mock, live Anthropic client.
  The matching core does not import it — enforced by a test.
- `/health` and `/ontology`; structured logging with a correlation id.
