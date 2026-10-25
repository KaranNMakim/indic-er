# indic-er — Plan

Entity resolution for messy Indian names, businesses and addresses.
Proof-of-value: a stakeholder should understand the value in 5 minutes.

---

## 1. Problem

One real person, business or address shows up many times across Indian datasets:

| Record | Name | Address | Phone |
|---|---|---|---|
| A | Jayeshbhai Patel | 12, Shreeji Soc., Nr. Swaminarayan Mandir, Maninagar, Ahmedabad 380008 | +91 98250 12345 |
| B | જયેશ પટેલ | 12 Shriji Society, opp Swaminarayan Temple, Maninagar, Ahmedabad | 9825012345 |
| C | J. Patil | Shreeji Society, Maninagar, Amdavad, 380 008 | — |

A fintech doing KYC dedupe, a logistics firm merging delivery addresses, or an
e-commerce firm building a single customer view all need these three rows
collapsed into one canonical entity — with an auditable reason for the merge.

Plain fuzzy matching fails here because the noise is **systematic**, not random:
transliteration choice (Jayesh / Jaishe / Jaysh), honorifics fused into the name
(bhai / ben / ji), legal suffixes (Pvt Ltd, & Sons), Indic address idiom
("Nr.", "opp.", "near ___ mandir"), and extremely common surnames
(Patel / Shah / Kumar) that make name similarity nearly uninformative on its own.

## 2. Architecture

```
                 +------------------------------------------+
  CSV / JSON --> |  Normalization                           |
                 |  script detect -> transliterate ->       |
                 |  honorific/suffix strip -> phonetic key  |
                 |  address parse (pincode/locality/landmark)|
                 +---------------+--------------------------+
                                 v
                 +------------------------------------------+
                 |  Blocking (phonetic key | pincode | phone)|
                 +---------------+--------------------------+
                                 v
                 +------------------------------------------+
                 |  Pairwise scoring (weights in YAML)      |
                 |  name string / name phonetic / address / |
                 |  phone / locality -> score + per-feature |
                 |  contributions                           |
                 +---------------+--------------------------+
                                 v
                 +------------------------------------------+
                 |  Clustering: connected components with a |
                 |  conflict check (+ human overrides)      |
                 +---------------+--------------------------+
                                 v
          CanonicalEntity  +  merge provenance  +  explanation
                                 |
              +------------------+------------------+
              v                  v                  v
          FastAPI            KG (NetworkX        Review queue
          /resolve /match     + SQLite)          (SQLite, overrides)
```

Layers:

- **`app/kg`** — ontology loader (YAML -> validated model -> JSON-LD/Turtle via
  rdflib), `GraphStore` interface with a NetworkX + SQLite implementation,
  provenance records, `LLMClient` interface.
- **`app/core`** — normalization, address parsing, features, blocking,
  clustering, pipeline, baseline, review store, explanations. No I/O, no LLM
  dependency.
- **`app/api`** — FastAPI surface over core.
- **`app/evaluation`** — pairwise + B-cubed metrics, threshold sweep, ablations.
- **`web`** — playground, review queue, cluster graph, metrics.
- **`config`** — ontology, honorific/suffix lists, phonetic folding rules,
  address abbreviations, scoring weights. All YAML, all hot-swappable.

### Why the KG layer at all

An ER result *is* a graph: records are nodes, `same_as` edges carry a score and
evidence, `located_at` and `contact_of` connect people to addresses and phones.
Keeping it behind a `GraphStore` interface means the POV runs on NetworkX+SQLite
today and can move to Neo4j/Kuzu without touching the matching code, and every
fact carries `(source, timestamp, confidence, method)` so a merge can be
defended to an auditor.

## 3. Build order

| Phase | Deliverable |
|---|---|
| 0 | Scaffold, task runner, lint/test config, ontology + GraphStore + provenance + LLMClient, `/health` + `/ontology` |
| 1 | Normalization: script detection, transliteration, honorific/suffix stripping, Indic phonetic keys, address parsing |
| 2 | Blocking -> pairwise scoring -> clustering -> canonical records; baseline (exact + Jaro-Winkler/Levenshtein on raw strings); review store + overrides; explanations |
| 3 | Full API: `/resolve`, `/match`, `/entities/{id}`, `/review-queue`, `/review/{id}`, `/metrics` |
| 4 | Web app: playground, review queue, Cytoscape cluster graph, metrics page |
| 5 | Synthetic data generator, evaluation, `docs/eval-report.md`, demo script, README polish |

After each phase: run tests, update README/CHANGELOG, commit, print a 5-line
works / mocked / next summary.

## 4. Scoring model (phase 2 detail)

Features computed per candidate pair, each in `[0,1]`, missing -> `None` (the
feature is dropped and the remaining weights renormalise, so a record with no
phone is not penalised for it):

| Feature | Signal |
|---|---|
| `name_string` | token-set Jaro-Winkler / ratio on the normalized Latin name |
| `name_phonetic` | agreement of Indic phonetic keys, per token |
| `name_rarity` | downweights a match carried only by a very common surname |
| `address_core` | similarity of parsed house/society/locality tokens, order-tolerant |
| `pincode` | exact, or same-district prefix |
| `landmark` | landmark phrase agreement after abbreviation expansion |
| `phone` | normalized last-10-digit equality |

Score = weighted mean of the present features (weights in
`config/weights.yaml`). Decision bands, also in YAML: `>= high` auto-merge,
`< low` reject, in between -> review queue.

Clustering is connected components over auto-merge edges, then a **conflict
check** splits a component when it contains a pair in strong disagreement
(e.g. two different pincodes *and* two different phones) — this is what stops
the Patel/Shah surname blobs from swallowing a whole block.

## 5. Human-in-the-loop

Borderline pairs land in a SQLite review queue. An accept/reject decision is
stored with its timestamp and reviewer, and on re-run is applied as a hard
override *before* clustering (accept -> forced edge, reject -> forbidden edge).
`/metrics` reports how the decisions made so far would shift the optimal
threshold, which is the honest way to show that review work compounds.

## 6. Assumptions

1. **POV, not production.** In-process, single node, no auth, no multi-tenancy.
2. **Synthetic data only.** No real PII, no scraping. Identifiers are masked
   (`XXXX1234`) even in synthetic records, so the shape is exercised without
   ever holding a real identifier.
3. **Transliteration is rule-based**, not a trained model: a curated
   Devanagari / Gujarati / Tamil -> Latin mapping plus Indic-aware folding. The
   right call for a POV (deterministic, offline, inspectable, no model download)
   and also the part a productised version would improve most.
4. **Matching never calls an LLM.** The LLM is optional, behind `LLMClient`,
   used only to phrase explanations and to opine on ambiguous pairs when
   `LLM_MODE=live`. `LLM_MODE=mock` is the default, is deterministic, and needs
   no API key and no network.
5. **English-script output.** Canonical names are Latin; the original script is
   kept as provenance, never discarded.
6. Ground truth for evaluation is the generator's entity id — so the reported
   numbers measure the pipeline against *our* noise model. That is a stated
   limitation in the eval report, not a claim about production data.

## 7. Risks and deviations from the brief

| Risk / deviation | Handling |
|---|---|
| **No Node/npm on this machine** — Next.js cannot be installed or built. | The web app is a dependency-free ES-module SPA (vendored Tailwind + Cytoscape, no build step) served by FastAPI at the same origin. Every required screen is delivered. Module boundaries mirror a Next.js App Router layout so a port is mechanical. Documented in README under "what's mocked vs real". |
| **No `make`** on this machine. | Targets live in `tasks.py`; a `Makefile` (POSIX) and `make.bat` (Windows `cmd`) both delegate to it. `py tasks.py demo` always works. |
| **Python 3.14, not 3.12.** | Code targets 3.12+ syntax and runs on 3.14; `requires-python = ">=3.12"`. |
| Common surnames inflate recall. | `name_rarity` feature plus the conflict check in clustering; measured as a named failure class in the eval report. |
| Transliteration quality ceiling. | Phonetic keys do the heavy lifting, so a transliteration miss degrades a match rather than breaking it. |
| Over-merging is worse than under-merging for KYC. | Report precision and recall separately, expose the threshold, default the band conservatively. |
| Eval numbers flattered by a too-easy generator. | The generator injects hard negatives: same name different people, shared common surname in the same pincode. Reported as its own slice. |

## 8. Definition of done

- `py tasks.py demo` (or `make demo`) boots API + web on seeded data, offline.
- `make test` green, >=80% coverage on `app/core`, API contract tests, web smoke test.
- `make eval` writes real numbers to `docs/eval-report.md`: baseline vs pipeline,
  pairwise P/R/F1, B-cubed, threshold sweep, ablations, failure analysis.
- `docs/demo-script.md`: 3 flows that work end to end.
- README: mermaid diagram, setup, mocked vs real, productisation notes
  (data acquisition, scale, DPDP compliance, pricing hypothesis).
