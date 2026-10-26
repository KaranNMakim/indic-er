# indic-er — demo script

Three flows, about five minutes. Each one works end to end on seeded data with
no network and no API key.

```bash
make demo          # or: py tasks.py demo
```

Boots the API and the web app on <http://127.0.0.1:8141> and opens a browser.
First boot resolves the seeded dataset (~1,500 records, about four seconds).

> If you want the ground-truth comparison on the Metrics page, run `make eval`
> once first. Everything else works without it.

---

## Flow 1 — "Show me the problem, and show me it solved" (2 min)

**Playground tab → Load example → Resolve.**

Fourteen hand-built records go in. They look like five different problems:

| What you see | What it really is |
|---|---|
| `Jayeshbhai Patel` / `જયેશ પટેલ` / `Shri Jayesh R. Pattel` | one person, three scripts, a fused honorific, a prefixed honorific, a typo |
| `Kokilaben Shah` / `કોકિલા શાહ` / `Smt. Kokila Sah` | one person; `ben` is a suffix, `Sah` and `Shah` are the same sound |
| `Patel Trading Pvt. Ltd.` / `M/s Patel Trading & Sons` | one business under two legal wrappers |
| `Laxmi Subramaniam` / `லக்ஷ்மி சுப்ரமணியன்` | one person, Tamil and Latin |
| `Jayesh Patel` in Bopal, `J. Patil`, `Jayesh Patel` with no address | **three traps** |

**Point at the summary strip.** 14 records in, 8 entities out, 43% duplicate
rate, about 5 ms.

**Then point at what did *not* happen** — this is the part that matters:

- The second `Jayesh Patel` (Bopal, 380058, different phone) is **its own
  entity**. Identical name, and the system did not merge it. The name features
  sum to 0.42, below the 0.84 merge threshold, so a name can never carry a
  merge on its own.
- `Jayesh Patel` with **no address and no phone** is also its own entity, and
  it sits in the review queue. Expand it and read the near-miss: the pair
  scores 1.00 on the name and is held anyway, because
  *"only 42% of the evidence could be computed."* That is the evidence gate.
- `J. Patil` scores **0.835** — just under the threshold. Also held for review.
  Flow 2 picks this up.

**Expand `Jayesh Patel` (3 records).** Each member shows its normalization
trace: `gujarati → jayesh patel`, `~~bhai~~`, `key JS-PTL`. Nothing is
silently discarded; an auditor can see why every token disappeared.

**Click "Why?" on the top merge pair.** You get the per-feature table —
value, weight, contribution, and a sentence in English. Note `pincode` marked
`n/a`: one record has no pincode, so the feature was dropped from both sides
of the weighted mean rather than scored as a disagreement.

**Optional, 15 seconds:** drag the *Merge at* slider down to 0.70 and press
Resolve again. Entity count drops; the two different Jayesh Patels merge. Drag
it back. The threshold is a business decision, not a constant.

---

## Flow 2 — "What happens when the machine won't decide" (1.5 min)

**Review queue tab.**

The queue holds the borderline pairs, ordered by *uncertainty* — closest to
the decision boundary first, because that is where a reviewer's five seconds
are worth the most.

The top card is a full pair view: both records, their normalization traces,
the score, the feature breakdown, and an LLM second opinion clearly marked
**advisory — it changes nothing**.

**Press `A`.** (Or `R` to reject, `S` to skip, `U` to undo.)

Three things happen, and all three are the point:

1. The decision is stored with the reviewer and a timestamp.
2. The seeded dataset is **re-resolved immediately** with that decision applied
   as a hard override *before* clustering. The toast reports the new entity
   count. The decision shaped the output; it is not an annotation on top of it.
3. The **"What your decisions imply"** banner updates: *"1 pair accepted so
   far, the lowest scoring 0.77. The auto-merge threshold looks too high —
   lowering it to that point would have merged them automatically."*

That banner is how review work compounds. Accept a few more and it converges
on a suggested threshold; if your decisions ever contradict each other it says
so plainly — *"the decisions overlap … that band is genuinely ambiguous on
score alone and is where a new feature, not a new threshold, is what would
help."*

A decided pair never returns to the queue.

---

## Flow 3 — "Is it actually better than what we'd build in a weekend?" (1.5 min)

**Metrics tab.**

Top strip, measured on 5,000 synthetic records with ground truth:

| | pairwise F1 | B³ F1 | precision | recall |
|---|---|---|---|---|
| **indic-er** | **0.922** | **0.957** | 0.923 | 0.922 |
| baseline (exact + fuzzy) | 0.074 | 0.324 | 0.040 | 0.452 |

**12.4× the baseline on pairwise F1.** The baseline is a real implementation,
not a strawman — exact match plus Jaro-Winkler on raw strings, blocked on two
characters. It catches a trailing suffix and a plain typo. It cannot see
through a leading honorific, a script change, or Patel/Patil.

Scroll down for the three things a sceptical buyer asks next:

- **Threshold sweep.** The curve is flat above 0.84, so the shipped default is
  not balanced on a knife edge. Want more precision? One line of config.
- **Ablations — what each piece is worth.** The evidence gate is the single
  biggest contributor (−0.335 F1 without it). Address evidence is next
  (−0.305). Phonetics move scoring by only −0.006 …
- **… which the blocking table explains.** Remove phonetic blocking and recall
  falls from **0.996 to 0.814**: 1,758 true pairs never become candidates and
  no threshold can recover them. The phonetic key's job is reach, not scoring.
  Reporting the ablation without this table would be misleading, so both are
  shown.
- **Recall by failure mode**, worst first. `name_only` records are the weak
  slice by design — and the honest answer is that they need a human or another
  field, which is what Flow 2 is for.

**Cluster graph tab** (optional, 20 seconds). Pick any merged entity. Records
are circles, the canonical entity is the green square, edges carry the pair
score. Click any node or edge for its provenance: source, method
(`rule` / `model` / `human`), confidence, timestamp. Every fact in the graph
has one — the ontology refuses the write otherwise.

---

## If something looks wrong

| Symptom | Cause |
|---|---|
| Metrics says *"Ground-truth comparison not available"* | `make eval` has not been run. Everything else still works. |
| Review queue empty | Resolve a batch in the playground with *Add borderline pairs to the queue* ticked. |
| `no seed data found` on boot | Run `make data` first (`make demo` does it for you). |
| LLM text looks templated | That is `LLM_MODE=mock`, the default — deterministic and offline by design. Set `LLM_MODE=live` and `ANTHROPIC_API_KEY` for the real thing. The *matching* never uses an LLM either way. |

## The 30-second version

```bash
make demo
```

Load example → Resolve → look at the two `Jayesh Patel` entities that did
**not** merge → Metrics tab → 0.922 against the baseline's 0.074.
