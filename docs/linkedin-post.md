# LinkedIn post — draft

## Main version (~2,000 characters)

---

**Jayeshbhai Patel. જયેશ પટેલ. J. Patil.**

Three rows in three systems. One person.

Every Indian fintech, logistics and e-commerce team has this problem, and most
of them are still throwing fuzzy string matching at it. I spent a week building
a proper entity resolution engine for Indian data to find out how much that
costs them.

The answer: a lot.

**Why fuzzy matching loses here**

The noise in Indian data isn't random — it's *systematic*. Transliteration
choice (Jayesh / Jaishe / Jaysh). Honorifics fused into the name (bhai, ben,
ji). Legal suffixes (Pvt Ltd, & Sons). Landmark-relative addresses ("Nr.
Swaminarayan Mandir", "opp. bus stand"). And surnames so common that
Patel, Shah or Kumar matching tells you almost nothing.

The hardest case is the opposite of the one people optimise for: **two
different people genuinely called Jayesh Patel.** Every real dataset has them.

**What I built**

Script detection → transliteration → honorific stripping → an Indic-tuned
phonetic key (Patel/Patil/Pattel all collapse to PTL) → landmark-aware address
parsing → blocking → weighted scoring → conflict-aware clustering. Every merge
carries provenance and a plain-English explanation. Borderline pairs go to a
human, and those decisions feed back as hard overrides.

**Measured on 5,000 records against ground truth:**

→ pairwise F1 **0.957** vs **0.074** for exact + Jaro-Winkler matching
→ 168 false-positive pairs vs 101,138
→ blocking recall 0.996

**The three things that actually mattered — none of which I'd have guessed**

1️⃣ **A pairwise score says nothing about transitivity.** A~B and B~C can both
pass while A~C was compared directly and *rejected*. Clustering merged all
three anyway. Refusing that one merge moved precision from 0.69 to 0.98.

2️⃣ **Treating a missing field as a disagreement quietly destroys you.** A
record with no phone must not score worse than one with a *wrong* phone.

3️⃣ **An identical name must never be enough to merge.** When the only
comparable fields are names, the score renormalises to ~1.00 on any common
name — and those pairs chain unrelated clusters together. In my first full run
that bug merged 4,702 of 5,000 records into a single "entity."

**The honest caveat:** this is a proof of value on synthetic data. The numbers
measure the pipeline against my own noise model, not anyone's production
system. Real validation needs a few thousand hand-labelled pairs from real
records — which is step one of any serious deployment anyway.

Code, evaluation report and demo: https://github.com/KaranNMakim/indic-er

#EntityResolution #DataEngineering #India #MachineLearning #DataQuality

---

## Short version (~900 characters)

---

**Jayeshbhai Patel. જયેશ પટેલ. J. Patil.** Three rows, one person.

I built an entity resolution engine for messy Indian names and addresses to see
how far fuzzy matching actually gets you.

Measured on 5,000 records with ground truth:
→ pairwise F1 **0.957** vs **0.074** for exact + Jaro-Winkler
→ 168 false-positive pairs vs 101,138

Three lessons I didn't expect:

**1.** A pairwise score says nothing about transitivity. A~B and B~C can both
pass while A~C was explicitly rejected — and clustering merges all three
anyway. Fixing that took precision from 0.69 to 0.98.

**2.** Treating a missing field as a disagreement is quietly fatal. No phone
must not score worse than a *wrong* phone.

**3.** An identical name can't be enough to merge. My first run chained 4,702
of 5,000 records into one "entity" because of this.

Proof of value on synthetic data — real validation needs labelled production
records.

Code + evaluation report: https://github.com/KaranNMakim/indic-er

#EntityResolution #DataEngineering #India

---

## Notes before you post

- The numbers are all reproducible with `make eval`; `docs/eval-report.md` is
  generated, so anyone who clones it gets the same figures.
- If you want a visual: the metrics page screenshot (pipeline vs baseline
  table) or the cluster graph both work well. LinkedIn favours a single clear
  image over a carousel for technical posts.
- The "honest caveat" paragraph is worth keeping. It is the thing that makes
  the rest of the numbers credible to anyone who has actually done this work.
