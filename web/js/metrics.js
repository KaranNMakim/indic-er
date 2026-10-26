// Metrics: our pipeline against the baseline.
//
// Two different things, kept visibly apart. The *evaluation* numbers come from
// `make eval` over 5,000 records with ground truth and are the ones worth
// quoting. The *seeded* numbers describe whatever is loaded in this process
// right now and have no ground truth at all.

import { api } from "./api.js";
import { $, esc, fixed } from "./util.js";

export async function loadMetrics() {
  const body = $("#metrics-body");
  body.innerHTML = '<p class="muted"><span class="spinner"></span> Loading…</p>';
  try {
    const data = await api.metrics();
    body.innerHTML = [
      data.evaluation_available ? evaluationSection(data.evaluation) : missingEval(data.note),
      seededSection(data),
      featureSection(data),
    ].join("");
  } catch (error) {
    body.innerHTML = `<p class="error">${esc(error.message)}</p>`;
  }
}

function missingEval(note) {
  return `
    <div class="metric-card">
      <h3>Ground-truth comparison not available</h3>
      <p class="muted">${esc(note || "Run `make eval` to populate it.")}</p>
      <p class="hint">The evaluation needs the labelled synthetic dataset, so it
      runs offline rather than in the request path.</p>
    </div>`;
}

function evaluationSection(evaluation) {
  const pipeline = evaluation.pipeline;
  const baselines = evaluation.baselines || [];
  const fuzzy = baselines.find((b) => b.label === "baseline_fuzzy");
  const exact = baselines.find((b) => b.label === "baseline_exact");
  const blocking = evaluation.blocking || {};

  const row = (label, report, ours = false) => {
    if (!report) return "";
    const p = report.pairwise;
    const b = report.bcubed;
    return `
      <tr class="${ours ? "ours" : ""}">
        <td>${esc(label)}</td>
        <td>${fixed(p.precision)}</td>
        <td>${fixed(p.recall)}</td>
        <td class="${ours ? "win" : ""}">${fixed(p.f1)}</td>
        <td>${fixed(b.precision)}</td>
        <td>${fixed(b.recall)}</td>
        <td class="${ours ? "win" : ""}">${fixed(b.f1)}</td>
        <td>${report.predicted_clusters}</td>
        <td>${report.largest_predicted}</td>
      </tr>`;
  };

  const lift = fuzzy ? pipeline.pairwise.f1 / Math.max(fuzzy.pairwise.f1, 0.0001) : null;

  return `
    <h3 class="section-title">Evaluation — ${pipeline.records.toLocaleString()} records,
      ${pipeline.true_clusters.toLocaleString()} true entities, with ground truth</h3>

    <div class="summary">
      <div class="stat good"><b>${fixed(pipeline.pairwise.f1)}</b><span>pairwise F1</span></div>
      <div class="stat good"><b>${fixed(pipeline.bcubed.f1)}</b><span>B³ F1</span></div>
      <div class="stat"><b>${fixed(pipeline.pairwise.precision)}</b><span>precision</span></div>
      <div class="stat"><b>${fixed(pipeline.pairwise.recall)}</b><span>recall</span></div>
      ${lift ? `<div class="stat good"><b>${lift.toFixed(1)}×</b><span>vs fuzzy baseline</span></div>` : ""}
      <div class="stat"><b>${fixed(blocking.blocking_recall, 4)}</b><span>blocking recall</span></div>
    </div>

    <table class="cmp-table">
      <thead>
        <tr>
          <th></th>
          <th colspan="3">pairwise</th>
          <th colspan="3">B-cubed</th>
          <th colspan="2">clusters</th>
        </tr>
        <tr>
          <th></th><th>P</th><th>R</th><th>F1</th>
          <th>P</th><th>R</th><th>F1</th>
          <th>count</th><th>largest</th>
        </tr>
      </thead>
      <tbody>
        ${row("indic-er", pipeline, true)}
        ${row("baseline — exact only", exact)}
        ${row("baseline — exact + fuzzy", fuzzy)}
        <tr>
          <td>ground truth</td><td>1.000</td><td>1.000</td><td>1.000</td>
          <td>1.000</td><td>1.000</td><td>1.000</td>
          <td>${pipeline.true_clusters}</td><td>${pipeline.largest_true}</td>
        </tr>
      </tbody>
    </table>
    <p class="hint">
      Pairwise F1 maps onto business cost — a false positive is two customers in
      one account. B³ asks the same question per record, so one enormous wrong
      cluster cannot hide. The fuzzy baseline's largest cluster holds
      ${fuzzy ? fuzzy.largest_predicted : "—"} records against a true maximum of
      ${pipeline.largest_true}, which is why both are reported.
    </p>

    ${sweepSection(evaluation.sweep, evaluation)}
    ${ablationSection(evaluation.ablations, pipeline)}
    ${blockingAblationSection(evaluation.blocking_ablations)}
    ${sliceSection(evaluation.slices)}`;
}

function sweepSection(sweep, evaluation) {
  if (!sweep?.length) return "";
  const scores = sweep.map((r) => r.pairwise_f1);
  const best = Math.max(...scores);
  const worst = Math.min(...scores);
  const current = evaluation.pipeline.auto_merge;
  const span = Math.max(best - worst, 0.0001);

  // Scaled between the worst and best point, not from zero. Every F1 here is
  // between 0.76 and 0.96, so a zero-based axis renders the whole sweep as a
  // flat wall of identical bars and hides the only thing the chart is for.
  const bars = sweep
    .map((row) => {
      const height = 8 + ((row.pairwise_f1 - worst) / span) * 92;
      const cls =
        row.auto_merge === current ? "current" : row.pairwise_f1 === best ? "best" : "";
      return `<div class="sweep-bar ${cls}" style="height:${height.toFixed(1)}%"
        data-label="${row.auto_merge} → F1 ${fixed(row.pairwise_f1)} · P ${fixed(row.pairwise_precision)} · R ${fixed(row.pairwise_recall)}"></div>`;
    })
    .join("");

  // Describe the plateau from the data rather than asserting a shape that
  // was true of an earlier configuration.
  const plateau = sweep.filter((r) => best - r.pairwise_f1 <= 0.002);
  const lo = Math.min(...plateau.map((r) => r.auto_merge));
  const hi = Math.max(...plateau.map((r) => r.auto_merge));
  const currentRow = sweep.find((r) => r.auto_merge === current);

  return `
    <h3 class="section-title">Threshold sweep — pairwise F1 by merge threshold</h3>
    <div class="metric-card">
      <div class="sweep-chart">${bars}</div>
      <div class="sweep-axis"><span>${sweep[0].auto_merge}</span><span>F1 ${fixed(worst)} – ${fixed(best)}</span><span>${sweep[sweep.length - 1].auto_merge}</span></div>
      <p class="hint" style="margin-top:8px">
        Blue is the shipped default (${current}); green is the measured optimum.
        F1 is within 0.002 of its peak across <strong>${lo}–${hi}</strong>, so the
        default sits on a plateau rather than a knife edge${
          currentRow ? ` — it scores ${fixed(currentRow.pairwise_f1)}` : ""
        }. A customer who wants more precision than recall moves one line of config.
      </p>
    </div>`;
}

function ablationSection(ablations, pipeline) {
  if (!ablations?.length) return "";
  const rows = [...ablations]
    .sort((a, b) => a.pairwise.f1 - b.pairwise.f1)
    .map((report) => {
      const delta = report.pairwise.f1 - pipeline.pairwise.f1;
      return `
        <tr>
          <td><code>${esc(report.label)}</code></td>
          <td>${fixed(report.pairwise.precision)}</td>
          <td>${fixed(report.pairwise.recall)}</td>
          <td>${fixed(report.pairwise.f1)}</td>
          <td>${fixed(report.bcubed.f1)}</td>
          <td style="color:var(--reject)">${delta.toFixed(3)}</td>
        </tr>`;
    })
    .join("");

  return `
    <h3 class="section-title">Ablations — what each component is worth</h3>
    <table class="cmp-table">
      <thead><tr><th>variant</th><th>P</th><th>R</th><th>F1</th><th>B³ F1</th><th>ΔF1</th></tr></thead>
      <tbody>
        <tr class="ours">
          <td>full pipeline</td>
          <td>${fixed(pipeline.pairwise.precision)}</td>
          <td>${fixed(pipeline.pairwise.recall)}</td>
          <td>${fixed(pipeline.pairwise.f1)}</td>
          <td>${fixed(pipeline.bcubed.f1)}</td>
          <td>—</td>
        </tr>
        ${rows}
      </tbody>
    </table>`;
}

function blockingAblationSection(rows) {
  if (!rows?.length) return "";
  return `
    <h3 class="section-title">Blocking — the recall ceiling</h3>
    <table class="cmp-table">
      <thead><tr><th>strategies</th><th>recall</th><th>true pairs missed</th><th>candidate pairs</th></tr></thead>
      <tbody>
        ${rows
          .map(
            (row) => `
          <tr class="${row.strategies === "all_strategies" ? "ours" : ""}">
            <td><code>${esc(row.strategies)}</code></td>
            <td>${fixed(row.blocking_recall, 4)}</td>
            <td>${row.missed_true_pairs}</td>
            <td>${row.candidate_pairs.toLocaleString()}</td>
          </tr>`
          )
          .join("")}
      </tbody>
    </table>
    <p class="hint">
      This is where the Indic phonetic key earns its place. It barely moves the
      scoring ablations above, because <code>name_string</code> already handles
      most spelling variants after transliteration — but remove it from blocking
      and the true pairs it would have proposed are out of reach of any
      threshold.
    </p>`;
}

function sliceSection(slices) {
  if (!slices?.length) return "";
  const rows = [...slices]
    .sort((a, b) => (a.recall ?? 1) - (b.recall ?? 1))
    .map(
      (row) => `
      <tr>
        <td><code>${esc(row.slice)}</code></td>
        <td>${row.records}</td>
        <td>${row.true_pairs}</td>
        <td class="${row.recall != null && row.recall < 0.8 ? "" : "win"}">${fixed(row.recall)}</td>
      </tr>`
    )
    .join("");
  return `
    <h3 class="section-title">Recall by failure mode — worst first</h3>
    <table class="cmp-table">
      <thead><tr><th>slice</th><th>records</th><th>true pairs</th><th>recall</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <p class="hint">One average hides a broken case. These are the slices worth
    labelling first on a customer's own data.</p>`;
}

function seededSection(data) {
  const seeded = data.seeded;
  const clustering = seeded.clustering;
  const blocking = seeded.blocking;
  return `
    <h3 class="section-title">This process — live, no ground truth</h3>
    <div class="metric-grid">
      <div class="metric-card">
        <h3>Seeded dataset</h3>
        <dl class="kv">
          <dt>source</dt><dd>${esc(seeded.source)}</dd>
          <dt>records</dt><dd>${seeded.records.toLocaleString()}</dd>
          <dt>entities</dt><dd>${clustering.entities.toLocaleString()}</dd>
          <dt>merged</dt><dd>${clustering.merged_entities.toLocaleString()}</dd>
          <dt>largest</dt><dd>${clustering.largest_entity}</dd>
          <dt>duplicate rate</dt><dd>${Math.round(seeded.duplicate_rate * 100)}%</dd>
          <dt>merges refused</dt><dd>${clustering.blocked_merges}</dd>
        </dl>
      </div>
      <div class="metric-card">
        <h3>Blocking</h3>
        <dl class="kv">
          <dt>candidate pairs</dt><dd>${blocking.candidate_pairs.toLocaleString()}</dd>
          <dt>all pairs</dt><dd>${blocking.all_pairs.toLocaleString()}</dd>
          <dt>reduction</dt><dd>${(blocking.reduction_ratio * 100).toFixed(2)}%</dd>
          <dt>blocks</dt><dd>${blocking.blocks.toLocaleString()}</dd>
          <dt>oversized skipped</dt><dd>${blocking.skipped_oversized_blocks}</dd>
        </dl>
      </div>
      <div class="metric-card">
        <h3>Decisions</h3>
        <dl class="kv">
          <dt>merged pairs</dt><dd>${seeded.pairs_merged.toLocaleString()}</dd>
          <dt>to review</dt><dd>${seeded.pairs_review.toLocaleString()}</dd>
          <dt>rejected</dt><dd>${seeded.pairs_rejected.toLocaleString()}</dd>
          <dt>overrides applied</dt><dd>${seeded.overrides_applied}</dd>
        </dl>
      </div>
      <div class="metric-card">
        <h3>Review queue</h3>
        <dl class="kv">
          <dt>pending</dt><dd>${data.review.counts.pending.toLocaleString()}</dd>
          <dt>accepted</dt><dd>${data.review.counts.accepted}</dd>
          <dt>rejected</dt><dd>${data.review.counts.rejected}</dd>
        </dl>
        <p class="hint" style="margin-top:7px">${esc(data.review.threshold_shift.note)}</p>
      </div>
      <div class="metric-card">
        <h3>Timings</h3>
        <dl class="kv">
          ${Object.entries(seeded.timings_ms)
            .map(([stage, ms]) => `<dt>${esc(stage)}</dt><dd>${Number(ms).toFixed(0)} ms</dd>`)
            .join("")}
        </dl>
      </div>
    </div>`;
}

function featureSection(data) {
  return `
    <h3 class="section-title">Scoring weights in force</h3>
    <div class="metric-card">
      <table class="cmp-table">
        <thead><tr><th>feature</th><th>weight</th><th>what it measures</th></tr></thead>
        <tbody>
          ${data.features
            .map(
              (f) => `
            <tr>
              <td><code>${esc(f.name)}</code></td>
              <td>${f.weight.toFixed(2)}</td>
              <td style="text-align:left" class="muted">${esc(f.description)}</td>
            </tr>`
            )
            .join("")}
        </tbody>
      </table>
      <p class="hint" style="margin-top:8px">
        Merge at <strong>${data.thresholds.auto_merge}</strong>, review from
        <strong>${data.thresholds.review_low}</strong>. The name features sum to
        less than the merge threshold on purpose: an identical name cannot carry
        a merge on its own.
      </p>
    </div>`;
}
