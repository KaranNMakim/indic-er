// Review queue: accept or reject borderline pairs, with keyboard shortcuts.
//
// A decision is stored and applied as a hard override before clustering on
// the next run, so the queue is the mechanism by which review work compounds
// rather than a list of annotations.

import { api } from "./api.js";
import { pairCard } from "./pair.js";
import { toInput } from "./playground.js";
import { $, el, esc, fixed, toast } from "./util.js";

let queue = [];
let cursor = 0;
let lastDecided = null;
let active = false;
let onChange = () => {};

export function initReview(hooks = {}) {
  onChange = hooks.onChange ?? (() => {});
  $("#review-order").addEventListener("change", load);
  $("#review-state").addEventListener("change", load);
  document.addEventListener("keydown", onKey);
}

export function setReviewActive(isActive) {
  active = isActive;
  if (isActive) load();
}

function onKey(event) {
  if (!active) return;
  if (event.target.matches("input, textarea, select")) return;
  const key = event.key.toLowerCase();
  if (key === "a") { event.preventDefault(); decide("accept"); }
  else if (key === "r") { event.preventDefault(); decide("reject"); }
  else if (key === "s") { event.preventDefault(); skip(); }
  else if (key === "u") { event.preventDefault(); undo(); }
}

export async function load() {
  const wrap = $("#review-card-wrap");
  wrap.innerHTML = '<p class="muted"><span class="spinner"></span> Loading queue…</p>';
  try {
    const data = await api.reviewQueue({
      limit: 50,
      order: $("#review-order").value,
      state: $("#review-state").value,
    });
    queue = data.items;
    cursor = 0;
    renderStats(data);
    renderThresholdNote(data.threshold_shift);
    renderCurrent();
    renderRest();
    updateBadge(data.counts);
  } catch (error) {
    wrap.innerHTML = `<p class="error">${esc(error.message)}</p>`;
  }
}

function renderStats(data) {
  const counts = data.counts;
  $("#queue-stats").innerHTML = `
    <div class="stat warn"><b>${counts.pending}</b><span>pending</span></div>
    <div class="stat good"><b>${counts.accepted}</b><span>accepted</span></div>
    <div class="stat"><b>${counts.rejected}</b><span>rejected</span></div>
    <div class="stat"><b>${counts.total}</b><span>seen</span></div>
  `;
}

function renderThresholdNote(shift) {
  if (!shift) return;
  const moved =
    shift.decisions > 0 && shift.suggested_auto_merge !== shift.current_auto_merge;
  $("#threshold-note").innerHTML = `
    <strong>What your decisions imply.</strong> ${esc(shift.note)}
    ${
      moved
        ? ` <span class="chip">current ${fixed(shift.current_auto_merge, 2)}</span>
            <span class="chip merge">suggested ${fixed(shift.suggested_auto_merge, 2)}</span>`
        : ""
    }
    ${shift.overlap ? '<span class="chip review">decisions overlap</span>' : ""}
  `;
}

async function renderCurrent() {
  const wrap = $("#review-card-wrap");
  const item = queue[cursor];
  if (!item) {
    wrap.innerHTML = `
      <div class="empty">
        <p><strong>Queue clear.</strong></p>
        <p>Nothing borderline is waiting. Resolve a batch in the playground with
        <em>Add borderline pairs to the queue</em> ticked to refill it.</p>
      </div>`;
    return;
  }

  wrap.innerHTML = `<p class="muted"><span class="spinner"></span> Scoring pair…</p>`;
  try {
    // Re-score through /match so the card shows the live feature breakdown and
    // the highlighted tokens, exactly as the playground does.
    const data = await api.match(
      toInput(item.left, item.left_id),
      toInput(item.right, item.right_id),
      { explain_with_llm: true, adjudicate: true }
    );

    const accept = el('<button class="btn btn-accept">Accept <kbd>A</kbd></button>');
    const reject = el('<button class="btn btn-reject">Reject <kbd>R</kbd></button>');
    const skipBtn = el('<button class="btn btn-ghost">Skip <kbd>S</kbd></button>');
    accept.addEventListener("click", () => decide("accept"));
    reject.addEventListener("click", () => decide("reject"));
    skipBtn.addEventListener("click", skip);

    const card = pairCard(data, { actions: [accept, reject, skipBtn] });
    card.prepend(
      el(`<p class="hint">Pair <code>${esc(item.pair_id)}</code> —
          ${cursor + 1} of ${queue.length} in this page</p>`)
    );

    if (data.adjudication) {
      card.append(
        el(`<p class="hint"><strong>LLM second opinion
            (${esc(data.adjudication.mode)}, advisory — it changes nothing):</strong>
            ${esc(data.adjudication.verdict)} at
            ${Math.round(data.adjudication.confidence * 100)}% —
            ${esc(data.adjudication.rationale)}</p>`)
      );
    }

    wrap.innerHTML = "";
    wrap.append(card);
  } catch (error) {
    wrap.innerHTML = `<p class="error">${esc(error.message)}</p>`;
  }
}

function renderRest() {
  const rest = queue.slice(cursor + 1, cursor + 13);
  $("#review-rest").innerHTML = rest.length
    ? `<h3 class="section-title">Next in the queue</h3>` +
      rest
        .map(
          (item) => `
        <div class="review-rest-row ${item.state === "decided" ? "decided" : ""}">
          <span class="chip ${item.state === "decided" ? "merge" : "review"}">${fixed(item.score, 3)}</span>
          <span>${esc(item.left?.raw_name ?? item.left_id)} <span class="muted">~</span>
                ${esc(item.right?.raw_name ?? item.right_id)}</span>
          <span class="ids">${esc(item.pair_id)}</span>
          ${item.decision ? `<span class="chip">${esc(item.decision)} by ${esc(item.reviewer ?? "")}</span>` : ""}
        </div>`
        )
        .join("")
    : "";
}

async function decide(decision) {
  const item = queue[cursor];
  if (!item) return;
  try {
    const result = await api.decide(item.pair_id, decision);
    lastDecided = item.pair_id;
    toast(
      `${decision === "accept" ? "Accepted" : "Rejected"} ${item.pair_id} — ` +
        `${result.entities.entities} entities, ${result.overrides_applied} override(s) applied`
    );
    renderStats(result);
    renderThresholdNote(result.threshold_shift);
    updateBadge(result.counts);
    queue.splice(cursor, 1);
    if (cursor >= queue.length) cursor = Math.max(0, queue.length - 1);
    renderCurrent();
    renderRest();
    onChange();
  } catch (error) {
    toast(error.message);
  }
}

function skip() {
  if (cursor < queue.length - 1) {
    cursor += 1;
    renderCurrent();
    renderRest();
  } else {
    toast("End of this page — reordering reloads the queue");
  }
}

async function undo() {
  if (!lastDecided) { toast("Nothing to undo"); return; }
  try {
    await api.undo(lastDecided);
    toast(`Undid ${lastDecided}`);
    lastDecided = null;
    onChange();
    load();
  } catch (error) {
    toast(error.message);
  }
}

export function updateBadge(counts) {
  const badge = $("#review-badge");
  if (!counts) return;
  badge.textContent = counts.pending;
  badge.hidden = !counts.pending;
}
