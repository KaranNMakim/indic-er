// The "Why?" card: a pair, its verdict, and the per-feature contributions
// that produced it. Shared by the playground and the review queue, because
// the reviewer and the analyst need to see exactly the same evidence.

import { DECISION_LABEL, esc, fixed, highlightName, pct } from "./util.js";

/**
 * @param {object} data  /match-shaped: {pair, explanation, features, highlights, left, right}
 * @param {object} opts  {actions: HTMLElement[]}
 */
export function pairCard(data, opts = {}) {
  const { pair, explanation, features, highlights, left, right } = data;
  const card = document.createElement("div");
  card.className = "pair-card";

  card.innerHTML = `
    <div class="pair-sides">
      ${side("Left", left, highlights?.left)}
      ${side("Right", right, highlights?.right)}
    </div>

    <div class="verdict ${esc(pair.decision)}">
      <span class="score-big">${fixed(pair.score, 3)}</span>
      <span class="chip ${esc(pair.decision)}">${esc(DECISION_LABEL[pair.decision] ?? pair.decision)}</span>
      ${pair.override ? `<span class="chip">${esc(pair.override.replace("human_", "reviewer "))}</span>` : ""}
      ${(pair.blocked_by || []).map((b) => `<span class="chip">blocked by ${esc(b)}</span>`).join("")}
    </div>

    <p class="explanation">${esc(explanation)}</p>
    ${pair.gate ? `<div class="gate-note"><strong>Held for review:</strong> ${esc(pair.gate)}</div>` : ""}

    ${featureTable(features)}
  `;

  if (opts.actions?.length) {
    const row = document.createElement("div");
    row.className = "row";
    opts.actions.forEach((node) => row.append(node));
    card.append(row);
  }
  return card;
}

function side(label, record, matched) {
  if (!record) return `<div class="side"><h4>${esc(label)}</h4><p class="muted">—</p></div>`;
  const name = record.name || {};
  const address = record.address;
  return `
    <div class="side">
      <h4>${esc(label)} · ${esc(record.record_id ?? "")}</h4>
      <div class="big">${highlightName(record.raw_name ?? "", matched)}</div>
      ${
        name.script && name.script !== "latin"
          ? `<div class="trace">${esc(name.script)} → ${esc(name.transliterated ?? "")}</div>`
          : ""
      }
      ${nameTrace(name)}
      ${address ? `<div class="addr">${esc(address.raw ?? "")}</div>` : `<div class="addr muted">no address</div>`}
      <div class="trace">
        ${record.phone_last10 ? `☎ ${esc(record.phone_last10)}` : "no phone"}
        ${address?.pincode ? ` · 📮 ${esc(address.pincode)}` : ""}
        ${record.source ? ` · ${esc(record.source)}` : ""}
      </div>
    </div>`;
}

/** Show what normalization removed — the audit trail, in the UI. */
function nameTrace(name) {
  const removed = [...(name.honorifics_removed || []), ...(name.suffixes_removed || [])];
  const bits = [];
  if (removed.length) bits.push(removed.map((r) => `<del>${esc(r)}</del>`).join(" "));
  if (name.legal_suffix) bits.push(`<del>${esc(name.legal_suffix)}</del>`);
  if (name.phonetic_key) bits.push(`key ${esc(name.phonetic_key)}`);
  return bits.length ? `<div class="trace">${bits.join(" · ")}</div>` : "";
}

function featureTable(features) {
  if (!features?.length) return "";
  const maxContribution = Math.max(...features.map((f) => f.contribution || 0), 0.001);
  const rows = features
    .map((f) => {
      const width = Math.round(((f.contribution || 0) / maxContribution) * 100);
      const tone = f.value == null ? "" : f.value >= 0.75 ? "hi" : f.value <= 0.35 ? "lo" : "";
      return `
      <tr class="${f.computed ? "" : "off"}">
        <td>${esc(f.label)}</td>
        <td class="num">${f.computed ? pct(f.value) : "n/a"}</td>
        <td><div class="bar ${tone}"><i style="width:${f.computed ? width : 0}%"></i></div></td>
        <td class="num">${f.weight.toFixed(2)}</td>
        <td class="num">${f.computed ? (f.contribution ?? 0).toFixed(3) : "—"}</td>
        <td class="muted">${esc(f.detail)}</td>
      </tr>`;
    })
    .join("");

  return `
    <table class="feature-table">
      <thead>
        <tr><th>feature</th><th>value</th><th>contribution</th><th>weight</th><th>×</th><th>why</th></tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
    <p class="hint" style="margin-top:8px">
      Features marked <em>n/a</em> could not be computed and were dropped from
      both sides of the weighted mean — a missing field is never scored as a
      disagreement.
    </p>`;
}
