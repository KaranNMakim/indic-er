// Playground: paste records in, get entities out, ask "Why?" about any pair.

import { api } from "./api.js";
import { pairCard } from "./pair.js";
import { $, download, el, esc, fixed, parseRecords, toCsv, toast } from "./util.js";

let lastResult = null;
let lastRecords = [];
let onResolved = () => {};

export function initPlayground(hooks = {}) {
  onResolved = hooks.onResolved ?? (() => {});

  const input = $("#records-input");
  const autoMerge = $("#auto-merge");
  const reviewLow = $("#review-low");

  // Thresholds start at whatever the service is actually configured with,
  // so the sliders never lie about the default.
  api.health().then((health) => {
    autoMerge.value = health.thresholds.auto_merge;
    reviewLow.value = health.thresholds.review_low;
    syncSliders();
  }).catch(() => syncSliders());

  const syncSliders = () => {
    $("#auto-merge-out").textContent = Number(autoMerge.value).toFixed(2);
    $("#review-low-out").textContent = Number(reviewLow.value).toFixed(2);
  };
  autoMerge.addEventListener("input", () => {
    if (Number(reviewLow.value) > Number(autoMerge.value)) reviewLow.value = autoMerge.value;
    syncSliders();
  });
  reviewLow.addEventListener("input", () => {
    if (Number(reviewLow.value) > Number(autoMerge.value)) autoMerge.value = reviewLow.value;
    syncSliders();
  });
  syncSliders();

  $("#run-resolve").addEventListener("click", runResolve);
  $("#clear-input").addEventListener("click", () => {
    input.value = "";
    $("#input-error").hidden = true;
  });

  $("#load-showcase").addEventListener("click", async () => {
    const { records } = await api.showcase();
    input.value = toCsv(records);
    toast(`Loaded ${records.length} example records`);
  });

  $("#load-seeded").addEventListener("click", async () => {
    const { entities } = await api.entities({ limit: 80, min_size: 1 });
    const ids = entities.flatMap((e) => e.member_ids).slice(0, 200);
    const rows = [];
    for (const chunk of chunked(ids, 40)) {
      const results = await Promise.all(chunk.map((id) => api.entity(id).catch(() => null)));
      for (const result of results) {
        if (!result) continue;
        for (const member of result.members) {
          if (!ids.includes(member.record_id)) continue;
          if (rows.some((r) => r.id === member.record_id)) continue;
          rows.push({
            id: member.record_id,
            name: member.raw_name,
            kind: member.kind,
            address: member.address?.raw ?? "",
            phone: member.phone_raw ?? "",
            source: member.source,
          });
        }
      }
    }
    input.value = toCsv(rows);
    toast(`Loaded ${rows.length} records from the seeded dataset`);
  });

  $("#entity-filter").addEventListener("input", (event) => {
    renderEntities(lastResult, event.target.value.trim().toLowerCase());
  });

  $("#download-result").addEventListener("click", () => {
    download("indic-er-result.json", JSON.stringify(lastResult, null, 2));
  });

  setupDrop(input);
}

function chunked(items, size) {
  const out = [];
  for (let i = 0; i < items.length; i += size) out.push(items.slice(i, i + size));
  return out;
}

function setupDrop(input) {
  const panel = input.closest(".panel");
  const stop = (event) => { event.preventDefault(); event.stopPropagation(); };
  ["dragenter", "dragover"].forEach((type) =>
    panel.addEventListener(type, (event) => { stop(event); panel.classList.add("dropping"); })
  );
  ["dragleave", "drop"].forEach((type) =>
    panel.addEventListener(type, (event) => { stop(event); panel.classList.remove("dropping"); })
  );
  panel.addEventListener("drop", async (event) => {
    const file = event.dataTransfer?.files?.[0];
    if (!file) return;
    input.value = await file.text();
    toast(`Loaded ${file.name}`);
  });
}

async function runResolve() {
  const button = $("#run-resolve");
  const errorBox = $("#input-error");
  errorBox.hidden = true;

  let records;
  try {
    records = parseRecords($("#records-input").value);
  } catch (error) {
    errorBox.textContent = error.message;
    errorBox.hidden = false;
    return;
  }

  button.disabled = true;
  button.innerHTML = '<span class="spinner"></span> Resolving…';
  try {
    const result = await api.resolve({
      records,
      include_normalized: true,
      apply_reviews: $("#apply-reviews").checked,
      enqueue_review: $("#enqueue-review").checked,
      auto_merge: Number($("#auto-merge").value),
      review_low: Number($("#review-low").value),
    });
    lastResult = result;
    lastRecords = records;
    renderSummary(result);
    renderEntities(result, $("#entity-filter").value.trim().toLowerCase());
    $("#download-result").hidden = false;
    if (result.enqueued) toast(`${result.enqueued} pairs added to the review queue`);
    onResolved(result);
  } catch (error) {
    errorBox.textContent = error.message;
    errorBox.hidden = false;
  } finally {
    button.disabled = false;
    button.textContent = "Resolve";
  }
}

function renderSummary(result) {
  const stats = result.stats;
  const clustering = stats.clustering;
  const box = $("#resolve-summary");
  box.hidden = false;
  box.innerHTML = `
    ${stat(stats.input_records, "records in")}
    ${stat(clustering.entities, "entities out", "good")}
    ${stat(`${Math.round(stats.duplicate_rate * 100)}%`, "duplicate rate", "good")}
    ${stat(clustering.merged_entities, "merged")}
    ${stat(stats.pairs_review, "to review", stats.pairs_review ? "warn" : "")}
    ${stat(clustering.blocked_merges, "merges refused", clustering.blocked_merges ? "warn" : "")}
    ${stat(`${stats.timings_ms.total.toFixed(0)}ms`, "elapsed")}
  `;
}

const stat = (value, label, tone = "") =>
  `<div class="stat ${tone}"><b>${esc(value)}</b><span>${esc(label)}</span></div>`;

function renderEntities(result, filter = "") {
  const list = $("#entity-list");
  if (!result) return;

  let entities = result.entities;
  if (filter) {
    entities = entities.filter((entity) =>
      (entity.canonical_name + " " + entity.variants.join(" ") + " " + entity.member_ids.join(" "))
        .toLowerCase()
        .includes(filter)
    );
  }

  if (!entities.length) {
    list.innerHTML = `<div class="empty"><p>No entities match <strong>${esc(filter)}</strong>.</p></div>`;
    return;
  }

  list.innerHTML = "";
  for (const entity of entities) {
    list.append(entityCard(entity, result));
  }
}

function entityCard(entity, result) {
  const stateClass =
    entity.review_state === "human_confirmed" ? "confirmed"
    : entity.review_state === "conflict_separated" ? "separated"
    : entity.size > 1 ? "merged" : "single";

  const scripts = entity.scripts.filter((s) => s !== "latin");
  const node = el(`
    <article class="entity ${stateClass}">
      <header class="entity-head" role="button" tabindex="0" aria-expanded="false"
              aria-label="${esc(entity.canonical_name)}, ${entity.size} record${entity.size === 1 ? "" : "s"}">
        <span class="caret" aria-hidden="true">▸</span>
        <div>
          <div class="entity-name">${esc(entity.canonical_name)}</div>
          <div class="entity-meta">
            <span>${entity.size} record${entity.size === 1 ? "" : "s"}</span>
            ${entity.canonical_address?.pincode ? `<span>📮 ${esc(entity.canonical_address.pincode)}</span>` : ""}
            ${entity.canonical_phone ? `<span>☎ ${esc(entity.canonical_phone)}</span>` : ""}
            <span>${esc(entity.sources.join(", "))}</span>
          </div>
        </div>
        <div class="right">
          ${scripts.map((s) => `<span class="chip script">${esc(s)}</span>`).join("")}
          ${entity.size > 1 ? `<span class="chip merge">${fixed(entity.cluster_score, 2)}</span>` : ""}
          ${entity.review_state !== "auto" ? `<span class="chip review">${esc(entity.review_state.replace("_", " "))}</span>` : ""}
        </div>
      </header>
      <div class="entity-body"></div>
    </article>`);

  const head = node.querySelector(".entity-head");
  const body = node.querySelector(".entity-body");

  const toggle = () => {
    const open = node.classList.toggle("open");
    head.setAttribute("aria-expanded", String(open));
    if (open && !body.dataset.filled) {
      fillEntityBody(body, entity, result);
      body.dataset.filled = "1";
    }
  };

  head.addEventListener("click", toggle);
  // A div with a click handler is invisible to a keyboard and to a screen
  // reader. role/tabindex/aria-expanded above plus this handler make the
  // disclosure behave like the button it is.
  head.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      toggle();
    }
  });
  return node;
}

function fillEntityBody(body, entity, result) {
  const normalized = result.normalized || {};

  body.innerHTML = entity.member_ids
    .map((id) => {
      const record = normalized[id];
      if (!record) return `<div class="member"><span class="member-id">${esc(id)}</span></div>`;
      const name = record.name;
      const removed = [...(name.honorifics_removed || []), ...(name.suffixes_removed || [])];
      return `
        <div class="member">
          <span class="member-id">${esc(id)}<br>${esc(record.source)}</span>
          <div>
            <div class="member-name">${esc(record.raw_name)}</div>
            ${record.address ? `<div class="member-detail">${esc(record.address.raw)}</div>` : ""}
            <div class="trace">
              ${name.script !== "latin" ? `${esc(name.script)} → ${esc(name.transliterated)} · ` : ""}
              ${removed.length ? `${removed.map((r) => `<del>${esc(r)}</del>`).join(" ")} · ` : ""}
              ${name.legal_suffix ? `<del>${esc(name.legal_suffix)}</del> · ` : ""}
              key ${esc(name.phonetic_key)}
              ${record.phone_last10 ? ` · ☎ ${esc(record.phone_last10)}` : ""}
            </div>
          </div>
        </div>`;
    })
    .join("");

  if (entity.split_reason) {
    body.append(el(`<div class="split-reason"><strong>Kept separate:</strong> ${esc(entity.split_reason)}</div>`));
  }

  if (entity.merge_evidence?.length) {
    const wrap = el(`<div class="evidence"><h4>Merge evidence</h4></div>`);
    for (const evidence of entity.merge_evidence) {
      const row = el(`
        <div class="evidence-row">
          <span class="chip merge">${fixed(evidence.score, 3)}</span>
          <span class="mono">${esc(evidence.left_id)} ~ ${esc(evidence.right_id)}</span>
          <span class="muted">${evidence.top_features.map((f) => `${esc(f.name)} ${Math.round(f.value * 100)}%`).join(", ")}</span>
        </div>`);
      row.append(whyButton(evidence.left_id, evidence.right_id, result, row));
      wrap.append(row);
    }
    body.append(wrap);
  }

  // Pairs involving this entity that were NOT merged are often the most
  // interesting thing on screen: they are the near-misses.
  const nearMisses = (result.review_pairs || []).filter(
    (pair) => entity.member_ids.includes(pair.left_id) || entity.member_ids.includes(pair.right_id)
  );
  if (nearMisses.length) {
    const wrap = el(`<div class="evidence"><h4>Near misses (held for review)</h4></div>`);
    for (const pair of nearMisses.slice(0, 6)) {
      const row = el(`
        <div class="evidence-row">
          <span class="chip review">${fixed(pair.score, 3)}</span>
          <span class="mono">${esc(pair.left_id)} ~ ${esc(pair.right_id)}</span>
          <span class="muted">${esc((pair.gate || "").slice(0, 70))}</span>
        </div>`);
      row.append(whyButton(pair.left_id, pair.right_id, result, row));
      wrap.append(row);
    }
    body.append(wrap);
  }
}

function whyButton(leftId, rightId, result, row) {
  const button = el(`<button class="why-btn">Why?</button>`);
  button.addEventListener("click", async () => {
    const existing = row.parentElement.querySelector(`[data-pair="${leftId}::${rightId}"]`);
    if (existing) { existing.remove(); return; }

    button.disabled = true;
    try {
      const left = toInput(result.normalized[leftId], leftId);
      const right = toInput(result.normalized[rightId], rightId);
      const data = await api.match(left, right, { explain_with_llm: true });
      const card = pairCard(data);
      card.dataset.pair = `${leftId}::${rightId}`;
      if (data.llm_explanation) {
        card.append(
          el(`<p class="hint"><strong>LLM (${esc(data.llm_explanation.mode)}, advisory only):</strong>
              ${esc(data.llm_explanation.text)}</p>`)
        );
      }
      row.after(card);
    } catch (error) {
      toast(error.message);
    } finally {
      button.disabled = false;
    }
  });
  return button;
}

/** Turn a normalized record back into an input record for /match. */
export function toInput(record, fallbackId) {
  if (!record) return { id: fallbackId, name: fallbackId };
  return {
    id: record.record_id,
    name: record.raw_name,
    kind: record.kind,
    ...(record.address?.raw ? { address: record.address.raw } : {}),
    ...(record.phone_raw ? { phone: record.phone_raw } : {}),
    ...(record.source ? { source: record.source } : {}),
  };
}

export const getLastResult = () => lastResult;
export const getLastRecords = () => lastRecords;
